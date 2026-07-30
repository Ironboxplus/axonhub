package emulation_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestControllerContinuesAnthropicPauseTurnForResponsesOverRealHTTP(t *testing.T) {
	t.Parallel()
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type string `json:"type"`
			} `json:"tools"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Type != "web_search_20250305" {
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = io.WriteString(writer, `{
				"id":"msg_pause_1","type":"message","role":"assistant","model":"fixture-model",
				"content":[
					{"type":"server_tool_use","id":"srv_pause_1","name":"web_search","input":{"query":"Axon pause turn"}},
					{"type":"web_search_tool_result","tool_use_id":"srv_pause_1","content":[{"type":"web_search_result","url":"https://example.invalid/axon","title":"Axon","encrypted_content":"opaque"}]}
				],
				"stop_reason":"pause_turn","usage":{"input_tokens":5,"output_tokens":2}
			}`)
			return
		}
		if round != 2 || len(payload.Messages) < 2 {
			http.Error(writer, "unexpected continuation", http.StatusBadRequest)
			return
		}
		last := payload.Messages[len(payload.Messages)-1]
		if last.Role != "assistant" ||
			!bytes.Contains(last.Content, []byte(`"type":"server_tool_use"`)) ||
			!bytes.Contains(last.Content, []byte(`"id":"srv_pause_1"`)) ||
			!bytes.Contains(last.Content, []byte(`"type":"web_search_tool_result"`)) ||
			!bytes.Contains(last.Content, []byte(`"encrypted_content":"opaque"`)) {
			http.Error(writer, "pause content not preserved", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(writer, `{
			"id":"msg_pause_2","type":"message","role":"assistant","model":"fixture-model",
			"content":[{"type":"text","text":"The continuation completed."}],
			"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":3}
		}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{ContinuePauseTurns: true, MaxRounds: 4})
	require.NoError(t, err)
	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","input":"search for Axon","tools":[{"type":"web_search"}]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, rounds.Load())
	var response struct {
		Status string `json:"status"`
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
			TotalTokens  int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &response), string(result.Response.Body))
	require.Equal(t, "completed", response.Status)
	require.Len(t, response.Output, 2)
	require.Equal(t, "web_search_call", response.Output[0].Type)
	require.Equal(t, "message", response.Output[1].Type)
	require.Equal(t, "The continuation completed.", response.Output[1].Content[0].Text)
	require.EqualValues(t, 12, response.Usage.InputTokens)
	require.EqualValues(t, 5, response.Usage.OutputTokens)
	require.EqualValues(t, 17, response.Usage.TotalTokens)
}

func TestControllerStreamsAnthropicPauseTurnAsOneResponsesLifecycleOverRealHTTP(t *testing.T) {
	t.Parallel()
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Type string `json:"type"`
			} `json:"tools"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || !payload.Stream || len(payload.Tools) != 1 || payload.Tools[0].Type != "web_search_20250305" {
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		if round == 2 {
			if len(payload.Messages) < 2 {
				http.Error(writer, "history", http.StatusBadRequest)
				return
			}
			last := payload.Messages[len(payload.Messages)-1]
			if last.Role != "assistant" || !bytes.Contains(last.Content, []byte(`"id":"srv_stream_pause"`)) ||
				!bytes.Contains(last.Content, []byte(`"encrypted_content":"opaque-stream"`)) {
				http.Error(writer, "pause content not preserved", http.StatusBadRequest)
				return
			}
		} else if round != 1 {
			http.Error(writer, "too many rounds", http.StatusBadRequest)
			return
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			http.Error(writer, "no flusher", http.StatusInternalServerError)
			return
		}
		write := func(event string, value any) bool {
			encoded, marshalErr := json.Marshal(value)
			if marshalErr != nil {
				return false
			}
			if _, writeErr := fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", event, encoded); writeErr != nil {
				return false
			}
			flusher.Flush()
			return true
		}
		if round == 1 {
			if !write("message_start", map[string]any{"type": "message_start", "message": map[string]any{
				"id": "msg_stream_pause_1", "type": "message", "role": "assistant", "model": "fixture-model",
				"content": []any{}, "usage": map[string]any{"input_tokens": 5, "output_tokens": 0},
			}}) ||
				!write("content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{
					"type": "server_tool_use", "id": "srv_stream_pause", "name": "web_search", "input": map[string]any{"query": "Axon stream pause"},
				}}) ||
				!write("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}) ||
				!write("content_block_start", map[string]any{"type": "content_block_start", "index": 1, "content_block": map[string]any{
					"type": "web_search_tool_result", "tool_use_id": "srv_stream_pause", "content": []any{map[string]any{
						"type": "web_search_result", "url": "https://example.invalid/stream", "title": "Axon", "encrypted_content": "opaque-stream",
					}},
				}}) ||
				!write("content_block_stop", map[string]any{"type": "content_block_stop", "index": 1}) ||
				!write("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "pause_turn"}, "usage": map[string]any{"output_tokens": 2}}) ||
				!write("message_stop", map[string]any{"type": "message_stop"}) {
				return
			}
			return
		}
		if !write("message_start", map[string]any{"type": "message_start", "message": map[string]any{
			"id": "msg_stream_pause_2", "type": "message", "role": "assistant", "model": "fixture-model",
			"content": []any{}, "usage": map[string]any{"input_tokens": 7, "output_tokens": 0},
		}}) ||
			!write("content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""}}) ||
			!write("content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "Streaming continuation completed."}}) ||
			!write("content_block_stop", map[string]any{"type": "content_block_stop", "index": 0}) ||
			!write("message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "end_turn"}, "usage": map[string]any{"output_tokens": 3}}) ||
			!write("message_stop", map[string]any{"type": "message_stop"}) {
			return
		}
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{ContinuePauseTurns: true, MaxRounds: 4})
	require.NoError(t, err)
	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","stream":true,"input":"search for Axon","tools":[{"type":"web_search"}]}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	defer result.EventStream.Close()
	var wire bytes.Buffer
	created, completed, incomplete := 0, 0, 0
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		switch event.Type {
		case "response.created":
			created++
		case "response.completed":
			completed++
		case "response.incomplete":
			incomplete++
		}
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.EqualValues(t, 2, rounds.Load())
	require.Equal(t, 1, created)
	require.Equal(t, 1, completed)
	require.Zero(t, incomplete)
	require.Contains(t, wire.String(), `"type":"web_search_call"`)
	require.Contains(t, wire.String(), "Streaming continuation completed.")
	require.Contains(t, wire.String(), `"total_tokens":17`)
}

func TestControllerPreservesPauseTurnForAnthropicClientOverRealHTTP(t *testing.T) {
	t.Parallel()
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		rounds.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{
			"id":"msg_native_pause","type":"message","role":"assistant","model":"fixture-model",
			"content":[{"type":"text","text":"Provider-managed work is paused."}],
			"stop_reason":"pause_turn","usage":{"input_tokens":4,"output_tokens":2}
		}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{ContinuePauseTurns: true, MaxRounds: 4})
	require.NoError(t, err)
	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).Pipeline(
		anthropic.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":"search for Axon"}],
			"tools":[{"type":"web_search_20250305","name":"web_search"}]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, rounds.Load(), "an Anthropic client owns its native pause_turn continuation")
	var response struct {
		StopReason string `json:"stop_reason"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &response), string(result.Response.Body))
	require.Equal(t, "pause_turn", response.StopReason)
}

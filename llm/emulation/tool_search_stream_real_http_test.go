package emulation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

func TestControllerStreamsAnthropicServerToolSearchThroughChatRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRounds atomic.Int64
	var searchName string
	providerErrors := make(chan error, 2)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			providerErrors <- err
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil || !payload.Stream {
			providerErrors <- fmt.Errorf("unexpected stream request: %s", body)
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		names := make([]string, 0, len(payload.Tools))
		for _, tool := range payload.Tools {
			names = append(names, tool.Function.Name)
			if strings.HasPrefix(tool.Function.Name, "axh_") {
				searchName = tool.Function.Name
			}
		}
		if round == 1 {
			if searchName == "" || len(names) != 2 || !slices.Contains(names, "always_available") ||
				slices.Contains(names, "create_event") || slices.Contains(names, "delete_event") {
				providerErrors <- fmt.Errorf("initial catalog leaked: %v", names)
				http.Error(writer, "catalog", http.StatusBadRequest)
				return
			}
		} else if round == 2 {
			if len(names) != 3 || !slices.Contains(names, searchName) || !slices.Contains(names, "always_available") ||
				!slices.Contains(names, "create_event") || slices.Contains(names, "delete_event") ||
				!strings.Contains(string(payload.Messages), "create_event") {
				providerErrors <- fmt.Errorf("discovered catalog missing: tools=%v messages=%s", names, payload.Messages)
				http.Error(writer, "catalog", http.StatusBadRequest)
				return
			}
		} else {
			providerErrors <- fmt.Errorf("too many rounds: %d", round)
			http.Error(writer, "rounds", http.StatusBadRequest)
			return
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			providerErrors <- fmt.Errorf("response writer has no flusher")
			return
		}
		writeFrame := func(value any) bool {
			encoded, marshalErr := json.Marshal(value)
			if marshalErr != nil {
				providerErrors <- marshalErr
				return false
			}
			if _, writeErr := fmt.Fprintf(writer, "data: %s\n\n", encoded); writeErr != nil {
				providerErrors <- writeErr
				return false
			}
			flusher.Flush()
			return true
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": fmt.Sprintf("chatcmpl-tool-search-stream-%d", round), "object": "chat.completion.chunk",
				"created": 1785432000 + round, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		toolName, callID, arguments := searchName, "search_stream_1", `{"query":"create"}`
		if round == 2 {
			toolName, callID, arguments = "create_event", "create_stream_1", `{"title":"Architecture review"}`
		}
		split := len(arguments) / 2
		if !writeFrame(chunk(map[string]any{"role": "assistant"}, nil)) ||
			!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
				"index": 0, "id": callID, "type": "function",
				"function": map[string]any{"name": toolName, "arguments": arguments[:split]},
			}}}, nil)) ||
			!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
				"index": 0, "function": map[string]any{"arguments": arguments[split:]},
			}}}, nil)) ||
			!writeFrame(chunk(map[string]any{}, "tool_calls")) ||
			!writeFrame(map[string]any{
				"id": fmt.Sprintf("chatcmpl-tool-search-stream-%d", round), "object": "chat.completion.chunk", "model": "fixture-model",
				"choices": []any{}, "usage": map[string]any{"prompt_tokens": 4 + round, "completion_tokens": 2, "total_tokens": 6 + round},
			}) {
			return
		}
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
		providerErrors <- nil
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted:    hosted.Config{SyntheticNameKey: []byte(strings.Repeat("stream-tool-search-key-", 2))},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "fixture-provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		anthropic.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","max_tokens":256,"stream":true,
			"messages":[{"role":"user","content":"create a calendar event"}],
			"tools":[
				{"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_bm25"},
				{"name":"always_available","description":"Always available","input_schema":{"type":"object"}},
				{"name":"create_event","description":"Create a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}},
				{"name":"delete_event","description":"Delete a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}
			]
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	defer result.EventStream.Close()
	var wire strings.Builder
	var eventTypes []string
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		eventTypes = append(eventTypes, event.Type)
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.NoError(t, <-providerErrors)
	require.NoError(t, <-providerErrors)
	require.EqualValues(t, 2, providerRounds.Load())
	encoded := wire.String()
	require.Contains(t, encoded, `"type":"server_tool_use"`)
	require.Contains(t, encoded, `"name":"tool_search_tool_bm25"`)
	require.Contains(t, encoded, `"type":"tool_search_tool_result"`)
	require.Contains(t, encoded, `"tool_name":"create_event"`)
	require.Contains(t, encoded, `"type":"tool_use"`)
	require.Contains(t, encoded, `"name":"create_event"`)
	require.NotContains(t, encoded, searchName)
	require.Equal(t, 1, countString(eventTypes, "message_start"))
	require.Equal(t, 1, countString(eventTypes, "message_stop"))
}

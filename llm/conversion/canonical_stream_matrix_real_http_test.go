package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/auth"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestResponsesWebSocketProviderUsesCanonicalParallelToolConversion(t *testing.T) {
	upgrader := websocket.Upgrader{}
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		connection, err := upgrader.Upgrade(writer, request, nil)
		if err != nil {
			t.Errorf("upgrade Responses WebSocket: %v", err)
			return
		}
		defer connection.Close()
		var create map[string]any
		if err := connection.ReadJSON(&create); err != nil {
			t.Errorf("read response.create: %v", err)
			return
		}
		if create["type"] != "response.create" {
			t.Errorf("WebSocket request type = %v", create["type"])
			return
		}
		wireEvents := []map[string]any{
			{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_ws_matrix", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}}},
			{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": map[string]any{"id": "fc_ws_a", "type": "function_call", "status": "in_progress", "call_id": "call_ws_a", "name": "lookup", "arguments": ""}},
			{"type": "response.output_item.added", "sequence_number": 2, "output_index": 1, "item": map[string]any{"id": "fc_ws_b", "type": "function_call", "status": "in_progress", "call_id": "call_ws_b", "name": "calculate", "arguments": ""}},
			{"type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 0, "item_id": "fc_ws_a", "delta": `{"q":"hel`},
			{"type": "response.function_call_arguments.delta", "sequence_number": 4, "output_index": 1, "item_id": "fc_ws_b", "delta": `{"x":1`},
			{"type": "response.function_call_arguments.delta", "sequence_number": 5, "output_index": 0, "item_id": "fc_ws_a", "delta": `lo"}`},
			{"type": "response.function_call_arguments.done", "sequence_number": 6, "output_index": 0, "item_id": "fc_ws_a", "call_id": "call_ws_a", "name": "lookup", "arguments": `{"q":"hello"}`},
			{"type": "response.output_item.done", "sequence_number": 7, "output_index": 0, "item": map[string]any{"id": "fc_ws_a", "type": "function_call", "status": "completed", "call_id": "call_ws_a", "name": "lookup", "arguments": `{"q":"hello"}`}},
			{"type": "response.function_call_arguments.delta", "sequence_number": 8, "output_index": 1, "item_id": "fc_ws_b", "delta": `}`},
			{"type": "response.function_call_arguments.done", "sequence_number": 9, "output_index": 1, "item_id": "fc_ws_b", "call_id": "call_ws_b", "name": "calculate", "arguments": `{"x":1}`},
			{"type": "response.output_item.done", "sequence_number": 10, "output_index": 1, "item": map[string]any{"id": "fc_ws_b", "type": "function_call", "status": "completed", "call_id": "call_ws_b", "name": "calculate", "arguments": `{"x":1}`}},
			{"type": "response.completed", "sequence_number": 11, "response": map[string]any{"id": "resp_ws_matrix", "object": "response", "model": "matrix-model", "status": "completed", "output": []any{}, "usage": map[string]any{"input_tokens": 5, "output_tokens": 4, "total_tokens": 9}}},
		}
		for _, event := range wireEvents {
			if err := connection.WriteJSON(event); err != nil {
				return
			}
		}
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformerWithConfig(&responses.Config{
		BaseURL: provider.URL, APIKeyProvider: auth.NewStaticKeyProvider("fixture-key"), Transport: responses.TransportWebSocket,
	})
	if err != nil {
		t.Fatalf("create Responses WebSocket outbound: %v", err)
	}
	t.Cleanup(outbound.Stop)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"matrix-model","stream":true,"messages":[{"role":"user","content":"run both"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},{"type":"function","function":{"name":"calculate","parameters":{"type":"object"}}}]}`),
		})
	if err != nil {
		t.Fatalf("start Responses WebSocket conversion: %v", err)
	}
	defer result.EventStream.Close()
	arguments := map[string]*strings.Builder{"call_ws_a": {}, "call_ws_b": {}}
	indexCalls := map[int]string{}
	finish, done := "", false
	for result.EventStream.Next() {
		raw := result.EventStream.Current().Data
		if bytes.Equal(raw, []byte("[DONE]")) {
			done = true
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
				Delta        struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(raw, &chunk); err != nil {
			t.Fatalf("decode WebSocket-converted Chat chunk: %v body=%s", err, raw)
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				finish = *choice.FinishReason
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.ID != "" {
					indexCalls[call.Index] = call.ID
				}
				if call.Function.Arguments != "" {
					arguments[indexCalls[call.Index]].WriteString(call.Function.Arguments)
				}
			}
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume WebSocket-converted Chat stream: %v", err)
	}
	if arguments["call_ws_a"].String() != `{"q":"hello"}` || arguments["call_ws_b"].String() != `{"x":1}` || finish != "tool_calls" || !done {
		t.Fatalf("WebSocket canonical conversion degraded: a=%q b=%q finish=%q done=%v", arguments["call_ws_a"].String(), arguments["call_ws_b"].String(), finish, done)
	}
}

func TestTruncatedProviderStreamsBecomeLegalIncompleteTerminals3x3(t *testing.T) {
	fragmentProfiles := []struct {
		name    string
		pattern []int
	}{
		{name: "1_byte", pattern: []int{1}},
		{name: "7_bytes", pattern: []int{7}},
		{name: "31_bytes", pattern: []int{31}},
		{name: "127_bytes", pattern: []int{127}},
		{name: "1024_bytes", pattern: []int{1024}},
		{name: "mixed", pattern: []int{1, 7, 31, 127, 1024}},
	}
	providers := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		write       func(*testing.T, http.ResponseWriter)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: func(t *testing.T, writer http.ResponseWriter) {
				writeNamedSSEJSON(t, writer, "", map[string]any{
					"id": "chat_truncated", "object": "chat.completion.chunk", "model": "matrix-model",
					"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant", "content": "partial"}, "finish_reason": nil}},
				})
			},
		},
		{
			name: "responses", path: "/v1/responses",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: func(t *testing.T, writer http.ResponseWriter) {
				writeNamedSSEJSON(t, writer, "response.created", map[string]any{
					"type": "response.created", "sequence_number": 0,
					"response": map[string]any{"id": "resp_truncated", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
				})
				writeNamedSSEJSON(t, writer, "response.output_item.added", map[string]any{
					"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
					"item": map[string]any{"id": "msg_truncated", "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}},
				})
				writeNamedSSEJSON(t, writer, "response.output_text.delta", map[string]any{
					"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0,
					"item_id": "msg_truncated", "content_index": 0, "delta": "partial",
				})
			},
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: func(t *testing.T, writer http.ResponseWriter) {
				writeNamedSSEJSON(t, writer, "message_start", map[string]any{
					"type": "message_start", "message": map[string]any{
						"id": "msg_truncated", "type": "message", "role": "assistant", "model": "matrix-model",
						"content": []any{}, "usage": map[string]any{"input_tokens": 4, "output_tokens": 0},
					},
				})
				writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
					"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "text", "text": ""},
				})
				writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
					"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "text_delta", "text": "partial"},
				})
			},
		},
	}
	clients := []struct {
		name       string
		path       string
		body       string
		newInbound func() transformer.Inbound
	}{
		{name: "chat", path: "/v1/chat/completions", body: `{"model":"matrix-model","stream":true,"messages":[{"role":"user","content":"go"}]}`, newInbound: func() transformer.Inbound { return openai.NewInboundTransformer() }},
		{name: "responses", path: "/v1/responses", body: `{"model":"matrix-model","stream":true,"input":"go"}`, newInbound: func() transformer.Inbound { return responses.NewInboundTransformer() }},
		{name: "anthropic", path: "/v1/messages", body: `{"model":"matrix-model","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"go"}]}`, newInbound: func() transformer.Inbound { return anthropic.NewInboundTransformer() }},
	}

	for _, providerCase := range providers {
		for _, clientCase := range clients {
			for _, fragmentProfile := range fragmentProfiles {
				providerCase, clientCase, fragmentProfile := providerCase, clientCase, fragmentProfile
				t.Run(providerCase.name+"_to_"+clientCase.name+"_"+fragmentProfile.name, func(t *testing.T) {
					provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
						if request.URL.Path != providerCase.path {
							http.Error(writer, "unexpected path", http.StatusNotFound)
							return
						}
						writer.Header().Set("Content-Type", "text/event-stream")
						providerCase.write(t, &fragmentingResponseWriter{ResponseWriter: writer, pattern: fragmentProfile.pattern})
					}))
					t.Cleanup(provider.Close)
					outbound, err := providerCase.newOutbound(provider.URL)
					if err != nil {
						t.Fatalf("create %s outbound: %v", providerCase.name, err)
					}
					executor := httpclient.NewHttpClientWithClient(provider.Client())
					t.Cleanup(executor.CloseIdleConnections)
					result, err := pipeline.NewFactory(executor).
						Pipeline(clientCase.newInbound(), conversion.NewOutbound(outbound)).
						Process(context.Background(), &httpclient.Request{
							Method: http.MethodPost, URL: clientCase.path,
							Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(clientCase.body),
						})
					if err != nil {
						t.Fatalf("start truncated %s-to-%s stream at %s: %v", providerCase.name, clientCase.name, fragmentProfile.name, err)
					}
					defer result.EventStream.Close()
					var events [][]byte
					for result.EventStream.Next() {
						events = append(events, append([]byte(nil), result.EventStream.Current().Data...))
					}
					if err := result.EventStream.Err(); err != nil {
						t.Fatalf("truncated %s-to-%s at %s leaked transport error: %v", providerCase.name, clientCase.name, fragmentProfile.name, err)
					}
					assertLegalIncompleteClientStream(t, clientCase.name, events)
				})
			}
		}
	}
}

type fragmentingResponseWriter struct {
	http.ResponseWriter
	pattern []int
	step    int
}

func (writer *fragmentingResponseWriter) Write(payload []byte) (int, error) {
	if len(writer.pattern) == 0 {
		return writer.ResponseWriter.Write(payload)
	}
	written := 0
	for written < len(payload) {
		size := writer.pattern[writer.step%len(writer.pattern)]
		writer.step++
		if size <= 0 {
			size = 1
		}
		end := min(written+size, len(payload))
		count, err := writer.ResponseWriter.Write(payload[written:end])
		written += count
		if flusher, ok := writer.ResponseWriter.(http.Flusher); ok {
			flusher.Flush()
		}
		if err != nil {
			return written, err
		}
		if count == 0 {
			return written, io.ErrShortWrite
		}
	}
	return written, nil
}

func (writer *fragmentingResponseWriter) Flush() {
	if flusher, ok := writer.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

func assertLegalIncompleteClientStream(t *testing.T, client string, events [][]byte) {
	t.Helper()
	var text strings.Builder
	switch client {
	case "chat":
		finish, done := "", false
		for _, raw := range events {
			if bytes.Equal(raw, []byte("[DONE]")) {
				done = true
				continue
			}
			var chunk struct {
				Choices []struct {
					Delta struct {
						Content string `json:"content"`
					} `json:"delta"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(raw, &chunk); err != nil {
				t.Fatalf("decode Chat truncated event: %v body=%s", err, raw)
			}
			for _, choice := range chunk.Choices {
				text.WriteString(choice.Delta.Content)
				if choice.FinishReason != nil {
					finish = *choice.FinishReason
				}
			}
		}
		if text.String() != "partial" || finish != "length" || !done {
			t.Fatalf("Chat incomplete terminal degraded: text=%q finish=%q done=%v events=%q", text.String(), finish, done, events)
		}
	case "responses":
		terminal := ""
		for index, raw := range events {
			var event struct {
				Type     string `json:"type"`
				Sequence int    `json:"sequence_number"`
				Delta    string `json:"delta"`
			}
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Fatalf("decode Responses truncated event: %v body=%s", err, raw)
			}
			if event.Sequence != index {
				t.Fatalf("Responses incomplete sequence[%d]=%d body=%s", index, event.Sequence, raw)
			}
			if event.Type == "response.output_text.delta" {
				text.WriteString(event.Delta)
			}
			if strings.HasPrefix(event.Type, "response.") && (event.Type == "response.incomplete" || event.Type == "response.completed" || event.Type == "response.failed") {
				terminal = event.Type
			}
		}
		if text.String() != "partial" || terminal != "response.incomplete" {
			t.Fatalf("Responses incomplete terminal degraded: text=%q terminal=%q events=%q", text.String(), terminal, events)
		}
	case "anthropic":
		stopReason, stopped := "", false
		for _, raw := range events {
			var event struct {
				Type  string `json:"type"`
				Delta *struct {
					Type       string `json:"type"`
					Text       string `json:"text"`
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Fatalf("decode Anthropic truncated event: %v body=%s", err, raw)
			}
			if event.Delta != nil {
				if event.Delta.Type == "text_delta" {
					text.WriteString(event.Delta.Text)
				}
				if event.Delta.StopReason != "" {
					stopReason = event.Delta.StopReason
				}
			}
			if event.Type == "message_stop" {
				stopped = true
			}
		}
		if text.String() != "partial" || stopReason != "max_tokens" || !stopped {
			t.Fatalf("Anthropic incomplete terminal degraded: text=%q stop_reason=%q stopped=%v events=%q", text.String(), stopReason, stopped, events)
		}
	default:
		t.Fatalf("unknown client %q", client)
	}
}

func TestTruncatedToolInputNeverSignalsExecutableCompletion(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_tool_truncated", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeNamedSSEJSON(t, writer, "response.output_item.added", map[string]any{
			"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
			"item": map[string]any{"id": "fc_truncated", "type": "function_call", "status": "in_progress", "call_id": "call_truncated", "name": "calculate", "arguments": ""},
		})
		writeNamedSSEJSON(t, writer, "response.function_call_arguments.delta", map[string]any{
			"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0,
			"item_id": "fc_truncated", "delta": `{"x":`,
		})
	}))
	t.Cleanup(provider.Close)

	clients := []struct {
		name       string
		path       string
		body       string
		newInbound func() transformer.Inbound
	}{
		{name: "chat", path: "/v1/chat/completions", body: `{"model":"matrix-model","stream":true,"messages":[{"role":"user","content":"calculate"}]}`, newInbound: func() transformer.Inbound { return openai.NewInboundTransformer() }},
		{name: "responses", path: "/v1/responses", body: `{"model":"matrix-model","stream":true,"input":"calculate"}`, newInbound: func() transformer.Inbound { return responses.NewInboundTransformer() }},
		{name: "anthropic", path: "/v1/messages", body: `{"model":"matrix-model","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"calculate"}]}`, newInbound: func() transformer.Inbound { return anthropic.NewInboundTransformer() }},
	}
	for _, clientCase := range clients {
		clientCase := clientCase
		t.Run(clientCase.name, func(t *testing.T) {
			outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
			if err != nil {
				t.Fatalf("create Responses outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(clientCase.newInbound(), conversion.NewOutbound(outbound)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: clientCase.path,
					Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(clientCase.body),
				})
			if err != nil {
				t.Fatalf("start truncated tool stream: %v", err)
			}
			defer result.EventStream.Close()
			var events [][]byte
			for result.EventStream.Next() {
				events = append(events, append([]byte(nil), result.EventStream.Current().Data...))
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("truncated tool stream leaked transport error: %v", err)
			}
			assertLegalIncompleteToolClientStream(t, clientCase.name, events)
		})
	}
}

func assertLegalIncompleteToolClientStream(t *testing.T, client string, events [][]byte) {
	t.Helper()
	var arguments strings.Builder
	switch client {
	case "chat":
		finish, done := "", false
		for _, raw := range events {
			if bytes.Equal(raw, []byte("[DONE]")) {
				done = true
				continue
			}
			var chunk struct {
				Choices []struct {
					FinishReason *string `json:"finish_reason"`
					Delta        struct {
						ToolCalls []struct {
							Function struct {
								Arguments string `json:"arguments"`
							} `json:"function"`
						} `json:"tool_calls"`
					} `json:"delta"`
				} `json:"choices"`
			}
			if err := json.Unmarshal(raw, &chunk); err != nil {
				t.Fatalf("decode Chat truncated tool event: %v body=%s", err, raw)
			}
			for _, choice := range chunk.Choices {
				if choice.FinishReason != nil {
					finish = *choice.FinishReason
				}
				for _, call := range choice.Delta.ToolCalls {
					arguments.WriteString(call.Function.Arguments)
				}
			}
		}
		if arguments.String() != `{"x":` || finish != "length" || !done {
			t.Fatalf("Chat truncated tool became executable: arguments=%q finish=%q done=%v", arguments.String(), finish, done)
		}
	case "responses":
		terminal, status := "", ""
		for _, raw := range events {
			var event struct {
				Type  string `json:"type"`
				Delta string `json:"delta"`
				Item  *struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				} `json:"item"`
			}
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Fatalf("decode Responses truncated tool event: %v body=%s", err, raw)
			}
			if event.Type == "response.function_call_arguments.delta" {
				arguments.WriteString(event.Delta)
			}
			if event.Type == "response.output_item.done" && event.Item != nil && event.Item.Type == "function_call" {
				status = event.Item.Status
			}
			if event.Type == "response.incomplete" || event.Type == "response.completed" {
				terminal = event.Type
			}
		}
		if arguments.String() != `{"x":` || status != "incomplete" || terminal != "response.incomplete" {
			t.Fatalf("Responses truncated tool became executable: arguments=%q status=%q terminal=%q", arguments.String(), status, terminal)
		}
	case "anthropic":
		stopReason, stopped := "", false
		for _, raw := range events {
			var event struct {
				Type  string `json:"type"`
				Delta *struct {
					Type        string `json:"type"`
					PartialJSON string `json:"partial_json"`
					StopReason  string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal(raw, &event); err != nil {
				t.Fatalf("decode Anthropic truncated tool event: %v body=%s", err, raw)
			}
			if event.Delta != nil {
				arguments.WriteString(event.Delta.PartialJSON)
				if event.Delta.StopReason != "" {
					stopReason = event.Delta.StopReason
				}
			}
			if event.Type == "message_stop" {
				stopped = true
			}
		}
		if arguments.String() != `{"x":` || stopReason != "max_tokens" || !stopped {
			t.Fatalf("Anthropic truncated tool became executable: arguments=%q stop_reason=%q stopped=%v", arguments.String(), stopReason, stopped)
		}
	default:
		t.Fatalf("unknown client %q", client)
	}
}

func TestAnthropicProviderToChatClientCanonicalParallelToolsOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/messages" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "message_start", map[string]any{
			"type": "message_start", "message": map[string]any{
				"id": "msg_anthropic_to_chat", "type": "message", "role": "assistant", "model": "matrix-model",
				"content": []any{}, "usage": map[string]any{"input_tokens": 7, "output_tokens": 0},
			},
		})
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "tool_use", "id": "call_a", "name": "lookup", "input": map[string]any{}},
		})
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 1,
			"content_block": map[string]any{"type": "tool_use", "id": "call_b", "name": "calculate", "input": map[string]any{}},
		})
		for _, delta := range []struct {
			index int
			text  string
		}{{0, `{"q":"hel`}, {1, `{"x":1`}, {0, `lo"}`}, {1, `}`}} {
			writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
				"type": "content_block_delta", "index": delta.index,
				"delta": map[string]any{"type": "input_json_delta", "partial_json": delta.text},
			})
		}
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 1})
		writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
			"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use"},
			"usage": map[string]any{"output_tokens": 5},
		})
		writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
	}))
	t.Cleanup(provider.Close)

	target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Anthropic outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"matrix-model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"run both"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},{"type":"function","function":{"name":"calculate","parameters":{"type":"object"}}}]}`),
		})
	if err != nil {
		t.Fatalf("run Anthropic-to-Chat pipeline: %v", err)
	}
	defer result.EventStream.Close()

	arguments := map[string]*strings.Builder{"call_a": {}, "call_b": {}}
	indexes := map[string]int{}
	indexCalls := map[int]string{}
	seenFinish, seenDone := false, false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if bytes.Equal(event.Data, []byte("[DONE]")) {
			seenDone = true
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
				Delta        struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			t.Fatalf("decode Chat event: %v body=%s", err, event.Data)
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				seenFinish = *choice.FinishReason == "tool_calls"
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.ID != "" {
					indexes[call.ID] = call.Index
					indexCalls[call.Index] = call.ID
				}
				if call.Function.Arguments != "" {
					arguments[indexCalls[call.Index]].WriteString(call.Function.Arguments)
				}
			}
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume Chat stream: %v", err)
	}
	if indexes["call_a"] != 0 || indexes["call_b"] != 1 ||
		arguments["call_a"].String() != `{"q":"hello"}` || arguments["call_b"].String() != `{"x":1}` || !seenFinish || !seenDone {
		t.Fatalf("Anthropic-to-Chat lifecycle degraded: indexes=%v a=%q b=%q finish=%v done=%v", indexes, arguments["call_a"].String(), arguments["call_b"].String(), seenFinish, seenDone)
	}
}

func TestChatProviderToAnthropicClientCanonicalRandomTCPSplits(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/chat/completions" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		var sse strings.Builder
		appendChatChunk := func(delta map[string]any, finish any) {
			payload, err := json.Marshal(map[string]any{
				"id": "chat_random_split", "object": "chat.completion.chunk", "model": "matrix-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			})
			if err != nil {
				t.Errorf("marshal Chat chunk: %v", err)
				return
			}
			sse.WriteString("data: ")
			sse.Write(payload)
			sse.WriteString("\n\n")
		}
		appendChatChunk(map[string]any{"role": "assistant", "tool_calls": []any{
			map[string]any{"index": 0, "id": "call_a", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"q":"hel`}},
			map[string]any{"index": 1, "id": "call_b", "type": "function", "function": map[string]any{"name": "calculate", "arguments": `{"x":1`}},
		}}, nil)
		appendChatChunk(map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "function": map[string]any{"arguments": `lo"}`}},
			map[string]any{"index": 1, "function": map[string]any{"arguments": `}`}},
		}}, nil)
		appendChatChunk(map[string]any{}, "tool_calls")
		sse.WriteString("data: [DONE]\n\n")

		payload := []byte(sse.String())
		pattern := []int{1, 7, 2, 13, 3, 5, 11}
		for offset, step := 0, 0; offset < len(payload); step++ {
			size := pattern[step%len(pattern)]
			end := offset + size
			if end > len(payload) {
				end = len(payload)
			}
			if _, err := writer.Write(payload[offset:end]); err != nil {
				return
			}
			offset = end
		}
	}))
	t.Cleanup(provider.Close)

	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"matrix-model","stream":true,"max_tokens":128,"messages":[{"role":"user","content":"run both"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}},{"name":"calculate","input_schema":{"type":"object"}}]}`),
		})
	if err != nil {
		t.Fatalf("run random-split Chat-to-Anthropic pipeline: %v", err)
	}
	defer result.EventStream.Close()

	arguments := map[string]*strings.Builder{"call_a": {}, "call_b": {}}
	blockCallIDs := map[int64]string{}
	seenStop := false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		var wire struct {
			Type         string `json:"type"`
			Index        *int64 `json:"index"`
			ContentBlock *struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content_block"`
			Delta *struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(event.Data, &wire); err != nil {
			t.Fatalf("decode random-split Anthropic event: %v body=%s", err, event.Data)
		}
		if wire.Type == "content_block_start" && wire.ContentBlock != nil && wire.ContentBlock.Type == "tool_use" {
			blockCallIDs[*wire.Index] = wire.ContentBlock.ID
		}
		if wire.Type == "content_block_delta" && wire.Delta != nil && wire.Delta.Type == "input_json_delta" {
			arguments[blockCallIDs[*wire.Index]].WriteString(wire.Delta.PartialJSON)
		}
		if wire.Type == "message_stop" {
			seenStop = true
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume random-split Anthropic stream: %v", err)
	}
	if arguments["call_a"].String() != `{"q":"hello"}` || arguments["call_b"].String() != `{"x":1}` || !seenStop {
		t.Fatalf("random TCP split corrupted lifecycle: blocks=%v a=%q b=%q stop=%v", blockCallIDs, arguments["call_a"].String(), arguments["call_b"].String(), seenStop)
	}
}

func TestResponsesSourceSequenceViolationIsObservableOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 4,
			"response": map[string]any{"id": "resp_bad_sequence", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeNamedSSEJSON(t, writer, "response.in_progress", map[string]any{
			"type": "response.in_progress", "sequence_number": 4,
			"response": map[string]any{"id": "resp_bad_sequence", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
	}))
	t.Cleanup(provider.Close)

	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	observer := &observationRecorder{}
	result, err := pipeline.NewFactory(executor).
		Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).
		Process(llm.WithConversionTrace(context.Background()), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/chat/completions",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"matrix-model","stream":true,"messages":[{"role":"user","content":"fail deterministically"}]}`),
		})
	if err != nil {
		t.Fatalf("start observable sequence test: %v", err)
	}
	defer result.EventStream.Close()
	for result.EventStream.Next() {
	}
	if err := result.EventStream.Err(); err == nil || !strings.Contains(err.Error(), "sequence_number 4 is not after 4") {
		t.Fatalf("stream error = %v", err)
	}

	for _, observation := range observer.snapshot() {
		if observation.Stage != pipeline.StageConversionRestore || observation.Conversion == nil {
			continue
		}
		if observation.Conversion.StreamViolations != 1 ||
			observation.Conversion.LastStreamViolation != string(llm.StreamInvariantSequence) {
			t.Fatalf("conversion violation summary = %#v", observation.Conversion)
		}
		return
	}
	t.Fatal("missing conversion restore observation for sequence violation")
}

func TestResponsesProviderToAnthropicClientCanonicalParallelToolsOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		if !bytes.Contains(body, []byte(`"type":"function"`)) || !bytes.Contains(body, []byte(`"stream":true`)) {
			http.Error(writer, "Anthropic tools did not become a streaming Responses request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_anthropic_matrix", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeNamedSSEJSON(t, writer, "response.in_progress", map[string]any{
			"type": "response.in_progress", "sequence_number": 1,
			"response": map[string]any{"id": "resp_anthropic_matrix", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeResponsesFunctionAdded(t, writer, 2, 0, "item_a", "call_a", "lookup")
		writeResponsesFunctionAdded(t, writer, 3, 1, "item_b", "call_b", "calculate")
		writeResponsesFunctionDelta(t, writer, 4, 0, "item_a", `{"q":"hel`)
		writeResponsesFunctionDelta(t, writer, 5, 1, "item_b", `{"x":1`)
		writeResponsesFunctionDelta(t, writer, 6, 0, "item_a", `lo"}`)
		writeResponsesFunctionDelta(t, writer, 7, 1, "item_b", `}`)
		writeResponsesFunctionDone(t, writer, 8, 0, "item_a", "call_a", "lookup", `{"q":"hello"}`)
		writeResponsesFunctionDone(t, writer, 10, 1, "item_b", "call_b", "calculate", `{"x":1}`)
		writeNamedSSEJSON(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 12,
			"response": map[string]any{
				"id": "resp_anthropic_matrix", "object": "response", "model": "matrix-model", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
			},
		})
	}))
	t.Cleanup(provider.Close)

	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"matrix-model","stream":true,"max_tokens":128,"messages":[{"role":"user","content":"run both"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}},{"name":"calculate","input_schema":{"type":"object"}}]}`),
		})
	if err != nil {
		t.Fatalf("run Responses-to-Anthropic pipeline: %v", err)
	}
	defer result.EventStream.Close()

	arguments := map[string]*strings.Builder{"call_a": {}, "call_b": {}}
	blockCallIDs := map[int64]string{}
	seenMessageStart, seenToolStop, seenMessageStop := false, false, false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		var wire struct {
			Type         string `json:"type"`
			Index        *int64 `json:"index"`
			ContentBlock *struct {
				Type string `json:"type"`
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"content_block"`
			Delta *struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
				StopReason  string `json:"stop_reason"`
			} `json:"delta"`
		}
		if err := json.Unmarshal(event.Data, &wire); err != nil {
			t.Fatalf("decode Anthropic SSE: %v body=%s", err, event.Data)
		}
		switch wire.Type {
		case "message_start":
			seenMessageStart = true
		case "content_block_start":
			if wire.ContentBlock != nil && wire.ContentBlock.Type == "tool_use" {
				blockCallIDs[*wire.Index] = wire.ContentBlock.ID
			}
		case "content_block_delta":
			if wire.Delta != nil && wire.Delta.Type == "input_json_delta" {
				arguments[blockCallIDs[*wire.Index]].WriteString(wire.Delta.PartialJSON)
			}
		case "content_block_stop":
			seenToolStop = true
		case "message_delta":
			if wire.Delta == nil || wire.Delta.StopReason != "tool_use" {
				t.Fatalf("Anthropic stop reason = %#v", wire.Delta)
			}
		case "message_stop":
			seenMessageStop = true
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume Anthropic SSE: %v", err)
	}
	if len(blockCallIDs) != 2 || arguments["call_a"].String() != `{"q":"hello"}` || arguments["call_b"].String() != `{"x":1}` {
		t.Fatalf("Anthropic parallel tools degraded: blocks=%v a=%q b=%q", blockCallIDs, arguments["call_a"].String(), arguments["call_b"].String())
	}
	if !seenMessageStart || !seenToolStop || !seenMessageStop {
		t.Fatalf("Anthropic lifecycle incomplete: start=%v tool_stop=%v message_stop=%v", seenMessageStart, seenToolStop, seenMessageStop)
	}
}

func TestResponsesProviderToChatClientCanonicalParallelToolsOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, err.Error(), http.StatusBadRequest)
			return
		}
		if !bytes.Contains(body, []byte(`"type":"function"`)) || !bytes.Contains(body, []byte(`"stream":true`)) {
			http.Error(writer, "Chat function request did not become a streaming Responses request", http.StatusBadRequest)
			return
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_matrix", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeNamedSSEJSON(t, writer, "response.in_progress", map[string]any{
			"type": "response.in_progress", "sequence_number": 1,
			"response": map[string]any{"id": "resp_matrix", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}},
		})
		writeResponsesFunctionAdded(t, writer, 2, 0, "item_a", "call_a", "lookup")
		writeResponsesFunctionAdded(t, writer, 3, 1, "item_b", "call_b", "calculate")
		writeResponsesFunctionDelta(t, writer, 4, 0, "item_a", `{"q":"hel`)
		writeResponsesFunctionDelta(t, writer, 5, 1, "item_b", `{"x":1`)
		writeResponsesFunctionDelta(t, writer, 6, 0, "item_a", `lo"}`)
		writeResponsesFunctionDelta(t, writer, 7, 1, "item_b", `}`)
		writeResponsesFunctionDone(t, writer, 8, 0, "item_a", "call_a", "lookup", `{"q":"hello"}`)
		writeResponsesFunctionDone(t, writer, 10, 1, "item_b", "call_b", "calculate", `{"x":1}`)
		writeNamedSSEJSON(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 12,
			"response": map[string]any{
				"id": "resp_matrix", "object": "response", "model": "matrix-model", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 7, "output_tokens": 5, "total_tokens": 12},
			},
		})
	}))
	t.Cleanup(provider.Close)

	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/chat/completions",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"matrix-model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"run both"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}},{"type":"function","function":{"name":"calculate","parameters":{"type":"object"}}}]}`),
		})
	if err != nil {
		t.Fatalf("run Responses-to-Chat pipeline: %v", err)
	}
	defer result.EventStream.Close()

	arguments := map[string]*strings.Builder{"call_a": {}, "call_b": {}}
	indexes := map[string]int{}
	finishPosition, usagePosition, donePosition := -1, -1, -1
	position := 0
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if bytes.Equal(event.Data, []byte("[DONE]")) {
			donePosition = position
			position++
			continue
		}
		var chunk struct {
			Choices []struct {
				FinishReason *string `json:"finish_reason"`
				Delta        struct {
					ToolCalls []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			t.Fatalf("decode Chat SSE chunk: %v body=%s", err, event.Data)
		}
		if len(chunk.Usage) > 0 && string(chunk.Usage) != "null" {
			usagePosition = position
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason != nil {
				if *choice.FinishReason != "tool_calls" {
					t.Fatalf("finish reason = %q", *choice.FinishReason)
				}
				finishPosition = position
			}
			for _, call := range choice.Delta.ToolCalls {
				if call.ID != "" {
					indexes[call.ID] = call.Index
				}
				callID := call.ID
				if callID == "" {
					for id, index := range indexes {
						if index == call.Index {
							callID = id
							break
						}
					}
				}
				if call.Function.Arguments != "" {
					arguments[callID].WriteString(call.Function.Arguments)
				}
			}
		}
		position++
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume Chat SSE: %v", err)
	}
	if indexes["call_a"] != 0 || indexes["call_b"] != 1 {
		t.Fatalf("call IDs/indexes collapsed: %#v", indexes)
	}
	if arguments["call_a"].String() != `{"q":"hello"}` || arguments["call_b"].String() != `{"x":1}` {
		t.Fatalf("parallel arguments corrupted: a=%q b=%q", arguments["call_a"].String(), arguments["call_b"].String())
	}
	if !(finishPosition >= 0 && usagePosition > finishPosition && donePosition > usagePosition) {
		t.Fatalf("invalid Chat terminal order: finish=%d usage=%d done=%d", finishPosition, usagePosition, donePosition)
	}
}

func writeResponsesFunctionAdded(t *testing.T, writer http.ResponseWriter, sequence, output int, itemID, callID, name string) {
	t.Helper()
	writeNamedSSEJSON(t, writer, "response.output_item.added", map[string]any{
		"type": "response.output_item.added", "sequence_number": sequence, "output_index": output,
		"item": map[string]any{"id": itemID, "type": "function_call", "status": "in_progress", "call_id": callID, "name": name, "arguments": ""},
	})
}

func writeResponsesFunctionDelta(t *testing.T, writer http.ResponseWriter, sequence, output int, itemID, delta string) {
	t.Helper()
	writeNamedSSEJSON(t, writer, "response.function_call_arguments.delta", map[string]any{
		"type": "response.function_call_arguments.delta", "sequence_number": sequence,
		"output_index": output, "item_id": itemID, "delta": delta,
	})
}

func writeResponsesFunctionDone(t *testing.T, writer http.ResponseWriter, sequence, output int, itemID, callID, name, arguments string) {
	t.Helper()
	writeNamedSSEJSON(t, writer, "response.function_call_arguments.done", map[string]any{
		"type": "response.function_call_arguments.done", "sequence_number": sequence,
		"output_index": output, "item_id": itemID, "call_id": callID, "name": name, "arguments": arguments,
	})
	writeNamedSSEJSON(t, writer, "response.output_item.done", map[string]any{
		"type": "response.output_item.done", "sequence_number": sequence + 1, "output_index": output,
		"item": map[string]any{"id": itemID, "type": "function_call", "status": "completed", "call_id": callID, "name": name, "arguments": arguments},
	})
}

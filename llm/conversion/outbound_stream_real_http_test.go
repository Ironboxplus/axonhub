package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestResponsesCustomToolStreamRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		providerPath  string
		newOutbound   func(string) (transformer.Outbound, error)
		serveProvider func(*testing.T, http.ResponseWriter, *http.Request)
	}{
		{
			name:         "chat_completions",
			providerPath: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
			serveProvider: serveChatCustomStreamFixture,
		},
		{
			name:         "anthropic_messages",
			providerPath: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
			serveProvider: serveAnthropicCustomStreamFixture,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != tt.providerPath {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				tt.serveProvider(t, writer, request)
			}))
			t.Cleanup(provider.Close)

			target, err := tt.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			observer := &observationRecorder{}
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).
				Process(context.Background(), responsesStreamFixtureRequest(t))
			if err != nil {
				t.Fatalf("run real streaming protocol pipeline: %v", err)
			}
			if result == nil || !result.Stream || result.EventStream == nil {
				t.Fatalf("unexpected streaming result: %#v", result)
			}
			defer result.EventStream.Close()

			var events [][]byte
			for result.EventStream.Next() {
				event := result.EventStream.Current()
				if event != nil && len(event.Data) > 0 && !bytes.Equal(event.Data, []byte("[DONE]")) {
					events = append(events, append([]byte(nil), event.Data...))
				}
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("consume real streaming result: %v", err)
			}
			assertResponsesCustomStream(t, events)
			assertConversionRestoreObservation(t, observer.snapshot(), target.APIFormat())
		})
	}
}

func TestResponsesCustomToolRepairsTruncatedAnthropicSSEOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/messages" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		name := streamSyntheticToolName(t, request)
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "message_start", map[string]any{
			"type": "message_start", "message": map[string]any{
				"id": "msg_repair", "type": "message", "role": "assistant", "model": "fixture-model",
				"content": []any{}, "usage": map[string]any{"input_tokens": 4, "output_tokens": 0},
			},
		})
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
			"type": "content_block_start", "index": 0,
			"content_block": map[string]any{"type": "tool_use", "id": continuedCallID, "name": name, "input": map[string]any{}},
		})
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": `{"input":"repaired over stream`},
		})
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
			"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
			"usage": map[string]any{"output_tokens": 2},
		})
		writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
	}))
	t.Cleanup(provider.Close)
	target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Anthropic outbound: %v", err)
	}
	observer := &observationRecorder{}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx := llm.WithConversionDebugTrace(llm.WithConversionTrace(context.Background()), []byte(t.Name()), 16)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).
		Process(ctx, responsesStreamFixtureRequest(t))
	if err != nil {
		t.Fatalf("run truncated Anthropic SSE: %v", err)
	}
	if result == nil || result.EventStream == nil {
		t.Fatalf("missing Responses event stream: %#v", result)
	}
	defer result.EventStream.Close()
	var delta, doneInput string
	var done bool
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || len(event.Data) == 0 || bytes.Equal(event.Data, []byte("[DONE]")) {
			continue
		}
		var wire struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
			Item  *struct {
				Type  string `json:"type"`
				Input string `json:"input"`
			} `json:"item"`
		}
		if json.Unmarshal(event.Data, &wire) != nil {
			t.Fatalf("decode repaired Responses SSE event: %s", event.Data)
		}
		if wire.Type == "response.custom_tool_call_input.delta" {
			delta += wire.Delta
		}
		if wire.Type == "response.output_item.done" && wire.Item != nil && wire.Item.Type == "custom_tool_call" {
			doneInput, done = wire.Item.Input, true
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume repaired Responses SSE: %v", err)
	}
	if !done || delta != "repaired over stream" || doneInput != "repaired over stream" {
		t.Fatalf("repaired custom SSE lifecycle delta=%q done=(%v,%q)", delta, done, doneInput)
	}
	var restore *llm.ConversionTraceSummary
	for _, event := range observer.snapshot() {
		if event.Stage == pipeline.StageConversionRestore {
			restore = event.Conversion
		}
	}
	if restore == nil || restore.CustomInputsRepaired != 1 || restore.RestoreMiss != 0 || !restore.Complete {
		t.Fatalf("stream custom repair observation = %#v", restore)
	}
}

func TestResponsesLocalShellStreamRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, *http.Request)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveChatLocalShellStream,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveAnthropicLocalShellStream,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				test.serve(t, writer, request)
			}))
			t.Cleanup(provider.Close)

			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), responsesLocalShellStreamRequest())
			if err != nil {
				t.Fatalf("run local shell stream: %v", err)
			}
			if result == nil || !result.Stream || result.EventStream == nil {
				t.Fatalf("unexpected streaming result: %#v", result)
			}
			defer result.EventStream.Close()
			var events [][]byte
			for result.EventStream.Next() {
				current := result.EventStream.Current()
				if current != nil && len(current.Data) > 0 && !bytes.Equal(current.Data, []byte("[DONE]")) {
					events = append(events, append([]byte(nil), current.Data...))
				}
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("consume local shell stream: %v", err)
			}
			assertResponsesLocalShellStream(t, events)
		})
	}
}

func responsesLocalShellStreamRequest() *httpclient.Request {
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,
			"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"show directory"}]}],
			"tools":[{"type":"local_shell"}]
		}`),
	}
}

func localShellSyntheticName(t *testing.T, request *http.Request) string {
	t.Helper()
	defer request.Body.Close()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read provider request: %v", err)
	}
	var payload struct {
		Stream bool              `json:"stream"`
		Tools  []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || !payload.Stream || len(payload.Tools) != 1 {
		t.Fatalf("decode local shell provider request: err=%v body=%s", err, body)
	}
	var tool struct {
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(payload.Tools[0], &tool); err != nil {
		t.Fatalf("decode local shell provider tool: %v", err)
	}
	name := tool.Name
	if name == "" {
		name = tool.Function.Name
	}
	if !strings.HasPrefix(name, "axc_") {
		t.Fatalf("local shell was not reversibly lowered: %s", body)
	}
	return name
}

func serveChatLocalShellStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := localShellSyntheticName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeChatSSEChunk(t, writer, map[string]any{
		"role": "assistant", "tool_calls": []any{map[string]any{
			"index": 0, "id": "shell_stream", "type": "function",
			"function": map[string]any{"name": name, "arguments": `{"type":"exec","command":`},
		}},
	}, nil)
	writeChatSSEChunk(t, writer, map[string]any{"tool_calls": []any{map[string]any{
		"index": 0, "function": map[string]any{"arguments": `["pwd"],"working_directory":"/workspace"}`},
	}}}, nil)
	writeChatSSEChunk(t, writer, map[string]any{}, "tool_calls")
	_, _ = io.WriteString(writer, "data: [DONE]\n\n")
}

func serveAnthropicLocalShellStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := localShellSyntheticName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeNamedSSEJSON(t, writer, "message_start", map[string]any{
		"type": "message_start", "message": map[string]any{
			"id": "msg_shell_stream", "type": "message", "role": "assistant", "model": "fixture-model",
			"content": []any{}, "usage": map[string]any{"input_tokens": 4, "output_tokens": 0},
		},
	})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "tool_use", "id": "shell_stream", "name": name, "input": map[string]any{}},
	})
	for _, fragment := range []string{`{"type":"exec","command":`, `["pwd"],"working_directory":"/workspace"}`} {
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": fragment},
		})
	}
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 3},
	})
	writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
}

func assertResponsesLocalShellStream(t *testing.T, events [][]byte) {
	t.Helper()
	var added, done, completed bool
	for _, raw := range events {
		if bytes.Contains(raw, []byte("axc_")) || bytes.Contains(raw, []byte("function_call_arguments")) {
			t.Fatalf("synthetic function lifecycle leaked to Responses client: %s", raw)
		}
		var event struct {
			Type string `json:"type"`
			Item *struct {
				Type   string `json:"type"`
				CallID string `json:"call_id"`
				Action struct {
					Type             string   `json:"type"`
					Command          []string `json:"command"`
					WorkingDirectory *string  `json:"working_directory"`
				} `json:"action"`
			} `json:"item"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatalf("decode local shell Responses event: %v body=%s", err, raw)
		}
		switch event.Type {
		case "response.output_item.added":
			added = event.Item != nil && event.Item.Type == "local_shell_call" && event.Item.CallID == "shell_stream" &&
				event.Item.Action.Type == "exec" && len(event.Item.Action.Command) == 0
		case "response.output_item.done":
			done = event.Item != nil && event.Item.Type == "local_shell_call" && event.Item.CallID == "shell_stream" &&
				event.Item.Action.Type == "exec" && len(event.Item.Action.Command) == 1 && event.Item.Action.Command[0] == "pwd" &&
				event.Item.Action.WorkingDirectory != nil && *event.Item.Action.WorkingDirectory == "/workspace"
		case "response.completed":
			completed = true
		case "response.failed":
			t.Fatalf("local shell Responses stream failed: %s", raw)
		}
	}
	if !added || !done || !completed {
		t.Fatalf("local shell Responses lifecycle degraded: added=%v done=%v completed=%v events=%q", added, done, completed, events)
	}
}

func TestResponsesClientToolSearchStreamRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, *http.Request)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			}, serve: serveChatToolSearchStream,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			}, serve: serveAnthropicToolSearchStream,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				test.serve(t, writer, request)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/responses",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model","stream":true,
						"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"find calendar tools"}]}],
						"tools":[{"type":"tool_search","execution":"client","description":"Search deferred tools","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]
					}`),
				})
			if err != nil {
				t.Fatalf("run tool search stream: %v", err)
			}
			defer result.EventStream.Close()
			var events [][]byte
			for result.EventStream.Next() {
				current := result.EventStream.Current()
				if current != nil && len(current.Data) > 0 && !bytes.Equal(current.Data, []byte("[DONE]")) {
					events = append(events, append([]byte(nil), current.Data...))
				}
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("consume tool search stream: %v", err)
			}
			assertResponsesToolSearchStream(t, events)
		})
	}
}

func toolSearchSyntheticName(t *testing.T, request *http.Request) string {
	t.Helper()
	defer request.Body.Close()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read provider request: %v", err)
	}
	var payload struct {
		Stream bool              `json:"stream"`
		Tools  []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || !payload.Stream || len(payload.Tools) != 1 {
		t.Fatalf("decode tool search provider request: err=%v body=%s", err, body)
	}
	var tool struct {
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(payload.Tools[0], &tool); err != nil {
		t.Fatalf("decode tool search provider tool: %v", err)
	}
	name := tool.Name
	if name == "" {
		name = tool.Function.Name
	}
	if !strings.HasPrefix(name, "axc_") {
		t.Fatalf("tool search was not reversibly lowered: %s", body)
	}
	return name
}

func serveChatToolSearchStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := toolSearchSyntheticName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeChatSSEChunk(t, writer, map[string]any{
		"role": "assistant", "tool_calls": []any{map[string]any{
			"index": 0, "id": "search_stream", "type": "function",
			"function": map[string]any{"name": name, "arguments": `{"query":`},
		}},
	}, nil)
	writeChatSSEChunk(t, writer, map[string]any{"tool_calls": []any{map[string]any{
		"index": 0, "function": map[string]any{"arguments": `"calendar","limit":2}`},
	}}}, nil)
	writeChatSSEChunk(t, writer, map[string]any{}, "tool_calls")
	_, _ = io.WriteString(writer, "data: [DONE]\n\n")
}

func serveAnthropicToolSearchStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := toolSearchSyntheticName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeNamedSSEJSON(t, writer, "message_start", map[string]any{
		"type": "message_start", "message": map[string]any{
			"id": "msg_search_stream", "type": "message", "role": "assistant", "model": "fixture-model",
			"content": []any{}, "usage": map[string]any{"input_tokens": 4, "output_tokens": 0},
		},
	})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "tool_use", "id": "search_stream", "name": name, "input": map[string]any{}},
	})
	for _, fragment := range []string{`{"query":`, `"calendar","limit":2}`} {
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0,
			"delta": map[string]any{"type": "input_json_delta", "partial_json": fragment},
		})
	}
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 3},
	})
	writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
}

func assertResponsesToolSearchStream(t *testing.T, events [][]byte) {
	t.Helper()
	var added, done, completed bool
	for _, raw := range events {
		if bytes.Contains(raw, []byte("axc_")) || bytes.Contains(raw, []byte("function_call_arguments")) {
			t.Fatalf("synthetic function lifecycle leaked to Responses client: %s", raw)
		}
		var event struct {
			Type string `json:"type"`
			Item *struct {
				Type      string          `json:"type"`
				CallID    string          `json:"call_id"`
				Execution string          `json:"execution"`
				Arguments json.RawMessage `json:"arguments"`
			} `json:"item"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatalf("decode tool search Responses event: %v body=%s", err, raw)
		}
		switch event.Type {
		case "response.output_item.added":
			added = event.Item != nil && event.Item.Type == "tool_search_call" && event.Item.CallID == "search_stream" &&
				event.Item.Execution == "client" && string(event.Item.Arguments) == `{}`
		case "response.output_item.done":
			done = event.Item != nil && event.Item.Type == "tool_search_call" && event.Item.CallID == "search_stream" &&
				event.Item.Execution == "client" && bytes.Contains(event.Item.Arguments, []byte(`"query":"calendar"`)) &&
				bytes.Contains(event.Item.Arguments, []byte(`"limit":2`))
		case "response.completed":
			completed = true
		case "response.failed":
			t.Fatalf("tool search Responses stream failed: %s", raw)
		}
	}
	if !added || !done || !completed {
		t.Fatalf("tool search Responses lifecycle degraded: added=%v done=%v completed=%v events=%q", added, done, completed, events)
	}
}

func TestResponsesAdditionalNamespaceToolsStreamRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, *http.Request)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveChatAdditionalNamespaceStream,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveAnthropicAdditionalNamespaceStream,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				test.serve(t, writer, request)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), responsesAdditionalNamespaceStreamRequest())
			if err != nil {
				t.Fatalf("run additional namespace stream: %v", err)
			}
			defer result.EventStream.Close()
			var events [][]byte
			for result.EventStream.Next() {
				current := result.EventStream.Current()
				if current != nil && len(current.Data) > 0 && !bytes.Equal(current.Data, []byte("[DONE]")) {
					events = append(events, append([]byte(nil), current.Data...))
				}
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("consume additional namespace stream: %v", err)
			}
			assertResponsesAdditionalNamespaceStream(t, events)
		})
	}
}

func responsesAdditionalNamespaceStreamRequest() *httpclient.Request {
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,
			"input":[
				{"type":"additional_tools","role":"developer","tools":[
					{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"send_message","parameters":{"type":"object","properties":{"message":{"type":"string"}}}}]},
					{"type":"namespace","name":"terminal","tools":[{"type":"custom","name":"exec"}]}
				]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"send then run"}]}
			]
		}`),
	}
}

func additionalNamespaceNamesFromProvider(t *testing.T, request *http.Request, anthropicWire bool) (string, string) {
	t.Helper()
	defer request.Body.Close()
	var raw struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.NewDecoder(request.Body).Decode(&raw); err != nil {
		t.Fatalf("decode provider stream request: %v", err)
	}
	tools := make([]additionalNamespaceToolFixture, 0, len(raw.Tools))
	for _, encoded := range raw.Tools {
		if anthropicWire {
			var tool additionalNamespaceToolFixture
			if err := json.Unmarshal(encoded, &tool); err != nil {
				t.Fatalf("decode Anthropic stream tool: %v", err)
			}
			tool.Schema = tool.InputSchema
			tools = append(tools, tool)
			continue
		}
		var wrapper struct {
			Function additionalNamespaceToolFixture `json:"function"`
		}
		if err := json.Unmarshal(encoded, &wrapper); err != nil {
			t.Fatalf("decode Chat stream tool: %v", err)
		}
		wrapper.Function.Schema = wrapper.Function.Parameters
		tools = append(tools, wrapper.Function)
	}
	return classifyAdditionalNamespaceTools(t, tools)
}

func serveChatAdditionalNamespaceStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	functionName, customName := additionalNamespaceNamesFromProvider(t, request, false)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeChatSSEChunk(t, writer, map[string]any{
		"role": "assistant",
		"tool_calls": []any{
			map[string]any{"index": 0, "id": "call_send", "type": "function", "function": map[string]any{"name": functionName, "arguments": `{"message":`}},
			map[string]any{"index": 1, "id": "call_exec", "type": "function", "function": map[string]any{"name": customName, "arguments": `{"input":`}},
		},
	}, nil)
	writeChatSSEChunk(t, writer, map[string]any{"tool_calls": []any{
		map[string]any{"index": 0, "function": map[string]any{"arguments": `"ping"}`}},
		map[string]any{"index": 1, "function": map[string]any{"arguments": `"pwd"}`}},
	}}, nil)
	writeChatSSEChunk(t, writer, map[string]any{}, "tool_calls")
	_, _ = io.WriteString(writer, "data: [DONE]\n\n")
}

func serveAnthropicAdditionalNamespaceStream(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	functionName, customName := additionalNamespaceNamesFromProvider(t, request, true)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeNamedSSEJSON(t, writer, "message_start", map[string]any{
		"type": "message_start", "message": map[string]any{"id": "msg_additional_stream", "type": "message", "role": "assistant", "model": "fixture-model", "content": []any{}, "usage": map[string]any{"input_tokens": 8, "output_tokens": 0}},
	})
	for index, tool := range []struct{ id, name string }{{"call_send", functionName}, {"call_exec", customName}} {
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
			"type": "content_block_start", "index": index, "content_block": map[string]any{"type": "tool_use", "id": tool.id, "name": tool.name, "input": map[string]any{}},
		})
	}
	for index, argument := range []string{`{"message":"ping"}`, `{"input":"pwd"}`} {
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": index, "delta": map[string]any{"type": "input_json_delta", "partial_json": argument},
		})
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
	}
	writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 4},
	})
	writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
}

func assertResponsesAdditionalNamespaceStream(t *testing.T, events [][]byte) {
	t.Helper()
	var functionAdded, functionDone, customAdded, customDone, completed bool
	var customInput strings.Builder
	for _, raw := range events {
		var event struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
			Item  *struct {
				Type      string `json:"type"`
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
				Input     string `json:"input"`
			} `json:"item"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatalf("decode additional namespace event: %v body=%s", err, raw)
		}
		switch event.Type {
		case "response.output_item.added", "response.output_item.done":
			if event.Item == nil {
				continue
			}
			if event.Item.CallID == "call_send" {
				valid := event.Item.Type == "function_call" && event.Item.Name == "send_message" && event.Item.Namespace == "collaboration"
				if event.Type == "response.output_item.added" {
					functionAdded = valid
				} else {
					functionDone = valid
				}
			}
			if event.Item.CallID == "call_exec" {
				valid := event.Item.Type == "custom_tool_call" && event.Item.Name == "exec" && event.Item.Namespace == "terminal"
				if event.Type == "response.output_item.added" {
					customAdded = valid
				} else {
					customDone = valid && event.Item.Input == "pwd"
				}
			}
		case "response.custom_tool_call_input.delta":
			customInput.WriteString(event.Delta)
		case "response.completed":
			completed = true
		case "response.failed":
			t.Fatalf("additional namespace stream failed: %s", raw)
		}
	}
	if !functionAdded || !functionDone || !customAdded || !customDone || customInput.String() != "pwd" || !completed {
		t.Fatalf("additional namespace lifecycle degraded: function=(%v,%v) custom=(%v,%v,%q) completed=%v events=%q",
			functionAdded, functionDone, customAdded, customDone, customInput.String(), completed, events)
	}
}

func assertConversionRestoreObservation(t *testing.T, events []pipeline.Observation, targetFormat llm.APIFormat) {
	t.Helper()
	for _, event := range events {
		if event.Stage != pipeline.StageConversionRestore {
			continue
		}
		if event.Conversion == nil || event.Conversion.SourceFormat != llm.APIFormatOpenAIResponse ||
			event.Conversion.TargetFormat != targetFormat || event.Conversion.Lowered < 4 ||
			event.Conversion.RestoreMiss != 0 || !event.Conversion.Complete {
			t.Fatalf("stream conversion restore summary = %#v", event.Conversion)
		}
		return
	}
	t.Fatal("stream conversion restore stage was not observed")
}

func responsesStreamFixtureRequest(t *testing.T) *httpclient.Request {
	t.Helper()
	request := responsesFixtureRequest(t)
	var payload map[string]any
	if err := json.Unmarshal(request.Body, &payload); err != nil {
		t.Fatalf("decode Responses stream fixture: %v", err)
	}
	payload["stream"] = true
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("encode Responses stream fixture: %v", err)
	}
	request.Body = body
	return request
}

func streamSyntheticToolName(t *testing.T, request *http.Request) string {
	t.Helper()
	defer request.Body.Close()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read provider request: %v", err)
	}
	var payload struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil || len(payload.Tools) != 1 {
		t.Fatalf("decode provider tools: err=%v body=%s", err, body)
	}
	var tool struct {
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(payload.Tools[0], &tool); err != nil {
		t.Fatalf("decode provider tool: %v", err)
	}
	name := tool.Name
	if name == "" {
		name = tool.Function.Name
	}
	if name == "" || name == originalCustomName || !bytes.Contains(body, []byte(`"stream":true`)) {
		t.Fatalf("custom stream definition was not lowered: %s", body)
	}
	return name
}

func serveChatCustomStreamFixture(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := streamSyntheticToolName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	arguments := []string{`{"input":"*** Begin`, ` Patch\n*** End`, ` Patch\n"}`}
	firstCall := map[string]any{
		"index": 0,
		"id":    continuedCallID,
		"type":  "function",
		"function": map[string]any{
			"name": name, "arguments": arguments[0],
		},
	}
	writeChatSSEChunk(t, writer, map[string]any{"role": "assistant", "tool_calls": []any{firstCall}}, nil)
	for _, fragment := range arguments[1:] {
		call := map[string]any{"index": 0, "function": map[string]any{"arguments": fragment}}
		writeChatSSEChunk(t, writer, map[string]any{"tool_calls": []any{call}}, nil)
	}
	writeChatSSEChunk(t, writer, map[string]any{}, "tool_calls")
	writeSSEJSON(t, writer, map[string]any{
		"id": "chatcmpl_stream_fixture", "object": "chat.completion.chunk", "model": "conversion-fixture-model",
		"choices": []any{}, "usage": map[string]any{"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
	})
	_, _ = io.WriteString(writer, "data: [DONE]\n\n")
}

func writeChatSSEChunk(t *testing.T, writer http.ResponseWriter, delta map[string]any, finishReason any) {
	t.Helper()
	writeSSEJSON(t, writer, map[string]any{
		"id":     "chatcmpl_stream_fixture",
		"object": "chat.completion.chunk",
		"model":  "conversion-fixture-model",
		"choices": []any{map[string]any{
			"index": 0, "delta": delta, "finish_reason": finishReason,
		}},
	})
}

func serveAnthropicCustomStreamFixture(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	name := streamSyntheticToolName(t, request)
	writer.Header().Set("Content-Type", "text/event-stream")
	writeNamedSSEJSON(t, writer, "message_start", map[string]any{
		"type": "message_start", "message": map[string]any{"id": "msg_stream_fixture", "type": "message", "role": "assistant", "model": "conversion-fixture-model", "content": []any{}, "usage": map[string]any{"input_tokens": 10, "output_tokens": 0}},
	})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "tool_use", "id": continuedCallID, "name": name, "input": map[string]any{}},
	})
	for _, fragment := range []string{`{"input":"*** Begin`, ` Patch\n*** End`, ` Patch\n"}`} {
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
			"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": fragment},
		})
	}
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 3},
	})
	writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
}

func writeSSEJSON(t *testing.T, writer http.ResponseWriter, value any) {
	t.Helper()
	writeNamedSSEJSON(t, writer, "", value)
}

func writeNamedSSEJSON(t *testing.T, writer http.ResponseWriter, eventType string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal SSE fixture: %v", err)
	}
	if eventType != "" {
		_, _ = fmt.Fprintf(writer, "event: %s\n", eventType)
	}
	_, _ = fmt.Fprintf(writer, "data: %s\n\n", data)
	if flusher, ok := writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func assertResponsesCustomStream(t *testing.T, events [][]byte) {
	t.Helper()
	var input strings.Builder
	var addedName, addedCallID, doneName, doneCallID, doneInput string
	seenCompleted := false
	failedEvent := ""
	eventTypes := make([]string, 0, len(events))
	for _, raw := range events {
		var event struct {
			Type  string `json:"type"`
			Delta string `json:"delta"`
			Item  *struct {
				Type   string `json:"type"`
				CallID string `json:"call_id"`
				Name   string `json:"name"`
				Input  string `json:"input"`
			} `json:"item"`
		}
		if err := json.Unmarshal(raw, &event); err != nil {
			t.Fatalf("decode Responses stream event: %v body=%s", err, raw)
		}
		eventTypes = append(eventTypes, event.Type)
		switch event.Type {
		case "response.output_item.added":
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				addedName, addedCallID = event.Item.Name, event.Item.CallID
			}
		case "response.custom_tool_call_input.delta":
			input.WriteString(event.Delta)
		case "response.output_item.done":
			if event.Item != nil && event.Item.Type == "custom_tool_call" {
				doneName, doneCallID, doneInput = event.Item.Name, event.Item.CallID, event.Item.Input
			}
		case "response.completed":
			seenCompleted = true
		case "response.failed":
			failedEvent = string(raw)
		}
	}
	if addedName != originalCustomName || addedCallID != continuedCallID || doneName != originalCustomName ||
		doneCallID != continuedCallID || input.String() != originalCustomInput || doneInput != originalCustomInput || !seenCompleted {
		t.Fatalf("custom stream lifecycle degraded: added=(%q,%q) delta=%q done=(%q,%q,%q) completed=%v events=%v failed=%s",
			addedName, addedCallID, input.String(), doneName, doneCallID, doneInput, seenCompleted, eventTypes, failedEvent)
	}
}

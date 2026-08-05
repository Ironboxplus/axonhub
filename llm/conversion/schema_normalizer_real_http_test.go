package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestChatEmptySchemaReachesAnthropicOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				InputSchema map[string]any `json:"input_schema"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Anthropic request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].InputSchema["type"] != "object" {
			http.Error(writer, "missing normalized object schema", http.StatusBadRequest)
			return
		}
		properties, ok := payload.Tools[0].InputSchema["properties"].(map[string]any)
		additional, additionalOK := payload.Tools[0].InputSchema["additionalProperties"].(bool)
		if !ok || len(properties) != 0 || !additionalOK || additional {
			http.Error(writer, "empty schema is not a closed empty object", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"msg_schema","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Anthropic outbound: %v", err)
	}
	runSchemaPipeline(t, provider, openai.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/chat/completions",
		`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"empty"}}]}`)
}

func TestAnthropicNestedSchemaReachesResponsesOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				Parameters map[string]any `json:"parameters"`
				Strict     *bool          `json:"strict"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Strict == nil || !*payload.Tools[0].Strict {
			http.Error(writer, "strict was not preserved", http.StatusBadRequest)
			return
		}
		properties := payload.Tools[0].Parameters["properties"].(map[string]any)
		nested := properties["nested"].(map[string]any)
		choice := properties["choice"].(map[string]any)
		if _, ok := nested["properties"].(map[string]any); !ok {
			http.Error(writer, "nested object properties missing", http.StatusBadRequest)
			return
		}
		if _, ok := choice["properties"].(map[string]any); !ok {
			http.Error(writer, "nullable object properties missing", http.StatusBadRequest)
			return
		}
		if additional, ok := payload.Tools[0].Parameters["additionalProperties"].(bool); !ok || additional {
			http.Error(writer, "additionalProperties changed", http.StatusBadRequest)
			return
		}
		required := jsonStringSet(payload.Tools[0].Parameters["required"])
		if !required["nested"] || !required["choice"] {
			http.Error(writer, "strict optional field was not made required", http.StatusBadRequest)
			return
		}
		choiceTypes, ok := choice["type"].([]any)
		if !ok || !jsonArrayContains(choiceTypes, "object") || !jsonArrayContains(choiceTypes, "null") {
			http.Error(writer, "strict optional field was not made nullable", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_schema","object":"response","status":"completed","model":"fixture-model","output":[{"id":"fc_schema","type":"function_call","status":"completed","call_id":"call_schema","name":"nested","arguments":"{\"nested\":{},\"choice\":null}"}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	result := runSchemaPipeline(t, provider, anthropic.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/messages",
		`{"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"nested","strict":true,"input_schema":{"type":"object","properties":{"nested":{"type":"object","properties":{},"additionalProperties":false},"choice":{"type":"object","properties":{},"additionalProperties":false}},"required":["nested"],"additionalProperties":false}}]}`)
	var clientBody struct {
		Content []struct {
			Type  string         `json:"type"`
			Input map[string]any `json:"input"`
		} `json:"content"`
	}
	if err := json.Unmarshal(result.Response.Body, &clientBody); err != nil {
		t.Fatalf("decode Anthropic client response: %v body=%s", err, result.Response.Body)
	}
	if len(clientBody.Content) != 1 || clientBody.Content[0].Type != "tool_use" {
		t.Fatalf("unexpected Anthropic client response: %s", result.Response.Body)
	}
	if _, leaked := clientBody.Content[0].Input["choice"]; leaked {
		t.Fatalf("synthetic optional null leaked over real HTTP: %s", result.Response.Body)
	}
}

func TestResponsesRootUnionSchemaReachesChatOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				Function struct {
					Parameters map[string]any `json:"parameters"`
					Strict     *bool          `json:"strict"`
				} `json:"function"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Chat request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Function.Strict == nil || *payload.Tools[0].Function.Strict {
			http.Error(writer, "incompatible strict schema did not fall back", http.StatusBadRequest)
			return
		}
		parameters := payload.Tools[0].Function.Parameters
		if parameters["type"] != "object" {
			http.Error(writer, "root object type missing", http.StatusBadRequest)
			return
		}
		if _, ok := parameters["oneOf"]; !ok {
			http.Error(writer, "Chat-compatible root union was unnecessarily removed", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chatcmpl_schema","object":"chat.completion","created":1,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses",
		`{"model":"fixture-model","input":"hi","tools":[{"type":"function","name":"lookup","strict":true,"parameters":{"oneOf":[{"type":"object","properties":{"query":{"type":"string"}}},{"type":"object","properties":{"id":{"type":"integer"}}}]}}]}`)
}

func TestResponsesIdentityImplicitRootUnionReachesStrictProviderOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				Parameters map[string]any `json:"parameters"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Parameters["type"] != "object" {
			http.Error(writer, "root object type missing", http.StatusBadRequest)
			return
		}
		branches, ok := payload.Tools[0].Parameters["anyOf"].([]any)
		if !ok || len(branches) != 2 {
			http.Error(writer, "root anyOf missing", http.StatusBadRequest)
			return
		}
		for _, branch := range branches {
			branchSchema, ok := branch.(map[string]any)
			if !ok || branchSchema["type"] != "object" {
				http.Error(writer, "tool parameter root union has a non-object branch", http.StatusBadRequest)
				return
			}
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_identity_schema","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_identity_schema","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses",
		`{"model":"fixture-model","input":"OpenAI Responses API official documentation","tools":[{"type":"function","name":"search","parameters":{"type":"object","properties":{"query":{"type":"string"},"queries":{"type":"array","items":{"type":"string"}}},"anyOf":[{"required":["query"]},{"required":["queries"]}],"additionalProperties":true}}]}`)
}

func TestInvalidNestedIdentitySchemaStopsBeforeProviderDispatchOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRequests atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		providerRequests.Add(1)
		http.Error(writer, "provider must not receive invalid schema", http.StatusBadRequest)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"fixture-model","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"query":[]}}}]}`),
		})
	if !errors.Is(err, conversion.ErrIncompletePlan) {
		t.Fatalf("pipeline error = %v, want ErrIncompletePlan; result=%#v", err, result)
	}
	if got := providerRequests.Load(); got != 0 {
		t.Fatalf("provider received %d invalid requests, want zero", got)
	}
}

func TestResponsesIdentityNamespacedFunctionRestoresChildNameOverRealHTTP(t *testing.T) {
	t.Parallel()
	historyCallID := "history call:" + strings.Repeat("x", 80)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				Type  string `json:"type"`
				Name  string `json:"name"`
				Tools []struct {
					Type string `json:"type"`
					Name string `json:"name"`
				} `json:"tools"`
			} `json:"tools"`
			Input []struct {
				Type      string `json:"type"`
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Type != "namespace" || payload.Tools[0].Name != "project" || len(payload.Tools[0].Tools) != 1 {
			http.Error(writer, "namespace tool missing", http.StatusBadRequest)
			return
		}
		providerChild := payload.Tools[0].Tools[0].Name
		if providerChild == "read.file" || !strings.HasPrefix(providerChild, "read_file_") {
			http.Error(writer, "namespace child was not normalized", http.StatusBadRequest)
			return
		}
		if len(payload.Input) != 2 || payload.Input[0].Type != "function_call" || payload.Input[0].Name != providerChild ||
			payload.Input[0].Namespace != "project" || payload.Input[0].CallID != historyCallID ||
			payload.Input[1].Type != "function_call_output" || payload.Input[1].CallID != historyCallID {
			http.Error(writer, "namespace history lifecycle diverged", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(writer, `{"id":"resp_namespace","object":"response","status":"completed","model":"fixture-model","output":[{"id":"fc_namespace","type":"function_call","call_id":"call_provider_new","name":%q,"namespace":"project","arguments":"{}","status":"completed"}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`, providerChild)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	result := runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses",
		fmt.Sprintf(`{"model":"fixture-model","input":[{"type":"function_call","call_id":%q,"name":"read.file","namespace":"project","arguments":"{}"},{"type":"function_call_output","call_id":%q,"output":"ok"}],"tools":[{"type":"namespace","name":"project","tools":[{"type":"function","name":"read.file","description":"read","parameters":{"type":"object"}}]}]}`, historyCallID, historyCallID))
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"output"`
	}
	if err := json.Unmarshal(result.Response.Body, &response); err != nil {
		t.Fatalf("decode client Responses body: %v; body=%s", err, result.Response.Body)
	}
	if len(response.Output) != 1 || response.Output[0].Name != "read.file" || response.Output[0].Namespace != "project" || response.Output[0].CallID != "call_provider_new" {
		t.Fatalf("restored client response = %#v", response.Output)
	}
}

func TestResponsesIdentityNamespacedPrefixedChildRestoresAcrossStreamOverRealHTTP(t *testing.T) {
	t.Parallel()
	const sourceChild = "project__read.file"
	const providerCallID = "call_provider_stream"
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Tools []struct {
				Type  string `json:"type"`
				Name  string `json:"name"`
				Tools []struct {
					Name string `json:"name"`
				} `json:"tools"`
			} `json:"tools"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Responses stream request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Type != "namespace" || payload.Tools[0].Name != "project" || len(payload.Tools[0].Tools) != 1 {
			http.Error(writer, "namespace stream tool missing", http.StatusBadRequest)
			return
		}
		providerChild := payload.Tools[0].Tools[0].Name
		if providerChild == sourceChild || strings.Contains(providerChild, ".") {
			http.Error(writer, "namespace stream child was not normalized", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_namespace_stream", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}},
		})
		writeNamedSSEJSON(t, writer, "response.output_item.added", map[string]any{
			"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
			"item": map[string]any{"id": "fc_namespace_stream", "type": "function_call", "status": "in_progress", "call_id": providerCallID, "name": providerChild, "namespace": "project", "arguments": ""},
		})
		writeResponsesFunctionDelta(t, writer, 2, 0, "fc_namespace_stream", `{"path":"README.md"}`)
		writeNamedSSEJSON(t, writer, "response.function_call_arguments.done", map[string]any{
			"type": "response.function_call_arguments.done", "sequence_number": 3, "output_index": 0,
			"item_id": "fc_namespace_stream", "call_id": providerCallID, "name": providerChild, "namespace": "project", "arguments": `{"path":"README.md"}`,
		})
		writeNamedSSEJSON(t, writer, "response.output_item.done", map[string]any{
			"type": "response.output_item.done", "sequence_number": 4, "output_index": 0,
			"item": map[string]any{"id": "fc_namespace_stream", "type": "function_call", "status": "completed", "call_id": providerCallID, "name": providerChild, "namespace": "project", "arguments": `{"path":"README.md"}`},
		})
		writeNamedSSEJSON(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 5,
			"response": map[string]any{"id": "resp_namespace_stream", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{}},
		})
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	result := runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses",
		`{"model":"fixture-model","stream":true,"input":"hi","tools":[{"type":"namespace","name":"project","tools":[{"type":"function","name":"project__read.file","parameters":{"type":"object"}}]}]}`)
	if !result.Stream || result.EventStream == nil {
		t.Fatalf("unexpected namespace stream result: %#v", result)
	}
	defer result.EventStream.Close()
	namedEvents := 0
	deltaSeen := false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || len(event.Data) == 0 || bytes.Equal(event.Data, []byte("[DONE]")) {
			continue
		}
		var wire map[string]any
		if json.Unmarshal(event.Data, &wire) != nil {
			continue
		}
		if wire["type"] == "response.function_call_arguments.delta" && wire["delta"] == `{"path":"README.md"}` {
			deltaSeen = true
		}
		candidate := wire
		if item, ok := wire["item"].(map[string]any); ok {
			candidate = item
		}
		name, _ := candidate["name"].(string)
		if name == "" {
			continue
		}
		namedEvents++
		if name != sourceChild || candidate["namespace"] != "project" || candidate["call_id"] != providerCallID {
			t.Fatalf("namespace stream identity was not restored: %#v", candidate)
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume namespace Responses stream: %v", err)
	}
	if namedEvents < 3 || !deltaSeen {
		t.Fatalf("namespace stream evidence incomplete: named=%d delta=%v", namedEvents, deltaSeen)
	}
}

func TestResponsesNamespacedHistoryUsesOneFlattenedNameAcrossProtocolsOverRealHTTP(t *testing.T) {
	t.Parallel()
	const historyCallID = "call_history"
	requestBody := `{"model":"fixture-model","input":[{"type":"function_call","call_id":"call_history","name":"project__read.file","namespace":"project","arguments":"{}"},{"type":"function_call_output","call_id":"call_history","output":"ok"}],"tools":[{"type":"namespace","name":"project","tools":[{"type":"function","name":"project__read.file","description":"read","parameters":{"type":"object"}}]}]}`

	t.Run("chat", func(t *testing.T) {
		provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var payload struct {
				Tools []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tools"`
				Messages []struct {
					Role       string `json:"role"`
					ToolCallID string `json:"tool_call_id"`
					ToolCalls  []struct {
						ID       string `json:"id"`
						Function struct {
							Name string `json:"name"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"messages"`
			}
			if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
				http.Error(writer, "decode Chat request", http.StatusBadRequest)
				return
			}
			definitionName := ""
			if len(payload.Tools) == 1 {
				definitionName = payload.Tools[0].Function.Name
			}
			callName, resultID := "", ""
			for _, message := range payload.Messages {
				if len(message.ToolCalls) > 0 {
					callName = message.ToolCalls[0].Function.Name
					if message.ToolCalls[0].ID != historyCallID {
						http.Error(writer, "Chat call ID changed", http.StatusBadRequest)
						return
					}
				}
				if message.Role == "tool" {
					resultID = message.ToolCallID
				}
			}
			if definitionName == "" || definitionName != callName || resultID != historyCallID || strings.Contains(definitionName, ".") {
				http.Error(writer, "Chat namespace identity diverged", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"chat_namespace","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
		}))
		t.Cleanup(provider.Close)
		target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
		if err != nil {
			t.Fatalf("create Chat outbound: %v", err)
		}
		runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses", requestBody)
	})

	t.Run("anthropic", func(t *testing.T) {
		provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var payload struct {
				Tools []struct {
					Name string `json:"name"`
				} `json:"tools"`
				Messages []struct {
					Content []struct {
						Type      string `json:"type"`
						ID        string `json:"id"`
						Name      string `json:"name"`
						ToolUseID string `json:"tool_use_id"`
					} `json:"content"`
				} `json:"messages"`
			}
			if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
				http.Error(writer, "decode Anthropic request", http.StatusBadRequest)
				return
			}
			definitionName := ""
			if len(payload.Tools) == 1 {
				definitionName = payload.Tools[0].Name
			}
			callName, resultID := "", ""
			for _, message := range payload.Messages {
				for _, block := range message.Content {
					switch block.Type {
					case "tool_use":
						callName = block.Name
						if block.ID != historyCallID {
							http.Error(writer, "Anthropic call ID changed", http.StatusBadRequest)
							return
						}
					case "tool_result":
						resultID = block.ToolUseID
					}
				}
			}
			if definitionName == "" || definitionName != callName || resultID != historyCallID || strings.Contains(definitionName, ".") {
				http.Error(writer, "Anthropic namespace identity diverged", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"msg_namespace","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
		}))
		t.Cleanup(provider.Close)
		target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
		if err != nil {
			t.Fatalf("create Anthropic outbound: %v", err)
		}
		runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses", requestBody)
	})
}

func TestResponsesStrictOptionalNullIsRemovedFromAnthropicStreamOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_schema_stream", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}},
		})
		writeResponsesFunctionAdded(t, writer, 1, 0, "fc_schema_stream", "call_schema_stream", "weather")
		writeResponsesFunctionDelta(t, writer, 2, 0, "fc_schema_stream", `{"location":"Paris",`)
		writeResponsesFunctionDelta(t, writer, 3, 0, "fc_schema_stream", `"unit":null}`)
		writeResponsesFunctionDone(t, writer, 4, 0, "fc_schema_stream", "call_schema_stream", "weather", `{"location":"Paris","unit":null}`)
		writeNamedSSEJSON(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 6,
			"response": map[string]any{
				"id": "resp_schema_stream", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
			},
		})
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	result := runSchemaPipeline(t, provider, anthropic.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/messages",
		`{"model":"fixture-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"weather"}],"tools":[{"name":"weather","strict":true,"input_schema":{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string"}},"required":["location"],"additionalProperties":false}}]}`)
	if !result.Stream || result.EventStream == nil {
		t.Fatalf("unexpected stream result: %#v", result)
	}
	defer result.EventStream.Close()
	var arguments strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || len(event.Data) == 0 || bytes.Equal(event.Data, []byte("[DONE]")) {
			continue
		}
		var wire struct {
			Type  string `json:"type"`
			Delta *struct {
				Type        string `json:"type"`
				PartialJSON string `json:"partial_json"`
			} `json:"delta"`
		}
		if json.Unmarshal(event.Data, &wire) == nil && wire.Type == "content_block_delta" && wire.Delta != nil && wire.Delta.Type == "input_json_delta" {
			arguments.WriteString(wire.Delta.PartialJSON)
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume Anthropic schema stream: %v", err)
	}
	if arguments.String() != `{"location":"Paris"}` {
		t.Fatalf("Anthropic stream arguments = %q, want optional null removed", arguments.String())
	}
}

// A Responses provider can require function_call.arguments to be returned
// byte-for-byte on a later tool-result turn. This uses the production shape:
// Chat -> Responses for the model turn and again for its tool-result
// continuation. In particular, schema compatibility cleanup must not
// re-marshal the provider-issued argument string before the Chat client sends
// that history back.
func TestResponsesFunctionArgumentsRemainByteExactAcrossReroutedToolTurnOverRealHTTP(t *testing.T) {
	const originalArguments = `{
  "path": "README.md",
  "recursive": null
}`
	var firstTurnReceived, secondTurnReceived bool
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/v1/responses":
			body, err := io.ReadAll(request.Body)
			if err != nil {
				http.Error(writer, "read Responses request", http.StatusBadRequest)
				return
			}
			var payload struct {
				Input []json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(body, &payload); err != nil {
				http.Error(writer, "decode Responses request", http.StatusBadRequest)
				return
			}
			var call struct {
				Type      string `json:"type"`
				Arguments string `json:"arguments"`
			}
			for _, item := range payload.Input {
				if json.Unmarshal(item, &call) == nil && call.Type == "function_call" {
					break
				}
			}
			writer.Header().Set("Content-Type", "application/json")
			if call.Type != "function_call" {
				firstTurnReceived = true
				_, _ = io.WriteString(writer, `{"id":"resp_first_turn","object":"response","status":"completed","model":"fixture-model","output":[{"id":"fc_read","type":"function_call","call_id":"call_read","name":"read_file","arguments":"{\n  \"path\": \"README.md\",\n  \"recursive\": null\n}","status":"completed"}]}`)
				return
			}
			if call.Arguments != originalArguments {
				http.Error(writer, "function_call.arguments was changed before its tool result", http.StatusBadRequest)
				return
			}
			secondTurnReceived = true
			_, _ = io.WriteString(writer, `{"id":"resp_second_turn","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_done","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}]}`)
		default:
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
		}
	}))
	t.Cleanup(provider.Close)

	responsesTarget, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	first := runSchemaPipeline(t, provider, openai.NewInboundTransformer(), conversion.NewOutbound(responsesTarget), "/v1/chat/completions",
		`{"model":"fixture-model","messages":[{"role":"user","content":"read the file"}],"tools":[{"type":"function","function":{"name":"read_file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}}}]}`)
	var firstWire struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(first.Response.Body, &firstWire); err != nil || len(firstWire.Choices) != 1 || len(firstWire.Choices[0].Message.ToolCalls) != 1 {
		t.Fatalf("decode first Responses tool call: %v; body=%s", err, first.Response.Body)
	}
	firstCall := firstWire.Choices[0].Message.ToolCalls[0]
	if firstCall.Function.Arguments != `{"path":"README.md"}` {
		t.Fatalf("Chat client did not receive source-native arguments: got %q", firstCall.Function.Arguments)
	}

	secondBody, err := json.Marshal(map[string]any{
		"model": "fixture-model",
		"messages": []any{
			map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
				"id": firstCall.ID, "type": "function",
				"function": map[string]any{"name": firstCall.Function.Name, "arguments": firstCall.Function.Arguments},
			}}},
			map[string]any{"role": "tool", "tool_call_id": firstCall.ID, "content": "file contents"},
		},
		"tools": []any{map[string]any{"type": "function", "function": map[string]any{
			"name": "read_file", "parameters": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}, "recursive": map[string]any{"type": "boolean"}}, "required": []string{"path"}, "additionalProperties": false},
		}}},
	})
	if err != nil {
		t.Fatalf("encode Chat tool-result turn: %v", err)
	}
	_ = runSchemaPipeline(t, provider, openai.NewInboundTransformer(), conversion.NewOutbound(responsesTarget), "/v1/chat/completions", string(secondBody))
	if !firstTurnReceived || !secondTurnReceived {
		t.Fatal("Responses provider did not receive both byte-exact tool turns")
	}
}

func TestResponsesStreamFunctionArgumentsReplayRawBytesOnChatContinuationOverRealHTTP(t *testing.T) {
	const originalArguments = `{
  "path": "README.md",
  "recursive": null
}`
	var secondTurnReceived bool
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var probe struct {
			Stream bool              `json:"stream"`
			Input  []json.RawMessage `json:"input"`
		}
		if json.NewDecoder(request.Body).Decode(&probe) != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		if probe.Stream {
			writer.Header().Set("Content-Type", "text/event-stream")
			writeNamedSSEJSON(t, writer, "response.created", map[string]any{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_stream_args", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}}})
			writeResponsesFunctionAdded(t, writer, 1, 0, "fc_stream_args", "call_stream_args", "read_file")
			writeResponsesFunctionDelta(t, writer, 2, 0, "fc_stream_args", "{\n  \"path\": \"README.md\",\n")
			writeResponsesFunctionDelta(t, writer, 3, 0, "fc_stream_args", "  \"recursive\": null\n}")
			writeResponsesFunctionDone(t, writer, 4, 0, "fc_stream_args", "call_stream_args", "read_file", originalArguments)
			writeNamedSSEJSON(t, writer, "response.completed", map[string]any{"type": "response.completed", "sequence_number": 6, "response": map[string]any{"id": "resp_stream_args", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{}}})
			return
		}
		for _, item := range probe.Input {
			var call struct{ Type, Arguments string }
			if json.Unmarshal(item, &call) == nil && call.Type == "function_call" {
				if call.Arguments != originalArguments {
					http.Error(writer, "streamed raw arguments were not restored", http.StatusBadRequest)
					return
				}
				secondTurnReceived = true
			}
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_stream_second","object":"response","status":"completed","model":"fixture-model","output":[]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx := shared.WithSessionScope(context.Background(), "fixture-tenant")
	first, err := pipeline.NewFactory(executor).Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(target)).Process(ctx, &httpclient.Request{Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}}, Metadata: map[string]string{"axon.trusted_continuity_identity": "fixture-tenant"}, Body: []byte(`{"model":"fixture-model","stream":true,"messages":[{"role":"user","content":"read"}],"tools":[{"type":"function","function":{"name":"read_file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}}}]}`)})
	if err != nil || first.EventStream == nil {
		t.Fatalf("start Responses stream: %v", err)
	}
	defer first.EventStream.Close()
	var clientArguments strings.Builder
	for first.EventStream.Next() {
		var event struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal(first.EventStream.Current().Data, &event) == nil {
			for _, choice := range event.Choices {
				for _, call := range choice.Delta.ToolCalls {
					clientArguments.WriteString(call.Function.Arguments)
				}
			}
		}
	}
	if err := first.EventStream.Err(); err != nil {
		t.Fatalf("consume Responses stream: %v", err)
	}
	if clientArguments.String() != `{"path":"README.md"}` {
		t.Fatalf("stream client args = %q", clientArguments.String())
	}
	secondBody := `{"model":"fixture-model","messages":[{"role":"assistant","tool_calls":[{"id":"call_stream_args","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]},{"role":"tool","tool_call_id":"call_stream_args","content":"ok"}],"tools":[{"type":"function","function":{"name":"read_file","strict":true,"parameters":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}}}]}`
	_ = runSchemaPipeline(t, provider, openai.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/chat/completions", secondBody)
	if !secondTurnReceived {
		t.Fatal("Responses provider did not receive raw streamed arguments")
	}
}

func TestAnthropicFunctionArgumentsReplayRawBytesToResponsesOverRealHTTP(t *testing.T) {
	const originalArguments = `{
  "path": "README.md",
  "recursive": null
}`
	var secondTurnReceived bool
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var payload struct {
			Input []json.RawMessage `json:"input"`
		}
		if json.NewDecoder(request.Body).Decode(&payload) != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		for _, item := range payload.Input {
			var call struct{ Type, Arguments string }
			if json.Unmarshal(item, &call) == nil && call.Type == "function_call" {
				if call.Arguments != originalArguments {
					http.Error(writer, "Anthropic continuation lost raw arguments", http.StatusBadRequest)
					return
				}
				secondTurnReceived = true
			}
		}
		writer.Header().Set("Content-Type", "application/json")
		if !secondTurnReceived {
			_, _ = io.WriteString(writer, `{"id":"resp_anthropic_first","object":"response","status":"completed","model":"fixture-model","output":[{"id":"fc_anthropic","type":"function_call","call_id":"call_anthropic","name":"read_file","arguments":"{\n  \"path\": \"README.md\",\n  \"recursive\": null\n}","status":"completed"}]}`)
			return
		}
		_, _ = io.WriteString(writer, `{"id":"resp_anthropic_second","object":"response","status":"completed","model":"fixture-model","output":[]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	first := runSchemaPipeline(t, provider, anthropic.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/messages", `{"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":"read"}],"tools":[{"name":"read_file","strict":true,"input_schema":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}}]}`)
	var firstWire struct {
		Content []struct {
			Type, ID string
			Input    json.RawMessage `json:"input"`
		} `json:"content"`
	}
	if err := json.Unmarshal(first.Response.Body, &firstWire); err != nil || len(firstWire.Content) != 1 {
		t.Fatalf("decode Anthropic tool use: %v body=%s", err, first.Response.Body)
	}
	if string(firstWire.Content[0].Input) != `{"path":"README.md"}` {
		t.Fatalf("Anthropic client args = %s", firstWire.Content[0].Input)
	}
	secondBody := `{"model":"fixture-model","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_anthropic","name":"read_file","input":{"path":"README.md"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_anthropic","content":"ok"}]}],"tools":[{"name":"read_file","strict":true,"input_schema":{"type":"object","properties":{"path":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}}]}`
	_ = runSchemaPipeline(t, provider, anthropic.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/messages", secondBody)
	if !secondTurnReceived {
		t.Fatal("Responses provider did not receive Anthropic raw arguments")
	}
}

func TestCrossProtocolStrictDefaultsReachExplicitWireValuesOverRealHTTP(t *testing.T) {
	t.Parallel()
	t.Run("Chat omitted strict to Responses false", func(t *testing.T) {
		t.Parallel()
		provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var payload struct {
				Tools []struct {
					Strict *bool `json:"strict"`
				} `json:"tools"`
			}
			if json.NewDecoder(request.Body).Decode(&payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Strict == nil || *payload.Tools[0].Strict {
				http.Error(writer, "Chat non-strict default was not made explicit", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"resp_default","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_default","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
		}))
		t.Cleanup(provider.Close)
		target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
		if err != nil {
			t.Fatalf("create Responses outbound: %v", err)
		}
		runSchemaPipeline(t, provider, openai.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/chat/completions",
			`{"model":"fixture-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}}]}`)
	})

	t.Run("Responses omitted strict to Anthropic true", func(t *testing.T) {
		t.Parallel()
		provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			var payload struct {
				Tools []struct {
					Strict *bool `json:"strict"`
				} `json:"tools"`
			}
			if json.NewDecoder(request.Body).Decode(&payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Strict == nil || !*payload.Tools[0].Strict {
				http.Error(writer, "Responses auto-strict default was not preserved", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"msg_default","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`)
		}))
		t.Cleanup(provider.Close)
		target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
		if err != nil {
			t.Fatalf("create Anthropic outbound: %v", err)
		}
		runSchemaPipeline(t, provider, responses.NewInboundTransformer(), conversion.NewOutbound(target), "/v1/responses",
			`{"model":"fixture-model","input":"hi","tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}}]}`)
	})
}

func runSchemaPipeline(
	t *testing.T,
	provider *httptest.Server,
	inbound transformer.Inbound,
	outbound transformer.Outbound,
	path, body string,
) *pipeline.Result {
	t.Helper()
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	// Trusted SessionScope is required for multi-turn argument continuity:
	// Capture/Resolve are no-ops without a host-provided tenant namespace.
	ctx := shared.WithSessionScope(context.Background(), "fixture-tenant")
	result, err := pipeline.NewFactory(executor).
		Pipeline(inbound, outbound).
		Process(ctx, &httpclient.Request{
			Method: http.MethodPost, URL: path,
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(body),
			Metadata: map[string]string{"axon.trusted_continuity_identity": "fixture-tenant"},
		})
	if err != nil {
		t.Fatalf("run schema conversion pipeline: %v", err)
	}
	if result == nil || result.Response == nil || result.Response.StatusCode != http.StatusOK {
		if result != nil && result.Stream && result.EventStream != nil {
			return result
		}
		t.Fatalf("unexpected schema conversion response: %#v", result)
	}
	return result
}

func jsonStringSet(value any) map[string]bool {
	result := make(map[string]bool)
	values, _ := value.([]any)
	for _, item := range values {
		if text, ok := item.(string); ok {
			result[text] = true
		}
	}
	return result
}

func jsonArrayContains(values []any, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

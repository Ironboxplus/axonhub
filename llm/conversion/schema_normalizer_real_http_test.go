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

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
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
	result, err := pipeline.NewFactory(executor).
		Pipeline(inbound, outbound).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: path,
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(body),
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

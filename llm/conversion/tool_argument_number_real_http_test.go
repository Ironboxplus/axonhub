package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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

const numericToolArguments = `{"timeout_ms":1000.0,"scientific":1e3,"negative_zero":-0.0,"nested":{"array":[1000.0,1e3,-0.0,9007199254740993.0]},"fraction":12.5,"tiny":1e-3,"string":"1000.0","1000.0":"key"}`
const numericToolArgumentsWant = `{"timeout_ms":1000,"scientific":1000,"negative_zero":0,"nested":{"array":[1000,1000,0,9007199254740993]},"fraction":12.5,"tiny":1e-3,"string":"1000.0","1000.0":"key"}`

type numericToolProtocol struct {
	name        string
	path        string
	requestBody string
	newInbound  func() transformer.Inbound
	newOutbound func(string) (transformer.Outbound, error)
}

func TestFunctionToolIntegralNumbersAreCanonicalAcrossRealHTTP3x3(t *testing.T) {
	for _, providerCase := range numericToolProtocols(false) {
		for _, clientCase := range numericToolProtocols(false) {
			providerCase, clientCase := providerCase, clientCase
			t.Run(providerCase.name+"_to_"+clientCase.name, func(t *testing.T) {
				t.Parallel()
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
					if request.URL.Path != providerCase.path {
						http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
						return
					}
					serveNumericToolHTTP(t, providerCase.name, writer, request)
				}))
				t.Cleanup(provider.Close)
				providerOutbound, err := providerCase.newOutbound(provider.URL)
				if err != nil {
					t.Fatalf("create %s provider: %v", providerCase.name, err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				result, err := pipeline.NewFactory(executor).Pipeline(clientCase.newInbound(), conversion.NewOutbound(providerOutbound)).Process(context.Background(), numericToolRequest(clientCase))
				if err != nil {
					t.Fatalf("run %s-to-%s HTTP fixture: %v", clientCase.name, providerCase.name, err)
				}
				if result == nil || result.Response == nil || result.Stream {
					t.Fatalf("unexpected HTTP pipeline result: %#v", result)
				}
				assertNumericToolClientWire(t, clientCase.name, result.Response.Body)
			})
		}
	}
}

func TestFunctionToolIntegralNumbersAreCanonicalAcrossRealSSE3x3(t *testing.T) {
	for _, providerCase := range numericToolProtocols(true) {
		for _, clientCase := range numericToolProtocols(true) {
			providerCase, clientCase := providerCase, clientCase
			t.Run(providerCase.name+"_to_"+clientCase.name, func(t *testing.T) {
				t.Parallel()
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
					if request.URL.Path != providerCase.path {
						http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
						return
					}
					writer.Header().Set("Content-Type", "text/event-stream")
					serveNumericToolSSE(t, providerCase.name, writer, request)
				}))
				t.Cleanup(provider.Close)
				providerOutbound, err := providerCase.newOutbound(provider.URL)
				if err != nil {
					t.Fatalf("create %s provider: %v", providerCase.name, err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				result, err := pipeline.NewFactory(executor).Pipeline(clientCase.newInbound(), conversion.NewOutbound(providerOutbound)).Process(context.Background(), numericToolRequest(clientCase))
				if err != nil {
					t.Fatalf("start %s-to-%s SSE fixture: %v", clientCase.name, providerCase.name, err)
				}
				if result == nil || result.EventStream == nil || !result.Stream {
					t.Fatalf("unexpected SSE pipeline result: %#v", result)
				}
				defer result.EventStream.Close()
				events := make([]*httpclient.StreamEvent, 0, 16)
				for result.EventStream.Next() {
					event := result.EventStream.Current()
					if event == nil {
						continue
					}
					events = append(events, &httpclient.StreamEvent{Type: event.Type, Data: append([]byte(nil), event.Data...)})
				}
				if err := result.EventStream.Err(); err != nil {
					t.Fatalf("consume %s-to-%s SSE fixture: %v", clientCase.name, providerCase.name, err)
				}
				aggregated, _, err := clientCase.newInbound().AggregateStreamChunks(context.Background(), events)
				if err != nil {
					t.Fatalf("aggregate %s output: %v", clientCase.name, err)
				}
				assertNumericToolClientWire(t, clientCase.name, aggregated)
			})
		}
	}
}

func TestToolArgumentNumericCanonicalizationEmitsPayloadFreeObjectEvidence(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
			return
		}
		serveNumericToolHTTP(t, "responses", writer, request)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses provider: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	observer := &observationRecorder{}
	ctx := llm.WithConversionDebugTrace(llm.WithConversionTrace(context.Background()), []byte(t.Name()), 8)
	client := numericToolProtocols(false)[1]
	result, err := pipeline.NewFactory(executor).
		Pipeline(client.newInbound(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).
		Process(ctx, numericToolRequest(client))
	if err != nil || result == nil || result.Response == nil {
		t.Fatalf("run observable numeric fixture: result=%#v err=%v", result, err)
	}
	assertNumericToolClientWire(t, "responses", result.Response.Body)

	found := false
	for _, event := range observer.snapshot() {
		if event.Stage != pipeline.StageConversionRestore || event.Conversion == nil || event.ConversionDebug == nil {
			continue
		}
		if event.Conversion.ToolArgumentsCanonicalized != 1 {
			t.Fatalf("numeric canonicalization summary = %#v", event.Conversion)
		}
		for _, action := range event.ConversionDebug.Actions {
			if action.Strategy != "tool_argument_numeric_canonicalization" {
				continue
			}
			found = action.ObjectKind == "tool_call" && action.FieldPath == "output[0].tool_call.arguments" &&
				action.Action == "normalize" && action.Reversible
			encoded, marshalErr := json.Marshal(action)
			if marshalErr != nil || bytes.Contains(encoded, []byte("1000.0")) || bytes.Contains(encoded, []byte("record_numbers")) {
				t.Fatalf("numeric canonicalization evidence leaked payload: err=%v evidence=%s", marshalErr, encoded)
			}
		}
	}
	if !found {
		t.Fatal("missing payload-free tool argument numeric canonicalization evidence")
	}
}

func TestCollaborationNamespaceFunctionArgumentsCanonicalizeOverRealHTTP(t *testing.T) {
	for _, provider := range []struct {
		name string
		path string
		new  func(string) (transformer.Outbound, error)
	}{
		{name: "chat", path: "/v1/chat/completions", new: func(baseURL string) (transformer.Outbound, error) {
			return openai.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "responses", path: "/v1/responses", new: func(baseURL string) (transformer.Outbound, error) {
			return responses.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "anthropic", path: "/v1/messages", new: func(baseURL string) (transformer.Outbound, error) {
			return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
		}},
	} {
		provider := provider
		t.Run(provider.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != provider.path {
					http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
					return
				}
				serveNumericCollaborationTool(t, provider.name, writer, request)
			}))
			t.Cleanup(server.Close)
			target, err := provider.new(server.URL)
			if err != nil {
				t.Fatalf("create %s provider: %v", provider.name, err)
			}
			executor := httpclient.NewHttpClientWithClient(server.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), numericCollaborationRequest())
			if err != nil || result == nil || result.Response == nil {
				t.Fatalf("run %s collaboration fixture: result=%#v err=%v", provider.name, result, err)
			}
			assertNumericCollaborationResponse(t, result.Response.Body)
		})
	}
}

func TestCodexV2CollaborationWaitAgentArgumentsCanonicalizeOverRealSSE(t *testing.T) {
	for _, provider := range []struct {
		name string
		path string
		new  func(string) (transformer.Outbound, error)
	}{
		{name: "chat", path: "/v1/chat/completions", new: func(baseURL string) (transformer.Outbound, error) {
			return openai.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "responses", path: "/v1/responses", new: func(baseURL string) (transformer.Outbound, error) {
			return responses.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "anthropic", path: "/v1/messages", new: func(baseURL string) (transformer.Outbound, error) {
			return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
		}},
	} {
		provider := provider
		t.Run(provider.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != provider.path {
					http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
					return
				}
				writer.Header().Set("Content-Type", "text/event-stream")
				serveNumericCollaborationSSE(t, provider.name, writer, request)
			}))
			t.Cleanup(server.Close)
			target, err := provider.new(server.URL)
			if err != nil {
				t.Fatalf("create %s provider: %v", provider.name, err)
			}
			executor := httpclient.NewHttpClientWithClient(server.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), numericCollaborationStreamRequest())
			if err != nil || result == nil || result.EventStream == nil {
				t.Fatalf("start %s collaboration stream: result=%#v err=%v", provider.name, result, err)
			}
			defer result.EventStream.Close()
			events := make([]*httpclient.StreamEvent, 0, 16)
			for result.EventStream.Next() {
				event := result.EventStream.Current()
				if event != nil {
					events = append(events, &httpclient.StreamEvent{Type: event.Type, Data: append([]byte(nil), event.Data...)})
				}
			}
			if err := result.EventStream.Err(); err != nil {
				t.Fatalf("consume %s collaboration stream: %v", provider.name, err)
			}
			aggregated, _, err := responses.NewInboundTransformer().AggregateStreamChunks(context.Background(), events)
			if err != nil {
				t.Fatalf("aggregate collaboration stream: %v", err)
			}
			assertNumericCollaborationResponse(t, aggregated)
		})
	}
}

func TestCustomToolFreeformInputNeverReceivesJSONNumberRewritingOverRealHTTP(t *testing.T) {
	const customInput = `timeout_ms=1000.0; scale=1e3; this is freeform text`
	for _, provider := range []struct {
		name string
		path string
		new  func(string) (transformer.Outbound, error)
	}{
		{name: "chat", path: "/v1/chat/completions", new: func(baseURL string) (transformer.Outbound, error) {
			return openai.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "responses", path: "/v1/responses", new: func(baseURL string) (transformer.Outbound, error) {
			return responses.NewOutboundTransformer(baseURL, "fixture-key")
		}},
		{name: "anthropic", path: "/v1/messages", new: func(baseURL string) (transformer.Outbound, error) {
			return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
		}},
	} {
		provider := provider
		t.Run(provider.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != provider.path {
					http.Error(writer, "unexpected provider endpoint", http.StatusNotFound)
					return
				}
				serveFreeformCustomTool(t, provider.name, customInput, writer, request)
			}))
			t.Cleanup(server.Close)
			target, err := provider.new(server.URL)
			if err != nil {
				t.Fatalf("create %s provider: %v", provider.name, err)
			}
			executor := httpclient.NewHttpClientWithClient(server.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), numericCustomRequest())
			if err != nil || result == nil || result.Response == nil {
				t.Fatalf("run %s custom fixture: result=%#v err=%v", provider.name, result, err)
			}
			var response struct {
				Output []struct {
					Type, Name, Input string
				} `json:"output"`
			}
			if err := json.Unmarshal(result.Response.Body, &response); err != nil || len(response.Output) != 1 {
				t.Fatalf("decode custom response: err=%v body=%s response=%#v", err, result.Response.Body, response)
			}
			call := response.Output[0]
			if call.Type != "custom_tool_call" || call.Name != "raw_numbers" || call.Input != customInput {
				t.Fatalf("custom freeform input was altered: %#v", call)
			}
		})
	}
}

func numericToolProtocols(stream bool) []numericToolProtocol {
	streamField := ""
	if stream {
		streamField = `,"stream":true`
	}
	return []numericToolProtocol{
		{name: "chat", path: "/v1/chat/completions", requestBody: `{"model":"numeric-model"` + streamField + `,"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"record_numbers","description":"record numeric arguments","parameters":{"type":"object"}}}]}`,
			newInbound: func() transformer.Inbound { return openai.NewInboundTransformer() }, newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			}},
		{name: "responses", path: "/v1/responses", requestBody: `{"model":"numeric-model"` + streamField + `,"input":"run","tools":[{"type":"function","name":"record_numbers","description":"record numeric arguments","parameters":{"type":"object"}}]}`,
			newInbound: func() transformer.Inbound { return responses.NewInboundTransformer() }, newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			}},
		{name: "anthropic", path: "/v1/messages", requestBody: `{"model":"numeric-model"` + streamField + `,"max_tokens":128,"messages":[{"role":"user","content":"run"}],"tools":[{"name":"record_numbers","description":"record numeric arguments","input_schema":{"type":"object"}}]}`,
			newInbound: func() transformer.Inbound { return anthropic.NewInboundTransformer() }, newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			}},
	}
}

func numericToolRequest(protocol numericToolProtocol) *httpclient.Request {
	return &httpclient.Request{Method: http.MethodPost, URL: protocol.path, Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(protocol.requestBody)}
}

func numericCollaborationRequest() *httpclient.Request {
	return &httpclient.Request{Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}, responses.ResponsesLiteHeader: []string{"true"}}, Body: []byte(`{
"model":"numeric-model","input":[
 {"type":"additional_tools","role":"developer","tools":[
  {"type":"namespace","name":"collaboration","tools":[
   {"type":"function","name":"wait_agent","description":"Wait agent.","strict":false,"parameters":{"type":"object","properties":{"timeout_ms":{"type":"number"}},"additionalProperties":false}}
  ]}
 ]},
 {"type":"message","role":"user","content":[{"type":"input_text","text":"send"}]}
]}`)}
}

func numericCollaborationStreamRequest() *httpclient.Request {
	request := numericCollaborationRequest()
	request.Body = bytes.Replace(request.Body, []byte(`"model":"numeric-model"`), []byte(`"model":"numeric-model","stream":true`), 1)
	return request
}

func numericCustomRequest() *httpclient.Request {
	return &httpclient.Request{Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{
"model":"numeric-model","input":"run","tools":[{"type":"custom","name":"raw_numbers","description":"accept freeform text"}]}`)}
}

func providerFunctionName(t *testing.T, provider string, request *http.Request) string {
	t.Helper()
	defer request.Body.Close()
	var body struct {
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil || len(body.Tools) != 1 {
		t.Fatalf("decode %s provider tool: err=%v tools=%d", provider, err, len(body.Tools))
	}
	if provider == "anthropic" || provider == "responses" {
		var tool struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal(body.Tools[0], &tool); err != nil || tool.Name == "" {
			t.Fatalf("decode %s native function tool: err=%v tool=%s", provider, err, body.Tools[0])
		}
		return tool.Name
	}
	var wrapper struct {
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(body.Tools[0], &wrapper); err != nil || wrapper.Function.Name == "" {
		t.Fatalf("decode Chat function tool: err=%v tool=%s", err, body.Tools[0])
	}
	return wrapper.Function.Name
}

func serveNumericCollaborationTool(t *testing.T, provider string, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	writer.Header().Set("Content-Type", "application/json")
	name := "wait_agent"
	if provider == "responses" {
		defer request.Body.Close()
		if _, err := io.Copy(io.Discard, request.Body); err != nil {
			t.Fatalf("read Responses collaboration request: %v", err)
		}
	} else {
		name = providerFunctionName(t, provider, request)
	}
	switch provider {
	case "chat":
		_, _ = fmt.Fprintf(writer, `{"id":"chat_collab","object":"chat.completion","model":"numeric-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_collab","type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}]}`, name, numericToolArguments)
	case "responses":
		_, _ = fmt.Fprintf(writer, `{"id":"resp_collab","object":"response","model":"numeric-model","status":"completed","output":[{"id":"fc_collab","type":"function_call","status":"completed","call_id":"call_collab","name":%q,"namespace":"collaboration","arguments":%q,"execution":"client"}]}`, name, numericToolArguments)
	case "anthropic":
		_, _ = fmt.Fprintf(writer, `{"id":"msg_collab","type":"message","role":"assistant","model":"numeric-model","content":[{"type":"tool_use","id":"call_collab","name":%q,"input":%s}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, name, numericToolArguments)
	default:
		t.Fatalf("unknown collaboration provider %q", provider)
	}
}

func serveNumericCollaborationSSE(t *testing.T, provider string, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	name := "wait_agent"
	if provider == "responses" {
		defer request.Body.Close()
		if _, err := io.Copy(io.Discard, request.Body); err != nil {
			t.Fatalf("read Responses collaboration stream request: %v", err)
		}
	} else {
		name = providerFunctionName(t, provider, request)
	}
	first, second := splitNumericToolArguments()
	switch provider {
	case "chat":
		writeSSEJSON(t, writer, map[string]any{
			"id": "chat_collab", "object": "chat.completion.chunk", "model": "numeric-model",
			"choices": []any{map[string]any{
				"index": 0,
				"delta": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
					"index": 0, "id": "call_collab", "type": "function", "function": map[string]any{"name": name, "arguments": first},
				}}},
				"finish_reason": nil,
			}},
		})
		writeSSEJSON(t, writer, map[string]any{
			"id": "chat_collab", "object": "chat.completion.chunk", "model": "numeric-model",
			"choices": []any{map[string]any{
				"index": 0,
				"delta": map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "function": map[string]any{"arguments": second},
				}}},
				"finish_reason": nil,
			}},
		})
		writeSSEJSON(t, writer, map[string]any{"id": "chat_collab", "object": "chat.completion.chunk", "model": "numeric-model", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}}})
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
	case "responses":
		for _, event := range []map[string]any{
			{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_collab", "object": "response", "model": "numeric-model", "status": "in_progress", "output": []any{}}},
			{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": map[string]any{"id": "fc_collab", "type": "function_call", "status": "in_progress", "call_id": "call_collab", "name": name, "namespace": "collaboration", "arguments": "", "execution": "client"}},
			{"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0, "item_id": "fc_collab", "delta": first},
			{"type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 0, "item_id": "fc_collab", "delta": second},
			{"type": "response.function_call_arguments.done", "sequence_number": 4, "output_index": 0, "item_id": "fc_collab", "call_id": "call_collab", "name": name, "namespace": "collaboration", "arguments": numericToolArguments},
			{"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": map[string]any{"id": "fc_collab", "type": "function_call", "status": "completed", "call_id": "call_collab", "name": name, "namespace": "collaboration", "arguments": numericToolArguments, "execution": "client"}},
			{"type": "response.completed", "sequence_number": 6, "response": map[string]any{"id": "resp_collab", "object": "response", "model": "numeric-model", "status": "completed", "output": []any{}}},
		} {
			writeNamedSSEJSON(t, writer, event["type"].(string), event)
		}
	case "anthropic":
		writeNamedSSEJSON(t, writer, "message_start", map[string]any{"type": "message_start", "message": map[string]any{"id": "msg_collab", "type": "message", "role": "assistant", "model": "numeric-model", "content": []any{}, "usage": map[string]any{"input_tokens": 1, "output_tokens": 0}}})
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "tool_use", "id": "call_collab", "name": name, "input": map[string]any{}}})
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": first}})
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": second}})
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		writeNamedSSEJSON(t, writer, "message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 1}})
		writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
	default:
		t.Fatalf("unknown collaboration stream provider %q", provider)
	}
}

func assertNumericCollaborationResponse(t *testing.T, body []byte) {
	t.Helper()
	var response struct {
		Output []struct {
			Type, Name, Namespace, Arguments string
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil || len(response.Output) != 1 {
		t.Fatalf("decode collaboration response: err=%v body=%s response=%#v", err, body, response)
	}
	call := response.Output[0]
	if call.Type != "function_call" || call.Name != "wait_agent" || call.Namespace != "collaboration" || call.Arguments != numericToolArgumentsWant {
		t.Fatalf("Codex V2 collaboration call restored incorrectly: %#v", call)
	}
}

func serveFreeformCustomTool(t *testing.T, provider, input string, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	writer.Header().Set("Content-Type", "application/json")
	name := providerFunctionName(t, provider, request)
	switch provider {
	case "chat":
		_, _ = fmt.Fprintf(writer, `{"id":"chat_custom","object":"chat.completion","model":"numeric-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_custom","type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}]}`, name, `{"input":`+mustJSONString(t, input)+`}`)
	case "responses":
		_, _ = fmt.Fprintf(writer, `{"id":"resp_custom","object":"response","model":"numeric-model","status":"completed","output":[{"id":"custom_numeric","type":"custom_tool_call","status":"completed","call_id":"call_custom","name":%q,"input":%q,"execution":"client"}]}`, name, input)
	case "anthropic":
		_, _ = fmt.Fprintf(writer, `{"id":"msg_custom","type":"message","role":"assistant","model":"numeric-model","content":[{"type":"tool_use","id":"call_custom","name":%q,"input":{"input":%q}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, name, input)
	default:
		t.Fatalf("unknown custom provider %q", provider)
	}
}

func mustJSONString(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("encode fixture string: %v", err)
	}
	return string(encoded)
}

func serveNumericToolHTTP(t *testing.T, provider string, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	if _, err := io.ReadAll(request.Body); err != nil {
		t.Fatalf("read %s provider request: %v", provider, err)
	}
	writer.Header().Set("Content-Type", "application/json")
	switch provider {
	case "chat":
		_, _ = fmt.Fprintf(writer, `{"id":"chat_numeric","object":"chat.completion","model":"numeric-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_numeric","type":"function","function":{"name":"record_numbers","arguments":%q}}]},"finish_reason":"tool_calls"}]}`, numericToolArguments)
	case "responses":
		_, _ = fmt.Fprintf(writer, `{"id":"resp_numeric","object":"response","model":"numeric-model","status":"completed","output":[{"id":"fc_numeric","type":"function_call","status":"completed","call_id":"call_numeric","name":"record_numbers","arguments":%q,"execution":"client"}]}`, numericToolArguments)
	case "anthropic":
		_, _ = fmt.Fprintf(writer, `{"id":"msg_numeric","type":"message","role":"assistant","model":"numeric-model","content":[{"type":"tool_use","id":"call_numeric","name":"record_numbers","input":%s}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, numericToolArguments)
	default:
		t.Fatalf("unknown numeric provider %q", provider)
	}
}

func serveNumericToolSSE(t *testing.T, provider string, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	if _, err := io.ReadAll(request.Body); err != nil {
		t.Fatalf("read %s provider stream request: %v", provider, err)
	}
	switch provider {
	case "chat":
		first, second := splitNumericToolArguments()
		writeSSEJSON(t, writer, map[string]any{
			"id": "chat_numeric", "object": "chat.completion.chunk", "model": "numeric-model",
			"choices": []any{map[string]any{
				"index": 0,
				"delta": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
					"index": 0, "id": "call_numeric", "type": "function",
					"function": map[string]any{"name": "record_numbers", "arguments": first},
				}}},
				"finish_reason": nil,
			}},
		})
		writeSSEJSON(t, writer, map[string]any{
			"id": "chat_numeric", "object": "chat.completion.chunk", "model": "numeric-model",
			"choices": []any{map[string]any{
				"index": 0,
				"delta": map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "function": map[string]any{"arguments": second},
				}}},
				"finish_reason": nil,
			}},
		})
		writeSSEJSON(t, writer, map[string]any{"id": "chat_numeric", "object": "chat.completion.chunk", "model": "numeric-model", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}}})
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
	case "responses":
		first, second := splitNumericToolArguments()
		for _, event := range []map[string]any{
			{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_numeric", "object": "response", "model": "numeric-model", "status": "in_progress", "output": []any{}}},
			{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": map[string]any{"id": "fc_numeric", "type": "function_call", "status": "in_progress", "call_id": "call_numeric", "name": "record_numbers", "arguments": "", "execution": "client"}},
			{"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0, "item_id": "fc_numeric", "delta": first},
			{"type": "response.function_call_arguments.delta", "sequence_number": 3, "output_index": 0, "item_id": "fc_numeric", "delta": second},
			{"type": "response.function_call_arguments.done", "sequence_number": 4, "output_index": 0, "item_id": "fc_numeric", "call_id": "call_numeric", "name": "record_numbers", "arguments": numericToolArguments},
			{"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": map[string]any{"id": "fc_numeric", "type": "function_call", "status": "completed", "call_id": "call_numeric", "name": "record_numbers", "arguments": numericToolArguments, "execution": "client"}},
			{"type": "response.completed", "sequence_number": 6, "response": map[string]any{"id": "resp_numeric", "object": "response", "model": "numeric-model", "status": "completed", "output": []any{}}},
		} {
			writeNamedSSEJSON(t, writer, event["type"].(string), event)
		}
	case "anthropic":
		first, second := splitNumericToolArguments()
		writeNamedSSEJSON(t, writer, "message_start", map[string]any{"type": "message_start", "message": map[string]any{"id": "msg_numeric", "type": "message", "role": "assistant", "model": "numeric-model", "content": []any{}, "usage": map[string]any{"input_tokens": 1, "output_tokens": 0}}})
		writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{"type": "content_block_start", "index": 0, "content_block": map[string]any{"type": "tool_use", "id": "call_numeric", "name": "record_numbers", "input": map[string]any{}}})
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": first}})
		writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "input_json_delta", "partial_json": second}})
		writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
		writeNamedSSEJSON(t, writer, "message_delta", map[string]any{"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 1}})
		writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
	default:
		t.Fatalf("unknown numeric stream provider %q", provider)
	}
}

func splitNumericToolArguments() (string, string) {
	// Split within the first integral decimal token. The test proves the
	// streaming boundary never leaks the first `1000.` fragment to clients.
	const splitAt = len(`{"timeout_ms":1000.`)
	return numericToolArguments[:splitAt], numericToolArguments[splitAt:]
}

func assertNumericToolClientWire(t *testing.T, protocol string, body []byte) {
	t.Helper()
	var got string
	switch protocol {
	case "chat":
		var response struct {
			Choices []struct {
				Message struct {
					ToolCalls []struct {
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(body, &response); err != nil || len(response.Choices) != 1 || len(response.Choices[0].Message.ToolCalls) != 1 {
			t.Fatalf("decode Chat numeric response: err=%v body=%s response=%#v", err, body, response)
		}
		got = response.Choices[0].Message.ToolCalls[0].Function.Arguments
	case "responses":
		var response struct {
			Output []struct {
				Arguments string `json:"arguments"`
			} `json:"output"`
		}
		if err := json.Unmarshal(body, &response); err != nil || len(response.Output) != 1 {
			t.Fatalf("decode Responses numeric response: err=%v body=%s response=%#v", err, body, response)
		}
		got = response.Output[0].Arguments
	case "anthropic":
		var response struct {
			Content []struct {
				Input json.RawMessage `json:"input"`
			} `json:"content"`
		}
		if err := json.Unmarshal(body, &response); err != nil || len(response.Content) != 1 {
			t.Fatalf("decode Anthropic numeric response: err=%v body=%s response=%#v", err, body, response)
		}
		got = string(response.Content[0].Input)
	default:
		t.Fatalf("unknown numeric client protocol %q", protocol)
	}
	if got != numericToolArgumentsWant {
		t.Fatalf("%s client function arguments = %s\nwant %s\nfull body: %s", protocol, got, numericToolArgumentsWant, bytes.TrimSpace(body))
	}
}

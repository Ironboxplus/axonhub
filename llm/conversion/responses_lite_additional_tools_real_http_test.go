package conversion_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestLatestCodexResponsesLiteAdditionalToolsPreserveNamespaceMetadataOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read provider request", http.StatusBadRequest)
			return
		}
		if err := validateLatestCodexResponsesLiteWire(body); err != nil {
			writer.Header().Set("Content-Type", "application/json")
			writer.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(writer, `{"error":{"message":"`+err.Error()+`","type":"invalid_request_error","code":"missing_required_parameter"}}`)
			return
		}
		if !strings.Contains(string(body), `"x_codex_marker":"preserve-input-envelope"`) ||
			!strings.Contains(string(body), `"x_namespace_marker":"preserve-namespace-envelope"`) {
			http.Error(writer, "Responses Lite private metadata was not preserved", http.StatusBadRequest)
			return
		}

		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_codex_lite","object":"response","created_at":1,"status":"completed","model":"gpt-5.6-sol","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(provider.Close)

	observer := &observationRecorder{}
	result, err := runResponsesLiteRequest(t, provider, observer, codexResponsesLiteRequest(true))
	if err != nil {
		t.Fatalf("latest Codex Responses Lite request failed: %v", err)
	}
	if result == nil || result.Response == nil {
		t.Fatalf("missing Responses result: %#v", result)
	}

	var foundIdentityEvidence bool
	for _, event := range observer.snapshot() {
		if event.Conversion == nil {
			continue
		}
		if event.Conversion.Unknown != 0 || !event.Conversion.Complete {
			t.Fatalf("identity Responses conversion became incomplete: %#v", event.Conversion)
		}
		if event.Conversion.Opaque > 0 {
			foundIdentityEvidence = true
		}
	}
	if !foundIdentityEvidence {
		t.Fatalf("missing same-protocol raw-envelope preservation evidence: %#v", observer.snapshot())
	}
}

func TestResponsesLiteRejectsUnrepairableAdditionalToolDescriptionsBeforeProviderHTTP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		description string
	}{
		{name: "function", description: `"description":"Wait for work.",`},
		{name: "custom", description: `"description":"Run JavaScript code to orchestrate tool calls.",`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				hits.Add(1)
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				if err := validateLatestCodexResponsesLiteWire(body); err != nil {
					http.Error(writer, err.Error(), http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"id":"resp_unexpected","object":"response","created_at":1,"status":"completed","model":"gpt-5.6-sol","output":[],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
			}))
			t.Cleanup(provider.Close)

			body := strings.Replace(string(codexResponsesLiteRequest(true)), test.description, "", 1)
			_, err := runResponsesLiteRequest(t, provider, &observationRecorder{}, []byte(body))
			if err == nil {
				t.Fatal("missing additional_tools description unexpectedly reached a successful provider response")
			}
			if got := hits.Load(); got != 0 {
				t.Fatalf("provider requests = %d, want 0 for local wire rejection", got)
			}
		})
	}
}

func validateLatestCodexResponsesLiteWire(body []byte) error {
	var payload struct {
		Input []struct {
			Type  string            `json:"type"`
			Tools []json.RawMessage `json:"tools"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return err
	}
	if len(payload.Input) == 0 || payload.Input[0].Type != "additional_tools" {
		return &responsesWireFixtureError{path: "input[0]", message: "must be additional_tools"}
	}
	for index, tool := range payload.Input[0].Tools {
		if err := validateLatestCodexTool(tool, "input[0].tools["+strconv.Itoa(index)+"]"); err != nil {
			return err
		}
	}
	return nil
}

type responsesWireFixtureError struct {
	path    string
	message string
}

func (err *responsesWireFixtureError) Error() string {
	return err.path + ": " + err.message
}

func validateLatestCodexTool(raw json.RawMessage, path string) error {
	var tool map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tool); err != nil {
		return &responsesWireFixtureError{path: path, message: "must be an object"}
	}
	var toolType string
	if err := json.Unmarshal(tool["type"], &toolType); err != nil || toolType == "" {
		return &responsesWireFixtureError{path: path + ".type", message: "is required"}
	}
	if toolType == "namespace" {
		if err := requireNonEmptyJSONText(tool, "name", path); err != nil {
			return err
		}
		if err := requireNonEmptyJSONText(tool, "description", path); err != nil {
			return err
		}
		var children []json.RawMessage
		if err := json.Unmarshal(tool["tools"], &children); err != nil || len(children) == 0 {
			return &responsesWireFixtureError{path: path + ".tools", message: "must be a non-empty array"}
		}
		for index, child := range children {
			if err := validateLatestCodexTool(child, path+".tools["+strconv.Itoa(index)+"]"); err != nil {
				return err
			}
		}
		return nil
	}
	if err := requireNonEmptyJSONText(tool, "description", path); err != nil {
		return err
	}
	if toolType == "function" || toolType == "custom" {
		if err := requireNonEmptyJSONText(tool, "name", path); err != nil {
			return err
		}
	}
	if toolType != "function" {
		return nil
	}
	parameters, ok := tool["parameters"]
	if !ok || string(parameters) == "null" {
		return &responsesWireFixtureError{path: path + ".parameters", message: "is required"}
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(parameters, &schema); err != nil {
		return &responsesWireFixtureError{path: path + ".parameters", message: "must be an object"}
	}
	return nil
}

func requireNonEmptyJSONText(object map[string]json.RawMessage, field, path string) error {
	var value string
	if err := json.Unmarshal(object[field], &value); err != nil || strings.TrimSpace(value) == "" {
		return &responsesWireFixtureError{path: path + "." + field, message: "is required"}
	}
	return nil
}

// TestResponsesLiteNamespaceCustomToolLowersLifecycleOverRealHTTP models the
// current Codex Responses Lite contract: namespace.tools accepts function
// declarations only. The test deliberately traverses the real HTTP body,
// rather than asserting a converter's in-memory view, and proves that both
// the initial definition and the next-turn call/result continuation use the
// reversible function representation.
func TestResponsesLiteNamespaceCustomToolLowersLifecycleOverRealHTTP(t *testing.T) {
	t.Parallel()

	const (
		namespace  = "agents"
		toolName   = "exec"
		callID     = "call_lite_namespace_1"
		toolInput  = "echo namespace-safe"
		toolOutput = "NAMESPACE_TOOL_DONE"
	)
	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		providerCall := providerCalls.Add(1)
		syntheticName := ""
		if providerCall == 1 {
			var strictErr error
			syntheticName, strictErr = strictResponsesLiteNamespaceFunction(body, namespace)
			if strictErr != nil {
				writer.Header().Set("Content-Type", "application/json")
				writer.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(writer, `{"error":{"message":"`+strictErr.Error()+`","type":"invalid_request_error","param":"input[0].tools[0].tools[0].type"}}`)
				return
			}
		}
		switch providerCall {
		case 1:
			response := map[string]any{
				"id": "resp_lite_namespace_call", "object": "response", "model": "gpt-5.6-sol", "status": "completed",
				"output": []any{map[string]any{
					"id": "fc_lite_namespace", "type": "function_call", "status": "completed", "call_id": callID,
					"name": syntheticName, "namespace": namespace, "arguments": `{"input":"` + toolInput + `"}`,
				}},
			}
			writer.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(writer).Encode(response)
		case 2:
			for _, required := range []string{`"type":"function_call"`, `"type":"function_call_output"`, syntheticName, namespace, toolOutput} {
				if !strings.Contains(string(body), required) {
					http.Error(writer, "function-only continuation lost "+required, http.StatusBadRequest)
					return
				}
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"resp_lite_namespace_final","object":"response","model":"gpt-5.6-sol","status":"completed","output":[{"id":"msg_lite_namespace_final","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"namespace continuation ok"}]}]}`)
		default:
			http.Error(writer, "unexpected provider call", http.StatusInternalServerError)
		}
	}))
	t.Cleanup(provider.Close)

	firstObserver := &observationRecorder{}
	first, err := runResponsesLiteRequest(t, provider, firstObserver, codexResponsesLiteNamespaceCustomRequest(false))
	if err != nil || first == nil || first.Response == nil || first.Stream {
		t.Fatalf("namespace custom first turn = %#v, err=%v", first, err)
	}
	assertResponsesLiteNamespaceCustomCall(t, first.Response.Body, callID, toolName, namespace, toolInput)
	assertResponsesLiteNamespaceLoweringObserved(t, firstObserver.snapshot(), 1)

	continuation := []byte(`{
		"model":"gpt-5.6-sol",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"run the namespace tool"}]},
			{"type":"custom_tool_call","call_id":"` + callID + `","name":"` + toolName + `","namespace":"` + namespace + `","input":"` + toolInput + `"},
			{"type":"custom_tool_call_output","call_id":"` + callID + `","output":"` + toolOutput + `"}
		]
	}`)
	secondObserver := &observationRecorder{}
	second, err := runResponsesLiteRequest(t, provider, secondObserver, continuation)
	if err != nil || second == nil || second.Response == nil || second.Stream {
		t.Fatalf("namespace custom continuation = %#v, err=%v", second, err)
	}
	assertResponsesLiteFinalText(t, second.Response.Body, "namespace continuation ok")
	assertResponsesLiteNamespaceLoweringObserved(t, secondObserver.snapshot(), 2)
	if providerCalls.Load() != 2 {
		t.Fatalf("provider calls = %d, want 2", providerCalls.Load())
	}
}

// TestResponsesLiteNamespaceCustomToolRestoresStreamOverRealHTTP proves the
// same lowering ledger restores function-call SSE back to Codex's native
// custom-tool events, including the namespace and free-form input bytes.
func TestResponsesLiteNamespaceCustomToolRestoresStreamOverRealHTTP(t *testing.T) {
	t.Parallel()

	const (
		namespace = "agents"
		toolName  = "exec"
		callID    = "call_lite_namespace_stream"
		toolInput = "echo streamed namespace-safe"
	)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		syntheticName, strictErr := strictResponsesLiteNamespaceFunction(body, namespace)
		if strictErr != nil {
			writer.Header().Set("Content-Type", "application/json")
			writer.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(writer, `{"error":{"message":"`+strictErr.Error()+`","type":"invalid_request_error","param":"input[0].tools[0].tools[0].type"}}`)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		arguments, err := json.Marshal(map[string]string{"input": toolInput})
		if err != nil {
			http.Error(writer, "encode function arguments", http.StatusInternalServerError)
			return
		}
		for _, event := range []struct {
			typeName string
			data     string
		}{
			{"response.created", responsesLiteSSEData(map[string]any{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_lite_namespace_stream", "object": "response", "model": "gpt-5.6-sol", "status": "in_progress", "output": []any{}}})},
			{"response.output_item.added", responsesLiteSSEData(map[string]any{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": map[string]any{"id": "fc_lite_namespace_stream", "type": "function_call", "status": "in_progress", "call_id": callID, "name": syntheticName, "namespace": namespace, "arguments": ""}})},
			{"response.function_call_arguments.delta", responsesLiteSSEData(map[string]any{"type": "response.function_call_arguments.delta", "sequence_number": 2, "item_id": "fc_lite_namespace_stream", "output_index": 0, "delta": string(arguments)})},
			{"response.function_call_arguments.done", responsesLiteSSEData(map[string]any{"type": "response.function_call_arguments.done", "sequence_number": 3, "item_id": "fc_lite_namespace_stream", "output_index": 0, "name": syntheticName, "namespace": namespace, "arguments": string(arguments)})},
			{"response.output_item.done", responsesLiteSSEData(map[string]any{"type": "response.output_item.done", "sequence_number": 4, "output_index": 0, "item": map[string]any{"id": "fc_lite_namespace_stream", "type": "function_call", "status": "completed", "call_id": callID, "name": syntheticName, "namespace": namespace, "arguments": string(arguments)}})},
			{"response.completed", responsesLiteSSEData(map[string]any{"type": "response.completed", "sequence_number": 5, "response": map[string]any{"id": "resp_lite_namespace_stream", "object": "response", "model": "gpt-5.6-sol", "status": "completed", "output": []any{}}})},
		} {
			_, _ = fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", event.typeName, event.data)
			if flusher, ok := writer.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
	t.Cleanup(provider.Close)

	observer := &observationRecorder{}
	result, err := runResponsesLiteRequest(t, provider, observer, codexResponsesLiteNamespaceCustomRequest(true))
	if err != nil || result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("namespace custom stream = %#v, err=%v", result, err)
	}
	var deltas strings.Builder
	var doneInput, addedName, addedNamespace, doneName, doneNamespace, doneItemInput string
	seenEvents := make([]string, 0, 8)
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		seenEvents = append(seenEvents, string(event.Data))
		var data struct {
			Type      string `json:"type"`
			Delta     string `json:"delta"`
			Input     string `json:"input"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
			Item      *struct {
				Type      string `json:"type"`
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
				Input     string `json:"input"`
			} `json:"item"`
		}
		if err := json.Unmarshal(event.Data, &data); err != nil {
			t.Fatalf("decode restored stream event: %v body=%s", err, event.Data)
		}
		switch data.Type {
		case "response.custom_tool_call_input.delta":
			deltas.WriteString(data.Delta)
		case "response.custom_tool_call_input.done":
			doneInput = data.Input
		case "response.output_item.added":
			if data.Item != nil && data.Item.Type == "custom_tool_call" {
				addedName, addedNamespace = data.Item.Name, data.Item.Namespace
			}
		case "response.output_item.done":
			if data.Item != nil && data.Item.Type == "custom_tool_call" {
				doneName, doneNamespace, doneItemInput = data.Item.Name, data.Item.Namespace, data.Item.Input
			}
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume namespace custom stream: %v", err)
	}
	_ = result.EventStream.Close()
	if deltas.String() != toolInput || doneInput != toolInput || addedName != toolName || addedNamespace != namespace ||
		doneName != toolName || doneNamespace != namespace || doneItemInput != toolInput {
		t.Fatalf("restored namespace custom stream degraded: delta=%q input=%q added=(%q,%q) done=(%q,%q,%q) events=%v", deltas.String(), doneInput, addedName, addedNamespace, doneName, doneNamespace, doneItemInput, seenEvents)
	}
	assertResponsesLiteNamespaceLoweringObserved(t, observer.snapshot(), 1)
}

// TestResponsesLiteTopLevelCustomStaysNativeOverRealHTTP proves that the Lite
// restriction is a namespace-child contract, rather than an unsupported
// blanket downgrade of the ordinary Responses custom-tool capability.
func TestResponsesLiteTopLevelCustomStaysNativeOverRealHTTP(t *testing.T) {
	t.Parallel()

	var providerCalls atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		providerCalls.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		var envelope struct {
			Tools []struct {
				Type        string `json:"type"`
				Name        string `json:"name"`
				Description string `json:"description"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(body, &envelope); err != nil {
			http.Error(writer, "decode request", http.StatusBadRequest)
			return
		}
		if len(envelope.Tools) != 1 || envelope.Tools[0].Type != "custom" ||
			envelope.Tools[0].Name != "top_level_exec" || envelope.Tools[0].Description == "" {
			http.Error(writer, "top-level Responses Lite custom tool must remain native", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_lite_top_level_custom","object":"response","model":"gpt-5.6-sol","status":"completed","output":[{"id":"msg_lite_top_level_custom","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"top-level custom remained native"}]}]}`)
	}))
	t.Cleanup(provider.Close)

	observer := &observationRecorder{}
	result, err := runResponsesLiteRequest(t, provider, observer, []byte(`{
		"model":"gpt-5.6-sol",
		"tools":[{"type":"custom","name":"top_level_exec","description":"Execute source code.","format":{"type":"grammar","syntax":"lark","definition":"start: source"}}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"run a command"}]}]
	}`))
	if err != nil || result == nil || result.Response == nil || result.Stream {
		t.Fatalf("top-level custom Responses Lite request = %#v, err=%v", result, err)
	}
	assertResponsesLiteFinalText(t, result.Response.Body, "top-level custom remained native")
	if providerCalls.Load() != 1 {
		t.Fatalf("provider calls = %d, want 1", providerCalls.Load())
	}
	assertResponsesLiteNamespaceLoweringObserved(t, observer.snapshot(), 0)
}

func assertResponsesLiteNamespaceLoweringObserved(t *testing.T, events []pipeline.Observation, wantLoweredAtLeast uint32) {
	t.Helper()
	for _, event := range events {
		if event.Stage != pipeline.StageConversionPlan || event.Conversion == nil {
			continue
		}
		if event.Conversion.TargetFormat != llm.APIFormatOpenAIResponse ||
			event.Conversion.ProfileID != "openai-responses/v1+responses-lite-namespace-function-only" ||
			event.Conversion.Lowered < wantLoweredAtLeast || !event.Conversion.Complete {
			t.Fatalf("Responses Lite conversion summary = %#v, want lower >= %d", event.Conversion, wantLoweredAtLeast)
		}
		return
	}
	t.Fatal("Responses Lite conversion plan was not observed")
}

func responsesLiteSSEData(value any) string {
	encoded, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	return string(encoded)
}

func codexResponsesLiteNamespaceCustomRequest(stream bool) []byte {
	return []byte(`{
		"model":"gpt-5.6-sol",
		"stream":` + fmt.Sprintf("%t", stream) + `,
		"input":[
			{"type":"additional_tools","role":"developer","tools":[
				{"type":"namespace","name":"agents","description":"Agent tools.","tools":[
					{"type":"custom","name":"exec","description":"Execute source code.","format":{"type":"grammar","syntax":"lark","definition":"start: source"}}
				]}
			]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"run a command"}]}
		]
	}`)
}

func strictResponsesLiteNamespaceFunction(body []byte, namespace string) (string, error) {
	var envelope struct {
		Input []struct {
			Type  string            `json:"type"`
			Tools []json.RawMessage `json:"tools"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return "", err
	}
	for inputIndex := range envelope.Input {
		item := envelope.Input[inputIndex]
		if item.Type != "additional_tools" {
			continue
		}
		for toolIndex := range item.Tools {
			var tool struct {
				Type  string            `json:"type"`
				Name  string            `json:"name"`
				Tools []json.RawMessage `json:"tools"`
			}
			if err := json.Unmarshal(item.Tools[toolIndex], &tool); err != nil {
				return "", err
			}
			if tool.Type != "namespace" {
				continue
			}
			for childIndex := range tool.Tools {
				var child struct {
					Type string `json:"type"`
					Name string `json:"name"`
				}
				if err := json.Unmarshal(tool.Tools[childIndex], &child); err != nil {
					return "", err
				}
				if child.Type != "function" {
					return "", fmt.Errorf("input[%d].tools[%d].tools[%d].type: namespace.tools can only contain function tools", inputIndex, toolIndex, childIndex)
				}
				if tool.Name == namespace && child.Name != "" {
					return child.Name, nil
				}
			}
		}
	}
	return "", fmt.Errorf("namespace %q function tool is missing", namespace)
}

func assertResponsesLiteNamespaceCustomCall(t *testing.T, body []byte, callID, name, namespace, input string) {
	t.Helper()
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
			Input     string `json:"input"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil || len(response.Output) != 1 ||
		response.Output[0].Type != "custom_tool_call" || response.Output[0].CallID != callID ||
		response.Output[0].Name != name || response.Output[0].Namespace != namespace || response.Output[0].Input != input {
		t.Fatalf("restored namespace custom call = %#v err=%v body=%s", response.Output, err, body)
	}
}

func assertResponsesLiteFinalText(t *testing.T, body []byte, want string) {
	t.Helper()
	var response struct {
		Output []struct {
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil || len(response.Output) != 1 ||
		len(response.Output[0].Content) != 1 || response.Output[0].Content[0].Text != want {
		t.Fatalf("Responses Lite final text mismatch: err=%v body=%s", err, body)
	}
}

func TestResponsesLitePreservedEnvelopeDoesNotBlockCrossProtocolPlanning(t *testing.T) {
	t.Parallel()

	inboundRequest, err := responses.NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method:  http.MethodPost,
		URL:     "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    codexResponsesLiteRequestWithoutWebSearch(),
	})
	if err != nil {
		t.Fatalf("decode latest Codex Responses Lite request: %v", err)
	}
	for _, target := range []llm.APIFormat{
		llm.APIFormatOpenAIChatCompletion,
		llm.APIFormatAnthropicMessage,
	} {
		plan, planErr := conversion.NewPlanner().Plan(inboundRequest, target)
		if planErr != nil {
			t.Fatalf("plan Responses Lite request for %s: %v", target, planErr)
		}
		if plan == nil || !plan.Complete() || plan.Summary.Unknown != 0 || plan.Summary.Lowered == 0 {
			t.Fatalf("cross-protocol plan treated represented envelope metadata as unknown: %#v", plan)
		}
	}
}

func codexResponsesLiteRequestWithoutWebSearch() []byte {
	return []byte(strings.Replace(string(codexResponsesLiteRequest(true)), `{"type":"web_search","external_web_access":true},`, "", 1))
}

func runResponsesLiteRequest(
	t *testing.T,
	provider *httptest.Server,
	observer *observationRecorder,
	requestBody []byte,
) (*pipeline.Result, error) {
	t.Helper()
	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	headers := http.Header{"Content-Type": []string{"application/json"}}
	headers.Set(responses.ResponsesLiteHeader, "true")
	return pipeline.NewFactory(executor).
		Pipeline(
			responses.NewInboundTransformer(),
			conversion.NewOutbound(outbound),
			pipeline.WithObserver(observer),
		).
		Process(llm.WithConversionTrace(context.Background()), &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/responses",
			Headers: headers,
			Body:    requestBody,
		})
}

func codexResponsesLiteRequest(withNamespaceDescription bool) []byte {
	description := `,"description":"Tools for spawning and managing sub-agents."`
	if !withNamespaceDescription {
		description = ""
	}
	return []byte(`{
		"model":"gpt-5.6-sol",
		"reasoning":{"effort":"low","context":"all_turns"},
		"input":[
			{"type":"additional_tools","role":"developer","x_codex_marker":"preserve-input-envelope","tools":[
				{"type":"custom","name":"exec","description":"Run JavaScript code to orchestrate tool calls.","format":{"type":"grammar","syntax":"lark","definition":"start: source"}},
				{"type":"function","name":"wait","description":"Wait for work.","parameters":{"type":"object","properties":{"cell_id":{"type":"string"}},"required":["cell_id"],"additionalProperties":false}},
				{"type":"web_search","external_web_access":true},
				{"type":"namespace","name":"multi_tool_use","description":"Run independent tools in parallel.","tools":[{"type":"function","name":"parallel","description":"Run tools concurrently.","parameters":{"type":"object","properties":{},"additionalProperties":false}}]},
				{"type":"namespace","name":"collaboration"` + description + `,"x_namespace_marker":"preserve-namespace-envelope","tools":[{"type":"function","name":"spawn_agent","description":"Spawn an agent.","parameters":{"type":"object","properties":{"task_name":{"type":"string"}},"required":["task_name"],"additionalProperties":false}}]}
			]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"reply with ok"}]}
		]
	}`)
}

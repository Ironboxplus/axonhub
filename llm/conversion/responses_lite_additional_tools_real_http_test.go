package conversion_test

import (
	"context"
	"encoding/json"
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

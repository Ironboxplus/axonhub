package conversion_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
		var payload struct {
			Input []struct {
				Type        string `json:"type"`
				CodexMarker string `json:"x_codex_marker"`
				Tools       []struct {
					Type            string `json:"type"`
					Name            string `json:"name"`
					Description     string `json:"description"`
					NamespaceMarker string `json:"x_namespace_marker"`
				} `json:"tools"`
			} `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil || len(payload.Input) == 0 || len(payload.Input[0].Tools) != 4 {
			http.Error(writer, "invalid Responses Lite request", http.StatusBadRequest)
			return
		}
		tool := payload.Input[0].Tools[3]
		if tool.Type != "namespace" || tool.Name != "collaboration" ||
			tool.Description != "Tools for spawning and managing sub-agents." {
			writer.Header().Set("Content-Type", "application/json")
			writer.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(writer, `{"error":{"message":"Missing required parameter: 'input[0].tools[3].description'.","type":"invalid_request_error","param":"input[0].tools[3].description","code":"missing_required_parameter"}}`)
			return
		}
		if payload.Input[0].CodexMarker != "preserve-input-envelope" || tool.NamespaceMarker != "preserve-namespace-envelope" {
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

func TestResponsesLitePreservedEnvelopeDoesNotBlockCrossProtocolPlanning(t *testing.T) {
	t.Parallel()

	inboundRequest, err := responses.NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method:  http.MethodPost,
		URL:     "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    codexResponsesLiteRequest(true),
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
	return pipeline.NewFactory(executor).
		Pipeline(
			responses.NewInboundTransformer(),
			conversion.NewOutbound(outbound),
			pipeline.WithObserver(observer),
		).
		Process(llm.WithConversionTrace(context.Background()), &httpclient.Request{
			Method: http.MethodPost,
			URL:    "/v1/responses",
			Headers: http.Header{
				"Content-Type": []string{"application/json"},
			},
			Body: requestBody,
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
				{"type":"namespace","name":"multi_tool_use","description":"Run independent tools in parallel.","tools":[{"type":"function","name":"parallel","description":"Run tools concurrently.","parameters":{"type":"object","properties":{},"additionalProperties":false}}]},
				{"type":"namespace","name":"collaboration"` + description + `,"x_namespace_marker":"preserve-namespace-envelope","tools":[{"type":"function","name":"spawn_agent","description":"Spawn an agent.","parameters":{"type":"object","properties":{"task_name":{"type":"string"}},"required":["task_name"],"additionalProperties":false}}]}
			]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"reply with ok"}]}
		]
	}`)
}

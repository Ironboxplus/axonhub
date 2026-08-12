package conversion_test

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"sync/atomic"
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

// codexAgentMessageWireFixture is a frozen Responses request fragment from the
// current Codex agent-message contract. It is deliberately an actual wire
// object: regression coverage must never infer agent_message from an opaque
// unknown-item log record.
const codexAgentMessageWireFixture = `{
  "id":"am_opaque_1",
  "type":"agent_message",
  "author":"/root",
  "recipient":"/root/worker",
  "content":[{"type":"input_text","text":"continue with the bounded task","future_agent_content_field":{"mode":"identity"}}],
  "future_agent_field":"identity-residual"
}`

func TestResponsesAgentMessagePlaintextLowersOverRealHTTP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		response    string
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			response: `{"id":"chat_agent_message","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			response: `{"id":"msg_agent_message","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				hits.Add(1)
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				legacy := providerLegacyInterAgentMessages(body)
				if len(legacy) != 1 || legacy[0].Role != "assistant" || legacy[0].Author != "/root" ||
					legacy[0].Recipient != "/root/worker" || len(legacy[0].OtherRecipients) != 0 ||
					legacy[0].Content != "continue with the bounded task" || !legacy[0].TriggerTurn || len(legacy[0].Raw) != 5 {
					http.Error(writer, "agent message was not lowered as a Codex legacy inter-agent assistant message", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.response)
			}))
			t.Cleanup(provider.Close)

			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), codexAgentMessageRequest())
			if err != nil {
				t.Fatalf("run plaintext agent-message conversion: %v", err)
			}
			if result == nil || result.Response == nil || hits.Load() != 1 {
				t.Fatalf("agent-message result=%#v provider_hits=%d", result, hits.Load())
			}
		})
	}
}

func TestResponsesAgentMessageEncryptedOrMixedBlocksProviderOverRealHTTP(t *testing.T) {
	t.Parallel()
	for _, targetFormat := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		targetFormat := targetFormat
		t.Run(string(targetFormat), func(t *testing.T) {
			t.Parallel()
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				hits.Add(1)
				http.Error(writer, "encrypted agent message must never reach provider", http.StatusInternalServerError)
			}))
			t.Cleanup(provider.Close)

			var target transformer.Outbound
			var err error
			if targetFormat == llm.APIFormatOpenAIChatCompletion {
				target, err = openai.NewOutboundTransformer(provider.URL, "fixture-key")
			} else {
				target, err = anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			}
			if err != nil {
				t.Fatalf("create outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)

			for _, content := range []string{
				`[{"type":"encrypted_content","encrypted_content":"PRIVATE_AGENT_FRAGMENT"}]`,
				`[{"type":"input_text","text":"visible"},{"type":"encrypted_content","encrypted_content":"PRIVATE_AGENT_FRAGMENT"}]`,
			} {
				request := strings.Replace(codexAgentMessageRequestBody(),
					`[{"type":"input_text","text":"continue with the bounded task","future_agent_content_field":{"mode":"identity"}}]`, content, 1)
				_, err := pipeline.NewFactory(executor).
					Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
					Process(context.Background(), responsesRequestFromBody(request))
				if !errors.Is(err, conversion.ErrIncompletePlan) {
					t.Fatalf("encrypted/mixed agent message error=%v, want incomplete plan", err)
				}
				plan, ok := conversion.PlanFromError(err)
				if !ok || plan == nil || plan.Summary.Unknown == 0 || plan.Debug == nil {
					t.Fatalf("missing incomplete plan evidence: plan=%#v ok=%v", plan, ok)
				}
				encoded, marshalErr := json.Marshal(plan.Debug)
				if marshalErr != nil || strings.Contains(string(encoded), "PRIVATE_AGENT_FRAGMENT") {
					t.Fatalf("conversion evidence leaked encrypted content: err=%v debug=%s", marshalErr, encoded)
				}
			}
			if got := hits.Load(); got != 0 {
				t.Fatalf("provider requests=%d, want 0 for encrypted/mixed agent messages", got)
			}
		})
	}
}

func TestResponsesAgentMessageWhitespacePlaintextBlocksProviderOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(writer, "blank agent-message commentary must not reach provider", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)
	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	body := strings.Replace(codexAgentMessageRequestBody(),
		`[{"type":"input_text","text":"continue with the bounded task","future_agent_content_field":{"mode":"identity"}}]`,
		`[{"type":"input_text","text":" \t"},{"type":"input_text","text":"\n"}]`, 1)
	_, processErr := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), responsesRequestFromBody(body))
	if !errors.Is(processErr, conversion.ErrIncompletePlan) || strings.Contains(processErr.Error(), " \\t") {
		t.Fatalf("blank plaintext error=%v, want payload-free incomplete plan", processErr)
	}
	plan, ok := conversion.PlanFromError(processErr)
	if !ok || plan == nil || plan.Summary.Unknown != 1 || len(plan.Actions) == 0 ||
		plan.Actions[len(plan.Actions)-1].SemanticClass != "agent_message_plaintext" {
		t.Fatalf("blank plaintext plan=%#v ok=%v", plan, ok)
	}
	debug, marshalErr := json.Marshal(plan.Debug)
	if marshalErr != nil || strings.Contains(string(debug), "\\t") {
		t.Fatalf("blank plaintext evidence leaked: err=%v debug=%s", marshalErr, debug)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("provider requests=%d, want 0", got)
	}
}

func TestResponsesManyPlaintextAgentMessagesLowerInOneProviderRequestOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read provider request", http.StatusBadRequest)
			return
		}
		legacy := providerLegacyInterAgentMessages(body)
		if len(legacy) != 13 {
			http.Error(writer, "missing legacy agent-message boundaries", http.StatusBadRequest)
			return
		}
		for index := range legacy {
			want := "part-" + strconv.Itoa(index) + "-a\npart-" + strconv.Itoa(index) + "-b"
			if legacy[index].Role != "assistant" || legacy[index].Content != want || !legacy[index].TriggerTurn ||
				legacy[index].Author != "/root" || legacy[index].Recipient != "/root/worker" || len(legacy[index].Raw) != 5 {
				http.Error(writer, "agent-message legacy boundaries were reordered or malformed", http.StatusBadRequest)
				return
			}
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"many_agent_messages","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	items := make([]string, 0, 13)
	for index := 0; index < 13; index++ {
		items = append(items, `{"id":"am_`+strconv.Itoa(index)+`","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"part-`+strconv.Itoa(index)+`-a"},{"type":"input_text","text":"part-`+strconv.Itoa(index)+`-b"}]}`)
	}
	body := `{"model":"fixture-model","input":[` + strings.Join(items, ",") + `]}`
	request := responsesRequestFromBody(body)
	requestForPlan, err := responses.NewInboundTransformer().TransformRequest(context.Background(), request)
	if err != nil {
		t.Fatalf("decode production-shaped agent messages: %v", err)
	}
	plan, err := conversion.NewOutbound(target).Preflight(requestForPlan)
	if err != nil || plan == nil || plan.Summary.Unknown != 0 || plan.Summary.Lowered != 13 {
		t.Fatalf("many agent-message plan=%#v err=%v", plan, err)
	}
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), request)
	if err != nil || result == nil || result.Response == nil || hits.Load() != 1 {
		t.Fatalf("many agent-message result=%#v err=%v hits=%d", result, err, hits.Load())
	}
}

func TestResponsesAgentMessageSameProtocolPreservesFrozenWireOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		hits.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read provider request", http.StatusBadRequest)
			return
		}
		var payload struct {
			Input []json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Input) != 2 ||
			!strings.Contains(string(payload.Input[1]), `"type":"agent_message"`) ||
			!strings.Contains(string(payload.Input[1]), `"author":"/root"`) ||
			!strings.Contains(string(payload.Input[1]), `"recipient":"/root/worker"`) ||
			!strings.Contains(string(payload.Input[1]), `"future_agent_content_field":{"mode":"identity"}`) ||
			!strings.Contains(string(payload.Input[1]), `"future_agent_field":"identity-residual"`) {
			http.Error(writer, "agent message identity was not preserved", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_agent_message","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[{"id":"out_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
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
		Process(context.Background(), codexAgentMessageRequest())
	if err != nil {
		t.Fatalf("run same-protocol agent-message identity route: %v", err)
	}
	if result == nil || result.Response == nil || hits.Load() != 1 {
		t.Fatalf("same-protocol result=%#v provider_hits=%d", result, hits.Load())
	}
}

func TestResponsesLiteTypedAgentAndContextRemainNativeOverRealHTTP(t *testing.T) {
	t.Parallel()
	fixtures := []struct {
		name string
		item string
	}{
		{name: "plaintext agent", item: `{"id":"lite_agent_plain","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"visible"}]}`},
		{name: "encrypted agent", item: `{"id":"lite_agent_encrypted","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"encrypted_content","encrypted_content":"PRIVATE_LITE_AGENT"}]}`},
		{name: "context checkpoint", item: `{"id":"lite_context","type":"context_compaction","encrypted_content":"PRIVATE_LITE_CONTEXT"}`},
	}
	for _, fixture := range fixtures {
		fixture := fixture
		t.Run(fixture.name, func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				hits.Add(1)
				body, _ := io.ReadAll(request.Body)
				var payload struct {
					Input []json.RawMessage `json:"input"`
				}
				if request.Header.Get(responses.ResponsesLiteHeader) != "true" || json.Unmarshal(body, &payload) != nil || len(payload.Input) != 1 || !semanticJSONEqual(payload.Input[0], []byte(fixture.item)) {
					http.Error(writer, "Responses Lite typed identity changed", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"id":"lite_identity","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[]}`)
			}))
			t.Cleanup(provider.Close)
			target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			request := responsesRequestFromBody(`{"model":"fixture-model","input":[` + fixture.item + `]}`)
			request.Headers.Set(responses.ResponsesLiteHeader, "true")
			result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), request)
			if err != nil || result == nil || result.Response == nil || hits.Load() != 1 {
				t.Fatalf("Lite %s result=%#v hits=%d err=%v", fixture.name, result, hits.Load(), err)
			}
		})
	}
}

func TestResponsesLiteFutureAgentContentStopsBeforeProviderOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(writer, "future Lite agent content reached provider", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	request := responsesRequestFromBody(`{"model":"fixture-model","input":[{"id":"lite_agent_future","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"future_agent_content","private":"PRIVATE_LITE_FUTURE"}]}]}`)
	request.Headers.Set(responses.ResponsesLiteHeader, "true")
	result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), request)
	if result != nil || processErr == nil || !strings.Contains(processErr.Error(), "invalid_agent_message_content") || strings.Contains(processErr.Error(), "PRIVATE_LITE_FUTURE") || hits.Load() != 0 {
		t.Fatalf("Lite future child result=%#v hits=%d err=%v", result, hits.Load(), processErr)
	}
}

func TestResponsesAgentMessageMalformedKnownContentStopsEveryTargetBeforeProviderOverRealHTTP(t *testing.T) {
	t.Parallel()
	for _, targetFormat := range []llm.APIFormat{llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		targetFormat := targetFormat
		t.Run(string(targetFormat), func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				hits.Add(1)
				http.Error(writer, "agent-message validation must stop before provider dispatch", http.StatusInternalServerError)
			}))
			t.Cleanup(provider.Close)
			var target transformer.Outbound
			var err error
			switch targetFormat {
			case llm.APIFormatOpenAIResponse:
				target, err = responses.NewOutboundTransformer(provider.URL, "fixture-key")
			case llm.APIFormatOpenAIChatCompletion:
				target, err = openai.NewOutboundTransformer(provider.URL, "fixture-key")
			default:
				target, err = anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			}
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			content := `[{"type":"input_text","text":"visible","encrypted_content":"PRIVATE_AGENT_FRAGMENT"}]`
			request := strings.Replace(codexAgentMessageRequestBody(), `[{"type":"input_text","text":"continue with the bounded task","future_agent_content_field":{"mode":"identity"}}]`, content, 1)
			result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(request))
			if result != nil || processErr == nil || !strings.Contains(processErr.Error(), "invalid_agent_message_content") || strings.Contains(processErr.Error(), "PRIVATE_AGENT_FRAGMENT") || hits.Load() != 0 {
				t.Fatalf("malformed known target=%s result=%#v hits=%d err=%v", targetFormat, result, hits.Load(), processErr)
			}
		})
	}
}

func TestResponsesIngressInvalidAgentPathIsRequestParseNotSentOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(writer, "invalid ingress path reached provider", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	observer := &observationRecorder{}
	request := responsesRequestFromBody(`{"model":"fixture-model","input":[{"type":"agent_message","author":"/root/Worker","recipient":"/root/worker","content":[{"type":"input_text","text":"PRIVATE_INVALID_PATH"}]}]}`)
	result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).Process(context.Background(), request)
	if result != nil || processErr == nil || !strings.Contains(processErr.Error(), "invalid_agent_message_path") || strings.Contains(processErr.Error(), "PRIVATE_INVALID_PATH") || hits.Load() != 0 {
		t.Fatalf("invalid path result=%#v hits=%d err=%v", result, hits.Load(), processErr)
	}
	diagnostic := llm.ErrorDiagnosticFrom(processErr)
	if diagnostic == nil || diagnostic.Component != "inbound_wire_validation" || diagnostic.Code != "invalid_agent_message_path" || diagnostic.StatusCode != http.StatusBadRequest {
		t.Fatalf("invalid path diagnostic=%#v", diagnostic)
	}
	observed := false
	for _, event := range observer.snapshot() {
		if event.Stage == pipeline.StageInboundTransform {
			observed = true
			if event.Outcome != pipeline.OutcomeFailure || event.ErrorClass != pipeline.ErrorClassTransform || event.StatusCode != 0 {
				t.Fatalf("invalid path inbound observation=%#v", event)
			}
		}
		if event.Stage == pipeline.StageProviderExchange {
			t.Fatalf("invalid path unexpectedly observed provider exchange=%#v", event)
		}
	}
	if !observed {
		t.Fatal("missing request-parse inbound observation")
	}
}

func TestResponsesAgentMessageFutureChildNeverMasksKnownMalformedMemberOverRealHTTP(t *testing.T) {
	t.Parallel()
	for _, content := range []string{
		`[{"type":"future_agent_content","payload":"opaque"},{"type":"input_text","text":"visible","encrypted_content":"PRIVATE_AGENT_FRAGMENT"}]`,
		`[{"type":"input_text","text":"visible","encrypted_content":"PRIVATE_AGENT_FRAGMENT"},{"type":"future_agent_content","payload":"opaque"}]`,
	} {
		content := content
		t.Run(content[:24], func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				hits.Add(1)
				http.Error(writer, "malformed agent message reached provider", http.StatusInternalServerError)
			}))
			t.Cleanup(provider.Close)
			target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			body := strings.Replace(codexAgentMessageRequestBody(), `[{"type":"input_text","text":"continue with the bounded task","future_agent_content_field":{"mode":"identity"}}]`, content, 1)
			result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(body))
			if result != nil || processErr == nil || !strings.Contains(processErr.Error(), "invalid_agent_message_content") || strings.Contains(processErr.Error(), "PRIVATE_AGENT_FRAGMENT") || hits.Load() != 0 {
				t.Fatalf("future/malformed ordering result=%#v hits=%d err=%v", result, hits.Load(), processErr)
			}
		})
	}
}

func TestResponsesAgentMessageFutureContentIsOpaqueIdentityOrCrossBlockOverRealHTTP(t *testing.T) {
	t.Parallel()
	const futureItem = `{ "recipient":"/root/..", "content":[{"payload":{"private":true},"type":"future_agent_content"}], "id":"am_future_child", "author":"/root/Worker", "type":"agent_message" }`
	for _, targetFormat := range []llm.APIFormat{llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		targetFormat := targetFormat
		t.Run(string(targetFormat), func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				hits.Add(1)
				if targetFormat != llm.APIFormatOpenAIResponse {
					http.Error(writer, "future agent child reached cross provider", http.StatusInternalServerError)
					return
				}
				body, _ := io.ReadAll(request.Body)
				var payload struct {
					Input []json.RawMessage `json:"input"`
				}
				if json.Unmarshal(body, &payload) != nil || len(payload.Input) != 1 || !semanticJSONEqual(payload.Input[0], []byte(futureItem)) {
					http.Error(writer, "future agent child identity changed", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"id":"future_child_identity","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[]}`)
			}))
			t.Cleanup(provider.Close)
			var target transformer.Outbound
			var err error
			switch targetFormat {
			case llm.APIFormatOpenAIResponse:
				target, err = responses.NewOutboundTransformer(provider.URL, "fixture-key")
			case llm.APIFormatOpenAIChatCompletion:
				target, err = openai.NewOutboundTransformer(provider.URL, "fixture-key")
			default:
				target, err = anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			}
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","input":[`+futureItem+`]}`))
			if targetFormat == llm.APIFormatOpenAIResponse {
				if processErr != nil || result == nil || result.Response == nil || hits.Load() != 1 {
					t.Fatalf("same future agent child result=%#v hits=%d err=%v", result, hits.Load(), processErr)
				}
				return
			}
			if !errors.Is(processErr, conversion.ErrIncompletePlan) || result != nil || hits.Load() != 0 || strings.Contains(processErr.Error(), "private") {
				t.Fatalf("cross future agent child result=%#v hits=%d err=%v", result, hits.Load(), processErr)
			}
			plan, ok := conversion.PlanFromError(processErr)
			if !ok || plan == nil || len(plan.Actions) != 1 || plan.Actions[0].Kind != conversion.ActionUnknown || plan.Actions[0].SourceType != "agent_message" ||
				plan.Actions[0].SemanticClass != "future_unknown_behavioral" || plan.Actions[0].RawBytes != uint32(len(futureItem)) || len(plan.Actions[0].SourceDigest) != 64 {
				t.Fatalf("cross future agent child plan=%#v ok=%v", plan, ok)
			}
		})
	}
}

func TestResponsesAgentMessageInvalidCodexPathStopsEveryTargetBeforeProvider(t *testing.T) {
	t.Parallel()
	for _, targetFormat := range []llm.APIFormat{llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		targetFormat := targetFormat
		t.Run(string(targetFormat), func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				hits.Add(1)
				http.Error(writer, "invalid AgentPath reached provider", http.StatusInternalServerError)
			}))
			t.Cleanup(provider.Close)
			var target transformer.Outbound
			var err error
			switch targetFormat {
			case llm.APIFormatOpenAIResponse:
				target, err = responses.NewOutboundTransformer(provider.URL, "fixture-key")
			case llm.APIFormatOpenAIChatCompletion:
				target, err = openai.NewOutboundTransformer(provider.URL, "fixture-key")
			default:
				target, err = anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			}
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			body := strings.Replace(codexAgentMessageRequestBody(), `"author":"/root"`, `"author":"/root/Worker"`, 1)
			result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(body))
			if result != nil || processErr == nil || !strings.Contains(processErr.Error(), "invalid_agent_message_path") || strings.Contains(processErr.Error(), "Worker") || hits.Load() != 0 {
				t.Fatalf("invalid AgentPath target=%s result=%#v hits=%d err=%v", targetFormat, result, hits.Load(), processErr)
			}
		})
	}
}

func TestResponsesFutureUnknownDoesNotApplyAgentPathValidationOverRealHTTP(t *testing.T) {
	t.Parallel()
	const futureItem = `{"id":"future_agent_path","type":"future_agent_message","author":"/root/Worker","recipient":"/root/..","content":[{"type":"input_text","text":"opaque"}]}`
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits.Add(1)
		body, _ := io.ReadAll(request.Body)
		var payload struct {
			Input []json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Input) != 1 || !semanticJSONEqual(payload.Input[0], []byte(futureItem)) {
			http.Error(writer, "future unknown identity changed", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"future_path_identity","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, processErr := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","input":[`+futureItem+`]}`))
	if processErr != nil || result == nil || result.Response == nil || hits.Load() != 1 {
		t.Fatalf("future AgentPath-like unknown result=%#v hits=%d err=%v", result, hits.Load(), processErr)
	}
}

func TestResponsesFutureAgentLikeItemDoesNotBecomeAgentMessageOverRealHTTP(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(writer, "future Responses item must not reach a Chat provider", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)

	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	const privatePayload = "PRIVATE_FUTURE_AGENT_FRAGMENT"
	futureItem := `{"id":"future_agent_1","type":"future_agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"` + privatePayload + `"}]}`
	body := strings.Replace(codexAgentMessageRequestBody(), codexAgentMessageWireFixture, futureItem, 1)
	_, processErr := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), responsesRequestFromBody(body))
	if !errors.Is(processErr, conversion.ErrIncompletePlan) {
		t.Fatalf("future agent-like item error=%v, want incomplete plan", processErr)
	}
	plan, ok := conversion.PlanFromError(processErr)
	if !ok || plan == nil || len(plan.Actions) == 0 || plan.Actions[len(plan.Actions)-1].Ref.Kind != conversion.ObjectInputItem ||
		plan.Actions[len(plan.Actions)-1].SourceType == "agent_message" {
		t.Fatalf("future agent-like item was not conservatively classified: plan=%#v ok=%v", plan, ok)
	}
	debug, err := json.Marshal(plan.Debug)
	if err != nil || strings.Contains(processErr.Error(), privatePayload) || strings.Contains(string(debug), privatePayload) {
		t.Fatalf("future opaque payload leaked: process=%v debug=%s marshal=%v", processErr, debug, err)
	}
	if got := hits.Load(); got != 0 {
		t.Fatalf("provider requests=%d, want 0 for future agent-like item", got)
	}
}

func TestResponsesAgentMessageSameProtocolStreamPreservesTypedWireOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writeAgentMessageSSE(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_agent_stream", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}},
		})
		writeAgentMessageSSE(t, writer, "response.output_item.added", map[string]any{
			"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
			"item": map[string]any{"id": "am_stream_1", "type": "agent_message", "author": "/root", "recipient": "/root/worker", "status": "in_progress",
				"content":            []any{map[string]any{"type": "input_text", "text": "typed stream", "future_agent_content_field": map[string]any{"mode": "identity"}}},
				"future_agent_field": "identity-residual"},
		})
		writeAgentMessageSSE(t, writer, "response.output_item.done", map[string]any{
			"type": "response.output_item.done", "sequence_number": 2, "output_index": 0,
			"item": map[string]any{"id": "am_stream_1", "type": "agent_message", "author": "/root", "recipient": "/root/worker", "status": "completed",
				"content":            []any{map[string]any{"type": "input_text", "text": "typed stream", "future_agent_content_field": map[string]any{"mode": "identity"}}},
				"future_agent_field": "identity-residual"},
		})
		writeAgentMessageSSE(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 3,
			"response": map[string]any{"id": "resp_agent_stream", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{}, "usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}},
		})
	}))
	t.Cleanup(provider.Close)

	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	request := codexAgentMessageRequest()
	request.Body = []byte(strings.Replace(codexAgentMessageRequestBody(), `"max_output_tokens":64`, `"max_output_tokens":64,"stream":true`, 1))
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), request)
	if err != nil || result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("start typed agent-message stream: result=%#v err=%v", result, err)
	}
	defer result.EventStream.Close()
	var added, done map[string]json.RawMessage
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || string(event.Data) == "[DONE]" {
			continue
		}
		var envelope map[string]json.RawMessage
		if err := json.Unmarshal(event.Data, &envelope); err != nil {
			t.Fatalf("decode Responses stream event: %v body=%s", err, event.Data)
		}
		var eventType string
		_ = json.Unmarshal(envelope["type"], &eventType)
		if eventType == "response.output_text.delta" {
			t.Fatalf("typed agent_message leaked as output text delta: %s", event.Data)
		}
		if eventType == "response.output_item.added" {
			_ = json.Unmarshal(envelope["item"], &added)
		}
		if eventType == "response.output_item.done" {
			_ = json.Unmarshal(envelope["item"], &done)
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume typed agent-message stream: %v", err)
	}
	for label, item := range map[string]map[string]json.RawMessage{"added": added, "done": done} {
		if item == nil || string(item["type"]) != `"agent_message"` || !strings.Contains(string(item["future_agent_field"]), "identity-residual") ||
			!strings.Contains(string(item["content"]), "future_agent_content_field") {
			t.Fatalf("%s stream agent-message identity degraded: %s", label, item)
		}
	}
}

func writeAgentMessageSSE(t *testing.T, writer http.ResponseWriter, eventType string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal agent-message SSE fixture: %v", err)
	}
	_, _ = writer.Write([]byte("event: " + eventType + "\n"))
	_, _ = writer.Write([]byte("data: " + string(data) + "\n\n"))
	if flusher, ok := writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func writeRawResponsesSSE(t *testing.T, writer http.ResponseWriter, data string) {
	t.Helper()
	if !json.Valid([]byte(data)) {
		t.Fatalf("invalid Responses SSE fixture: %s", data)
	}
	_, _ = writer.Write([]byte("data: " + data + "\n\n"))
	if flusher, ok := writer.(http.Flusher); ok {
		flusher.Flush()
	}
}

func codexAgentMessageRequest() *httpclient.Request {
	return responsesRequestFromBody(codexAgentMessageRequestBody())
}

func responsesRequestFromBody(body string) *httpclient.Request {
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(body),
	}
}

func codexAgentMessageRequestBody() string {
	return `{
  "model":"fixture-model",
  "max_output_tokens":64,
  "input":[
    {"type":"message","role":"user","content":[{"type":"input_text","text":"perform the handoff"}]},
    ` + codexAgentMessageWireFixture + `
  ]
}`
}

func TestResponsesProviderNonOutputItemsFailClosedForLegacyClientsOverRealHTTP(t *testing.T) {
	t.Parallel()
	fixtures := []struct {
		name, item, semantic string
	}{
		{name: "plaintext agent", semantic: "agent_message_provider_output", item: `{"id":"am_plain","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"PRIVATE_AGENT"}]}`},
		{name: "encrypted agent", semantic: "agent_message_provider_output", item: `{"id":"am_secret","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"encrypted_content","encrypted_content":"PRIVATE_AGENT"}]}`},
		{name: "mixed agent", semantic: "agent_message_provider_output", item: `{"id":"am_mixed","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"visible"},{"type":"encrypted_content","encrypted_content":"PRIVATE_AGENT"}]}`},
		{name: "compaction checkpoint", semantic: "compaction_checkpoint", item: `{"id":"compact_private","type":"compaction","encrypted_content":"PRIVATE_COMPACTION"}`},
		{name: "context checkpoint", semantic: "context_compaction_checkpoint", item: `{"id":"ctx_private","type":"context_compaction","encrypted_content":"PRIVATE_CONTEXT"}`},
		{name: "future behavioral", semantic: "future_unknown_behavioral", item: `{"id":"future_private","type":"future_behavior","secret":"PRIVATE_FUTURE"}`},
	}
	for _, targetClient := range []struct {
		name    string
		inbound transformer.Inbound
		request *httpclient.Request
	}{
		{name: "chat", inbound: openai.NewInboundTransformer(), request: &httpclient.Request{Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{"model":"fixture-model","messages":[{"role":"user","content":"continue"}]}`)}},
		{name: "anthropic", inbound: anthropic.NewInboundTransformer(), request: &httpclient.Request{Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{"model":"fixture-model","max_tokens":16,"messages":[{"role":"user","content":"continue"}]}`)}},
	} {
		targetClient := targetClient
		for _, fixture := range fixtures {
			fixture := fixture
			t.Run(targetClient.name+"/"+fixture.name, func(t *testing.T) {
				var hits atomic.Int32
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
					hits.Add(1)
					writer.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(writer, `{"id":"resp_private","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[`+fixture.item+`]}`)
				}))
				t.Cleanup(provider.Close)
				target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
				if err != nil {
					t.Fatalf("Responses outbound: %v", err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				observer := &observationRecorder{}
				result, err := pipeline.NewFactory(executor).Pipeline(targetClient.inbound, conversion.NewOutbound(target), pipeline.WithObserver(observer)).Process(context.Background(), targetClient.request)
				if err == nil || result != nil || hits.Load() != 1 || strings.Contains(err.Error(), "PRIVATE_") || !strings.Contains(err.Error(), "canonical output") {
					t.Fatalf("non-output client projection result=%#v hits=%d err=%v", result, hits.Load(), err)
				}
				action, ok := outputBlockerFromObservations(observer.snapshot())
				if !ok || action.Direction != llm.ConversionDirectionResponse || action.ObjectID != "output[0]" ||
					action.Action != string(conversion.ActionUnknown) || action.Result != llm.ConversionResultUnknown || action.Severity != llm.ConversionSeverityCritical ||
					action.SourceType != fixtureSourceType(fixture.item) || action.SemanticClass != fixture.semantic || action.RawBytes != uint32(len(fixture.item)) || action.SourceDigest != exactSHA256(fixture.item) ||
					strings.Contains(mustMarshalJSON(t, action), "PRIVATE_") {
					t.Fatalf("response output blocker fixture=%s action=%#v ok=%v", fixture.name, action, ok)
				}
			})
		}
	}
}

func TestResponsesUnknownFullRawIdentityWithEmptyKnownNamesOverRealHTTP(t *testing.T) {
	t.Parallel()
	// The order, spaces, [], null, and empty string are deliberately retained.
	const futureItem = `{ "future": {"mode":"identity"}, "tools": [], "type": "future_behavior", "content": null, "arguments": "", "status": null, "id":"future_empty_1" }`
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits.Add(1)
		body, _ := io.ReadAll(request.Body)
		var payload struct {
			Input []json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil || len(payload.Input) != 1 || !semanticJSONEqual(payload.Input[0], []byte(futureItem)) {
			http.Error(writer, "future raw input identity changed", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_identity","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	request := responsesRequestFromBody(`{"model":"fixture-model","input":[` + futureItem + `]}`)
	result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), request)
	if err != nil || result == nil || result.Response == nil || hits.Load() != 1 {
		t.Fatalf("future raw identity result=%#v hits=%d err=%v", result, hits.Load(), err)
	}
}

func TestResponsesProviderFutureOutputWithEmptyKnownNamesKeepsSameProtocolIdentityOverRealHTTP(t *testing.T) {
	t.Parallel()
	const futureItem = `{ "future": {"mode":"output-identity"}, "tools": [], "type": "future_behavior", "content": null, "arguments": "", "status": null, "id":"future_output_1" }`
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_output_identity","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[`+futureItem+`]}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","input":"continue"}`))
	if err != nil || result == nil || result.Response == nil {
		t.Fatalf("future output identity result=%#v err=%v", result, err)
	}
	var output struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(result.Response.Body, &output); err != nil || len(output.Output) != 1 || !semanticJSONEqual(output.Output[0], []byte(futureItem)) {
		t.Fatalf("future output identity body=%s err=%v", result.Response.Body, err)
	}
}

func TestResponsesProviderAgentMessageFailsClosedForChatStreamOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeAgentMessageSSE(t, writer, "response.created", map[string]any{"type": "response.created", "response": map[string]any{"id": "resp_stream_project", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}}})
		writeAgentMessageSSE(t, writer, "response.output_item.added", map[string]any{"type": "response.output_item.added", "output_index": 0, "item": map[string]any{"id": "am_stream_project", "type": "agent_message", "author": "/root", "recipient": "/root/worker", "content": []any{map[string]any{"type": "input_text", "text": "stream handoff"}}}})
		writeAgentMessageSSE(t, writer, "response.output_item.done", map[string]any{"type": "response.output_item.done", "output_index": 0, "item": map[string]any{"id": "am_stream_project", "type": "agent_message", "author": "/root", "recipient": "/root/worker", "status": "completed", "content": []any{map[string]any{"type": "input_text", "text": "stream handoff"}}}})
		writeAgentMessageSSE(t, writer, "response.completed", map[string]any{"type": "response.completed", "response": map[string]any{"id": "resp_stream_project", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{}}})
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	request := &httpclient.Request{Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{"model":"fixture-model","stream":true,"messages":[{"role":"user","content":"continue"}]}`)}
	observer := &observationRecorder{}
	result, err := pipeline.NewFactory(executor).Pipeline(openai.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).Process(context.Background(), request)
	if err != nil || result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("start stream projection result=%#v err=%v", result, err)
	}
	defer result.EventStream.Close()
	seenPrivate := false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event != nil && strings.Contains(string(event.Data), "stream handoff") {
			seenPrivate = true
		}
	}
	if err := result.EventStream.Err(); err == nil || seenPrivate || strings.Contains(err.Error(), "stream handoff") {
		t.Fatalf("Chat stream agent-message projection leaked=%v err=%v", seenPrivate, err)
	}
	action, ok := outputBlockerFromObservations(observer.snapshot())
	if !ok || action.Direction != llm.ConversionDirectionStream || action.ObjectID != "output[0]" ||
		action.SourceType != "agent_message" || action.SemanticClass != "agent_message_provider_output" ||
		action.Action != string(conversion.ActionUnknown) || action.Result != llm.ConversionResultUnknown || action.Severity != llm.ConversionSeverityCritical ||
		action.RawBytes == 0 || len(action.SourceDigest) != 64 || strings.Contains(mustMarshalJSON(t, action), "stream handoff") {
		t.Fatalf("stream output blocker action=%#v ok=%v", action, ok)
	}
}

func TestResponsesProviderNonOutputItemsFailClosedForLegacyStreamsOverRealHTTP(t *testing.T) {
	t.Parallel()
	fixtures := []struct {
		name, item, sourceType, semantic string
	}{
		{
			name: "agent message", sourceType: "agent_message", semantic: "agent_message_provider_output",
			item: `{"id":"am_stream_private","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"PRIVATE_STREAM_AGENT"}]}`,
		},
		{
			name: "compaction checkpoint", sourceType: "compaction", semantic: "compaction_checkpoint",
			item: `{"id":"compact_stream_private","type":"compaction","encrypted_content":"PRIVATE_STREAM_COMPACTION"}`,
		},
		{
			name: "context checkpoint", sourceType: "context_compaction", semantic: "context_compaction_checkpoint",
			item: `{"id":"ctx_stream_private","type":"context_compaction","encrypted_content":"PRIVATE_STREAM_CONTEXT"}`,
		},
		{
			name: "future behavioral", sourceType: "future_stream_behavior", semantic: "future_unknown_behavioral",
			item: `{"id":"future_stream_private","type":"future_stream_behavior","secret":"PRIVATE_STREAM_FUTURE"}`,
		},
	}
	clients := []struct {
		name string
		in   transformer.Inbound
		req  *httpclient.Request
	}{
		{name: "chat", in: openai.NewInboundTransformer(), req: &httpclient.Request{Method: http.MethodPost, URL: "/v1/chat/completions", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{"model":"fixture-model","stream":true,"messages":[{"role":"user","content":"continue"}]}`)}},
		{name: "anthropic", in: anthropic.NewInboundTransformer(), req: &httpclient.Request{Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(`{"model":"fixture-model","max_tokens":16,"stream":true,"messages":[{"role":"user","content":"continue"}]}`)}},
	}
	for _, client := range clients {
		client := client
		for _, fixture := range fixtures {
			fixture := fixture
			t.Run(client.name+"/"+fixture.name, func(t *testing.T) {
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
					writer.Header().Set("Content-Type", "text/event-stream")
					writeRawResponsesSSE(t, writer, `{"type":"response.created","response":{"id":"resp_stream_private","object":"response","model":"fixture-model","status":"in_progress","output":[]}}`)
					writeRawResponsesSSE(t, writer, `{"type":"response.output_item.added","output_index":0,"item":`+fixture.item+`}`)
					writeRawResponsesSSE(t, writer, `{"type":"response.output_item.done","output_index":0,"item":`+fixture.item+`}`)
					writeRawResponsesSSE(t, writer, `{"type":"response.completed","response":{"id":"resp_stream_private","object":"response","model":"fixture-model","status":"completed","output":[]}}`)
				}))
				t.Cleanup(provider.Close)
				target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
				if err != nil {
					t.Fatal(err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				observer := &observationRecorder{}
				result, err := pipeline.NewFactory(executor).Pipeline(client.in, conversion.NewOutbound(target), pipeline.WithObserver(observer)).Process(context.Background(), client.req)
				if err != nil || result == nil || !result.Stream || result.EventStream == nil {
					t.Fatalf("start %s stream result=%#v err=%v", fixture.name, result, err)
				}
				defer result.EventStream.Close()
				for result.EventStream.Next() {
					if event := result.EventStream.Current(); event != nil && strings.Contains(string(event.Data), "PRIVATE_STREAM") {
						t.Fatalf("%s leaked provider payload through %s stream: %s", fixture.name, client.name, event.Data)
					}
				}
				if streamErr := result.EventStream.Err(); streamErr == nil || strings.Contains(streamErr.Error(), "PRIVATE_STREAM") {
					t.Fatalf("%s stream err=%v", fixture.name, streamErr)
				}
				action, ok := outputBlockerFromObservations(observer.snapshot())
				if !ok || action.Direction != llm.ConversionDirectionStream || action.ObjectID != "output[0]" || action.SourceType != fixture.sourceType ||
					action.SemanticClass != fixture.semantic || action.Action != string(conversion.ActionUnknown) || action.Result != llm.ConversionResultUnknown ||
					action.Severity != llm.ConversionSeverityCritical || action.RawBytes != uint32(len(fixture.item)) || action.SourceDigest != exactSHA256(fixture.item) ||
					strings.Contains(mustMarshalJSON(t, action), "PRIVATE_STREAM") {
					t.Fatalf("%s stream blocker=%#v ok=%v", fixture.name, action, ok)
				}
				if countOutputBlockers(observer.snapshot()) != 1 {
					t.Fatalf("%s added/done blocker was not deduplicated: %#v", fixture.name, observer.snapshot())
				}
			})
		}
	}
}

func TestResponsesProviderContextCompactionKeepsSameProtocolStreamIdentityOverRealHTTP(t *testing.T) {
	t.Parallel()
	const contextAdded = `{ "future_context":{"checkpoint":1}, "encrypted_content":"PRIVATE_CONTEXT_STREAM", "type":"context_compaction", "id":"ctx_stream_1" }`
	// ContextCompaction has no lifecycle-status field in the Responses union.
	// Keep the terminal snapshot statusless just like its added snapshot.
	const contextDone = `{ "future_context":{"checkpoint":1}, "encrypted_content":"PRIVATE_CONTEXT_STREAM", "type":"context_compaction", "id":"ctx_stream_1" }`
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeRawResponsesSSE(t, writer, `{"type":"response.created","response":{"id":"resp_context_stream","object":"response","model":"fixture-model","status":"in_progress","output":[]}}`)
		writeRawResponsesSSE(t, writer, `{"type":"response.output_item.added","output_index":0,"item":`+contextAdded+`}`)
		writeRawResponsesSSE(t, writer, `{"type":"response.output_item.done","output_index":0,"item":`+contextDone+`}`)
		writeRawResponsesSSE(t, writer, `{"type":"response.completed","response":{"id":"resp_context_stream","object":"response","model":"fixture-model","status":"completed","output":[]}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","stream":true,"input":"continue"}`))
	if err != nil || result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("start context identity stream result=%#v err=%v", result, err)
	}
	defer result.EventStream.Close()
	found := 0
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || string(event.Data) == "[DONE]" {
			continue
		}
		var wire struct {
			Type string          `json:"type"`
			Item json.RawMessage `json:"item"`
		}
		if err := json.Unmarshal(event.Data, &wire); err != nil {
			t.Fatalf("decode context identity stream=%s err=%v", event.Data, err)
		}
		if wire.Type == "response.output_item.added" || wire.Type == "response.output_item.done" {
			want := []byte(contextAdded)
			if wire.Type == "response.output_item.done" {
				want = []byte(contextDone)
			}
			if !semanticJSONEqual(wire.Item, want) {
				t.Fatalf("context stream identity changed: %s", wire.Item)
			}
			found++
		}
	}
	if err := result.EventStream.Err(); err != nil || found != 2 {
		t.Fatalf("context identity stream found=%d err=%v", found, err)
	}
}

func TestResponsesProviderFutureOutputKeepsSameProtocolStreamIdentityOverRealHTTP(t *testing.T) {
	t.Parallel()
	const futureItem = `{ "future": {"mode":"stream-identity"}, "tools": [], "type": "future_behavior", "content": null, "arguments": "", "status": null, "id":"future_stream_1" }`
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		for _, event := range []string{
			`{"type":"response.created","response":{"id":"resp_future_stream","object":"response","model":"fixture-model","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","output_index":0,"item":` + futureItem + `}`,
			`{"type":"response.output_item.done","output_index":0,"item":` + futureItem + `}`,
			`{"type":"response.completed","response":{"id":"resp_future_stream","object":"response","model":"fixture-model","status":"completed","output":[]}}`,
		} {
			_, _ = writer.Write([]byte("data: " + event + "\n\n"))
			if flusher, ok := writer.(http.Flusher); ok {
				flusher.Flush()
			}
		}
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatal(err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	request := responsesRequestFromBody(`{"model":"fixture-model","stream":true,"input":"continue"}`)
	result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), request)
	if err != nil || result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("start future identity stream result=%#v err=%v", result, err)
	}
	defer result.EventStream.Close()
	found := 0
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || string(event.Data) == "[DONE]" {
			continue
		}
		var wire struct {
			Type string          `json:"type"`
			Item json.RawMessage `json:"item"`
		}
		if err := json.Unmarshal(event.Data, &wire); err != nil {
			t.Fatalf("decode future client SSE=%s err=%v", event.Data, err)
		}
		if wire.Type == "response.output_item.added" || wire.Type == "response.output_item.done" {
			if !semanticJSONEqual(wire.Item, []byte(futureItem)) {
				t.Fatalf("future stream identity changed: %s", wire.Item)
			}
			found++
		}
	}
	if err := result.EventStream.Err(); err != nil || found != 2 {
		t.Fatalf("future identity stream found=%d err=%v", found, err)
	}
}

func TestResponsesContextCompactionSameProtocolAndCrossFailClosedOverRealHTTP(t *testing.T) {
	t.Parallel()
	const contextItem = `{"id":"ctx_real_1","type":"context_compaction","encrypted_content":"PRIVATE_CONTEXT","future_context":{"checkpoint":1}}`
	t.Run("same Responses identity", func(t *testing.T) {
		var hits atomic.Int32
		provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
			hits.Add(1)
			body, _ := io.ReadAll(request.Body)
			var payload struct {
				Input []json.RawMessage `json:"input"`
			}
			if err := json.Unmarshal(body, &payload); err != nil || len(payload.Input) != 1 || !semanticJSONEqual(payload.Input[0], []byte(contextItem)) {
				http.Error(writer, "context identity changed", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(writer, `{"id":"resp_context","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[]}`)
		}))
		t.Cleanup(provider.Close)
		target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
		if err != nil {
			t.Fatal(err)
		}
		executor := httpclient.NewHttpClientWithClient(provider.Client())
		t.Cleanup(executor.CloseIdleConnections)
		result, err := pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","input":[`+contextItem+`]}`))
		if err != nil || result == nil || result.Response == nil || hits.Load() != 1 {
			t.Fatalf("context same result=%#v hits=%d err=%v", result, hits.Load(), err)
		}
	})
	for _, targetFormat := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		targetFormat := targetFormat
		t.Run("cross "+string(targetFormat), func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				hits.Add(1)
				http.Error(writer, "context must not reach cross provider", http.StatusInternalServerError)
			}))
			t.Cleanup(provider.Close)
			var target transformer.Outbound
			var err error
			if targetFormat == llm.APIFormatOpenAIChatCompletion {
				target, err = openai.NewOutboundTransformer(provider.URL, "fixture-key")
			} else {
				target, err = anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			}
			if err != nil {
				t.Fatal(err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			_, err = pipeline.NewFactory(executor).Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).Process(context.Background(), responsesRequestFromBody(`{"model":"fixture-model","input":[`+contextItem+`]}`))
			if !errors.Is(err, conversion.ErrIncompletePlan) || hits.Load() != 0 || strings.Contains(err.Error(), "PRIVATE_CONTEXT") {
				t.Fatalf("context cross error=%v hits=%d", err, hits.Load())
			}
			plan, ok := conversion.PlanFromError(err)
			if !ok || plan == nil || plan.Summary.Unknown != 1 || len(plan.Actions) != 1 || plan.Actions[0].Ref.Kind != conversion.ObjectContextCompaction ||
				plan.Actions[0].SourceType != "context_compaction" || plan.Actions[0].SemanticClass != "context_compaction_checkpoint" || plan.Actions[0].RawBytes != uint32(len(contextItem)) || len(plan.Actions[0].SourceDigest) != 64 {
				t.Fatalf("context cross plan=%#v ok=%v", plan, ok)
			}
			debug, marshalErr := json.Marshal(plan.Debug)
			if marshalErr != nil || strings.Contains(string(debug), "PRIVATE_CONTEXT") {
				t.Fatalf("context evidence leaked: %s err=%v", debug, marshalErr)
			}
		})
	}
}

func semanticJSONEqual(left, right []byte) bool {
	var leftValue, rightValue any
	return json.Unmarshal(left, &leftValue) == nil && json.Unmarshal(right, &rightValue) == nil && reflect.DeepEqual(leftValue, rightValue)
}

func outputBlockerFromObservations(events []pipeline.Observation) (llm.ConversionActionTrace, bool) {
	for index := range events {
		event := &events[index]
		if event.Stage != pipeline.StageConversionRestore || event.ConversionDebug == nil {
			continue
		}
		for actionIndex := range event.ConversionDebug.Actions {
			action := event.ConversionDebug.Actions[actionIndex]
			if action.Action == string(conversion.ActionUnknown) && action.Severity == llm.ConversionSeverityCritical && strings.HasPrefix(action.ObjectID, "output[") {
				return action, true
			}
		}
	}
	return llm.ConversionActionTrace{}, false
}

func countOutputBlockers(events []pipeline.Observation) int {
	seen := make(map[string]struct{})
	for index := range events {
		event := &events[index]
		if event.Stage != pipeline.StageConversionRestore || event.ConversionDebug == nil {
			continue
		}
		for actionIndex := range event.ConversionDebug.Actions {
			action := event.ConversionDebug.Actions[actionIndex]
			if action.Action == string(conversion.ActionUnknown) && action.Severity == llm.ConversionSeverityCritical && strings.HasPrefix(action.ObjectID, "output[") {
				seen[fmt.Sprintf("%s:%s:%s:%s", action.Direction, action.ObjectID, action.SourceType, action.SourceDigest)] = struct{}{}
			}
		}
	}
	return len(seen)
}

func fixtureSourceType(item string) string {
	var wire struct {
		Type string `json:"type"`
	}
	if json.Unmarshal([]byte(item), &wire) != nil || wire.Type == "" {
		return "unknown"
	}
	return wire.Type
}

func exactSHA256(raw string) string {
	digest := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", digest)
}

func mustMarshalJSON(t *testing.T, value any) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal payload-free evidence: %v", err)
	}
	return string(encoded)
}

type providerLegacyInterAgent struct {
	Role            string
	Author          string
	Recipient       string
	OtherRecipients []string
	Content         string
	TriggerTurn     bool
	Raw             map[string]json.RawMessage
}

func providerLegacyInterAgentMessages(body []byte) []providerLegacyInterAgent {
	var payload struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil
	}
	legacy := make([]providerLegacyInterAgent, 0)
	for index := range payload.Messages {
		message := &payload.Messages[index]
		for _, text := range providerMessageTexts(message.Content) {
			var fields map[string]json.RawMessage
			if json.Unmarshal([]byte(text), &fields) != nil {
				continue
			}
			var candidate providerLegacyInterAgent
			if json.Unmarshal(fields["author"], &candidate.Author) != nil || json.Unmarshal(fields["recipient"], &candidate.Recipient) != nil ||
				json.Unmarshal(fields["other_recipients"], &candidate.OtherRecipients) != nil || json.Unmarshal(fields["content"], &candidate.Content) != nil ||
				json.Unmarshal(fields["trigger_turn"], &candidate.TriggerTurn) != nil {
				continue
			}
			candidate.Role, candidate.Raw = message.Role, fields
			legacy = append(legacy, candidate)
		}
	}
	return legacy
}

func providerMessageTexts(content json.RawMessage) []string {
	var text string
	if json.Unmarshal(content, &text) == nil {
		return []string{text}
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if json.Unmarshal(content, &parts) != nil {
		return nil
	}
	texts := make([]string, 0, len(parts))
	for partIndex := range parts {
		if parts[partIndex].Text != "" {
			texts = append(texts, parts[partIndex].Text)
		}
	}
	return texts
}

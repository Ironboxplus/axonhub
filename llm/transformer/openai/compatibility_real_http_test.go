package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

func TestOpenAIChatCompatibilityThroughRealHTTP(t *testing.T) {
	t.Parallel()

	providerBody := make(chan []byte, 1)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read provider request: %v", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		providerBody <- body
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chatcmpl_compat","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer provider.Close()

	rawBody := []byte(`{
		"model":"deepseek-v4-flash",
		"messages":[{"role":"user","content":"hello"}],
		"thinking":{"type":"disabled","budget_tokens":4096},
		"reasoning_effort":"high",
		"metadata":{"user_id":"anthropic-user","trace_id":"keep-canonical"}
	}`)
	rawRequest := &httpclient.Request{
		Method:  http.MethodPost,
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    rawBody,
	}

	inbound := NewInboundTransformer()
	canonical, err := inbound.TransformRequest(context.Background(), rawRequest)
	if err != nil {
		t.Fatalf("transform inbound request: %v", err)
	}
	if canonical.ProviderExtensions == nil || canonical.ProviderExtensions.OpenAIChat == nil ||
		canonical.ProviderExtensions.OpenAIChat.Request == nil {
		t.Fatal("raw thinking extension was not captured")
	}
	var thinking map[string]any
	if err := json.Unmarshal(canonical.ProviderExtensions.OpenAIChat.Request.RawThinking, &thinking); err != nil {
		t.Fatalf("decode captured thinking: %v", err)
	}
	if thinking["type"] != "disabled" || thinking["budget_tokens"] != float64(4096) {
		t.Fatalf("captured thinking lost fields: %#v", thinking)
	}

	outbound, err := NewOutboundTransformer(provider.URL, "provider-key")
	if err != nil {
		t.Fatalf("create outbound transformer: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).Pipeline(inbound, outbound).Process(context.Background(), rawRequest)
	if err != nil {
		t.Fatalf("run real HTTP pipeline: %v", err)
	}
	if result == nil || result.Response == nil || result.Response.StatusCode != http.StatusOK {
		t.Fatalf("unexpected pipeline result: %#v", result)
	}

	var sent map[string]any
	if err := json.Unmarshal(<-providerBody, &sent); err != nil {
		t.Fatalf("decode provider request: %v", err)
	}
	if _, exists := sent["metadata"]; exists {
		t.Fatalf("Chat Completions provider body leaked metadata: %#v", sent["metadata"])
	}
	if sent["user"] != "anthropic-user" {
		t.Fatalf("metadata.user_id was not mapped to user: %#v", sent["user"])
	}
	sentThinking, ok := sent["thinking"].(map[string]any)
	if !ok || sentThinking["type"] != "disabled" || sentThinking["budget_tokens"] != float64(4096) {
		t.Fatalf("thinking sidecar was not replayed intact: %#v", sent["thinking"])
	}
	if sent["reasoning_effort"] != "high" {
		t.Fatalf("reasoning_effort did not coexist with thinking: %#v", sent["reasoning_effort"])
	}
	if canonical.Metadata["trace_id"] != "keep-canonical" || canonical.Metadata["user_id"] != "anthropic-user" {
		t.Fatalf("outbound mutated canonical metadata: %#v", canonical.Metadata)
	}
	if canonical.User != nil {
		t.Fatalf("outbound mutated canonical user: %#v", canonical.User)
	}
}

func TestRequestFromLLMPreservesExplicitUserWhileOmittingMetadata(t *testing.T) {
	t.Parallel()

	explicitUser := "explicit-user"
	request := &llm.Request{
		User:     &explicitUser,
		Metadata: map[string]string{"user_id": "metadata-user"},
	}
	wire := RequestFromLLM(request, ReasoningFieldContent)
	if wire.User == nil || *wire.User != explicitUser {
		t.Fatalf("explicit user was replaced: %#v", wire.User)
	}
	if wire.Metadata != nil {
		t.Fatalf("Chat Completions metadata must be omitted: %#v", wire.Metadata)
	}
	if request.Metadata["user_id"] != "metadata-user" || request.User == nil || *request.User != explicitUser {
		t.Fatalf("conversion mutated canonical request: %#v", request)
	}
}

package llm

import (
	"bytes"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/looplj/axonhub/llm/httpclient"
)

func TestRequestCloneIsolatesAttemptMutableState(t *testing.T) {
	t.Parallel()

	stream := true
	text := "original text"
	rawHTTPRequest, err := http.NewRequest(http.MethodPost, "https://client.example/v1/responses", bytes.NewBufferString("raw"))
	if err != nil {
		t.Fatalf("create raw request: %v", err)
	}

	cyclicMetadata := map[string]any{}
	cyclicMetadata["self"] = cyclicMetadata
	request := &Request{
		Model:  "original-model",
		Stream: &stream,
		Messages: []Message{{
			Role: "user",
			Content: MessageContent{MultipleContent: []MessageContentPart{{
				Type: "text",
				Text: &text,
				TransformerMetadata: map[string]any{
					"nested": []any{map[string]any{"value": "original"}},
				},
			}}},
			InlineToolResults: []InlineToolResult{{
				ToolCallID: "call_1",
				TransformerMetadata: map[string]any{
					"raw": []byte("original-result"),
				},
			}},
		}},
		TransformerMetadata: cyclicMetadata,
		ProviderExtensions: &ProviderExtensions{
			OpenAIChat: &OpenAIChatProviderExtensions{
				Request: &OpenAIChatRequestExtensions{
					RawThinking: []byte(`{"type":"disabled","budget_tokens":4096}`),
				},
			},
			OpenAIResponses: &OpenAIResponsesProviderExtensions{
				Request: &OpenAIResponsesRequestExtensions{
					RawTools: []OpenAIResponsesRawFragment{{
						Type: "custom",
						Name: "shell",
						Raw:  []byte(`{"type":"custom","name":"shell"}`),
					}},
					RawToolChoice: []byte(`{"type":"custom","name":"shell"}`),
				},
			},
		},
		RawRequest: &httpclient.Request{
			Method:  http.MethodPost,
			Query:   url.Values{"trace": {"original"}},
			Headers: http.Header{"X-Test": {"original"}},
			Body:    []byte("original-body"),
			Auth:    &httpclient.AuthConfig{Type: httpclient.AuthTypeBearer, APIKey: "original-key"},
			TransformerMetadata: map[string]any{
				"nested": map[string]any{"value": "original"},
			},
			RawRequest: rawHTTPRequest,
		},
	}
	request.RawRequest.TransformerMetadata["request"] = request.RawRequest

	cloned := request.Clone()
	if cloned == request {
		t.Fatal("Clone returned the original request pointer")
	}
	if cloned.RawRequest == request.RawRequest {
		t.Fatal("Clone shared the mutable httpclient request wrapper")
	}
	if cloned.RawRequest.RawRequest != rawHTTPRequest {
		t.Fatal("Clone should retain the read-only raw net/http request pointer")
	}

	cloned.Model = "changed-model"
	*cloned.Stream = false
	*cloned.Messages[0].Content.MultipleContent[0].Text = "changed text"
	cloned.Messages[0].Content.MultipleContent[0].TransformerMetadata["nested"].([]any)[0].(map[string]any)["value"] = "changed"
	cloned.Messages[0].InlineToolResults[0].TransformerMetadata["raw"].([]byte)[0] = 'X'
	cloned.ProviderExtensions.OpenAIChat.Request.RawThinking[0] = 'X'
	cloned.ProviderExtensions.OpenAIResponses.Request.RawTools[0].Raw[0] = 'X'
	cloned.ProviderExtensions.OpenAIResponses.Request.RawToolChoice[0] = 'X'
	cloned.RawRequest.Query.Set("trace", "changed")
	cloned.RawRequest.Headers.Set("X-Test", "changed")
	cloned.RawRequest.Body[0] = 'X'
	cloned.RawRequest.Auth.APIKey = "changed-key"
	cloned.RawRequest.TransformerMetadata["nested"].(map[string]any)["value"] = "changed"
	cloned.TransformerMetadata["clone-only"] = true

	if request.Model != "original-model" || !*request.Stream || *request.Messages[0].Content.MultipleContent[0].Text != "original text" {
		t.Fatal("top-level or nested request fields leaked from the clone")
	}
	if got := request.Messages[0].Content.MultipleContent[0].TransformerMetadata["nested"].([]any)[0].(map[string]any)["value"]; got != "original" {
		t.Fatalf("message metadata leaked from clone: %v", got)
	}
	if got := string(request.Messages[0].InlineToolResults[0].TransformerMetadata["raw"].([]byte)); got != "original-result" {
		t.Fatalf("inline tool metadata leaked from clone: %q", got)
	}
	if request.ProviderExtensions.OpenAIResponses.Request.RawTools[0].Raw[0] == 'X' || request.ProviderExtensions.OpenAIResponses.Request.RawToolChoice[0] == 'X' {
		t.Fatal("provider extension raw JSON leaked from clone")
	}
	if request.ProviderExtensions.OpenAIChat.Request.RawThinking[0] == 'X' {
		t.Fatal("OpenAI chat provider extension raw JSON leaked from clone")
	}
	if request.RawRequest.Query.Get("trace") != "original" || request.RawRequest.Headers.Get("X-Test") != "original" {
		t.Fatal("raw request query or headers leaked from clone")
	}
	if string(request.RawRequest.Body) != "original-body" || request.RawRequest.Auth.APIKey != "original-key" {
		t.Fatal("raw request body or auth leaked from clone")
	}
	if got := request.RawRequest.TransformerMetadata["nested"].(map[string]any)["value"]; got != "original" {
		t.Fatalf("raw request metadata leaked from clone: %v", got)
	}
	if got := cloned.RawRequest.TransformerMetadata["request"].(*httpclient.Request); got != cloned.RawRequest {
		t.Fatal("Clone did not preserve a cycle through the httpclient request wrapper")
	}
	if _, exists := request.TransformerMetadata["clone-only"]; exists {
		t.Fatal("cyclic transformer metadata leaked from clone")
	}

	clonedSelf := cloned.TransformerMetadata["self"].(map[string]any)
	if reflect.ValueOf(clonedSelf).Pointer() != reflect.ValueOf(cloned.TransformerMetadata).Pointer() {
		t.Fatal("Clone did not preserve a cycle inside transformer metadata")
	}
	if reflect.ValueOf(cloned.TransformerMetadata).Pointer() == reflect.ValueOf(request.TransformerMetadata).Pointer() {
		t.Fatal("Clone shared the original transformer metadata map")
	}

	// Prove the retained raw HTTP body was not consumed or replaced by cloning.
	rawBody, err := io.ReadAll(rawHTTPRequest.Body)
	if err != nil {
		t.Fatalf("read retained raw request body: %v", err)
	}
	if string(rawBody) != "raw" {
		t.Fatalf("raw HTTP request body changed: %q", rawBody)
	}
}

func TestNilRequestClone(t *testing.T) {
	t.Parallel()

	var request *Request
	if request.Clone() != nil {
		t.Fatal("nil request clone must be nil")
	}
}

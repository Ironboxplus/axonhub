package llm

import (
	"net/http"
	"testing"

	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesLiteWireProfileDetectionUsesMetadataAndCaseInsensitiveRawHeader(t *testing.T) {
	t.Parallel()

	if HasResponsesLiteWireHeader(nil) || UsesResponsesLiteWireRequest(nil) || UsesResponsesLiteWireProfile(nil) {
		t.Fatal("nil request selected Responses Lite")
	}
	canonicalHeader := &httpclient.Request{Headers: http.Header{
		"X-Other":                                []string{"ignored"},
		"x-openai-internal-codex-responses-lite": []string{" false ", " TRUE "},
	}}
	if !HasResponsesLiteWireHeader(canonicalHeader) || !UsesResponsesLiteWireRequest(canonicalHeader) ||
		!UsesResponsesLiteWireProfile(&Request{RawRequest: canonicalHeader}) {
		t.Fatalf("case-insensitive Lite header was not recognized: %#v", canonicalHeader.Headers)
	}
	notLite := &httpclient.Request{Headers: http.Header{
		"X-OpenAI-Internal-Codex-Responses-Lite": []string{"false"},
	}}
	if HasResponsesLiteWireHeader(notLite) || UsesResponsesLiteWireRequest(notLite) ||
		UsesResponsesLiteWireProfile(&Request{RawRequest: notLite}) {
		t.Fatalf("non-Lite header selected Lite: %#v", notLite.Headers)
	}
	metadata := &httpclient.Request{TransformerMetadata: map[string]any{
		ResponsesWireProfileMetadataKey: ResponsesWireProfileLiteValue,
	}}
	if !UsesResponsesLiteWireRequest(metadata) || !UsesResponsesLiteWireProfile(&Request{TransformerMetadata: metadata.TransformerMetadata}) {
		t.Fatalf("Lite metadata was not recognized: %#v", metadata.TransformerMetadata)
	}
	if UsesResponsesLiteWireProfile(&Request{TransformerMetadata: map[string]any{
		ResponsesWireProfileMetadataKey: "responses",
	}}) {
		t.Fatal("non-Lite metadata selected Lite")
	}
}

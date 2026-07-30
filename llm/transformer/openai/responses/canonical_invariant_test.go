package responses

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestResponsesCanonicalRequestRejectsUnencodableItemInsteadOfDroppingIt(t *testing.T) {
	t.Parallel()
	_, _, ok, err := canonicalRequestInput(&llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		Input: []llm.Item{{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{
			Type: "future_behavior", Raw: json.RawMessage(`{"type":"future_behavior"}`), Behavioral: true,
		}}},
	})
	if !ok || err == nil || !strings.Contains(err.Error(), "canonical input item") {
		t.Fatalf("unencodable canonical item was not rejected: ok=%v err=%v", ok, err)
	}
}

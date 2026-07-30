package llm

import (
	"encoding/json"
	"testing"
)

// FuzzCanonicalItemsJSONRoundTrip keeps arbitrary client-owned documents on the
// same validation boundary used by all three protocol decoders. Valid canonical
// documents must remain valid after a JSON round trip; malformed documents may
// be rejected, but must never panic or become silently valid with different
// tool/result identity.
func FuzzCanonicalItemsJSONRoundTrip(f *testing.F) {
	f.Add([]byte(`[{
		"kind":"message","role":"user","content":[{"kind":"text","text":"hello"}]
	}]`))
	f.Add([]byte(`[
		{"kind":"tool_call","tool_call":{"kind":"function","id":"item_1","call_id":"call_1","logical_name":"lookup","arguments_json":{"q":"x"}}},
		{"kind":"tool_result","tool_result":{"kind":"function","call_id":"call_1","content":[{"kind":"text","text":"ok"}]}}
	]`))
	f.Add([]byte(`[{"kind":"tool_result","tool_result":{"kind":"function","call_id":"orphan"}}]`))

	f.Fuzz(func(t *testing.T, document []byte) {
		if len(document) > 1<<20 {
			t.Skip()
		}
		var items []Item
		if err := json.Unmarshal(document, &items); err != nil {
			return
		}
		if err := ValidateItems(items); err != nil {
			return
		}
		encoded, err := json.Marshal(items)
		if err != nil {
			t.Fatalf("marshal valid canonical items: %v", err)
		}
		var roundTripped []Item
		if err := json.Unmarshal(encoded, &roundTripped); err != nil {
			t.Fatalf("unmarshal canonical round trip: %v", err)
		}
		if err := ValidateItems(roundTripped); err != nil {
			t.Fatalf("canonical round trip became invalid: %v\n%s", err, encoded)
		}
	})
}

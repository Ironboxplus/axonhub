package responses

import (
	"encoding/json"
	"testing"
)

// FuzzResponsesCanonicalStreamDecoder feeds the real Responses wire event
// decoder, including sequence-number and lifecycle validation. Invalid or
// unknown events may return a stable error, but arbitrary JSON must not panic.
func FuzzResponsesCanonicalStreamDecoder(f *testing.F) {
	f.Add([]byte(`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","model":"fixture","status":"in_progress","output":[]}}`))
	f.Add([]byte(`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"item_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"","status":"in_progress"}}`))
	f.Add([]byte(`{"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","object":"response","model":"fixture","status":"completed","output":[]}}`))
	f.Add([]byte(`{"type":"future.provider.event","sequence_number":-1,"payload":{"nested":[1,2,3]}}`))

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 1<<20 {
			t.Skip()
		}
		var wire StreamEvent
		if err := json.Unmarshal(data, &wire); err != nil {
			return
		}
		decoder := newCanonicalStreamDecoder()
		_, _ = decoder.decode(&wire)
	})
}

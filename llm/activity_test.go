package llm

import (
	"encoding/json"
	"testing"
)

func TestResponseHasModelActivityUsesCanonicalEvents(t *testing.T) {
	outputIndex := 0
	ref := ItemRef{OutputIndex: &outputIndex, ItemID: "item_1", CallID: "call_1"}
	tests := []struct {
		name     string
		response *Response
		want     bool
	}{
		{
			name:     "response lifecycle metadata is not activity",
			response: &Response{Events: []Event{{Kind: EventKindResponseStarted}}},
		},
		{
			name: "empty assistant message declaration is not activity",
			response: &Response{Events: []Event{{
				Kind: EventKindItemAdded, ItemRef: ref,
				Snapshot: &Item{Kind: ItemKindMessage, Role: RoleAssistant, Content: []ContentBlock{{Kind: ContentKindText}}},
			}}},
		},
		{
			name:     "text delta is activity",
			response: &Response{Events: []Event{{Kind: EventKindTextDelta, ItemRef: ref, Delta: Delta{Text: "hello"}}}},
			want:     true,
		},
		{
			name:     "reasoning delta is activity",
			response: &Response{Events: []Event{{Kind: EventKindReasoningDelta, ItemRef: ref, Delta: Delta{Text: "thinking"}}}},
			want:     true,
		},
		{
			name: "tool declaration is activity before arguments arrive",
			response: &Response{Events: []Event{{
				Kind: EventKindItemAdded, ItemRef: ref,
				Snapshot: &Item{Kind: ItemKindToolCall, ToolCall: &ToolInvocation{Kind: ToolKindFunction, CallID: "call_1", LogicalName: "lookup"}},
			}}},
			want: true,
		},
		{
			name: "unknown provider output is activity",
			response: &Response{Events: []Event{{
				Kind: EventKindItemDone, ItemRef: ref,
				Snapshot: &Item{Kind: ItemKindUnknown, Unknown: &UnknownItem{Type: "future_output", Raw: json.RawMessage(`{"type":"future_output"}`)}},
			}}},
			want: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.response.HasModelActivity(); got != test.want {
				t.Fatalf("HasModelActivity() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestResponseHasModelActivityKeepsLegacyCompatibility(t *testing.T) {
	text := "legacy reasoning"
	response := &Response{Choices: []Choice{{Delta: &Message{ReasoningContent: &text}}}}
	if !response.HasModelActivity() {
		t.Fatal("legacy reasoning delta must remain model activity")
	}
}

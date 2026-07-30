package conversion

import (
	"context"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestDecodeCustomInputDeterministicRepair(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		arguments string
		want      string
		status    customInputStatus
	}{
		{
			name:      "exact function envelope",
			arguments: `{"input":"*** Begin Patch\n*** End Patch"}`,
			want:      "*** Begin Patch\n*** End Patch",
			status:    customInputExact,
		},
		{
			name:      "markdown fenced envelope",
			arguments: "```json\n{\"input\":\"pwd\"}\n```",
			want:      "pwd",
			status:    customInputRepaired,
		},
		{
			name:      "double encoded envelope",
			arguments: `"{\"input\":\"pwd\"}"`,
			want:      "pwd",
			status:    customInputRepaired,
		},
		{
			name:      "bare json string",
			arguments: `"pwd"`,
			want:      "pwd",
			status:    customInputRepaired,
		},
		{
			name:      "literal newline inside input string",
			arguments: "{\"input\":\"line one\nline two\"}",
			want:      "line one\nline two",
			status:    customInputRepaired,
		},
		{
			name:      "missing string and object terminators",
			arguments: `{"input":"partial patch`,
			want:      "partial patch",
			status:    customInputRepaired,
		},
		{
			name:      "trailing object comma",
			arguments: `{"input":"pwd",}`,
			want:      "pwd",
			status:    customInputRepaired,
		},
		{
			name:      "raw freeform fallback stays byte exact",
			arguments: "  *** Begin Patch\r\n*** End Patch  ",
			want:      "  *** Begin Patch\r\n*** End Patch  ",
			status:    customInputRaw,
		},
		{
			name:      "wrong envelope stays byte exact",
			arguments: `{"command":"pwd"}`,
			want:      `{"command":"pwd"}`,
			status:    customInputRaw,
		},
		{
			name:      "non string input stays byte exact",
			arguments: `{"input":{"command":"pwd"}}`,
			want:      `{"input":{"command":"pwd"}}`,
			status:    customInputRaw,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			got, status := decodeCustomInput(test.arguments)
			if got != test.want || status != test.status {
				t.Fatalf("decodeCustomInput(%q) = (%q, %v), want (%q, %v)", test.arguments, got, status, test.want, test.status)
			}
		})
	}
}

func TestRestoreCustomInputRecordsRepairWithoutPayload(t *testing.T) {
	t.Parallel()

	trace := llm.NewConversionDebugTrace(
		llm.WithConversionDebugTrace(context.Background(), []byte("custom-repair"), 4),
		1,
	)
	session := &Session{plan: &Plan{Debug: trace}, traceEnabled: true}
	ref := ObjectRef{Kind: ObjectToolCall, ItemIndex: 7, ToolIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}

	input, valid := restoreCustomInput(`{"input":"pwd",}`, "call-1", session, llm.ConversionDirectionResponse, ref)
	if !valid || input != "pwd" {
		t.Fatalf("repaired input = (%q, %v), want (pwd, true)", input, valid)
	}
	summary := session.Summary()
	if summary.CustomInputsRepaired != 1 || summary.RestoreMiss != 0 {
		t.Fatalf("repair summary = %#v", summary)
	}
	snapshot := session.DebugTrace()
	if snapshot == nil || len(snapshot.Actions) != 1 {
		t.Fatalf("repair debug trace missing: %#v", snapshot)
	}
	action := snapshot.Actions[0]
	if action.Action != "repair" || action.Strategy != string(StrategyCustomAsFunction) || !action.Reversible {
		t.Fatalf("repair action = %#v", action)
	}

	raw := `{"wrong":"do-not-log-me"}`
	input, valid = restoreCustomInput(raw, "call-2", session, llm.ConversionDirectionStream, ref)
	if valid || input != raw {
		t.Fatalf("raw fallback = (%q, %v), want byte-exact false", input, valid)
	}
	if got := session.Summary(); got.CustomInputsRepaired != 1 || got.RestoreMiss != 1 || got.Complete {
		t.Fatalf("raw fallback summary = %#v", got)
	}
	for _, recorded := range session.DebugTrace().Actions {
		if recorded.LogicalIDHash == raw || recorded.Reason == raw || recorded.Strategy == raw {
			t.Fatalf("raw arguments leaked into trace: %#v", recorded)
		}
	}
}

package conversion

import (
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestStreamRestorerKeysStateByWireChoiceIndex(t *testing.T) {
	session := &Session{bySyntheticName: map[string]toolIdentity{
		"synthetic_a": {SourceKind: llm.ToolKindCustom, SourceName: "custom_a"},
		"synthetic_b": {SourceKind: llm.ToolKindCustom, SourceName: "custom_b"},
	}}
	restorer := newStreamRestorer(session)

	restorer.restore(&llm.Response{Choices: []llm.Choice{
		{Index: 7, Delta: streamFunctionDelta(0, "call_a", "synthetic_a", `{"input":"a`)},
		{Index: 2, Delta: streamFunctionDelta(0, "call_b", "synthetic_b", `{"input":"b`)},
	}})
	// Providers may reorder the choice array in a later chunk. The stable
	// choice.index must keep each accumulator attached to its own call.
	completed := restorer.restore(&llm.Response{Choices: []llm.Choice{
		{Index: 2, Delta: streamFunctionDelta(0, "", "", `-done"}`)},
		{Index: 7, Delta: streamFunctionDelta(0, "", "", `-done"}`)},
	}})

	assertRestoredCustomInput(t, completed.Choices[0].Delta, "call_b", "custom_b", "b-done")
	assertRestoredCustomInput(t, completed.Choices[1].Delta, "call_a", "custom_a", "a-done")
}

func TestStreamRestorerFlushesIncompleteCallsInCreationOrder(t *testing.T) {
	session := &Session{plan: &Plan{Summary: llm.ConversionTraceSummary{Complete: true}}, bySyntheticName: map[string]toolIdentity{
		"synthetic_first":  {SourceKind: llm.ToolKindCustom, SourceName: "custom_first"},
		"synthetic_second": {SourceKind: llm.ToolKindCustom, SourceName: "custom_second"},
	}}
	restorer := newStreamRestorer(session)
	restorer.restore(&llm.Response{Choices: []llm.Choice{{
		Index: 4,
		Delta: &llm.Message{ToolCalls: []llm.ToolCall{
			streamFunctionDelta(3, "call_first", "synthetic_first", `{"input":"first`).ToolCalls[0],
			streamFunctionDelta(1, "call_second", "synthetic_second", `{"input":"second`).ToolCalls[0],
		}},
	}}})

	finish := "tool_calls"
	terminal := restorer.restore(&llm.Response{Choices: []llm.Choice{{Index: 4, FinishReason: &finish}}})
	got := terminal.Choices[0].Delta.ToolCalls
	if len(got) != 2 || got[0].Index != 3 || got[1].Index != 1 {
		t.Fatalf("incomplete call flush order = %#v, want tool indexes [3 1]", got)
	}
	assertRestoredCustomInput(t, &llm.Message{ToolCalls: got[:1]}, "call_first", "custom_first", "first")
	assertRestoredCustomInput(t, &llm.Message{ToolCalls: got[1:]}, "call_second", "custom_second", "second")
	if summary := session.Summary(); summary.RestoreMiss != 0 || summary.CustomInputsRepaired != 2 || !summary.Complete {
		t.Fatalf("repaired incomplete stream summary = %#v", summary)
	}
}

func TestStreamRestorerPreservesUnrepairableArgumentsAtTerminal(t *testing.T) {
	raw := `{"wrong":"keep every byte"}`
	session := &Session{plan: &Plan{Summary: llm.ConversionTraceSummary{Complete: true}}, bySyntheticName: map[string]toolIdentity{
		"synthetic": {SourceKind: llm.ToolKindCustom, SourceName: "custom"},
	}}
	restorer := newStreamRestorer(session)
	restored := restorer.restore(&llm.Response{Choices: []llm.Choice{{
		Index: 0,
		Delta: streamFunctionDelta(0, "call_raw", "synthetic", raw),
	}}})
	assertRestoredCustomInput(t, restored.Choices[0].Delta, "call_raw", "custom", raw)
	if summary := session.Summary(); summary.RestoreMiss != 1 || summary.CustomInputsRepaired != 0 || summary.Complete {
		t.Fatalf("raw stream fallback summary = %#v", summary)
	}
}

func TestConversionSummaryRecordsIncompleteCanonicalTerminal(t *testing.T) {
	session := &Session{
		plan:         &Plan{Summary: llm.ConversionTraceSummary{Complete: true}},
		traceEnabled: true,
	}
	session.recordTerminalEvent(llm.EventKindResponseIncomplete)
	summary := session.Summary()
	if summary.TerminalEvent != llm.EventKindResponseIncomplete || summary.Complete {
		t.Fatalf("incomplete terminal summary = %#v", summary)
	}
}

func streamFunctionDelta(index int, callID, name, arguments string) *llm.Message {
	return &llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{
		Index: index, ID: callID, Type: llm.ToolTypeFunction,
		Function: llm.FunctionCall{Name: name, Arguments: arguments},
	}}}
}

func assertRestoredCustomInput(t *testing.T, message *llm.Message, callID, name, input string) {
	t.Helper()
	if message == nil || len(message.ToolCalls) != 1 {
		t.Fatalf("restored message = %#v", message)
	}
	call := message.ToolCalls[0]
	if call.ResponseCustomToolCall == nil || call.ResponseCustomToolCall.CallID != callID ||
		call.ResponseCustomToolCall.Name != name || call.ResponseCustomToolCall.Input != input {
		t.Fatalf("restored call = %#v", call)
	}
}

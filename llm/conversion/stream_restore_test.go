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

func TestStreamRestorerReplaysUnchangedFunctionArgumentFragments(t *testing.T) {
	outputIndex := 0
	ref := llm.ItemRef{ItemID: "fc_unchanged", CallID: "call_unchanged", OutputIndex: &outputIndex}
	restorer := newStreamRestorer(&Session{})
	restorer.restore(&llm.Response{Events: []llm.Event{{
		Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref,
		Snapshot: &llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, ID: "fc_unchanged", CallID: "call_unchanged", LogicalName: "lookup",
		}},
	}}})

	first := `{"query":"hel`
	if response := restorer.restore(&llm.Response{Events: []llm.Event{{
		Kind: llm.EventKindToolInputDelta, Sequence: 2, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: first},
	}}}); len(response.Events) != 0 {
		t.Fatalf("incomplete unchanged arguments must remain pending: %#v", response.Events)
	}

	second := `lo"}`
	response := restorer.restore(&llm.Response{Events: []llm.Event{{
		Kind: llm.EventKindToolInputDelta, Sequence: 3, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: second},
	}}})
	if len(response.Events) != 2 || response.Events[0].Sequence != 2 || response.Events[1].Sequence != 3 ||
		response.Events[0].Delta.ArgumentsJSON != first || response.Events[1].Delta.ArgumentsJSON != second {
		t.Fatalf("unchanged function fragments must replay in source order: %#v", response.Events)
	}
}

func TestStreamRestorerReplaysParallelUnchangedFunctionArgumentFragmentsInSequenceOrder(t *testing.T) {
	firstOutput, secondOutput := 0, 1
	firstRef := llm.ItemRef{ItemID: "fc_first", CallID: "call_first", OutputIndex: &firstOutput}
	secondRef := llm.ItemRef{ItemID: "fc_second", CallID: "call_second", OutputIndex: &secondOutput}
	restorer := newStreamRestorer(&Session{})
	for _, item := range []struct {
		ref      llm.ItemRef
		name     string
		sequence uint64
	}{
		{ref: firstRef, name: "lookup", sequence: 1},
		{ref: secondRef, name: "weather", sequence: 2},
	} {
		restorer.restore(&llm.Response{Events: []llm.Event{{
			Kind: llm.EventKindItemAdded, Sequence: item.sequence, ItemRef: item.ref,
			Snapshot: &llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, ID: item.ref.ItemID, CallID: item.ref.CallID, LogicalName: item.name}},
		}}})
	}
	if response := restorer.restore(&llm.Response{Events: []llm.Event{
		{Kind: llm.EventKindToolInputDelta, Sequence: 3, ItemRef: firstRef, Delta: llm.Delta{ArgumentsJSON: `{"city":"`}},
		{Kind: llm.EventKindToolInputDelta, Sequence: 4, ItemRef: secondRef, Delta: llm.Delta{ArgumentsJSON: `{"unit":"`}},
	}}); len(response.Events) != 0 {
		t.Fatalf("parallel partial function arguments must remain pending: %#v", response.Events)
	}
	response := restorer.restore(&llm.Response{Events: []llm.Event{
		{Kind: llm.EventKindToolInputDelta, Sequence: 5, ItemRef: firstRef, Delta: llm.Delta{ArgumentsJSON: `Tokyo"}`}},
		{Kind: llm.EventKindToolInputDelta, Sequence: 6, ItemRef: secondRef, Delta: llm.Delta{ArgumentsJSON: `C"}`}},
	}})
	if len(response.Events) != 4 {
		t.Fatalf("parallel fragment replay count = %d events=%#v", len(response.Events), response.Events)
	}
	for index, want := range []uint64{3, 4, 5, 6} {
		if response.Events[index].Sequence != want {
			t.Fatalf("parallel fragment sequence at %d = %d, want %d: %#v", index, response.Events[index].Sequence, want, response.Events)
		}
	}
}

func TestStreamRestorerRebasesDelayedParallelFragmentsAcrossResponses(t *testing.T) {
	firstOutput, secondOutput := 0, 1
	firstRef := llm.ItemRef{ItemID: "fc_first", CallID: "call_first", OutputIndex: &firstOutput}
	secondRef := llm.ItemRef{ItemID: "fc_second", CallID: "call_second", OutputIndex: &secondOutput}
	restorer := newStreamRestorer(&Session{})
	for _, item := range []struct {
		ref      llm.ItemRef
		name     string
		sequence uint64
	}{
		{ref: firstRef, name: "lookup", sequence: 1},
		{ref: secondRef, name: "weather", sequence: 2},
	} {
		restorer.restore(&llm.Response{Events: []llm.Event{{
			Kind: llm.EventKindItemAdded, Sequence: item.sequence, ItemRef: item.ref,
			Snapshot: &llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, ID: item.ref.ItemID, CallID: item.ref.CallID, LogicalName: item.name}},
		}}})
	}
	for _, event := range []llm.Event{
		{Kind: llm.EventKindToolInputDelta, Sequence: 3, ItemRef: firstRef, Delta: llm.Delta{ArgumentsJSON: `{"city":"`}},
		{Kind: llm.EventKindToolInputDelta, Sequence: 4, ItemRef: secondRef, Delta: llm.Delta{ArgumentsJSON: `{"unit":"`}},
	} {
		if response := restorer.restore(&llm.Response{Events: []llm.Event{event}}); len(response.Events) != 0 {
			t.Fatalf("parallel partial arguments must remain pending: %#v", response.Events)
		}
	}
	var emitted []llm.Event
	for _, event := range []llm.Event{
		{Kind: llm.EventKindToolInputDelta, Sequence: 5, ItemRef: firstRef, Delta: llm.Delta{ArgumentsJSON: `Tokyo"}`}},
		{Kind: llm.EventKindToolInputDelta, Sequence: 6, ItemRef: secondRef, Delta: llm.Delta{ArgumentsJSON: `C"}`}},
	} {
		response := restorer.restore(&llm.Response{Events: []llm.Event{event}})
		emitted = append(emitted, response.Events...)
	}
	if len(emitted) != 4 {
		t.Fatalf("delayed parallel fragment count = %d events=%#v", len(emitted), emitted)
	}
	for index := 1; index < len(emitted); index++ {
		if emitted[index-1].Sequence >= emitted[index].Sequence {
			t.Fatalf("delayed replay must have strictly increasing output sequences: %#v", emitted)
		}
	}
	if emitted[0].Delta.ArgumentsJSON != `{"city":"` || emitted[1].Delta.ArgumentsJSON != `Tokyo"}` ||
		emitted[2].Delta.ArgumentsJSON != `{"unit":"` || emitted[3].Delta.ArgumentsJSON != `C"}` {
		t.Fatalf("delayed parallel fragments lost per-call order: %#v", emitted)
	}
}

func TestStreamRestorerPreservesNativeCustomInputDelta(t *testing.T) {
	outputIndex := 0
	ref := llm.ItemRef{ItemID: "custom_unchanged", CallID: "call_custom", OutputIndex: &outputIndex}
	restorer := newStreamRestorer(&Session{})
	restorer.restore(&llm.Response{Events: []llm.Event{{
		Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref,
		Snapshot: &llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindCustom, ID: "custom_unchanged", CallID: "call_custom", LogicalName: "exec",
		}},
	}}})

	response := restorer.restore(&llm.Response{Events: []llm.Event{{
		Kind: llm.EventKindToolInputDelta, Sequence: 2, ItemRef: ref, Delta: llm.Delta{InputText: "echo 1000.0"},
	}}})
	if len(response.Events) != 1 || response.Events[0].Delta.InputText != "echo 1000.0" || response.Events[0].Delta.ArgumentsJSON != "" {
		t.Fatalf("native custom input delta must be untouched: %#v", response.Events)
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

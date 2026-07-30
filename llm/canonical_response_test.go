package llm

import "testing"

func TestCanonicalResponseAccumulatorMaterializesInterleavedParallelTools(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	events := parallelToolLifecycleEvents()
	for index := range events {
		response := &Response{ID: "provider_response_1", Model: "fixture-model", Events: []Event{events[index]}}
		if err := accumulator.Observe(response); err != nil {
			t.Fatalf("observe event %d %s: %v", index, events[index].Kind, err)
		}
	}
	if !accumulator.IsTerminal() || accumulator.TerminalKind() != EventKindResponseCompleted {
		t.Fatalf("terminal = %q", accumulator.TerminalKind())
	}
	snapshot := accumulator.Snapshot()
	if snapshot.ID != "provider_response_1" || len(snapshot.Output) != 2 {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if got := string(snapshot.Output[0].ToolCall.ArgumentsJSON); got != `{"q":"hello"}` {
		t.Fatalf("first arguments = %q", got)
	}
	if got := string(snapshot.Output[1].ToolCall.ArgumentsJSON); got != `{"x":1}` {
		t.Fatalf("second arguments = %q", got)
	}

	// Snapshot must not retain mutable decoder state.
	snapshot.Output[0].ToolCall.ArgumentsJSON[0] = '['
	if got := string(accumulator.Snapshot().Output[0].ToolCall.ArgumentsJSON); got != `{"q":"hello"}` {
		t.Fatalf("snapshot mutation leaked into accumulator: %q", got)
	}
}

func TestCanonicalResponseAccumulatorRejectsDuplicateObservation(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	event := Event{Kind: EventKindResponseStarted, Sequence: 0}
	if err := accumulator.Observe(&Response{Events: []Event{event}}); err != nil {
		t.Fatalf("observe first event: %v", err)
	}
	if err := accumulator.Observe(&Response{Events: []Event{event}}); err == nil {
		t.Fatal("duplicate event observation was accepted")
	}
}

func TestCanonicalResponseAccumulatorConsumesTerminalAttachedToDoneMarker(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	events := []Event{
		{Kind: EventKindResponseStarted, Sequence: 0},
		{Kind: EventKindResponseInProgress, Sequence: 1},
	}
	if err := accumulator.Observe(&Response{Events: events}); err != nil {
		t.Fatalf("observe start: %v", err)
	}
	if err := accumulator.Observe(&Response{
		Object: "[DONE]",
		Events: []Event{{Kind: EventKindResponseCompleted, Sequence: 2}},
	}); err != nil {
		t.Fatalf("observe done terminal: %v", err)
	}
	if !accumulator.IsTerminal() || accumulator.TerminalKind() != EventKindResponseCompleted {
		t.Fatalf("terminal = %q", accumulator.TerminalKind())
	}
}

func TestCanonicalTerminalReasonSurvivesMaterializeAndAggregate(t *testing.T) {
	response := &Response{
		ID: "provider_pause", Status: ResponseStatusIncomplete, TerminalReason: "pause_turn",
		Output: []Item{{
			Kind: ItemKindMessage, Role: RoleAssistant, Status: ItemStatusIncomplete,
			Content: []ContentBlock{{Kind: ContentKindText, Text: "continuing"}},
		}},
	}
	events, err := CanonicalEventsFromResponse(response)
	if err != nil {
		t.Fatalf("render pause_turn response: %v", err)
	}
	terminal := events[len(events)-1]
	if terminal.Kind != EventKindResponseIncomplete || terminal.TerminalReason != "pause_turn" {
		t.Fatalf("terminal event = %#v", terminal)
	}
	accumulator := NewCanonicalResponseAccumulator()
	for index := range events {
		if err := accumulator.Observe(&Response{Events: []Event{events[index]}}); err != nil {
			t.Fatalf("observe pause_turn event %d: %v", index, err)
		}
	}
	snapshot := accumulator.Snapshot()
	if snapshot.Status != ResponseStatusIncomplete || snapshot.TerminalReason != "pause_turn" {
		t.Fatalf("pause_turn snapshot = %#v", snapshot)
	}

	invalid := Event{Kind: EventKindTextDelta, ItemRef: ItemRef{ItemID: "msg"}, Delta: Delta{Text: "x"}, TerminalReason: "pause_turn"}
	if err := invalid.Validate(); err == nil {
		t.Fatal("non-terminal event accepted a terminal reason")
	}
}

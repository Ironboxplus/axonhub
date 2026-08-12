package llm

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestEventValidateAllPayloadAndResidualBranches(t *testing.T) {
	t.Parallel()
	if err := (*Event)(nil).Validate(); err == nil {
		t.Fatal("nil event was accepted")
	}
	validItem := completedMessageStateItem("msg_validate")
	invalid := []Event{
		{Kind: EventKindResponseStarted, Usage: &Usage{}},
		{Kind: EventKindResponseFailed},
		{Kind: EventKindItemAdded},
		{Kind: EventKindItemAdded, Snapshot: &Item{Kind: ItemKind("future")}, ItemRef: ItemRef{ItemID: "bad"}},
		{Kind: EventKindItemAdded, Snapshot: validItem},
		{Kind: EventKindTextDelta},
		{Kind: EventKindToolInputDelta, Delta: Delta{ArgumentsJSON: `{}`}},
		{Kind: EventKindToolInputDelta, ItemRef: ItemRef{ItemID: "tool"}},
		{Kind: EventKindToolInputDelta, ItemRef: ItemRef{ItemID: "tool"}, Delta: Delta{ArgumentsJSON: `{}`, InputText: "both"}},
		{Kind: EventKindToolInputDone},
		{Kind: EventKindHostedStatus, Delta: Delta{HostedStatus: "running"}},
		{Kind: EventKindHostedStatus, ItemRef: ItemRef{ItemID: "hosted"}},
		{Kind: EventKindMCPStatus, Delta: Delta{MCPStatus: "running"}},
		{Kind: EventKindMCPStatus, ItemRef: ItemRef{ItemID: "mcp"}},
		{Kind: EventKindUsage},
		{Kind: EventKind("future")},
		{Kind: EventKindResponseStarted, Delta: Delta{ProviderData: json.RawMessage(`{`)}},
		{Kind: EventKindResponseStarted, SourceResidual: json.RawMessage(`{`)},
		{Kind: EventKindResponseStarted, ResponseSourceResidual: json.RawMessage(`{`)},
		{Kind: EventKindResponseStarted, ProtocolFrames: []ProtocolFrameHint{{}}},
		{Kind: EventKindResponseStarted, ProtocolFrames: []ProtocolFrameHint{{SourceType: "frame", SourceResidual: json.RawMessage(`{`)}}},
		{Kind: EventKindResponseStarted, ProtocolFrames: []ProtocolFrameHint{{SourceType: "frame", PayloadResidual: json.RawMessage(`{`)}}},
		{Kind: EventKindResponseStarted, TerminalReason: "pause_turn"},
	}
	for index := range invalid {
		if err := invalid[index].Validate(); err == nil {
			t.Fatalf("invalid event %d was accepted: %#v", index, invalid[index])
		}
	}

	valid := Event{
		Kind: EventKindResponseCompleted, TerminalReason: "pause_turn",
		SourceResidual:         json.RawMessage(`{"future":1}`),
		ResponseSourceResidual: json.RawMessage(`{"future_response":1}`),
		ProtocolFrames: []ProtocolFrameHint{{
			SourceType:      "response.content_part.done",
			SourceResidual:  json.RawMessage(`{"future_event":1}`),
			PayloadResidual: json.RawMessage(`{"future_part":1}`),
		}},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid residual event: %v", err)
	}
	if err := (&Event{
		Kind: EventKindResponseFailed, TerminalReason: "provider_failure",
		Error: &ResponseError{Detail: ErrorDetail{Message: "failed"}},
	}).Validate(); err != nil {
		t.Fatalf("response-failed terminal reason must be valid: %v", err)
	}
}

func TestItemRefStableKeyCoversEveryIdentityShape(t *testing.T) {
	t.Parallel()
	choice, tool, output := 2, 3, 4
	for _, test := range []struct {
		ref  ItemRef
		want string
	}{
		{ref: ItemRef{ItemID: "item"}, want: "item:item"},
		{ref: ItemRef{CallID: "call"}, want: "call:call"},
		{ref: ItemRef{ChoiceIndex: &choice, ToolIndex: &tool}, want: "choice:2/tool:3"},
		{ref: ItemRef{OutputIndex: &output}, want: "output:4"},
		{ref: ItemRef{ChoiceIndex: &choice}, want: "choice:2"},
	} {
		got, err := test.ref.StableKey()
		if err != nil || got != test.want {
			t.Fatalf("stable key %#v = %q, %v; want %q", test.ref, got, err, test.want)
		}
	}
	if _, err := (ItemRef{}).StableKey(); err == nil {
		t.Fatal("empty item reference was accepted")
	}
}

func TestStreamInvariantErrorAndNilMachineBranches(t *testing.T) {
	t.Parallel()
	if got := (*StreamInvariantError)(nil).Error(); got != "<nil>" {
		t.Fatalf("nil invariant error = %q", got)
	}
	cause := errors.New("cause")
	violation := &StreamInvariantError{Code: StreamInvariantInvalidEvent, EventKind: EventKindError, Sequence: 7, Cause: cause}
	if got := violation.Error(); got != "stream invariant invalid_event at sequence 7 event error: cause" {
		t.Fatalf("invariant error = %q", got)
	}
	if !errors.Is(violation, cause) || violation.Unwrap() != cause {
		t.Fatal("invariant cause is not unwrap-compatible")
	}

	var machine *StreamStateMachine
	if machine.ResponseState() != ResponseStreamStateInit || machine.TerminalKind() != "" {
		t.Fatal("nil machine getters returned non-zero state")
	}
	err := machine.Apply(Event{Kind: EventKindResponseStarted})
	assertInvariantCode(t, err, StreamInvariantResponseState)
}

func TestStreamStateMachineResponseAndGenericItemBranches(t *testing.T) {
	t.Parallel()
	machine := NewStreamStateMachine()
	assertInvariantCode(t, machine.Apply(Event{Kind: EventKind("future"), Sequence: 0}), StreamInvariantInvalidEvent)

	machine = NewStreamStateMachine()
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseStarted, Sequence: 0})
	assertInvariantCode(t, machine.Apply(Event{Kind: EventKindResponseStarted, Sequence: 1}), StreamInvariantResponseState)

	machine = NewStreamStateMachine()
	assertInvariantCode(t, machine.Apply(Event{Kind: EventKindResponseInProgress, Sequence: 0}), StreamInvariantResponseState)

	machine = NewStreamStateMachine()
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseStarted, Sequence: 0})
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseInProgress, Sequence: 1})
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseInProgress, Sequence: 2})
	mustApplyEvent(t, machine, Event{Kind: EventKindUsage, Sequence: 3, Usage: &Usage{}})

	inactive := NewStreamStateMachine()
	assertInvariantCode(t, inactive.Apply(messageAddedStateEvent(0, "msg_inactive")), StreamInvariantResponseState)
	assertInvariantCode(t, inactive.Apply(Event{Kind: EventKindUsage, Sequence: 1, Usage: &Usage{}}), StreamInvariantResponseState)

	machine = NewStreamStateMachine()
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseStarted, Sequence: 0})
	added := messageAddedStateEvent(1, "msg_1")
	mustApplyEvent(t, machine, added)
	duplicate := added
	duplicate.Sequence = 2
	assertInvariantCode(t, machine.Apply(duplicate), StreamInvariantItemState)

	for _, kind := range []EventKind{EventKindTextDelta, EventKindReasoningDelta, EventKindRefusalDelta} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			m := NewStreamStateMachine()
			mustApplyEvent(t, m, Event{Kind: EventKindResponseStarted, Sequence: 0})
			mustApplyEvent(t, m, messageAddedStateEvent(1, "msg_"+string(kind)))
			mustApplyEvent(t, m, Event{Kind: kind, Sequence: 2, ItemRef: ItemRef{ItemID: "msg_" + string(kind)}, Delta: Delta{Text: "x"}})
			mustApplyEvent(t, m, Event{Kind: EventKindItemDone, Sequence: 3, ItemRef: ItemRef{ItemID: "msg_" + string(kind)}, Snapshot: completedMessageStateItem("msg_" + string(kind))})
		})
	}

	machine = NewStreamStateMachine()
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseStarted, Sequence: 0})
	mustApplyEvent(t, machine, Event{Kind: EventKindError, Sequence: 1, Error: &ResponseError{Detail: ErrorDetail{Message: "provider"}}})
	mustApplyEvent(t, machine, Event{Kind: EventKindResponseCompleted, Sequence: 2})
	assertInvariantCode(t, machine.Apply(Event{Kind: EventKindUsage, Sequence: 3, Usage: &Usage{}}), StreamInvariantResponseState)
}

func TestStreamStateMachineToolInputBranches(t *testing.T) {
	t.Parallel()
	ref := ItemRef{ItemID: "tool_1"}
	delta := Event{Kind: EventKindToolInputDelta, ItemRef: ref, Delta: Delta{ArgumentsJSON: `{}`}}
	done := Event{Kind: EventKindToolInputDone, ItemRef: ref}

	machine := NewStreamStateMachine()
	assertInvariantCode(t, machine.streamItem(delta, true), StreamInvariantItemState)
	machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindMessage}
	assertInvariantCode(t, machine.streamItem(delta, true), StreamInvariantToolState)
	machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindToolCall, tool: StreamToolStateInputDone}
	assertInvariantCode(t, machine.streamItem(delta, true), StreamInvariantToolState)
	machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindToolCall, tool: StreamToolStateDeclared}
	if err := machine.streamItem(delta, true); err != nil {
		t.Fatalf("first tool delta: %v", err)
	}
	if err := machine.streamItem(delta, true); err != nil {
		t.Fatalf("continued tool delta: %v", err)
	}

	machine = NewStreamStateMachine()
	assertInvariantCode(t, machine.finishToolInput(done), StreamInvariantToolState)
	machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindMessage}
	assertInvariantCode(t, machine.finishToolInput(done), StreamInvariantToolState)
	machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindToolCall, tool: StreamToolStateResultDone}
	assertInvariantCode(t, machine.finishToolInput(done), StreamInvariantToolState)
	for _, initial := range []StreamToolState{StreamToolStateDeclared, StreamToolStateInputStreaming} {
		machine.items["item:tool_1"] = streamItemLifecycle{state: StreamItemStateAdded, kind: ItemKindToolCall, tool: initial}
		if err := machine.finishToolInput(done); err != nil {
			t.Fatalf("finish tool input from %s: %v", initial, err)
		}
	}
}

func TestStreamStateMachineHostedStatusBranches(t *testing.T) {
	t.Parallel()
	ref := ItemRef{ItemID: "hosted_1"}
	event := func(status string) Event {
		return Event{Kind: EventKindHostedStatus, ItemRef: ref, Delta: Delta{HostedStatus: status}}
	}
	applyMachine := NewStreamStateMachine()
	mustApplyEvent(t, applyMachine, Event{Kind: EventKindResponseStarted, Sequence: 0})
	applyMachine.items["item:hosted_1"] = streamItemLifecycle{kind: ItemKindHostedCall, state: StreamItemStateAdded, tool: StreamToolStateDeclared}
	applyEvent := event("queued")
	applyEvent.Sequence = 1
	mustApplyEvent(t, applyMachine, applyEvent)

	machine := NewStreamStateMachine()
	assertInvariantCode(t, machine.updateHostedTool(event("queued")), StreamInvariantToolState)
	machine.items["item:hosted_1"] = streamItemLifecycle{kind: ItemKindMessage, state: StreamItemStateAdded}
	assertInvariantCode(t, machine.updateHostedTool(event("queued")), StreamInvariantToolState)

	for _, status := range []string{"queued", "in_progress", "searching", "generating"} {
		for _, initial := range []StreamToolState{StreamToolStateDeclared, StreamToolStateHostedRunning} {
			machine.items["item:hosted_1"] = streamItemLifecycle{kind: ItemKindHostedCall, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateHostedTool(event(status)); err != nil {
				t.Fatalf("hosted %s from %s: %v", status, initial, err)
			}
		}
	}
	for _, status := range []string{"completed", "failed"} {
		for _, initial := range []StreamToolState{StreamToolStateDeclared, StreamToolStateHostedRunning} {
			machine.items["item:hosted_1"] = streamItemLifecycle{kind: ItemKindHostedCall, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateHostedTool(event(status)); err != nil {
				t.Fatalf("hosted %s from %s: %v", status, initial, err)
			}
		}
	}
	machine.items["item:hosted_1"] = streamItemLifecycle{kind: ItemKindHostedCall, state: StreamItemStateAdded, tool: StreamToolStateInputDone}
	assertInvariantCode(t, machine.updateHostedTool(event("queued")), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateHostedTool(event("completed")), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateHostedTool(event("future")), StreamInvariantToolState)

	addMachine := NewStreamStateMachine()
	addMachine.response = ResponseStreamStateStarted
	if err := addMachine.addItem(Event{ItemRef: ItemRef{ItemID: "hosted_added"}, Snapshot: &Item{Kind: ItemKindHostedCall}}); err != nil {
		t.Fatalf("add hosted item: %v", err)
	}
	if addMachine.items["item:hosted_added"].tool != StreamToolStateDeclared {
		t.Fatalf("hosted item tool state = %s", addMachine.items["item:hosted_added"].tool)
	}
}

func TestStreamStateMachineMCPStatusBranches(t *testing.T) {
	t.Parallel()
	ref := ItemRef{ItemID: "mcp_1"}
	event := func(status MCPCallStatus) Event {
		return Event{Kind: EventKindMCPStatus, ItemRef: ref, Delta: Delta{MCPStatus: string(status)}}
	}
	machine := NewStreamStateMachine()
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusInProgress)), StreamInvariantToolState)
	machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMessage, state: StreamItemStateAdded}
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusInProgress)), StreamInvariantToolState)

	for _, status := range []MCPCallStatus{MCPCallStatusCalling, MCPCallStatusInProgress} {
		for _, initial := range []StreamToolState{StreamToolStateInputDone, StreamToolStateHostedRunning} {
			machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPCall, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateMCP(event(status)); err != nil {
				t.Fatalf("MCP call %s from %s: %v", status, initial, err)
			}
		}
	}
	for _, status := range []MCPCallStatus{MCPCallStatusCompleted, MCPCallStatusIncomplete, MCPCallStatusFailed} {
		for _, initial := range []StreamToolState{StreamToolStateInputDone, StreamToolStateHostedRunning} {
			machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPCall, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateMCP(event(status)); err != nil {
				t.Fatalf("MCP call %s from %s: %v", status, initial, err)
			}
		}
	}
	machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPCall, state: StreamItemStateAdded, tool: StreamToolStateDeclared}
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusInProgress)), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusCompleted)), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatus("future"))), StreamInvariantToolState)

	for _, status := range []MCPCallStatus{MCPCallStatusInProgress} {
		for _, initial := range []StreamToolState{StreamToolStateDeclared, StreamToolStateHostedRunning} {
			machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPListTools, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateMCP(event(status)); err != nil {
				t.Fatalf("MCP list %s from %s: %v", status, initial, err)
			}
		}
	}
	for _, status := range []MCPCallStatus{MCPCallStatusCompleted, MCPCallStatusIncomplete, MCPCallStatusFailed} {
		for _, initial := range []StreamToolState{StreamToolStateDeclared, StreamToolStateHostedRunning} {
			machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPListTools, state: StreamItemStateAdded, tool: initial}
			if err := machine.updateMCP(event(status)); err != nil {
				t.Fatalf("MCP list %s from %s: %v", status, initial, err)
			}
		}
	}
	machine.items["item:mcp_1"] = streamItemLifecycle{kind: ItemKindMCPListTools, state: StreamItemStateAdded, tool: StreamToolStateInputDone}
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusInProgress)), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatusCompleted)), StreamInvariantToolState)
	assertInvariantCode(t, machine.updateMCP(event(MCPCallStatus("future"))), StreamInvariantToolState)
}

func TestStreamStateMachineFinishItemAndTerminalBranches(t *testing.T) {
	t.Parallel()
	ref := ItemRef{ItemID: "item_1"}
	done := Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: completedMessageStateItem("item_1")}
	machine := NewStreamStateMachine()
	assertInvariantCode(t, machine.finishItem(done), StreamInvariantItemState)
	machine.items["item:item_1"] = streamItemLifecycle{kind: ItemKindMessage, state: StreamItemStateDone}
	assertInvariantCode(t, machine.finishItem(done), StreamInvariantItemState)
	machine.items["item:item_1"] = streamItemLifecycle{kind: ItemKindToolCall, state: StreamItemStateAdded, tool: StreamToolStateDeclared}
	assertInvariantCode(t, machine.finishItem(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: &Item{Kind: ItemKindMessage}}), StreamInvariantItemState)
	machine.items["item:item_1"] = streamItemLifecycle{kind: ItemKindToolCall, state: StreamItemStateAdded, tool: StreamToolStateDeclared}
	assertInvariantCode(t, machine.finishItem(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: &Item{Kind: ItemKindToolCall}}), StreamInvariantToolState)
	machine.items["item:item_1"] = streamItemLifecycle{kind: ItemKindHostedCall, state: StreamItemStateAdded, tool: StreamToolStateHostedRunning}
	assertInvariantCode(t, machine.finishItem(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: &Item{Kind: ItemKindHostedCall}}), StreamInvariantToolState)
	for _, kind := range []ItemKind{ItemKindMCPCall, ItemKindMCPListTools} {
		machine.items["item:item_1"] = streamItemLifecycle{kind: kind, state: StreamItemStateAdded, tool: StreamToolStateHostedRunning}
		assertInvariantCode(t, machine.finishItem(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: &Item{Kind: kind}}), StreamInvariantToolState)
	}

	for _, lifecycle := range []streamItemLifecycle{
		{kind: ItemKindMessage, state: StreamItemStateAdded},
		{kind: ItemKindMessage, state: StreamItemStateStreaming},
		{kind: ItemKindToolCall, state: StreamItemStateStreaming, tool: StreamToolStateInputDone},
		{kind: ItemKindHostedCall, state: StreamItemStateStreaming, tool: StreamToolStateResultDone},
		{kind: ItemKindMCPCall, state: StreamItemStateStreaming, tool: StreamToolStateResultDone},
		{kind: ItemKindMCPListTools, state: StreamItemStateStreaming, tool: StreamToolStateResultDone},
	} {
		machine.items["item:item_1"] = lifecycle
		snapshot := &Item{Kind: lifecycle.kind}
		if err := machine.finishItem(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: snapshot}); err != nil {
			t.Fatalf("finish %s/%s/%s: %v", lifecycle.kind, lifecycle.state, lifecycle.tool, err)
		}
	}

	machine = NewStreamStateMachine()
	assertInvariantCode(t, machine.finishResponse(Event{Kind: EventKindResponseCompleted}), StreamInvariantResponseState)
	machine.response = ResponseStreamStateStarted
	machine.items["item:item_1"] = streamItemLifecycle{kind: ItemKindMessage, state: StreamItemStateAdded}
	assertInvariantCode(t, machine.finishResponse(Event{Kind: EventKindResponseCompleted}), StreamInvariantOpenItems)

	for _, terminal := range []EventKind{EventKindResponseCompleted, EventKindResponseFailed, EventKindResponseIncomplete, EventKindResponseCancelled} {
		m := NewStreamStateMachine()
		m.response = ResponseStreamStateInProgress
		event := Event{Kind: terminal}
		if terminal == EventKindResponseFailed {
			event.Error = &ResponseError{Detail: ErrorDetail{Message: "failed"}}
		}
		if err := m.finishResponse(event); err != nil {
			t.Fatalf("finish response %s: %v", terminal, err)
		}
		if m.response != ResponseStreamStateTerminal || m.terminal != terminal {
			t.Fatalf("terminal state = %s/%s, want %s", m.response, m.terminal, terminal)
		}
	}
}

func messageAddedStateEvent(sequence uint64, id string) Event {
	outputIndex := 0
	return Event{
		Kind: EventKindItemAdded, Sequence: sequence,
		ItemRef: ItemRef{ItemID: id, OutputIndex: &outputIndex},
		Snapshot: &Item{
			Kind: ItemKindMessage, ID: id, Role: RoleAssistant, Status: ItemStatusInProgress,
			Content: []ContentBlock{{Kind: ContentKindText, Text: "x"}},
		},
	}
}

func completedMessageStateItem(id string) *Item {
	return &Item{
		Kind: ItemKindMessage, ID: id, Role: RoleAssistant, Status: ItemStatusCompleted,
		Content: []ContentBlock{{Kind: ContentKindText, Text: "x"}},
	}
}

func mustApplyEvent(t *testing.T, machine *StreamStateMachine, event Event) {
	t.Helper()
	if err := machine.Apply(event); err != nil {
		t.Fatalf("apply %s: %v", event.Kind, err)
	}
}

func assertInvariantCode(t *testing.T, err error, code StreamInvariantCode) {
	t.Helper()
	var violation *StreamInvariantError
	if !errors.As(err, &violation) || violation.Code != code {
		t.Fatalf("invariant error = %T %v, want %s", err, err, code)
	}
}

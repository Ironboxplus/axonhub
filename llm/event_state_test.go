package llm

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
)

func TestStreamStateMachineParallelToolLifecycle(t *testing.T) {
	machine := NewStreamStateMachine()
	events := parallelToolLifecycleEvents()

	for _, event := range events {
		require.NoError(t, machine.Apply(event), "event %s sequence %d", event.Kind, event.Sequence)
	}
	require.Equal(t, ResponseStreamStateTerminal, machine.ResponseState())
	require.Equal(t, EventKindResponseCompleted, machine.TerminalKind())
}

func BenchmarkStreamStateMachineParallelToolLifecycle(b *testing.B) {
	events := parallelToolLifecycleEvents()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		machine := NewStreamStateMachine()
		for _, event := range events {
			if err := machine.Apply(event); err != nil {
				b.Fatalf("apply %s: %v", event.Kind, err)
			}
		}
	}
}

func parallelToolLifecycleEvents() []Event {
	return []Event{
		{Kind: EventKindResponseStarted, Sequence: 0},
		{Kind: EventKindResponseInProgress, Sequence: 1},
		toolAddedEvent(2, 0, "item-a", "call-a", "lookup"),
		toolAddedEvent(3, 1, "item-b", "call-b", "calculate"),
		{Kind: EventKindToolInputDelta, Sequence: 4, ItemRef: toolRef(0, "item-a", "call-a"), Delta: Delta{ArgumentsJSON: `{"q":"hel`}},
		{Kind: EventKindToolInputDelta, Sequence: 5, ItemRef: toolRef(1, "item-b", "call-b"), Delta: Delta{ArgumentsJSON: `{"x":1`}},
		{Kind: EventKindToolInputDelta, Sequence: 6, ItemRef: toolRef(0, "item-a", "call-a"), Delta: Delta{ArgumentsJSON: `lo"}`}},
		{Kind: EventKindToolInputDone, Sequence: 7, ItemRef: toolRef(0, "item-a", "call-a")},
		{Kind: EventKindToolInputDelta, Sequence: 8, ItemRef: toolRef(1, "item-b", "call-b"), Delta: Delta{ArgumentsJSON: `}`}},
		{Kind: EventKindToolInputDone, Sequence: 9, ItemRef: toolRef(1, "item-b", "call-b")},
		toolDoneEvent(10, 0, "item-a", "call-a", "lookup", `{"q":"hello"}`),
		toolDoneEvent(11, 1, "item-b", "call-b", "calculate", `{"x":1}`),
		{Kind: EventKindUsage, Sequence: 12, Usage: &Usage{PromptTokens: 3, CompletionTokens: 5}},
		{Kind: EventKindResponseCompleted, Sequence: 13},
	}
}

func TestStreamStateMachineRejectsLifecycleViolations(t *testing.T) {
	tests := []struct {
		name   string
		events []Event
		code   StreamInvariantCode
	}{
		{
			name: "delta before item added",
			events: []Event{
				{Kind: EventKindResponseStarted, Sequence: 0},
				{Kind: EventKindTextDelta, Sequence: 1, ItemRef: ItemRef{ChoiceIndex: lo.ToPtr(0)}, Delta: Delta{Text: "x"}},
			},
			code: StreamInvariantItemState,
		},
		{
			name: "terminal with open tool",
			events: []Event{
				{Kind: EventKindResponseStarted, Sequence: 0},
				toolAddedEvent(1, 0, "item-a", "call-a", "lookup"),
				{Kind: EventKindResponseCompleted, Sequence: 2},
			},
			code: StreamInvariantOpenItems,
		},
		{
			name: "duplicate sequence",
			events: []Event{
				{Kind: EventKindResponseStarted, Sequence: 7},
				{Kind: EventKindResponseInProgress, Sequence: 7},
			},
			code: StreamInvariantSequence,
		},
		{
			name: "event after terminal",
			events: []Event{
				{Kind: EventKindResponseStarted, Sequence: 0},
				{Kind: EventKindResponseCompleted, Sequence: 1},
				{Kind: EventKindUsage, Sequence: 2, Usage: &Usage{}},
			},
			code: StreamInvariantResponseState,
		},
		{
			name: "different items reuse one output index",
			events: []Event{
				{Kind: EventKindResponseStarted, Sequence: 0},
				toolAddedEvent(1, 0, "item-a", "call-a", "lookup"),
				toolAddedEvent(2, 0, "item-b", "call-b", "calculate"),
			},
			code: StreamInvariantOutputIndex,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			machine := NewStreamStateMachine()
			var got error
			for _, event := range tt.events {
				got = machine.Apply(event)
				if got != nil {
					break
				}
			}
			require.Error(t, got)
			var violation *StreamInvariantError
			require.True(t, errors.As(got, &violation))
			require.Equal(t, tt.code, violation.Code)
		})
	}
}

func TestStreamStateMachineMCPCallRequiresArgumentsThenTerminalStatus(t *testing.T) {
	t.Parallel()
	ref := ItemRef{ItemID: "mcp_call_1", OutputIndex: lo.ToPtr(0)}
	added := Event{
		Kind: EventKindItemAdded, Sequence: 1, ItemRef: ref,
		Snapshot: &Item{
			Kind: ItemKindMCPCall, ID: "mcp_call_1", Status: ItemStatusInProgress,
			MCPCall: &MCPCall{ServerLabel: "inventory", LogicalName: "lookup", Status: MCPCallStatusInProgress},
		},
	}

	machine := NewStreamStateMachine()
	require.NoError(t, machine.Apply(Event{Kind: EventKindResponseStarted, Sequence: 0}))
	require.NoError(t, machine.Apply(added))
	err := machine.Apply(Event{
		Kind: EventKindMCPStatus, Sequence: 2, ItemRef: ref,
		Delta: Delta{MCPStatus: string(MCPCallStatusInProgress)},
	})
	require.Error(t, err)
	var violation *StreamInvariantError
	require.True(t, errors.As(err, &violation))
	require.Equal(t, StreamInvariantToolState, violation.Code)

	machine = NewStreamStateMachine()
	events := []Event{
		{Kind: EventKindResponseStarted, Sequence: 0},
		added,
		{Kind: EventKindToolInputDelta, Sequence: 2, ItemRef: ref, Delta: Delta{ArgumentsJSON: `{"sku":"A-1"}`}},
		{Kind: EventKindToolInputDone, Sequence: 3, ItemRef: ref},
		{Kind: EventKindMCPStatus, Sequence: 4, ItemRef: ref, Delta: Delta{MCPStatus: string(MCPCallStatusInProgress)}},
		{Kind: EventKindMCPStatus, Sequence: 5, ItemRef: ref, Delta: Delta{MCPStatus: string(MCPCallStatusCompleted)}},
		{
			Kind: EventKindItemDone, Sequence: 6, ItemRef: ref,
			Snapshot: &Item{
				Kind: ItemKindMCPCall, ID: "mcp_call_1", Status: ItemStatusCompleted,
				MCPCall: &MCPCall{
					ServerLabel: "inventory", LogicalName: "lookup",
					ArgumentsJSON: json.RawMessage(`{"sku":"A-1"}`), Output: "7", Status: MCPCallStatusCompleted,
				},
			},
		},
		{Kind: EventKindResponseCompleted, Sequence: 7},
	}
	for _, event := range events {
		require.NoError(t, machine.Apply(event), "event %s", event.Kind)
	}
}

func toolRef(index int, itemID, callID string) ItemRef {
	return ItemRef{ChoiceIndex: lo.ToPtr(0), OutputIndex: lo.ToPtr(index), ToolIndex: lo.ToPtr(index), ItemID: itemID, CallID: callID}
}

func toolAddedEvent(sequence uint64, index int, itemID, callID, name string) Event {
	return Event{
		Kind: EventKindItemAdded, Sequence: sequence, ItemRef: toolRef(index, itemID, callID),
		Snapshot: &Item{Kind: ItemKindToolCall, ID: itemID, Status: ItemStatusInProgress, ToolCall: &ToolInvocation{
			Kind: ToolKindFunction, ID: itemID, CallID: callID, LogicalName: name, Status: ToolCallStatusInProgress, Execution: ExecutionOwnerClient,
		}},
	}
}

func toolDoneEvent(sequence uint64, index int, itemID, callID, name, arguments string) Event {
	return Event{
		Kind: EventKindItemDone, Sequence: sequence, ItemRef: toolRef(index, itemID, callID),
		Snapshot: &Item{Kind: ItemKindToolCall, ID: itemID, Status: ItemStatusCompleted, ToolCall: &ToolInvocation{
			Kind: ToolKindFunction, ID: itemID, CallID: callID, LogicalName: name, ArgumentsJSON: []byte(arguments), Status: ToolCallStatusCompleted, Execution: ExecutionOwnerClient,
		}},
	}
}

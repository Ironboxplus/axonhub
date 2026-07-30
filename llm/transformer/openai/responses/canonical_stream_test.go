package responses

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
)

func TestCanonicalStreamEncoderDefersNativeClientItemsWithoutDeltaEvents(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name     string
		kind     llm.ToolKind
		itemType string
		args     json.RawMessage
	}{
		{name: "computer", kind: llm.ToolKindComputer, itemType: "computer_call", args: json.RawMessage(`{"type":"click","x":12,"y":34,"button":"left"}`)},
		{name: "apply_patch", kind: llm.ToolKindApplyPatch, itemType: "apply_patch_call", args: json.RawMessage(`{"type":"update_file","path":"main.go","diff":"@@"}`)},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			encoder := newCanonicalStreamEncoder()
			source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
			outputIndex := 0
			ref := llm.ItemRef{ItemID: test.name + "_1", CallID: test.name + "_call_1", OutputIndex: &outputIndex}
			events := []llm.Event{
				{Kind: llm.EventKindResponseStarted, Sequence: 0},
				{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref, Snapshot: &llm.Item{
					Kind: llm.ItemKindToolCall, ID: ref.ItemID, Status: llm.ItemStatusInProgress,
					ToolCall: &llm.ToolInvocation{Kind: test.kind, ID: ref.ItemID, CallID: ref.CallID, LogicalName: test.name, Execution: llm.ExecutionOwnerClient},
				}},
				{Kind: llm.EventKindToolInputDelta, Sequence: 2, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: string(test.args)}},
				{Kind: llm.EventKindToolInputDone, Sequence: 3, ItemRef: ref},
				{Kind: llm.EventKindItemDone, Sequence: 4, ItemRef: ref, Snapshot: &llm.Item{
					Kind: llm.ItemKindToolCall, ID: ref.ItemID, Status: llm.ItemStatusCompleted,
					ToolCall: &llm.ToolInvocation{Kind: test.kind, ID: ref.ItemID, CallID: ref.CallID, LogicalName: test.name, ArgumentsJSON: test.args, Execution: llm.ExecutionOwnerClient},
				}},
				{Kind: llm.EventKindResponseCompleted, Sequence: 5},
			}
			for _, event := range events {
				require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
			}
			want := []StreamEventType{StreamEventTypeResponseCreated, StreamEventTypeOutputItemAdded, StreamEventTypeOutputItemDone, StreamEventTypeResponseCompleted}
			require.Len(t, source.eventQueue, len(want))
			for index, queued := range source.eventQueue {
				var wire StreamEvent
				require.NoError(t, json.Unmarshal(queued.Data, &wire))
				require.Equal(t, want[index], wire.Type)
				if wire.Item != nil {
					require.Equal(t, test.itemType, wire.Item.Type)
				}
			}
		})
	}
}

func TestCanonicalStreamEncoderEmitsHostedShellResultAsAdjacentOutputItem(t *testing.T) {
	t.Parallel()
	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	outputIndex := 0
	ref := llm.ItemRef{ItemID: "shell_1", CallID: "shell_call_1", OutputIndex: &outputIndex}
	call := llm.ToolInvocation{
		Kind: llm.ToolKindShell, ID: ref.ItemID, CallID: ref.CallID, LogicalName: "shell", Execution: llm.ExecutionOwnerProvider,
		ArgumentsJSON: json.RawMessage(`{"commands":["go test ./conversion"],"timeout_ms":30000}`),
	}
	result := &llm.ToolResult{
		Kind: llm.ToolKindShell, CallID: ref.CallID, LogicalName: "shell", Execution: llm.ExecutionOwnerProvider,
		Status:  llm.ToolResultStatusCompleted,
		Content: []llm.ContentBlock{{Kind: llm.ContentKindUnknown, UnknownRaw: json.RawMessage(`[{"stdout":"ok\\n","stderr":"","outcome":{"type":"exit","exit_code":0}}]`)}},
	}
	events := []llm.Event{
		{Kind: llm.EventKindResponseStarted, Sequence: 0},
		{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref, Snapshot: &llm.Item{Kind: llm.ItemKindHostedCall, ID: ref.ItemID, Status: llm.ItemStatusInProgress, HostedCall: &llm.HostedToolCall{Invocation: call}}},
		{Kind: llm.EventKindHostedStatus, Sequence: 2, ItemRef: ref, Delta: llm.Delta{HostedStatus: "completed"}},
		{Kind: llm.EventKindItemDone, Sequence: 3, ItemRef: ref, Snapshot: &llm.Item{Kind: llm.ItemKindHostedCall, ID: ref.ItemID, Status: llm.ItemStatusCompleted, HostedCall: &llm.HostedToolCall{Invocation: call, Result: result}}},
		{Kind: llm.EventKindResponseCompleted, Sequence: 4},
	}
	for _, event := range events {
		require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
	}
	want := []StreamEventType{
		StreamEventTypeResponseCreated, StreamEventTypeOutputItemAdded, StreamEventTypeOutputItemDone,
		StreamEventTypeOutputItemAdded, StreamEventTypeOutputItemDone, StreamEventTypeResponseCompleted,
	}
	require.Len(t, source.eventQueue, len(want))
	for index, queued := range source.eventQueue {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal(queued.Data, &wire))
		require.Equal(t, want[index], wire.Type)
		if index == 1 || index == 2 {
			require.Equal(t, "shell_call", wire.Item.Type)
		}
		if index == 3 || index == 4 {
			require.Equal(t, "shell_call_output", wire.Item.Type)
			require.Equal(t, 1, wire.OutputIndex)
		}
	}
}

func TestCanonicalStreamEncoderEmitsHostedComputerLifecycleAsAdjacentOutputItem(t *testing.T) {
	t.Parallel()
	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	outputIndex := 0
	ref := llm.ItemRef{ItemID: "computer_1", CallID: "computer_call_1", OutputIndex: &outputIndex}
	check := llm.ToolSafetyCheck{ID: "safe_1", Code: "external_side_effect", Message: "confirm"}
	call := llm.ToolInvocation{
		Kind: llm.ToolKindComputer, ID: ref.ItemID, CallID: ref.CallID, LogicalName: "computer", Execution: llm.ExecutionOwnerProvider,
		ArgumentsJSON: json.RawMessage(`{"type":"click","x":12,"y":34,"button":"left"}`), PendingSafetyChecks: []llm.ToolSafetyCheck{check},
	}
	result := &llm.ToolResult{
		Kind: llm.ToolKindComputer, CallID: ref.CallID, LogicalName: "computer", Execution: llm.ExecutionOwnerGateway,
		Status:                   llm.ToolResultStatusCompleted,
		StructuredContent:        json.RawMessage(`{"type":"computer_screenshot","image_url":"data:image/png;base64,Q09NUFVURVI="}`),
		Content:                  []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,Q09NUFVURVI="}}},
		AcknowledgedSafetyChecks: []llm.ToolSafetyCheck{check},
	}
	events := []llm.Event{
		{Kind: llm.EventKindResponseStarted, Sequence: 0},
		{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref, Snapshot: &llm.Item{Kind: llm.ItemKindHostedCall, ID: ref.ItemID, Status: llm.ItemStatusInProgress, HostedCall: &llm.HostedToolCall{Invocation: call}}},
		{Kind: llm.EventKindHostedStatus, Sequence: 2, ItemRef: ref, Delta: llm.Delta{HostedStatus: "completed"}},
		{Kind: llm.EventKindItemDone, Sequence: 3, ItemRef: ref, Snapshot: &llm.Item{Kind: llm.ItemKindHostedCall, ID: ref.ItemID, Status: llm.ItemStatusCompleted, HostedCall: &llm.HostedToolCall{Invocation: call, Result: result}}},
		{Kind: llm.EventKindResponseCompleted, Sequence: 4},
	}
	for _, event := range events {
		require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
	}
	want := []StreamEventType{
		StreamEventTypeResponseCreated, StreamEventTypeOutputItemAdded, StreamEventTypeOutputItemDone,
		StreamEventTypeOutputItemAdded, StreamEventTypeOutputItemDone, StreamEventTypeResponseCompleted,
	}
	require.Len(t, source.eventQueue, len(want))
	for index, queued := range source.eventQueue {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal(queued.Data, &wire))
		require.Equal(t, want[index], wire.Type)
		if index == 1 || index == 2 {
			require.Equal(t, "computer_call", wire.Item.Type)
			require.Len(t, wire.Item.PendingSafetyChecks, 1)
		}
		if index == 3 || index == 4 {
			require.Equal(t, "computer_call_output", wire.Item.Type)
			require.Equal(t, 1, wire.OutputIndex)
			require.NotNil(t, wire.Item.ComputerOutput)
			require.Equal(t, "data:image/png;base64,Q09NUFVURVI=", wire.Item.ComputerOutput.ImageURL)
			require.Len(t, wire.Item.AcknowledgedSafetyChecks, 1)
		}
	}
}

func TestCanonicalStreamDecoderRealResponsesParallelTools(t *testing.T) {
	file, err := os.Open("testdata/tool-2.stream.jsonl")
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, file.Close()) })

	decoder := newCanonicalStreamDecoder()
	var events []llm.Event
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var transport struct {
			Type string
			Data string
		}
		require.NoError(t, json.Unmarshal(scanner.Bytes(), &transport))
		if transport.Data == "[DONE]" {
			continue
		}
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(transport.Data), &wire))
		decoded, decodeErr := decoder.decode(&wire)
		require.NoError(t, decodeErr, "wire event %s", wire.Type)
		events = append(events, decoded...)
	}
	require.NoError(t, scanner.Err())
	require.NotEmpty(t, events)
	require.Equal(t, llm.EventKindResponseStarted, events[0].Kind)
	require.Equal(t, llm.EventKindResponseCompleted, events[len(events)-1].Kind)

	added := make(map[string]llm.Event)
	done := make(map[string]llm.Event)
	argumentDeltas := make(map[string]string)
	for _, event := range events {
		switch event.Kind {
		case llm.EventKindItemAdded:
			if event.Snapshot.Kind == llm.ItemKindToolCall {
				added[event.ItemRef.CallID] = event
			}
		case llm.EventKindToolInputDelta:
			argumentDeltas[event.ItemRef.CallID] += event.Delta.ArgumentsJSON
		case llm.EventKindItemDone:
			if event.Snapshot.Kind == llm.ItemKindToolCall {
				done[event.ItemRef.CallID] = event
			}
		}
	}
	require.GreaterOrEqual(t, len(added), 2)
	require.Equal(t, len(added), len(done))
	for callID, addedEvent := range added {
		doneEvent, ok := done[callID]
		require.True(t, ok, "missing done event for %s", callID)
		require.NotEmpty(t, addedEvent.ItemRef.ItemID)
		require.Equal(t, addedEvent.ItemRef.ItemID, doneEvent.ItemRef.ItemID)
		require.NotNil(t, addedEvent.ItemRef.OutputIndex)
		require.Equal(t, addedEvent.ItemRef.OutputIndex, doneEvent.ItemRef.OutputIndex)
		require.JSONEq(t, string(doneEvent.Snapshot.ToolCall.ArgumentsJSON), argumentDeltas[callID])
	}
}

func TestCanonicalStreamDecoderRejectsUnknownAndOutOfOrderEvents(t *testing.T) {
	decoder := newCanonicalStreamDecoder()
	_, err := decoder.decode(&StreamEvent{Type: "response.future_behavior"})
	require.ErrorContains(t, err, "unsupported Responses stream event type")

	decoder = newCanonicalStreamDecoder()
	_, err = decoder.decode(&StreamEvent{
		Type: StreamEventTypeOutputTextDelta, ItemID: lo.ToPtr("missing"), Delta: "x",
	})
	require.ErrorContains(t, err, "response is init")

	decoder = newCanonicalStreamDecoder()
	_, err = decoder.decode(&StreamEvent{Type: StreamEventTypeResponseCreated, SequenceNumber: lo.ToPtr(5)})
	require.NoError(t, err)
	_, err = decoder.decode(&StreamEvent{Type: StreamEventTypeResponseInProgress, SequenceNumber: lo.ToPtr(5)})
	require.ErrorContains(t, err, "sequence_number 5 is not after 5")

	decoder = newCanonicalStreamDecoder()
	_, err = decoder.decode(&StreamEvent{Type: StreamEventTypeResponseCreated, SequenceNumber: lo.ToPtr(9)})
	require.NoError(t, err)
	_, err = decoder.decode(&StreamEvent{Type: StreamEventTypeResponseInProgress, SequenceNumber: lo.ToPtr(8)})
	require.ErrorContains(t, err, "sequence_number 8 is not after 9")
}

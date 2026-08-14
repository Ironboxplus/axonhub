package responses

import (
	"bufio"
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
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

func TestCanonicalResponsesStreamPreservesFutureFieldsOnKnownOutputObjects(t *testing.T) {
	t.Parallel()

	var wire StreamEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"response.output_item.added",
		"output_index":0,
		"future_event":{"checkpoint":"added"},
		"item":{
			"id":"msg_future_1",
			"type":"message",
			"role":"assistant",
			"status":"in_progress",
			"future_item":{"routing":"blue"},
			"content":[{
				"type":"output_text",
				"text":"hello",
				"annotations":[],
				"future_content":{"confidence":0.75}
			}]
		}
	}`), &wire))

	decoder := newCanonicalStreamDecoder()
	events, err := decoder.decode(&StreamEvent{Type: StreamEventTypeResponseCreated})
	require.NoError(t, err)
	itemEvents, err := decoder.decode(&wire)
	require.NoError(t, err)
	events = append(events, itemEvents...)
	var doneWire StreamEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"response.output_item.done",
		"output_index":0,
		"future_event":{"checkpoint":"done"},
		"item":{
			"id":"msg_future_1",
			"type":"message",
			"role":"assistant",
			"status":"completed",
			"future_item":{"routing":"blue"},
			"content":[{
				"type":"output_text",
				"text":"hello",
				"annotations":[],
				"future_content":{"confidence":0.75}
			}]
		}
	}`), &doneWire))
	doneEvents, err := decoder.decode(&doneWire)
	require.NoError(t, err)
	events = append(events, doneEvents...)
	require.Len(t, events, 3)
	added := events[1]
	require.Equal(t, llm.EventKindItemAdded, added.Kind)
	require.JSONEq(t, `{"future_item":{"routing":"blue"}}`, string(added.Snapshot.ProtocolHints.SourceResidual))
	require.Len(t, added.Snapshot.Content, 1)
	require.JSONEq(t, `{"future_content":{"confidence":0.75}}`, string(added.Snapshot.Content[0].SourceResidual))

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	for _, event := range events {
		require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
	}

	var encoded map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(source.eventQueue[len(source.eventQueue)-1].Data, &encoded))
	require.JSONEq(t, `{"checkpoint":"done"}`, string(encoded["future_event"]))
	var item map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded["item"], &item))
	require.JSONEq(t, `{"routing":"blue"}`, string(item["future_item"]))
	var content []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(item["content"], &content))
	require.Len(t, content, 1)
	require.JSONEq(t, `{"confidence":0.75}`, string(content[0]["future_content"]))
}

func TestCanonicalResponsesStreamKeepsStatuslessOpaqueSnapshotsUntouched(t *testing.T) {
	t.Parallel()

	fixtures := []string{
		`{"id":"am_statusless","type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"handoff"}]}`,
		`{"id":"ctx_statusless","type":"context_compaction","encrypted_content":"opaque"}`,
		`{"id":"future_statusless","type":"future_behavior","content":null,"tools":[]}`,
	}
	for _, rawItem := range fixtures {
		rawItem := rawItem
		t.Run(rawItem, func(t *testing.T) {
			t.Parallel()
			decoder := newCanonicalStreamDecoder()
			created, err := decoder.decode(&StreamEvent{Type: StreamEventTypeResponseCreated})
			require.NoError(t, err)
			var addedWire StreamEvent
			require.NoError(t, json.Unmarshal([]byte(`{"type":"response.output_item.added","output_index":0,"item":`+rawItem+`}`), &addedWire))
			added, err := decoder.decode(&addedWire)
			require.NoError(t, err)
			require.Len(t, added, 1)

			encoder := newCanonicalStreamEncoder()
			source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
			require.NoError(t, encoder.encode(source, created[0]))
			require.NoError(t, encoder.encode(source, added[0]))
			require.Len(t, source.eventQueue, 2)
			var encoded struct {
				Item json.RawMessage `json:"item"`
			}
			require.NoError(t, json.Unmarshal(source.eventQueue[1].Data, &encoded))
			require.JSONEq(t, rawItem, string(encoded.Item))
		})
	}
}

func TestCanonicalResponsesStreamStatuslessMessageAddedKeepsIdentityAndDoneOwnsFinalStatus(t *testing.T) {
	t.Parallel()
	// Codex's current ev_message_item_added fixture omits status entirely. Its
	// response_item.done snapshot is the authoritative terminal update. Required
	// empty output_text annotations are a wire repair, not a lifecycle mutation.
	const addedItem = `{"id":"msg_statusless","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}`
	const repairedAddedItem = `{"id":"msg_statusless","type":"message","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}`
	const doneItem = `{"id":"msg_statusless","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello"}]}`
	decoder := newCanonicalStreamDecoder()
	var canonicalEvents []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_statusless","object":"response","model":"fixture","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":` + addedItem + `}`,
		`{"type":"response.output_item.done","output_index":0,"item":` + doneItem + `}`,
		`{"type":"response.completed","response":{"id":"resp_statusless","object":"response","model":"fixture","status":"completed","output":[]}}`,
	} {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(raw), &wire))
		events, err := decoder.decode(&wire)
		require.NoError(t, err)
		canonicalEvents = append(canonicalEvents, events...)
	}

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	for index := range canonicalEvents {
		require.NoError(t, encoder.encode(source, canonicalEvents[index]))
	}
	var added, completed struct {
		Item   json.RawMessage `json:"item"`
		Output []Item          `json:"response"`
	}
	for _, event := range source.eventQueue {
		var envelope struct {
			Type     StreamEventType `json:"type"`
			Item     json.RawMessage `json:"item"`
			Response *Response       `json:"response"`
		}
		require.NoError(t, json.Unmarshal(event.Data, &envelope))
		switch envelope.Type {
		case StreamEventTypeOutputItemAdded:
			added.Item = envelope.Item
		case StreamEventTypeResponseCompleted:
			if envelope.Response != nil {
				completed.Output = envelope.Response.Output
			}
		}
	}
	require.JSONEq(t, repairedAddedItem, string(added.Item))
	require.Len(t, completed.Output, 1)
	require.NotNil(t, completed.Output[0].Status)
	require.Equal(t, "completed", *completed.Output[0].Status)
}

func TestResponsesItemHasLifecycleStatusIsClosed(t *testing.T) {
	t.Parallel()
	for _, itemType := range []string{"message", "function_call", "mcp_call"} {
		if !responsesItemHasLifecycleStatus(itemType) {
			t.Fatalf("%s unexpectedly has no lifecycle status", itemType)
		}
	}
	for _, itemType := range []string{"agent_message", "compaction", "context_compaction", "future_behavior"} {
		if responsesItemHasLifecycleStatus(itemType) {
			t.Fatalf("%s unexpectedly has lifecycle status", itemType)
		}
	}
}

func TestCanonicalResponsesStreamAttachesEventResidualOnlyToMatchingWireEvent(t *testing.T) {
	t.Parallel()
	decoder := newCanonicalStreamDecoder()
	var events []llm.Event
	for _, raw := range []string{
		`{"type":"response.created"}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello","future_delta":{"trace":7}}`,
	} {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(raw), &wire))
		decoded, err := decoder.decode(&wire)
		require.NoError(t, err)
		events = append(events, decoded...)
	}

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	for _, event := range events {
		require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
	}

	foundDelta := false
	for _, queued := range source.eventQueue {
		var envelope map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(queued.Data, &envelope))
		var eventType StreamEventType
		require.NoError(t, json.Unmarshal(envelope["type"], &eventType))
		if eventType == StreamEventTypeOutputTextDelta {
			foundDelta = true
			require.JSONEq(t, `{"trace":7}`, string(envelope["future_delta"]))
			continue
		}
		require.NotContains(t, envelope, "future_delta", "source event residual leaked onto %s", eventType)
	}
	require.True(t, foundDelta)
}

func TestCanonicalResponsesStreamPreservesNestedResponseResidualPerLifecycleEvent(t *testing.T) {
	t.Parallel()

	decoder := newCanonicalStreamDecoder()
	var created StreamEvent
	require.NoError(t, json.Unmarshal([]byte(`{
		"type":"response.created",
		"future_event":{"checkpoint":"created"},
		"response":{
			"id":"resp_stream_nested","object":"response","created_at":1,"model":"fixture-model","status":"in_progress","output":[],
			"conversation":{"id":"conv_stream","future_conversation":{"shard":"sg"}},
			"future_response":{"trace":"keep"}
		}
	}`), &created))
	events, err := decoder.decode(&created)
	require.NoError(t, err)
	require.Len(t, events, 1)

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	require.NoError(t, encoder.encode(source, events[0]))
	require.Len(t, source.eventQueue, 1)

	var encoded map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(source.eventQueue[0].Data, &encoded))
	require.JSONEq(t, `{"checkpoint":"created"}`, string(encoded["future_event"]))
	var response map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(encoded["response"], &response))
	require.JSONEq(t, `{"trace":"keep"}`, string(response["future_response"]))
	var conversation map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(response["conversation"], &conversation))
	require.JSONEq(t, `{"shard":"sg"}`, string(conversation["future_conversation"]))
}

// TestCanonicalResponsesTerminalUsageRoundTrips verifies the wire contract at
// the exact boundary exercised by an SSE relay: JSON SSE frame -> canonical
// lifecycle -> JSON SSE frame. ResponseUsage is optional on a Response, but
// its three token totals are required whenever a usage object is present.
// The latter is enforced by the currently pinned async-openai client types.
func TestCanonicalResponsesTerminalUsageRoundTrips(t *testing.T) {
	t.Parallel()

	for _, test := range terminalUsageCases() {
		test := test
		t.Run(test.name, func(t *testing.T) {
			decoder := newCanonicalStreamDecoder()
			canonicalEvents := decodeTerminalUsageSSE(t, decoder, test.terminalType, test.status, true, test.hasError)

			usageIndex := -1
			terminalIndex := -1
			for index := range canonicalEvents {
				switch canonicalEvents[index].Kind {
				case llm.EventKindUsage:
					usageIndex = index
				case llm.EventKindResponseCompleted, llm.EventKindResponseIncomplete,
					llm.EventKindResponseFailed, llm.EventKindResponseCancelled:
					terminalIndex = index
				}
			}
			require.GreaterOrEqual(t, usageIndex, 0, "terminal usage was lost in canonical lifecycle")
			require.Greater(t, terminalIndex, usageIndex, "usage must precede its terminal response event")

			encoded := encodeCanonicalResponsesEvents(t, canonicalEvents)
			assertTerminalUsageWireContract(t, encoded, test.terminalType, true)
			if test.hasError {
				require.NotNil(t, encoded.Response)
				require.NotNil(t, encoded.Response.Error)
				require.Equal(t, "upstream_failed", encoded.Response.Error.Code)
				require.Equal(t, "provider returned failure", encoded.Response.Error.Message)
			}
		})
	}
}

func TestCanonicalResponsesTerminalUsageDoesNotInventUsage(t *testing.T) {
	t.Parallel()

	for _, test := range terminalUsageCases() {
		test := test
		t.Run(test.name, func(t *testing.T) {
			decoder := newCanonicalStreamDecoder()
			canonicalEvents := decodeTerminalUsageSSE(t, decoder, test.terminalType, test.status, false, test.hasError)
			for index := range canonicalEvents {
				require.NotEqual(t, llm.EventKindUsage, canonicalEvents[index].Kind)
			}

			encoded := encodeCanonicalResponsesEvents(t, canonicalEvents)
			assertTerminalUsageWireContract(t, encoded, test.terminalType, false)
		})
	}
}

// TestResponsesTerminalUsageSurvivesTransformerBridge covers the production
// streaming path rather than the encoder in isolation. In particular,
// response.failed must retain its terminal usage and its semantic error when
// the legacy outbound adapter reports failure to the next layer.
func TestResponsesTerminalUsageSurvivesTransformerBridge(t *testing.T) {
	t.Parallel()

	for _, test := range terminalUsageCases() {
		test := test
		t.Run(test.name, func(t *testing.T) {
			providerEvents := terminalUsageProviderEvents(t, test.terminalType, test.status, test.hasError)
			outbound, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
			require.NoError(t, err)
			canonical, err := outbound.TransformStream(t.Context(), nil, streams.SliceStream(providerEvents))
			require.NoError(t, err)
			client, err := NewInboundTransformer().TransformStream(t.Context(), canonical)
			require.NoError(t, err)

			var terminal *StreamEvent
			for client.Next() {
				var event StreamEvent
				require.NoError(t, json.Unmarshal(client.Current().Data, &event))
				if event.Type == test.terminalType {
					terminal = &event
				}
			}
			require.NoError(t, client.Err())
			assertTerminalUsageWireContract(t, terminal, test.terminalType, true)
			if test.hasError {
				require.NotNil(t, terminal.Response.Error)
				require.Equal(t, "upstream_failed", terminal.Response.Error.Code)
			}
		})
	}
}

func decodeTerminalUsageSSE(
	t *testing.T,
	decoder *canonicalStreamDecoder,
	terminalType StreamEventType,
	status string,
	includeUsage bool,
	includeError bool,
) []llm.Event {
	t.Helper()

	var events []llm.Event
	for _, raw := range terminalUsageSSEFrames(t, terminalType, status, includeUsage, includeError) {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(raw), &wire))
		decoded, err := decoder.decode(&wire)
		require.NoError(t, err, "decode %s", wire.Type)
		events = append(events, decoded...)
	}
	return events
}

func terminalUsageProviderEvents(t *testing.T, terminalType StreamEventType, status string, includeError bool) []*httpclient.StreamEvent {
	t.Helper()

	frames := terminalUsageSSEFrames(t, terminalType, status, true, includeError)
	return []*httpclient.StreamEvent{
		{Type: string(StreamEventTypeResponseCreated), Data: []byte(frames[0])},
		{Type: string(terminalType), Data: []byte(frames[1])},
	}
}

type terminalUsageCase struct {
	name         string
	terminalType StreamEventType
	status       string
	hasError     bool
}

func terminalUsageCases() []terminalUsageCase {
	return []terminalUsageCase{
		{name: "completed", terminalType: StreamEventTypeResponseCompleted, status: "completed"},
		{name: "incomplete", terminalType: StreamEventTypeResponseIncomplete, status: "incomplete"},
		{name: "failed", terminalType: StreamEventTypeResponseFailed, status: "failed", hasError: true},
		{name: "cancelled", terminalType: StreamEventTypeResponseCancelled, status: "cancelled"},
	}
}

func terminalUsageSSEFrames(t *testing.T, terminalType StreamEventType, status string, includeUsage bool, includeError bool) []string {
	t.Helper()

	terminal := `{"type":` + stringMustMarshal(t, string(terminalType)) + `,"sequence_number":2,"response":{"id":"resp_terminal_usage","object":"response","created_at":1,"model":"fixture-model","status":` + stringMustMarshal(t, status) + `,"output":[]`
	if includeUsage {
		terminal += `,"usage":{"input_tokens":17,"output_tokens":5,"total_tokens":22,"future_usage":{"billing":"reserved"}}`
	}
	if includeError {
		terminal += `,"error":{"type":"server_error","code":"upstream_failed","message":"provider returned failure"}`
	}
	terminal += `},"future_terminal":{"checkpoint":"terminal"}}`
	return []string{
		`{"type":"response.created","sequence_number":1,"response":{"id":"resp_terminal_usage","object":"response","created_at":1,"model":"fixture-model","status":"in_progress","output":[]}}`,
		terminal,
	}
}

func encodeCanonicalResponsesEvents(t *testing.T, events []llm.Event) *StreamEvent {
	t.Helper()

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{
		ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator(),
	}
	for index := range events {
		require.NoError(t, encoder.encode(source, events[index]), "encode %s", events[index].Kind)
	}
	require.NotEmpty(t, source.eventQueue)

	var encoded StreamEvent
	require.NoError(t, json.Unmarshal(source.eventQueue[len(source.eventQueue)-1].Data, &encoded))
	return &encoded
}

func assertTerminalUsageWireContract(t *testing.T, event *StreamEvent, terminalType StreamEventType, wantUsage bool) {
	t.Helper()
	require.NotNil(t, event)
	require.Equal(t, terminalType, event.Type)
	require.NotNil(t, event.Response)

	if !wantUsage {
		require.Nil(t, event.Response.Usage)
		return
	}

	require.NotNil(t, event.Response.Usage)
	require.Equal(t, int64(17), event.Response.Usage.InputTokens)
	require.Equal(t, int64(5), event.Response.Usage.OutputTokens)
	require.Equal(t, int64(22), event.Response.Usage.TotalTokens)

	serialized, err := json.Marshal(event)
	require.NoError(t, err)
	var envelope struct {
		Response struct {
			Usage map[string]json.RawMessage `json:"usage"`
		} `json:"response"`
	}
	require.NoError(t, json.Unmarshal(serialized, &envelope))
	require.Contains(t, envelope.Response.Usage, "input_tokens")
	require.Contains(t, envelope.Response.Usage, "output_tokens")
	require.Contains(t, envelope.Response.Usage, "total_tokens")
	require.JSONEq(t, `{"billing":"reserved"}`, string(envelope.Response.Usage["future_usage"]))
}

func stringMustMarshal(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return string(encoded)
}

func TestCanonicalResponsesStreamPreservesFramingEventAndPartResiduals(t *testing.T) {
	t.Parallel()

	decoder := newCanonicalStreamDecoder()
	var canonicalEvents []llm.Event
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_frames","object":"response","created_at":1,"model":"fixture-model","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_frames","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
		`{"type":"response.content_part.added","item_id":"msg_frames","output_index":0,"content_index":0,"future_added_event":{"trace":1},"part":{"type":"output_text","text":"","annotations":[],"future_added_part":{"owner":"part"}}}`,
		`{"type":"response.output_text.delta","item_id":"msg_frames","output_index":0,"content_index":0,"delta":"hello","future_delta_event":{"trace":2}}`,
		`{"type":"response.output_text.done","item_id":"msg_frames","output_index":0,"content_index":0,"text":"hello","future_text_done":{"trace":3}}`,
		`{"type":"response.content_part.done","item_id":"msg_frames","output_index":0,"content_index":0,"future_done_event":{"trace":4},"part":{"type":"output_text","text":"hello","annotations":[],"future_done_part":{"owner":"part"}}}`,
		`{"type":"response.output_item.done","output_index":0,"future_item_done":{"trace":5},"item":{"id":"msg_frames","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
		`{"type":"response.completed","response":{"id":"resp_frames","object":"response","created_at":1,"model":"fixture-model","status":"completed","output":[{"id":"msg_frames","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}]}}`,
	} {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(raw), &wire))
		events, err := decoder.decode(&wire)
		require.NoError(t, err, "decode %s", wire.Type)
		canonicalEvents = append(canonicalEvents, events...)
	}

	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator()}
	for index := range canonicalEvents {
		require.NoError(t, encoder.encode(source, canonicalEvents[index]), "encode canonical event %s", canonicalEvents[index].Kind)
	}

	encodedByType := make(map[StreamEventType]map[string]json.RawMessage)
	for index := range source.eventQueue {
		var envelope map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(source.eventQueue[index].Data, &envelope))
		var eventType StreamEventType
		require.NoError(t, json.Unmarshal(envelope["type"], &eventType))
		encodedByType[eventType] = envelope
	}

	added := encodedByType[StreamEventTypeContentPartAdded]
	require.JSONEq(t, `{"trace":1}`, string(added["future_added_event"]))
	var addedPart map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(added["part"], &addedPart))
	require.JSONEq(t, `{"owner":"part"}`, string(addedPart["future_added_part"]))
	delta := encodedByType[StreamEventTypeOutputTextDelta]
	require.JSONEq(t, `{"trace":2}`, string(delta["future_delta_event"]))
	textDone := encodedByType[StreamEventTypeOutputTextDone]
	require.JSONEq(t, `{"trace":3}`, string(textDone["future_text_done"]))
	done := encodedByType[StreamEventTypeContentPartDone]
	require.JSONEq(t, `{"trace":4}`, string(done["future_done_event"]))
	var donePart map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(done["part"], &donePart))
	require.JSONEq(t, `{"owner":"part"}`, string(donePart["future_done_part"]))
	itemDone := encodedByType[StreamEventTypeOutputItemDone]
	require.JSONEq(t, `{"trace":5}`, string(itemDone["future_item_done"]))
}

func TestCanonicalResponsesStreamBoundsPendingProtocolFrames(t *testing.T) {
	t.Parallel()
	decoder := newCanonicalStreamDecoder()
	_, err := decoder.decode(&StreamEvent{Type: StreamEventTypeResponseCreated})
	require.NoError(t, err)
	itemID := "msg_frame_bound"
	_, err = decoder.decode(&StreamEvent{
		Type: StreamEventTypeOutputItemAdded, OutputIndex: 0,
		Item: &Item{ID: itemID, Type: "message", Role: "assistant", Content: &Input{Items: []Item{}}, Status: lo.ToPtr("in_progress")},
	})
	require.NoError(t, err)
	frame := &StreamEvent{
		Type: StreamEventTypeContentPartDone, ItemID: &itemID, OutputIndex: 0,
		Part: &StreamEventContentPart{Type: "output_text"},
	}
	for index := 0; index < maxPendingProtocolFramesPerItem; index++ {
		_, err = decoder.decode(frame)
		require.NoError(t, err, "frame %d", index)
	}
	_, err = decoder.decode(frame)
	require.ErrorContains(t, err, "exceeded 128 pending events")
}

func TestCanonicalProtocolFrameHelperBranches(t *testing.T) {
	t.Parallel()
	if hasProtocolFrame(nil, StreamEventTypeContentPartAdded) {
		t.Fatal("empty frame list reported a match")
	}
	frames := []llm.ProtocolFrameHint{
		{SourceType: string(StreamEventTypeContentPartAdded), SourceResidual: json.RawMessage(`{"a":1}`)},
		{SourceType: string(StreamEventTypeContentPartDone), SourceResidual: json.RawMessage(`{"b":2}`)},
	}
	if !hasProtocolFrame(frames, StreamEventTypeContentPartDone) || hasProtocolFrame(frames, StreamEventTypeOutputTextDone) {
		t.Fatal("protocol frame lookup returned the wrong result")
	}

	decoder := newCanonicalStreamDecoder()
	if got := decoder.takeProtocolFrames("missing"); got != nil {
		t.Fatalf("missing frames = %#v", got)
	}
	decoder.protocolFrames["item:msg"] = cloneProtocolFrameHints(frames)
	if got := decoder.takeProtocolFrames("item:msg", StreamEventTypeOutputTextDone); got != nil {
		t.Fatalf("unmatched frames = %#v", got)
	}
	selected := decoder.takeProtocolFrames("item:msg", StreamEventTypeContentPartAdded)
	require.Len(t, selected, 1)
	selected[0].SourceResidual[0] = '['
	require.JSONEq(t, `{"a":1}`, string(frames[0].SourceResidual), "selected frame shared residual bytes")
	remaining := decoder.takeProtocolFrames("item:msg")
	require.Len(t, remaining, 1)
	require.Equal(t, string(StreamEventTypeContentPartDone), remaining[0].SourceType)
	if got := decoder.takeProtocolFrames("item:msg"); got != nil {
		t.Fatalf("drained frames = %#v", got)
	}

	unknownItem := "unknown"
	err := decoder.rememberProtocolFrame(&StreamEvent{
		Type: StreamEventTypeContentPartDone, ItemID: &unknownItem,
		Part: &StreamEventContentPart{Type: "output_text"},
	})
	require.ErrorContains(t, err, "unknown item_id")
}

func TestCanonicalStreamItemCloneIsolatesResidualOwnership(t *testing.T) {
	t.Parallel()

	item := &llm.Item{
		ProtocolHints: llm.ProtocolHints{SourceResidual: json.RawMessage(`{"item":1}`)},
		Content: []llm.ContentBlock{{
			Kind: llm.ContentKindCitation, SourceResidual: json.RawMessage(`{"block":1}`),
			Citation: &llm.URLCitation{URL: "https://example.invalid", SourceResidual: json.RawMessage(`{"citation":1}`)},
		}},
		ToolCall:     &llm.ToolInvocation{PendingSafetyChecks: []llm.ToolSafetyCheck{{SourceResidual: json.RawMessage(`{"pending":1}`)}}},
		ToolResult:   &llm.ToolResult{AcknowledgedSafetyChecks: []llm.ToolSafetyCheck{{SourceResidual: json.RawMessage(`{"ack":1}`)}}},
		MCPListTools: &llm.MCPListTools{Tools: []llm.MCPDiscoveredTool{{Name: "lookup", SourceResidual: json.RawMessage(`{"mcp":1}`)}}},
		Reasoning: &llm.ReasoningItem{
			SummaryParts: []llm.ReasoningPart{{Type: "summary_text", SourceResidual: json.RawMessage(`{"summary":1}`)}},
			ContentParts: []llm.ReasoningPart{{Type: "reasoning_text", SourceResidual: json.RawMessage(`{"reasoning":1}`)}},
		},
	}
	clone := cloneCanonicalItem(item)
	require.NotNil(t, clone)

	clone.ProtocolHints.SourceResidual[0] = '['
	clone.Content[0].SourceResidual[0] = '['
	clone.Content[0].Citation.SourceResidual[0] = '['
	clone.ToolCall.PendingSafetyChecks[0].SourceResidual[0] = '['
	clone.ToolResult.AcknowledgedSafetyChecks[0].SourceResidual[0] = '['
	clone.MCPListTools.Tools[0].SourceResidual[0] = '['
	clone.Reasoning.SummaryParts[0].SourceResidual[0] = '['
	clone.Reasoning.ContentParts[0].SourceResidual[0] = '['

	require.JSONEq(t, `{"item":1}`, string(item.ProtocolHints.SourceResidual))
	require.JSONEq(t, `{"block":1}`, string(item.Content[0].SourceResidual))
	require.JSONEq(t, `{"citation":1}`, string(item.Content[0].Citation.SourceResidual))
	require.JSONEq(t, `{"pending":1}`, string(item.ToolCall.PendingSafetyChecks[0].SourceResidual))
	require.JSONEq(t, `{"ack":1}`, string(item.ToolResult.AcknowledgedSafetyChecks[0].SourceResidual))
	require.JSONEq(t, `{"mcp":1}`, string(item.MCPListTools.Tools[0].SourceResidual))
	require.JSONEq(t, `{"summary":1}`, string(item.Reasoning.SummaryParts[0].SourceResidual))
	require.JSONEq(t, `{"reasoning":1}`, string(item.Reasoning.ContentParts[0].SourceResidual))
}

func TestCanonicalStreamItemCloneCoversOptionalLifecycleBranches(t *testing.T) {
	t.Parallel()

	require.Nil(t, cloneCanonicalItem(nil))
	require.Nil(t, cloneCanonicalToolResult(nil))
	require.Nil(t, cloneCanonicalContent(nil))

	item := &llm.Item{
		Content: []llm.ContentBlock{
			{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "https://example.invalid/image.png"}},
			{Kind: llm.ContentKindAudio, Audio: &llm.InputAudio{Data: "audio", Format: "wav"}},
			{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{Filename: "fixture.txt", Data: "document"}},
		},
		HostedCall: &llm.HostedToolCall{
			Invocation: llm.ToolInvocation{ArgumentsJSON: json.RawMessage(`{"x":1}`), ProviderData: json.RawMessage(`{"p":1}`)},
			Result: &llm.ToolResult{
				StructuredContent: json.RawMessage(`{"result":1}`), ProviderData: json.RawMessage(`{"provider":1}`),
				DiscoveredTools: []llm.ToolDefinition{{
					Kind: llm.ToolKindFunction, LogicalName: "lookup",
					Function:      &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)},
					ProtocolHints: llm.ProtocolHints{SourceResidual: json.RawMessage(`{"future":1}`)},
				}},
			},
		},
		MCPApprovalRequest:  &llm.MCPApprovalRequest{ArgumentsJSON: json.RawMessage(`{"approve":1}`)},
		MCPApprovalResponse: &llm.MCPApprovalResponse{Reason: "approved"},
		MCPCall:             &llm.MCPCall{ArgumentsJSON: json.RawMessage(`{"call":1}`)},
		Compaction:          &llm.CompactionItem{EncryptedContent: "encrypted"},
		AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{
			Kind: llm.AgentMessageContentInputText, Text: "handoff", SourceResidual: json.RawMessage(`{"agent_part":1}`),
		}}},
		Unknown: &llm.UnknownItem{Raw: json.RawMessage(`{"unknown":1}`)},
	}
	clone := cloneCanonicalItem(item)
	require.NotNil(t, clone)
	require.NotSame(t, item.Content[0].Image, clone.Content[0].Image)
	require.NotSame(t, item.Content[1].Audio, clone.Content[1].Audio)
	require.NotSame(t, item.Content[2].Document, clone.Content[2].Document)

	clone.HostedCall.Invocation.ArgumentsJSON[0] = '['
	clone.HostedCall.Invocation.ProviderData[0] = '['
	clone.HostedCall.Result.StructuredContent[0] = '['
	clone.HostedCall.Result.ProviderData[0] = '['
	clone.HostedCall.Result.DiscoveredTools[0].Function.Parameters[0] = '['
	clone.HostedCall.Result.DiscoveredTools[0].ProtocolHints.SourceResidual[0] = '['
	clone.MCPApprovalRequest.ArgumentsJSON[0] = '['
	clone.MCPCall.ArgumentsJSON[0] = '['
	clone.AgentMessage.Content[0].SourceResidual[0] = '['
	clone.Unknown.Raw[0] = '['

	require.JSONEq(t, `{"x":1}`, string(item.HostedCall.Invocation.ArgumentsJSON))
	require.JSONEq(t, `{"p":1}`, string(item.HostedCall.Invocation.ProviderData))
	require.JSONEq(t, `{"result":1}`, string(item.HostedCall.Result.StructuredContent))
	require.JSONEq(t, `{"provider":1}`, string(item.HostedCall.Result.ProviderData))
	require.JSONEq(t, `{"type":"object"}`, string(item.HostedCall.Result.DiscoveredTools[0].Function.Parameters))
	require.JSONEq(t, `{"future":1}`, string(item.HostedCall.Result.DiscoveredTools[0].ProtocolHints.SourceResidual))
	require.JSONEq(t, `{"approve":1}`, string(item.MCPApprovalRequest.ArgumentsJSON))
	require.JSONEq(t, `{"call":1}`, string(item.MCPCall.ArgumentsJSON))
	require.JSONEq(t, `{"agent_part":1}`, string(item.AgentMessage.Content[0].SourceResidual))
	require.JSONEq(t, `{"unknown":1}`, string(item.Unknown.Raw))
}

func TestCanonicalResidualOwnerHelpers(t *testing.T) {
	t.Parallel()

	require.Nil(t, contentResidualForWire(nil, "input_text"))
	require.Nil(t, contentResidualForWire(&llm.ContentBlock{}, "input_text"))
	require.JSONEq(t, `{"future":1}`, string(contentResidualForWire(&llm.ContentBlock{
		SourceResidual: json.RawMessage(`{"future":1}`),
	}, "input_image")))
	require.JSONEq(t, `{"future":1}`, string(contentResidualForWire(&llm.ContentBlock{
		ResidualOwnerType: "text", SourceResidual: json.RawMessage(`{"future":1}`),
	}, "input_text")))
	require.Nil(t, contentResidualForWire(&llm.ContentBlock{
		ResidualOwnerType: "input_text", SourceResidual: json.RawMessage(`{"future":1}`),
	}, "input_image"))

	require.Nil(t, reasoningPartResidualForWire(nil, "summary_text"))
	require.Nil(t, reasoningPartResidualForWire(&llm.ReasoningPart{}, "summary_text"))
	require.JSONEq(t, `{"future":1}`, string(reasoningPartResidualForWire(&llm.ReasoningPart{
		SourceResidual: json.RawMessage(`{"future":1}`),
	}, "summary_text")))
	require.Nil(t, reasoningPartResidualForWire(&llm.ReasoningPart{
		ResidualOwnerType: "summary_text", SourceResidual: json.RawMessage(`{"future":1}`),
	}, "reasoning_text"))
}

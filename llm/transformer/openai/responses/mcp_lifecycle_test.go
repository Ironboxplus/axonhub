package responses

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesMCPLifecycleItemsUseTypedCanonicalRoundTrip(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"mcp_list_tools","id":"list_1","server_label":"inventory","tools":[{"name":"lookup","description":"find stock","input_schema":{"type":"object","properties":{"sku":{"type":"string"}}},"annotations":{"readOnlyHint":true}}]},
			{"type":"mcp_approval_request","id":"approval_1","server_label":"inventory","name":"reserve","arguments":"{\"sku\":\"A-1\"}"},
			{"type":"mcp_approval_response","id":"decision_1","approval_request_id":"approval_1","approve":false,"reason":"not now"},
			{"type":"mcp_call","id":"call_1","server_label":"inventory","name":"lookup","approval_request_id":"approval_0","arguments":"{\"sku\":\"A-1\"}","output":"{\"content\":[{\"type\":\"text\",\"text\":\"7\"}]}","status":"completed"}
		]
	}`)

	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	require.NoError(t, err)
	require.Len(t, request.Input, 4)
	require.Equal(t, []llm.ItemKind{
		llm.ItemKindMCPListTools, llm.ItemKindMCPApprovalRequest,
		llm.ItemKindMCPApprovalResponse, llm.ItemKindMCPCall,
	}, []llm.ItemKind{request.Input[0].Kind, request.Input[1].Kind, request.Input[2].Kind, request.Input[3].Kind})
	require.NoError(t, llm.ValidateItemStructure(request.Input))
	require.JSONEq(t, `{"type":"object","properties":{"sku":{"type":"string"}}}`, string(request.Input[0].MCPListTools.Tools[0].InputSchema))
	require.JSONEq(t, `{"sku":"A-1"}`, string(request.Input[1].MCPApprovalRequest.ArgumentsJSON))
	require.False(t, request.Input[2].MCPApprovalResponse.Approve)
	require.Equal(t, "approval_1", request.Input[2].MCPApprovalResponse.ApprovalRequestID)
	require.Equal(t, llm.MCPCallStatusCompleted, request.Input[3].MCPCall.Status)
	require.Contains(t, request.Input[3].MCPCall.Output, `"text":"7"`)
	clone := request.Clone()
	clone.Input[0].MCPListTools.Tools[0].InputSchema[0] = '['
	clone.Input[1].MCPApprovalRequest.ArgumentsJSON[0] = '['
	clone.Input[3].MCPCall.ArgumentsJSON[0] = '['
	require.Equal(t, byte('{'), request.Input[0].MCPListTools.Tools[0].InputSchema[0])
	require.Equal(t, byte('{'), request.Input[1].MCPApprovalRequest.ArgumentsJSON[0])
	require.Equal(t, byte('{'), request.Input[3].MCPCall.ArgumentsJSON[0])

	if request.ProviderExtensions != nil && request.ProviderExtensions.OpenAIResponses != nil &&
		request.ProviderExtensions.OpenAIResponses.Request != nil {
		require.Empty(t, request.ProviderExtensions.OpenAIResponses.Request.RawInputItems,
			"typed MCP lifecycle must not also survive as an opaque sidecar")
	}

	for index := range request.Input {
		wire, ok := canonicalItemToResponses(&request.Input[index])
		require.True(t, ok, "item %d", index)
		encoded, marshalErr := json.Marshal(wire)
		require.NoError(t, marshalErr)
		var decoded Item
		require.NoError(t, json.Unmarshal(encoded, &decoded), "item %d: %s", index, encoded)
		restored, restoreErr := responseItemToCanonical(&decoded, encoded, index)
		require.NoError(t, restoreErr)
		require.NoError(t, restored.Validate())
		require.Equal(t, request.Input[index].Kind, restored.Kind)
	}

	decisionWire, ok := canonicalItemToResponses(&request.Input[2])
	require.True(t, ok)
	decisionJSON, err := json.Marshal(decisionWire)
	require.NoError(t, err)
	require.Contains(t, string(decisionJSON), `"approve":false`)
}

func TestCanonicalResponsesMCPStreamLifecycle(t *testing.T) {
	t.Parallel()
	decoder := newCanonicalStreamDecoder()
	sequence := 0
	itemID := func(value string) *string { return &value }
	status := func(value string) *string { return &value }
	approve := false

	wireEvents := []*StreamEvent{
		{Type: StreamEventTypeResponseCreated, SequenceNumber: &sequence},
		{Type: StreamEventTypeOutputItemAdded, SequenceNumber: intPointer(1), OutputIndex: 0, Item: &Item{
			ID: "list_1", Type: "mcp_list_tools", ServerLabel: "inventory", Tools: []MCPListedTool{}, Status: status("in_progress"),
		}},
		{Type: StreamEventTypeMCPListToolsInProgress, SequenceNumber: intPointer(2), OutputIndex: 0, ItemID: itemID("list_1")},
		{Type: StreamEventTypeMCPListToolsCompleted, SequenceNumber: intPointer(3), OutputIndex: 0, ItemID: itemID("list_1")},
		{Type: StreamEventTypeOutputItemDone, SequenceNumber: intPointer(4), OutputIndex: 0, Item: &Item{
			ID: "list_1", Type: "mcp_list_tools", ServerLabel: "inventory", Status: status("completed"),
			Tools: []MCPListedTool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}},
		}},
		{Type: StreamEventTypeOutputItemAdded, SequenceNumber: intPointer(5), OutputIndex: 1, Item: &Item{
			ID: "approval_1", Type: "mcp_approval_response", ApprovalRequestID: "request_1", Approve: &approve,
		}},
		{Type: StreamEventTypeOutputItemDone, SequenceNumber: intPointer(6), OutputIndex: 1, Item: &Item{
			ID: "approval_1", Type: "mcp_approval_response", ApprovalRequestID: "request_1", Approve: &approve,
		}},
		{Type: StreamEventTypeOutputItemAdded, SequenceNumber: intPointer(7), OutputIndex: 2, Item: &Item{
			ID: "call_1", Type: "mcp_call", ServerLabel: "inventory", Name: "lookup", Status: status("in_progress"),
		}},
		{Type: StreamEventTypeMCPCallArgumentsDelta, SequenceNumber: intPointer(8), OutputIndex: 2, ItemID: itemID("call_1"), Delta: `{"sku":"`},
		{Type: StreamEventTypeMCPCallArgumentsDelta, SequenceNumber: intPointer(9), OutputIndex: 2, ItemID: itemID("call_1"), Delta: `A-1"}`},
		{Type: StreamEventTypeMCPCallArgumentsDone, SequenceNumber: intPointer(10), OutputIndex: 2, ItemID: itemID("call_1"), Arguments: `{"sku":"A-1"}`},
		{Type: StreamEventTypeMCPCallInProgress, SequenceNumber: intPointer(11), OutputIndex: 2, ItemID: itemID("call_1")},
		{Type: StreamEventTypeMCPCallCompleted, SequenceNumber: intPointer(12), OutputIndex: 2, ItemID: itemID("call_1")},
		{Type: StreamEventTypeOutputItemDone, SequenceNumber: intPointer(13), OutputIndex: 2, Item: &Item{
			ID: "call_1", Type: "mcp_call", ServerLabel: "inventory", Name: "lookup",
			Arguments: `{"sku":"A-1"}`, Output: &Input{Text: itemID("7")}, Status: status("completed"),
		}},
		{Type: StreamEventTypeResponseCompleted, SequenceNumber: intPointer(14), Response: &Response{}},
	}

	var canonical []llm.Event
	for _, wire := range wireEvents {
		events, err := decoder.decode(wire)
		require.NoError(t, err, "wire event %s", wire.Type)
		canonical = append(canonical, events...)
	}
	require.NotEmpty(t, canonical)
	require.Equal(t, llm.EventKindResponseCompleted, canonical[len(canonical)-1].Kind)

	var sawArgumentsDone, sawCallCompleted, sawListCompleted bool
	for _, event := range canonical {
		if event.Kind == llm.EventKindToolInputDone && event.ItemRef.ItemID == "call_1" {
			sawArgumentsDone = true
		}
		if event.Kind == llm.EventKindMCPStatus && event.ItemRef.ItemID == "call_1" && event.Delta.MCPStatus == "completed" {
			sawCallCompleted = true
		}
		if event.Kind == llm.EventKindMCPStatus && event.ItemRef.ItemID == "list_1" && event.Delta.MCPStatus == "completed" {
			sawListCompleted = true
		}
	}
	require.True(t, sawArgumentsDone)
	require.True(t, sawCallCompleted)
	require.True(t, sawListCompleted)
}

func TestCanonicalMCPStreamEncoderEmitsResponsesNativeEvents(t *testing.T) {
	t.Parallel()
	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{
		ctx: context.Background(), transformerMetadata: make(map[string]any),
		aggregator: newStreamAggregator(),
	}
	outputIndex := 0
	ref := llm.ItemRef{ItemID: "call_1", OutputIndex: &outputIndex}
	events := []llm.Event{
		{Kind: llm.EventKindResponseStarted, Sequence: 0},
		{Kind: llm.EventKindResponseInProgress, Sequence: 1},
		{
			Kind: llm.EventKindItemAdded, Sequence: 2, ItemRef: ref,
			Snapshot: &llm.Item{
				Kind: llm.ItemKindMCPCall, ID: "call_1", Status: llm.ItemStatusInProgress,
				MCPCall: &llm.MCPCall{ServerLabel: "inventory", LogicalName: "lookup", Status: llm.MCPCallStatusInProgress},
			},
		},
		{Kind: llm.EventKindToolInputDelta, Sequence: 3, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: `{"sku":"A-1"}`}},
		{Kind: llm.EventKindToolInputDone, Sequence: 4, ItemRef: ref},
		{Kind: llm.EventKindMCPStatus, Sequence: 5, ItemRef: ref, Delta: llm.Delta{MCPStatus: "in_progress"}},
		{Kind: llm.EventKindMCPStatus, Sequence: 6, ItemRef: ref, Delta: llm.Delta{MCPStatus: "completed"}},
		{
			Kind: llm.EventKindItemDone, Sequence: 7, ItemRef: ref,
			Snapshot: &llm.Item{
				Kind: llm.ItemKindMCPCall, ID: "call_1", Status: llm.ItemStatusCompleted,
				MCPCall: &llm.MCPCall{
					ServerLabel: "inventory", LogicalName: "lookup", ArgumentsJSON: json.RawMessage(`{"sku":"A-1"}`),
					Output: "7", Status: llm.MCPCallStatusCompleted,
				},
			},
		},
		{Kind: llm.EventKindResponseCompleted, Sequence: 8},
	}
	for _, event := range events {
		require.NoError(t, encoder.encode(source, event), "canonical event %s", event.Kind)
	}

	want := []StreamEventType{
		StreamEventTypeResponseCreated, StreamEventTypeResponseInProgress,
		StreamEventTypeOutputItemAdded, StreamEventTypeMCPCallArgumentsDelta,
		StreamEventTypeMCPCallArgumentsDone, StreamEventTypeMCPCallInProgress,
		StreamEventTypeMCPCallCompleted, StreamEventTypeOutputItemDone,
		StreamEventTypeResponseCompleted,
	}
	require.Len(t, source.eventQueue, len(want))
	decoder := newCanonicalStreamDecoder()
	for index, event := range source.eventQueue {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal(event.Data, &wire))
		require.Equal(t, want[index], wire.Type)
		_, err := decoder.decode(&wire)
		require.NoError(t, err, "encoded wire event %s", wire.Type)
	}
}

func intPointer(value int) *int { return &value }

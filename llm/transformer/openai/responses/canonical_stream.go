package responses

import (
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
)

// canonicalStreamDecoder converts Responses wire events into the shared
// lifecycle without first flattening them into Chat Completion deltas.
// One decoder belongs to one request and therefore needs no synchronization.
const maxPendingProtocolFramesPerItem = 128

type canonicalStreamDecoder struct {
	machine            *llm.StreamStateMachine
	nextSequence       uint64
	items              map[string]*llm.Item
	itemOrder          []string
	refByWireID        map[string]llm.ItemRef
	inputDone          map[string]bool
	hostedDone         map[string]bool
	itemDone           map[string]bool
	argumentsByID      map[string]string
	inputByID          map[string]string
	protocolFrames     map[string][]llm.ProtocolFrameHint
	lastSourceSequence int
	hasSourceSequence  bool
}

func newCanonicalStreamDecoder() *canonicalStreamDecoder {
	return &canonicalStreamDecoder{
		machine:        llm.NewStreamStateMachine(),
		items:          make(map[string]*llm.Item),
		refByWireID:    make(map[string]llm.ItemRef),
		inputDone:      make(map[string]bool),
		hostedDone:     make(map[string]bool),
		itemDone:       make(map[string]bool),
		argumentsByID:  make(map[string]string),
		inputByID:      make(map[string]string),
		protocolFrames: make(map[string][]llm.ProtocolFrameHint),
	}
}

func (decoder *canonicalStreamDecoder) decode(wire *StreamEvent) ([]llm.Event, error) {
	if wire == nil {
		return nil, fmt.Errorf("nil Responses stream event")
	}

	var events []llm.Event
	var sourceSequence *uint64
	if wire.SequenceNumber != nil {
		if *wire.SequenceNumber < 0 {
			return nil, responsesSourceSequenceError(wire, fmt.Errorf("negative sequence_number %d", *wire.SequenceNumber))
		}
		if decoder.hasSourceSequence && *wire.SequenceNumber <= decoder.lastSourceSequence {
			return nil, responsesSourceSequenceError(wire, fmt.Errorf("sequence_number %d is not after %d", *wire.SequenceNumber, decoder.lastSourceSequence))
		}
		decoder.hasSourceSequence = true
		decoder.lastSourceSequence = *wire.SequenceNumber
		value := uint64(*wire.SequenceNumber)
		sourceSequence = &value
	}
	add := func(event llm.Event) error {
		event.Sequence = decoder.nextSequence
		event.SourceSequence = sourceSequence
		event.SourceType = string(wire.Type)
		event.SourceResidual = cloneRaw(wire.Residual)
		if wire.Response != nil {
			event.ResponseSourceResidual = cloneRaw(wire.Response.Residual)
		}
		decoder.nextSequence++
		if err := decoder.machine.Apply(event); err != nil {
			return fmt.Errorf("Responses %s: %w", wire.Type, err)
		}
		events = append(events, event)
		return nil
	}
	if decoder.machine.ResponseState() == llm.ResponseStreamStateInit && responsesEventNeedsLifecycle(wire.Type) {
		if err := add(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
			return nil, err
		}
		if wire.Type != StreamEventTypeResponseInProgress && wire.Type != StreamEventTypeResponseQueued {
			if err := add(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
				return nil, err
			}
		}
	}

	switch wire.Type {
	case StreamEventTypeResponseCreated:
		if err := add(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
			return nil, err
		}
	case StreamEventTypeResponseInProgress, StreamEventTypeResponseQueued:
		if err := add(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
			return nil, err
		}
	case StreamEventTypeOutputItemAdded:
		item, err := canonicalStreamItem(wire.Item, wire.ItemRaw, wire.OutputIndex, true)
		if err != nil {
			return nil, err
		}
		ref := decoder.rememberItem(wire.OutputIndex, item)
		if err := add(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: ref, Snapshot: cloneCanonicalItem(item)}); err != nil {
			return nil, err
		}
	case StreamEventTypeOutputTextDelta, StreamEventTypeRefusalDelta:
		ref, found := decoder.knownRefForWire(wire)
		if !found {
			var err error
			ref, err = decoder.synthesizeMessageAdded(wire, add)
			if err != nil {
				return nil, err
			}
		}
		key := decoder.storageKey(ref)
		if wire.Type == StreamEventTypeRefusalDelta {
			if item := decoder.items[key]; item != nil && len(item.Content) > 0 {
				item.Content[0].Kind = llm.ContentKindRefusal
			}
		}
		decoder.appendText(key, wire.Delta)
		eventKind := llm.EventKindTextDelta
		if wire.Type == StreamEventTypeRefusalDelta {
			eventKind = llm.EventKindRefusalDelta
		}
		event := llm.Event{Kind: eventKind, ItemRef: ref, ContentIndex: wire.ContentIndex, Delta: llm.Delta{Text: wire.Delta}}
		event.ProtocolFrames = decoder.takeProtocolFrames(key, StreamEventTypeContentPartAdded)
		if err := add(event); err != nil {
			return nil, err
		}
	case StreamEventTypeReasoningSummaryTextDelta:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		decoder.appendReasoning(decoder.storageKey(ref), wire.Delta)
		event := llm.Event{Kind: llm.EventKindReasoningDelta, ItemRef: ref, ContentIndex: wire.SummaryIndex, Delta: llm.Delta{Text: wire.Delta}}
		event.ProtocolFrames = decoder.takeProtocolFrames(decoder.storageKey(ref), StreamEventTypeReasoningSummaryPartAdded)
		if err := add(event); err != nil {
			return nil, err
		}
	case StreamEventTypeFunctionCallArgumentsDelta:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		decoder.argumentsByID[decoder.storageKey(ref)] += wire.Delta
		if err := add(llm.Event{Kind: llm.EventKindToolInputDelta, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: wire.Delta}}); err != nil {
			return nil, err
		}
	case StreamEventTypeFunctionCallArgumentsDone:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		key := decoder.storageKey(ref)
		decoder.argumentsByID[key] = wire.Arguments
		decoder.inputDone[key] = true
		decoder.updateFunctionSnapshot(key, wire)
		if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
			return nil, err
		}
	case StreamEventTypeCustomToolCallInputDelta:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		decoder.inputByID[decoder.storageKey(ref)] += wire.Delta
		if err := add(llm.Event{Kind: llm.EventKindToolInputDelta, ItemRef: ref, Delta: llm.Delta{InputText: wire.Delta}}); err != nil {
			return nil, err
		}
	case StreamEventTypeCustomToolCallInputDone:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		key := decoder.storageKey(ref)
		decoder.inputByID[key] = wire.Input
		decoder.inputDone[key] = true
		decoder.updateCustomSnapshot(key, wire.Input)
		if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
			return nil, err
		}
	case StreamEventTypeMCPCallArgumentsDelta:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		key := decoder.storageKey(ref)
		decoder.argumentsByID[key] += wire.Delta
		decoder.updateMCPArguments(key, decoder.argumentsByID[key])
		if err := add(llm.Event{Kind: llm.EventKindToolInputDelta, ItemRef: ref, Delta: llm.Delta{ArgumentsJSON: wire.Delta}}); err != nil {
			return nil, err
		}
	case StreamEventTypeMCPCallArgumentsDone:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		key := decoder.storageKey(ref)
		decoder.argumentsByID[key] = wire.Arguments
		decoder.inputDone[key] = true
		decoder.updateMCPArguments(key, wire.Arguments)
		if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
			return nil, err
		}
	case StreamEventTypeMCPCallInProgress, StreamEventTypeMCPCallCompleted, StreamEventTypeMCPCallFailed,
		StreamEventTypeMCPListToolsInProgress, StreamEventTypeMCPListToolsCompleted, StreamEventTypeMCPListToolsFailed:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		key := decoder.storageKey(ref)
		item := decoder.items[key]
		if item != nil && item.Kind == llm.ItemKindMCPCall && !decoder.inputDone[key] {
			decoder.updateMCPArguments(key, item.MCPCall.ArgumentsText)
			if len(item.MCPCall.ArgumentsJSON) > 0 {
				decoder.updateMCPArguments(key, string(item.MCPCall.ArgumentsJSON))
			}
			if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
				return nil, err
			}
			decoder.inputDone[key] = true
		}
		status := responsesMCPEventStatus(wire.Type)
		decoder.updateMCPStatus(key, status)
		if err := add(llm.Event{Kind: llm.EventKindMCPStatus, ItemRef: ref, Delta: llm.Delta{MCPStatus: string(status)}}); err != nil {
			return nil, err
		}
		if status == llm.MCPCallStatusCompleted || status == llm.MCPCallStatusFailed {
			decoder.hostedDone[key] = true
		}
	case StreamEventTypeImageGenerationGenerating, StreamEventTypeImageGenerationInProgress,
		StreamEventTypeImageGenerationCompleted:
		ref, err := decoder.refForWire(wire)
		if err != nil {
			return nil, err
		}
		status := "in_progress"
		if wire.Type == StreamEventTypeImageGenerationGenerating {
			status = "generating"
		} else if wire.Type == StreamEventTypeImageGenerationCompleted {
			status = "completed"
		}
		if err := add(llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: ref, Delta: llm.Delta{HostedStatus: status}}); err != nil {
			return nil, err
		}
		if status == "completed" {
			decoder.hostedDone[decoder.storageKey(ref)] = true
		}
	case StreamEventTypeImageGenerationPartialImage:
		// The image bytes are provider output rather than status. Preserve them as
		// typed item state on output_item.done; emitting every base64 preview as
		// trace metadata would be both lossy and prohibitively expensive.
	case StreamEventTypeOutputItemDone:
		item, err := canonicalStreamItem(wire.Item, wire.ItemRaw, wire.OutputIndex, false)
		if err != nil {
			return nil, err
		}
		ref := decoder.itemRef(wire.OutputIndex, item)
		key := decoder.storageKey(ref)
		if _, exists := decoder.items[key]; !exists {
			decoder.rememberItem(wire.OutputIndex, item)
			addedSnapshot := cloneCanonicalItem(item)
			markItemInProgress(addedSnapshot)
			if err := add(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: ref, Snapshot: addedSnapshot}); err != nil {
				return nil, err
			}
		}
		if (item.Kind == llm.ItemKindToolCall || item.Kind == llm.ItemKindMCPCall) && !decoder.inputDone[key] {
			// Responses normally emits a dedicated *.done event. Some compatible
			// providers omit it while still supplying a final output_item.done
			// snapshot; synthesize the semantic boundary deterministically.
			if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
				return nil, err
			}
			decoder.inputDone[key] = true
		}
		if (item.Kind == llm.ItemKindMCPCall || item.Kind == llm.ItemKindMCPListTools) && !decoder.hostedDone[key] {
			status := llm.MCPCallStatusCompleted
			if item.Status == llm.ItemStatusFailed {
				status = llm.MCPCallStatusFailed
			} else if item.Status == llm.ItemStatusIncomplete {
				status = llm.MCPCallStatusIncomplete
			}
			decoder.updateMCPStatus(key, status)
			if err := add(llm.Event{Kind: llm.EventKindMCPStatus, ItemRef: ref, Delta: llm.Delta{MCPStatus: string(status)}}); err != nil {
				return nil, err
			}
			decoder.hostedDone[key] = true
		}
		if item.Kind == llm.ItemKindHostedCall && !decoder.hostedDone[key] {
			status := "completed"
			if item.Status == llm.ItemStatusFailed {
				status = "failed"
			}
			if err := add(llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: ref, Delta: llm.Delta{HostedStatus: status}}); err != nil {
				return nil, err
			}
			decoder.hostedDone[key] = true
		}
		decoder.items[key] = cloneCanonicalItem(item)
		doneEvent := llm.Event{Kind: llm.EventKindItemDone, ItemRef: ref, Snapshot: cloneCanonicalItem(item)}
		doneEvent.ProtocolFrames = decoder.takeProtocolFrames(key)
		if err := add(doneEvent); err != nil {
			return nil, err
		}
		decoder.itemDone[key] = true
	case StreamEventTypeResponseCompleted:
		if err := decoder.closeOpenItems(llm.ItemStatusCompleted, add); err != nil {
			return nil, err
		}
		if wire.Response != nil && wire.Response.Usage != nil {
			if err := add(llm.Event{Kind: llm.EventKindUsage, Usage: wire.Response.Usage.ToUsage()}); err != nil {
				return nil, err
			}
		}
		if err := add(llm.Event{Kind: llm.EventKindResponseCompleted}); err != nil {
			return nil, err
		}
	case StreamEventTypeResponseFailed:
		if err := decoder.closeOpenItems(llm.ItemStatusFailed, add); err != nil {
			return nil, err
		}
		responseErr := responsesStreamError(wire)
		if err := add(llm.Event{Kind: llm.EventKindResponseFailed, Error: responseErr}); err != nil {
			return nil, err
		}
	case StreamEventTypeResponseIncomplete:
		if err := decoder.closeOpenItems(llm.ItemStatusIncomplete, add); err != nil {
			return nil, err
		}
		if err := add(llm.Event{Kind: llm.EventKindResponseIncomplete}); err != nil {
			return nil, err
		}
	case StreamEventTypeResponseCancelled:
		if err := decoder.closeOpenItems(llm.ItemStatusIncomplete, add); err != nil {
			return nil, err
		}
		if err := add(llm.Event{Kind: llm.EventKindResponseCancelled}); err != nil {
			return nil, err
		}
	case StreamEventTypeError:
		if err := add(llm.Event{Kind: llm.EventKindError, Error: responsesStreamError(wire)}); err != nil {
			return nil, err
		}
	case StreamEventTypeContentPartAdded, StreamEventTypeContentPartDone,
		StreamEventTypeOutputTextDone, StreamEventTypeRefusalDone,
		StreamEventTypeReasoningSummaryPartAdded, StreamEventTypeReasoningSummaryPartDone,
		StreamEventTypeReasoningSummaryTextDone:
		if err := decoder.rememberProtocolFrame(wire); err != nil {
			return nil, err
		}
	case StreamEventType("keepalive"):
		// Transport keepalives do not belong to the model lifecycle.
	default:
		return nil, fmt.Errorf("unsupported Responses stream event type %q", wire.Type)
	}

	return events, nil
}

func responsesEventNeedsLifecycle(eventType StreamEventType) bool {
	switch eventType {
	case StreamEventTypeResponseCompleted, StreamEventTypeResponseFailed,
		StreamEventTypeResponseIncomplete, StreamEventTypeResponseCancelled:
		return true
	default:
		return false
	}
}

func responsesMCPEventStatus(eventType StreamEventType) llm.MCPCallStatus {
	switch eventType {
	case StreamEventTypeMCPCallInProgress, StreamEventTypeMCPListToolsInProgress:
		return llm.MCPCallStatusInProgress
	case StreamEventTypeMCPCallCompleted, StreamEventTypeMCPListToolsCompleted:
		return llm.MCPCallStatusCompleted
	case StreamEventTypeMCPCallFailed, StreamEventTypeMCPListToolsFailed:
		return llm.MCPCallStatusFailed
	default:
		return ""
	}
}

func responsesSourceSequenceError(wire *StreamEvent, cause error) error {
	eventKind := llm.EventKind(wire.Type)
	switch wire.Type {
	case StreamEventTypeResponseCreated:
		eventKind = llm.EventKindResponseStarted
	case StreamEventTypeResponseInProgress, StreamEventTypeResponseQueued:
		eventKind = llm.EventKindResponseInProgress
	case StreamEventTypeResponseCompleted:
		eventKind = llm.EventKindResponseCompleted
	case StreamEventTypeResponseFailed:
		eventKind = llm.EventKindResponseFailed
	case StreamEventTypeResponseIncomplete:
		eventKind = llm.EventKindResponseIncomplete
	case StreamEventTypeResponseCancelled:
		eventKind = llm.EventKindResponseCancelled
	}
	sequence := uint64(0)
	if wire.SequenceNumber != nil && *wire.SequenceNumber >= 0 {
		sequence = uint64(*wire.SequenceNumber)
	}
	return &llm.StreamInvariantError{
		Code: llm.StreamInvariantSequence, EventKind: eventKind, Sequence: sequence,
		Cause: cause,
	}
}

func canonicalStreamItem(item *Item, raw json.RawMessage, outputIndex int, added bool) (*llm.Item, error) {
	if item == nil {
		return nil, fmt.Errorf("Responses output item event at index %d has no item", outputIndex)
	}
	canonical, err := responseItemToCanonical(item, raw, outputIndex)
	if err != nil {
		return nil, fmt.Errorf("decode Responses stream item %q: %w", item.Type, err)
	}
	if canonical != nil {
		return canonical, nil
	}
	if item.Type == "message" && added {
		return &llm.Item{
			Kind: llm.ItemKindMessage, ID: item.ID, Role: llm.RoleAssistant,
			Status:        llm.ItemStatusInProgress,
			Content:       []llm.ContentBlock{{Kind: llm.ContentKindText}},
			ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse, SourceType: item.Type, Ordinal: outputIndex},
		}, nil
	}
	return nil, fmt.Errorf("Responses stream item %q at index %d has no canonical payload", item.Type, outputIndex)
}

func (decoder *canonicalStreamDecoder) rememberItem(outputIndex int, item *llm.Item) llm.ItemRef {
	ref := decoder.itemRef(outputIndex, item)
	key := decoder.storageKey(ref)
	if _, exists := decoder.items[key]; !exists {
		decoder.itemOrder = append(decoder.itemOrder, key)
	}
	decoder.items[key] = cloneCanonicalItem(item)
	if item.MCPCall != nil {
		arguments := item.MCPCall.ArgumentsText
		if len(item.MCPCall.ArgumentsJSON) > 0 {
			arguments = string(item.MCPCall.ArgumentsJSON)
		}
		decoder.argumentsByID[key] = arguments
	}
	if ref.ItemID != "" {
		decoder.refByWireID[ref.ItemID] = ref
	}
	if ref.CallID != "" {
		decoder.refByWireID[ref.CallID] = ref
	}
	return ref
}

func (decoder *canonicalStreamDecoder) itemRef(outputIndex int, item *llm.Item) llm.ItemRef {
	ref := llm.ItemRef{OutputIndex: &outputIndex, ItemID: item.ID}
	switch item.Kind {
	case llm.ItemKindToolCall:
		ref.CallID = item.ToolCall.CallID
	case llm.ItemKindHostedCall:
		ref.CallID = item.HostedCall.Invocation.CallID
	}
	return ref
}

func (decoder *canonicalStreamDecoder) refForWire(wire *StreamEvent) (llm.ItemRef, error) {
	ref, found := decoder.knownRefForWire(wire)
	if !found {
		itemID := ""
		if wire.ItemID != nil {
			itemID = *wire.ItemID
		}
		if itemID == "" {
			return llm.ItemRef{}, fmt.Errorf("Responses %s at output index %d has no item_id", wire.Type, wire.OutputIndex)
		}
		return llm.ItemRef{}, fmt.Errorf("Responses %s references unknown item_id %q", wire.Type, itemID)
	}
	return ref, nil
}

func (decoder *canonicalStreamDecoder) knownRefForWire(wire *StreamEvent) (llm.ItemRef, bool) {
	if wire == nil || wire.ItemID == nil || *wire.ItemID == "" {
		return llm.ItemRef{}, false
	}
	ref, found := decoder.refByWireID[*wire.ItemID]
	if !found {
		return llm.ItemRef{}, false
	}
	outputIndex := wire.OutputIndex
	ref.OutputIndex = &outputIndex
	return ref, true
}

func (decoder *canonicalStreamDecoder) synthesizeMessageAdded(wire *StreamEvent, add func(llm.Event) error) (llm.ItemRef, error) {
	if wire.ItemID == nil || *wire.ItemID == "" {
		return llm.ItemRef{}, fmt.Errorf("Responses %s at output index %d has no item_id", wire.Type, wire.OutputIndex)
	}
	item := &llm.Item{
		Kind: llm.ItemKindMessage, ID: *wire.ItemID, Role: llm.RoleAssistant,
		Status: llm.ItemStatusInProgress, Content: []llm.ContentBlock{{Kind: llm.ContentKindText}},
		ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "message", Ordinal: wire.OutputIndex},
	}
	ref := decoder.rememberItem(wire.OutputIndex, item)
	if err := add(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: ref, Snapshot: cloneCanonicalItem(item)}); err != nil {
		return llm.ItemRef{}, err
	}
	return ref, nil
}

func (decoder *canonicalStreamDecoder) storageKey(ref llm.ItemRef) string {
	if ref.ItemID != "" {
		return "item:" + ref.ItemID
	}
	return "call:" + ref.CallID
}

func (decoder *canonicalStreamDecoder) rememberProtocolFrame(wire *StreamEvent) error {
	ref, err := decoder.refForWire(wire)
	if err != nil {
		return err
	}
	hint := llm.ProtocolFrameHint{
		SourceType:     string(wire.Type),
		SourceResidual: cloneRaw(wire.Residual),
	}
	if wire.Part != nil {
		hint.PayloadResidual = cloneRaw(wire.Part.Residual)
	}
	key := decoder.storageKey(ref)
	if len(decoder.protocolFrames[key]) >= maxPendingProtocolFramesPerItem {
		return fmt.Errorf("Responses protocol framing for %s exceeded %d pending events", key, maxPendingProtocolFramesPerItem)
	}
	decoder.protocolFrames[key] = append(decoder.protocolFrames[key], hint)
	return nil
}

func (decoder *canonicalStreamDecoder) takeProtocolFrames(key string, types ...StreamEventType) []llm.ProtocolFrameHint {
	frames := decoder.protocolFrames[key]
	if len(frames) == 0 {
		return nil
	}
	if len(types) == 0 {
		delete(decoder.protocolFrames, key)
		return cloneProtocolFrameHints(frames)
	}
	wanted := make(map[string]bool, len(types))
	for _, eventType := range types {
		wanted[string(eventType)] = false
	}
	selected := make([]llm.ProtocolFrameHint, 0, len(frames))
	remaining := make([]llm.ProtocolFrameHint, 0, len(frames))
	for index := range frames {
		frame := frames[index]
		if consumed, ok := wanted[frame.SourceType]; ok && !consumed {
			selected = append(selected, frame)
			wanted[frame.SourceType] = true
			continue
		}
		remaining = append(remaining, frame)
	}
	if len(remaining) == 0 {
		delete(decoder.protocolFrames, key)
	} else {
		decoder.protocolFrames[key] = remaining
	}
	return cloneProtocolFrameHints(selected)
}

func cloneProtocolFrameHints(frames []llm.ProtocolFrameHint) []llm.ProtocolFrameHint {
	if len(frames) == 0 {
		return nil
	}
	clone := make([]llm.ProtocolFrameHint, len(frames))
	copy(clone, frames)
	for index := range clone {
		clone[index].SourceResidual = cloneRaw(frames[index].SourceResidual)
		clone[index].PayloadResidual = cloneRaw(frames[index].PayloadResidual)
	}
	return clone
}

func (decoder *canonicalStreamDecoder) appendText(itemID, delta string) {
	item := decoder.items[itemID]
	if item == nil || len(item.Content) == 0 {
		return
	}
	item.Content[0].Text += delta
}

func (decoder *canonicalStreamDecoder) appendReasoning(itemID, delta string) {
	item := decoder.items[itemID]
	if item == nil || item.Reasoning == nil {
		return
	}
	item.Reasoning.Content += delta
}

func (decoder *canonicalStreamDecoder) updateFunctionSnapshot(itemID string, wire *StreamEvent) {
	item := decoder.items[itemID]
	if item == nil || item.ToolCall == nil {
		return
	}
	if wire.Name != "" {
		item.ToolCall.LogicalName = wire.Name
	}
	if wire.Namespace != "" {
		item.ToolCall.Namespace = wire.Namespace
	}
	if json.Valid([]byte(wire.Arguments)) {
		item.ToolCall.ArgumentsJSON = append(json.RawMessage(nil), wire.Arguments...)
		item.ToolCall.ArgumentsText = ""
	} else {
		item.ToolCall.ArgumentsJSON = nil
		item.ToolCall.ArgumentsText = wire.Arguments
	}
}

func (decoder *canonicalStreamDecoder) updateCustomSnapshot(itemID, input string) {
	item := decoder.items[itemID]
	if item == nil || item.ToolCall == nil {
		return
	}
	item.ToolCall.InputText = input
}

func (decoder *canonicalStreamDecoder) updateMCPArguments(itemID, arguments string) {
	item := decoder.items[itemID]
	if item == nil || item.MCPCall == nil {
		return
	}
	if json.Valid([]byte(arguments)) {
		item.MCPCall.ArgumentsJSON = append(json.RawMessage(nil), arguments...)
		item.MCPCall.ArgumentsText = ""
	} else {
		item.MCPCall.ArgumentsJSON = nil
		item.MCPCall.ArgumentsText = arguments
	}
}

func (decoder *canonicalStreamDecoder) updateMCPStatus(itemID string, status llm.MCPCallStatus) {
	item := decoder.items[itemID]
	if item == nil {
		return
	}
	if item.MCPCall != nil {
		item.MCPCall.Status = status
	}
	switch status {
	case llm.MCPCallStatusCompleted:
		item.Status = llm.ItemStatusCompleted
	case llm.MCPCallStatusFailed:
		item.Status = llm.ItemStatusFailed
	case llm.MCPCallStatusIncomplete:
		item.Status = llm.ItemStatusIncomplete
	default:
		item.Status = llm.ItemStatusInProgress
	}
}

func (decoder *canonicalStreamDecoder) closeOpenItems(status llm.ItemStatus, add func(llm.Event) error) error {
	for _, key := range decoder.itemOrder {
		if decoder.itemDone[key] {
			continue
		}
		item := decoder.items[key]
		if item == nil {
			continue
		}
		ref := decoder.itemRef(item.ProtocolHints.Ordinal, item)
		if item.Kind == llm.ItemKindToolCall {
			if !decoder.inputDone[key] {
				if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
					return err
				}
				decoder.inputDone[key] = true
			}
			if item.ToolCall.Kind == llm.ToolKindFunction {
				arguments := decoder.argumentsByID[key]
				if json.Valid([]byte(arguments)) {
					item.ToolCall.ArgumentsJSON = append(json.RawMessage(nil), arguments...)
					item.ToolCall.ArgumentsText = ""
				} else {
					item.ToolCall.ArgumentsJSON = nil
					item.ToolCall.ArgumentsText = arguments
				}
			} else {
				item.ToolCall.InputText = decoder.inputByID[key]
			}
		} else if item.Kind == llm.ItemKindMCPCall {
			if !decoder.inputDone[key] {
				if err := add(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: ref}); err != nil {
					return err
				}
				decoder.inputDone[key] = true
			}
			if arguments, ok := decoder.argumentsByID[key]; ok {
				decoder.updateMCPArguments(key, arguments)
			}
		}
		if item.Kind == llm.ItemKindMCPCall || item.Kind == llm.ItemKindMCPListTools {
			mcpStatus := llm.MCPCallStatusCompleted
			if status == llm.ItemStatusFailed {
				mcpStatus = llm.MCPCallStatusFailed
			} else if status != llm.ItemStatusCompleted {
				mcpStatus = llm.MCPCallStatusIncomplete
			}
			if err := add(llm.Event{Kind: llm.EventKindMCPStatus, ItemRef: ref, Delta: llm.Delta{MCPStatus: string(mcpStatus)}}); err != nil {
				return err
			}
			decoder.hostedDone[key] = true
		} else if item.Kind == llm.ItemKindHostedCall {
			hostedStatus := "completed"
			if status != llm.ItemStatusCompleted {
				hostedStatus = "failed"
			}
			if err := add(llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: ref, Delta: llm.Delta{HostedStatus: hostedStatus}}); err != nil {
				return err
			}
			decoder.hostedDone[key] = true
		}

		markItemTerminal(item, status)
		doneEvent := llm.Event{Kind: llm.EventKindItemDone, ItemRef: ref, Snapshot: cloneCanonicalItem(item)}
		doneEvent.ProtocolFrames = decoder.takeProtocolFrames(key)
		if err := add(doneEvent); err != nil {
			return err
		}
		decoder.itemDone[key] = true
	}
	return nil
}

func markItemInProgress(item *llm.Item) {
	if item == nil {
		return
	}
	item.Status = llm.ItemStatusInProgress
	if item.ToolCall != nil {
		item.ToolCall.Status = llm.ToolCallStatusInProgress
	}
	if item.HostedCall != nil {
		item.HostedCall.Invocation.Status = llm.ToolCallStatusInProgress
		item.HostedCall.Result = nil
	}
	if item.MCPCall != nil {
		item.MCPCall.Output = ""
		item.MCPCall.Error = ""
		item.MCPCall.Status = llm.MCPCallStatusInProgress
	}
}

func markItemTerminal(item *llm.Item, status llm.ItemStatus) {
	if item == nil {
		return
	}
	item.Status = status
	if item.ToolCall != nil {
		switch status {
		case llm.ItemStatusCompleted:
			item.ToolCall.Status = llm.ToolCallStatusCompleted
		case llm.ItemStatusFailed:
			item.ToolCall.Status = llm.ToolCallStatusFailed
		default:
			item.ToolCall.Status = llm.ToolCallStatusIncomplete
		}
	}
	if item.HostedCall != nil {
		switch status {
		case llm.ItemStatusCompleted:
			item.HostedCall.Invocation.Status = llm.ToolCallStatusCompleted
		case llm.ItemStatusFailed:
			item.HostedCall.Invocation.Status = llm.ToolCallStatusFailed
		default:
			item.HostedCall.Invocation.Status = llm.ToolCallStatusIncomplete
		}
	}
	if item.MCPCall != nil {
		switch status {
		case llm.ItemStatusCompleted:
			item.MCPCall.Status = llm.MCPCallStatusCompleted
		case llm.ItemStatusFailed:
			item.MCPCall.Status = llm.MCPCallStatusFailed
		default:
			item.MCPCall.Status = llm.MCPCallStatusIncomplete
		}
	}
}

func cloneCanonicalItem(item *llm.Item) *llm.Item {
	if item == nil {
		return nil
	}
	clone := *item
	clone.ProtocolHints.SourceResidual = cloneRaw(item.ProtocolHints.SourceResidual)
	clone.Content = cloneCanonicalContent(item.Content)
	if item.ToolCall != nil {
		call := *item.ToolCall
		call.ArgumentsJSON = append(json.RawMessage(nil), item.ToolCall.ArgumentsJSON...)
		call.ProviderData = append(json.RawMessage(nil), item.ToolCall.ProviderData...)
		call.PendingSafetyChecks = cloneCanonicalSafetyChecks(item.ToolCall.PendingSafetyChecks)
		clone.ToolCall = &call
	}
	if item.ToolResult != nil {
		result := cloneCanonicalToolResult(item.ToolResult)
		clone.ToolResult = result
	}
	if item.HostedCall != nil {
		hosted := *item.HostedCall
		hosted.Invocation.ArgumentsJSON = append(json.RawMessage(nil), item.HostedCall.Invocation.ArgumentsJSON...)
		hosted.Invocation.ProviderData = append(json.RawMessage(nil), item.HostedCall.Invocation.ProviderData...)
		hosted.Invocation.PendingSafetyChecks = cloneCanonicalSafetyChecks(item.HostedCall.Invocation.PendingSafetyChecks)
		hosted.Result = cloneCanonicalToolResult(item.HostedCall.Result)
		clone.HostedCall = &hosted
	}
	if item.MCPListTools != nil {
		list := *item.MCPListTools
		list.Tools = append([]llm.MCPDiscoveredTool(nil), item.MCPListTools.Tools...)
		for index := range list.Tools {
			list.Tools[index].InputSchema = append(json.RawMessage(nil), item.MCPListTools.Tools[index].InputSchema...)
			list.Tools[index].OutputSchema = append(json.RawMessage(nil), item.MCPListTools.Tools[index].OutputSchema...)
			list.Tools[index].Annotations = append(json.RawMessage(nil), item.MCPListTools.Tools[index].Annotations...)
			list.Tools[index].Meta = append(json.RawMessage(nil), item.MCPListTools.Tools[index].Meta...)
			list.Tools[index].SourceResidual = cloneRaw(item.MCPListTools.Tools[index].SourceResidual)
		}
		clone.MCPListTools = &list
	}
	if item.MCPApprovalRequest != nil {
		request := *item.MCPApprovalRequest
		request.ArgumentsJSON = append(json.RawMessage(nil), item.MCPApprovalRequest.ArgumentsJSON...)
		clone.MCPApprovalRequest = &request
	}
	if item.MCPApprovalResponse != nil {
		response := *item.MCPApprovalResponse
		clone.MCPApprovalResponse = &response
	}
	if item.MCPCall != nil {
		call := *item.MCPCall
		call.ArgumentsJSON = append(json.RawMessage(nil), item.MCPCall.ArgumentsJSON...)
		clone.MCPCall = &call
	}
	if item.Reasoning != nil {
		reasoning := *item.Reasoning
		reasoning.SummaryParts = cloneCanonicalReasoningParts(item.Reasoning.SummaryParts)
		reasoning.ContentParts = cloneCanonicalReasoningParts(item.Reasoning.ContentParts)
		clone.Reasoning = &reasoning
	}
	if item.AgentMessage != nil {
		message := *item.AgentMessage
		message.Content = append([]llm.AgentMessageContentPart(nil), item.AgentMessage.Content...)
		for index := range message.Content {
			message.Content[index].SourceResidual = cloneRaw(item.AgentMessage.Content[index].SourceResidual)
		}
		clone.AgentMessage = &message
	}
	if item.Compaction != nil {
		compaction := *item.Compaction
		clone.Compaction = &compaction
	}
	if item.ContextCompaction != nil {
		compaction := *item.ContextCompaction
		compaction.EncryptedContent = stringPointerClone(item.ContextCompaction.EncryptedContent)
		clone.ContextCompaction = &compaction
	}
	if item.Unknown != nil {
		unknown := *item.Unknown
		unknown.Raw = append(json.RawMessage(nil), item.Unknown.Raw...)
		clone.Unknown = &unknown
	}
	return &clone
}

func cloneCanonicalToolResult(result *llm.ToolResult) *llm.ToolResult {
	if result == nil {
		return nil
	}
	clone := *result
	clone.Content = cloneCanonicalContent(result.Content)
	clone.StructuredContent = append(json.RawMessage(nil), result.StructuredContent...)
	clone.ProviderData = append(json.RawMessage(nil), result.ProviderData...)
	if len(result.DiscoveredTools) > 0 {
		clone.DiscoveredTools = (&llm.Request{ToolDefinitions: result.DiscoveredTools}).Clone().ToolDefinitions
	}
	clone.AcknowledgedSafetyChecks = cloneCanonicalSafetyChecks(result.AcknowledgedSafetyChecks)
	return &clone
}

func cloneCanonicalSafetyChecks(checks []llm.ToolSafetyCheck) []llm.ToolSafetyCheck {
	if len(checks) == 0 {
		return nil
	}
	clone := append([]llm.ToolSafetyCheck(nil), checks...)
	for index := range clone {
		clone[index].SourceResidual = cloneRaw(checks[index].SourceResidual)
	}
	return clone
}

func cloneCanonicalReasoningParts(parts []llm.ReasoningPart) []llm.ReasoningPart {
	if len(parts) == 0 {
		return nil
	}
	clone := append([]llm.ReasoningPart(nil), parts...)
	for index := range clone {
		clone[index].SourceResidual = cloneRaw(parts[index].SourceResidual)
	}
	return clone
}

func cloneCanonicalContent(content []llm.ContentBlock) []llm.ContentBlock {
	if len(content) == 0 {
		return nil
	}
	clone := append([]llm.ContentBlock(nil), content...)
	for index := range clone {
		clone[index].UnknownRaw = append(json.RawMessage(nil), content[index].UnknownRaw...)
		clone[index].SourceResidual = cloneRaw(content[index].SourceResidual)
		if content[index].Image != nil {
			image := *content[index].Image
			clone[index].Image = &image
		}
		if content[index].Audio != nil {
			audio := *content[index].Audio
			clone[index].Audio = &audio
		}
		if content[index].Document != nil {
			document := *content[index].Document
			clone[index].Document = &document
		}
		if content[index].Citation != nil {
			citation := *content[index].Citation
			citation.SourceResidual = cloneRaw(content[index].Citation.SourceResidual)
			clone[index].Citation = &citation
		}
	}
	return clone
}

func responsesStreamError(wire *StreamEvent) *llm.ResponseError {
	detail := llm.ErrorDetail{Type: "server_error", Code: wire.Code, Message: wire.Message}
	if wire.Response != nil && wire.Response.Error != nil {
		if wire.Response.Error.Type != "" {
			detail.Type = wire.Response.Error.Type
		}
		if wire.Response.Error.Code != "" {
			detail.Code = wire.Response.Error.Code
		}
		if wire.Response.Error.Message != "" {
			detail.Message = wire.Response.Error.Message
		}
	}
	return &llm.ResponseError{Detail: detail}
}

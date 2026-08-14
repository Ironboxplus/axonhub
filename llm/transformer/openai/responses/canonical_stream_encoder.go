package responses

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
)

// canonicalStreamEncoder writes a canonical lifecycle directly as Responses
// wire events. It does not inspect the legacy Chat Choice projection.
type canonicalStreamEncoder struct {
	machine        *llm.StreamStateMachine
	items          map[string]Item
	itemKinds      map[string]llm.ItemKind
	itemIDs        map[string]string
	outputIndexes  map[string]int
	outputByIndex  map[int]Item
	arguments      map[string]*strings.Builder
	inputs         map[string]*strings.Builder
	contentStarted map[string]bool
	reasonStarted  map[string]bool
	deferredAdded  map[string]bool
	nextOutput     int
	usage          *llm.Usage
}

func newCanonicalStreamEncoder() *canonicalStreamEncoder {
	return &canonicalStreamEncoder{
		machine:        llm.NewStreamStateMachine(),
		items:          make(map[string]Item),
		itemKinds:      make(map[string]llm.ItemKind),
		itemIDs:        make(map[string]string),
		outputIndexes:  make(map[string]int),
		outputByIndex:  make(map[int]Item),
		arguments:      make(map[string]*strings.Builder),
		inputs:         make(map[string]*strings.Builder),
		contentStarted: make(map[string]bool),
		reasonStarted:  make(map[string]bool),
		deferredAdded:  make(map[string]bool),
	}
}

func (encoder *canonicalStreamEncoder) encode(source *responsesInboundStream, event llm.Event) error {
	if err := encoder.machine.Apply(event); err != nil {
		return fmt.Errorf("encode canonical %s as Responses: %w", event.Kind, err)
	}
	emit := func(wire *StreamEvent) error {
		if event.SourceType == string(wire.Type) {
			wire.Residual = cloneRaw(event.SourceResidual)
			if wire.Response != nil {
				wire.Response.Residual = cloneRaw(event.ResponseSourceResidual)
			}
		}
		applyProtocolFrameHints(wire, event.ProtocolFrames)
		if err := source.enqueueEvent(wire); err != nil {
			return fmt.Errorf("emit Responses %s: %w", wire.Type, err)
		}
		return nil
	}

	switch event.Kind {
	case llm.EventKindResponseStarted:
		source.hasStarted = true
		source.hasResponseCreated = true
		return emit(&StreamEvent{Type: StreamEventTypeResponseCreated, Response: source.canonicalResponse("in_progress", encoder.output())})
	case llm.EventKindResponseInProgress:
		return emit(&StreamEvent{Type: StreamEventTypeResponseInProgress, Response: source.canonicalResponse("in_progress", encoder.output())})
	case llm.EventKindItemAdded:
		return encoder.addItem(source, event, emit)
	case llm.EventKindTextDelta, llm.EventKindRefusalDelta:
		return encoder.textDelta(event, emit)
	case llm.EventKindReasoningDelta:
		return encoder.reasoningDelta(event, emit)
	case llm.EventKindToolInputDelta:
		return encoder.toolInputDelta(event, emit)
	case llm.EventKindToolInputDone:
		return encoder.toolInputDone(event, emit)
	case llm.EventKindHostedStatus:
		return encoder.hostedStatus(event, emit)
	case llm.EventKindMCPStatus:
		return encoder.mcpStatus(event, emit)
	case llm.EventKindItemDone:
		return encoder.doneItem(event, emit)
	case llm.EventKindUsage:
		encoder.usage = event.Usage
		source.usage = event.Usage
		return nil
	case llm.EventKindResponseCompleted:
		source.hasFinished = true
		source.responseCompleted = true
		return emit(&StreamEvent{Type: StreamEventTypeResponseCompleted, Response: source.canonicalResponse("completed", encoder.output())})
	case llm.EventKindResponseIncomplete:
		source.hasFinished = true
		source.responseCompleted = true
		return emit(&StreamEvent{Type: StreamEventTypeResponseIncomplete, Response: source.canonicalResponse("incomplete", encoder.output())})
	case llm.EventKindResponseCancelled:
		source.hasFinished = true
		source.responseCompleted = true
		return emit(&StreamEvent{Type: StreamEventTypeResponseCancelled, Response: source.canonicalResponse("cancelled", encoder.output())})
	case llm.EventKindResponseFailed:
		source.hasFinished = true
		source.responseCompleted = true
		response := source.canonicalResponse("failed", encoder.output())
		response.Error = responsesWireError(event.Error)
		return emit(&StreamEvent{Type: StreamEventTypeResponseFailed, Response: response})
	case llm.EventKindError:
		return emit(&StreamEvent{
			Type: StreamEventTypeError, Code: event.Error.Detail.Code,
			Message: event.Error.Detail.Message, Param: lo.ToPtr(event.Error.Detail.Param),
		})
	default:
		return fmt.Errorf("canonical event %q has no Responses stream encoding", event.Kind)
	}
}

func applyProtocolFrameHints(wire *StreamEvent, frames []llm.ProtocolFrameHint) {
	if wire == nil || len(frames) == 0 {
		return
	}
	for index := range frames {
		frame := &frames[index]
		if frame.SourceType != string(wire.Type) {
			continue
		}
		wire.Residual = cloneRaw(frame.SourceResidual)
		if wire.Part != nil {
			wire.Part.Residual = cloneRaw(frame.PayloadResidual)
		}
		return
	}
}

func hasProtocolFrame(frames []llm.ProtocolFrameHint, eventType StreamEventType) bool {
	for index := range frames {
		if frames[index].SourceType == string(eventType) {
			return true
		}
	}
	return false
}

func (encoder *canonicalStreamEncoder) addItem(source *responsesInboundStream, event llm.Event, emit func(*StreamEvent) error) error {
	key := canonicalEventRefKey(event.ItemRef)
	if key == "" {
		return fmt.Errorf("canonical item_added has no stable reference")
	}
	wire, ok := canonicalItemToResponses(event.Snapshot)
	if !ok && event.Snapshot != nil && event.Snapshot.Kind == llm.ItemKindToolCall &&
		event.Snapshot.ToolCall != nil && event.Snapshot.ToolCall.Kind == llm.ToolKindLocalShell &&
		len(event.Snapshot.ToolCall.ArgumentsJSON) == 0 && event.Snapshot.ToolCall.ArgumentsText == "" {
		// Chat and Anthropic announce a tool call before its streamed JSON input
		// exists. Responses has no local-shell argument delta events, so expose a
		// legal in-progress shell item now and replace its action in item.done.
		wire = Item{
			ID: event.Snapshot.ID, Type: "local_shell_call",
			CallID: event.Snapshot.ToolCall.CallID,
			Action: NewLocalShellAction(&LocalShellAction{Type: "exec", Command: []string{}}),
			Status: responsesStatus(event.Snapshot.Status),
		}
		ok = true
	}
	if !ok {
		return fmt.Errorf("canonical item kind %q has no Responses item encoding", event.Snapshot.Kind)
	}
	outputIndex := encoder.outputIndex(key, event.ItemRef.OutputIndex)
	itemID := wire.ID
	// Opaque same-Responses identity keeps the whole union in Raw, so its
	// convenience Item.ID is intentionally empty. The canonical snapshot still
	// carries the typed structural ID needed by the lifecycle index; do not lose
	// it before item.done merely because the wire encoder correctly avoids
	// rebuilding the opaque object.
	if itemID == "" && event.Snapshot != nil {
		itemID = event.Snapshot.ID
	}
	if itemID == "" && event.Snapshot.ProtocolHints.SourceFormat != llm.APIFormatOpenAIResponse {
		itemID = generateItemID()
		wire.ID = itemID
	}
	referenceID := itemID
	if referenceID == "" {
		referenceID = event.ItemRef.CallID
	}
	encoder.itemIDs[key] = referenceID
	encoder.itemKinds[key] = event.Snapshot.Kind
	// A Responses stream may legally omit status on an output_item.added
	// snapshot. Preserve every same-Responses source snapshot: Codex's current
	// ev_message_item_added fixture does exactly that, and its SSE decoder
	// accepts the item without synthesizing a status. The canonical lifecycle
	// still records ItemAdded internally. Only a snapshot synthesized from a
	// different protocol needs an in-progress lifecycle projection.
	if event.Snapshot == nil || event.Snapshot.ProtocolHints.SourceFormat != llm.APIFormatOpenAIResponse {
		markResponsesWireItemInProgress(&wire)
	}
	// Same-protocol identity preserves provider-owned values and lifecycle
	// omissions, but it must not forward a structurally invalid Responses
	// output item. Strict SDKs require annotations on every output_text part,
	// including the item snapshot carried by output_item.added.
	ensureResponsesOutputAnnotations(&wire)
	encoder.items[key] = wire
	if event.Snapshot.Kind == llm.ItemKindToolCall || event.Snapshot.Kind == llm.ItemKindMCPCall {
		encoder.arguments[key] = &strings.Builder{}
		encoder.inputs[key] = &strings.Builder{}
	}
	if event.Snapshot.Kind == llm.ItemKindToolCall && event.Snapshot.ToolCall != nil &&
		(event.Snapshot.ToolCall.Kind == llm.ToolKindComputer || event.Snapshot.ToolCall.Kind == llm.ToolKindApplyPatch) &&
		len(event.Snapshot.ToolCall.ArgumentsJSON) == 0 && event.Snapshot.ToolCall.ArgumentsText == "" {
		encoder.deferredAdded[key] = true
		return nil
	}
	return emit(&StreamEvent{Type: StreamEventTypeOutputItemAdded, OutputIndex: outputIndex, Item: &wire})
}

func (encoder *canonicalStreamEncoder) textDelta(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	contentIndex := intValue(event.ContentIndex)
	if !encoder.contentStarted[key] {
		encoder.contentStarted[key] = true
		partType := "output_text"
		if event.Kind == llm.EventKindRefusalDelta {
			partType = "refusal"
		}
		if err := emit(&StreamEvent{
			Type: StreamEventTypeContentPartAdded, ItemID: &itemID, OutputIndex: outputIndex,
			ContentIndex: &contentIndex, Part: &StreamEventContentPart{Type: partType},
		}); err != nil {
			return err
		}
	}
	eventType := StreamEventTypeOutputTextDelta
	if event.Kind == llm.EventKindRefusalDelta {
		eventType = StreamEventTypeRefusalDelta
	}
	return emit(&StreamEvent{Type: eventType, ItemID: &itemID, OutputIndex: outputIndex, ContentIndex: &contentIndex, Delta: event.Delta.Text})
}

func (encoder *canonicalStreamEncoder) reasoningDelta(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	summaryIndex := intValue(event.ContentIndex)
	if !encoder.reasonStarted[key] {
		encoder.reasonStarted[key] = true
		if err := emit(&StreamEvent{
			Type: StreamEventTypeReasoningSummaryPartAdded, ItemID: &itemID,
			OutputIndex: outputIndex, SummaryIndex: &summaryIndex,
			Part: &StreamEventContentPart{Type: "summary_text"},
		}); err != nil {
			return err
		}
	}
	return emit(&StreamEvent{
		Type: StreamEventTypeReasoningSummaryTextDelta, ItemID: &itemID,
		OutputIndex: outputIndex, SummaryIndex: &summaryIndex, Delta: event.Delta.Text,
	})
}

func (encoder *canonicalStreamEncoder) toolInputDelta(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	wire := encoder.items[key]
	if wire.Type == "local_shell_call" || wire.Type == "tool_search_call" || wire.Type == "computer_call" || wire.Type == "apply_patch_call" {
		// Responses models local-shell actions and tool-search arguments on the
		// output item itself and defines no function_call_arguments event for
		// either. Keep provider fragments private; item.done carries the value.
		encoder.arguments[key].WriteString(event.Delta.ArgumentsJSON)
		return nil
	}
	if wire.Type == "mcp_call" {
		encoder.arguments[key].WriteString(event.Delta.ArgumentsJSON)
		return emit(&StreamEvent{Type: StreamEventTypeMCPCallArgumentsDelta, ItemID: &itemID, OutputIndex: outputIndex, Delta: event.Delta.ArgumentsJSON})
	}
	if wire.Type == "custom_tool_call" {
		encoder.inputs[key].WriteString(event.Delta.InputText)
		return emit(&StreamEvent{Type: StreamEventTypeCustomToolCallInputDelta, ItemID: &itemID, OutputIndex: outputIndex, Delta: event.Delta.InputText})
	}
	encoder.arguments[key].WriteString(event.Delta.ArgumentsJSON)
	return emit(&StreamEvent{Type: StreamEventTypeFunctionCallArgumentsDelta, ItemID: &itemID, OutputIndex: outputIndex, Delta: event.Delta.ArgumentsJSON})
}

func (encoder *canonicalStreamEncoder) toolInputDone(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	wire := encoder.items[key]
	if wire.Type == "local_shell_call" || wire.Type == "tool_search_call" || wire.Type == "computer_call" || wire.Type == "apply_patch_call" {
		return nil
	}
	if wire.Type == "mcp_call" {
		return emit(&StreamEvent{
			Type: StreamEventTypeMCPCallArgumentsDone, ItemID: &itemID,
			OutputIndex: outputIndex, Arguments: encoder.arguments[key].String(),
		})
	}
	if wire.Type == "custom_tool_call" {
		input := encoder.inputs[key].String()
		return emit(&StreamEvent{Type: StreamEventTypeCustomToolCallInputDone, ItemID: &itemID, OutputIndex: outputIndex, Input: input})
	}
	return emit(&StreamEvent{
		Type: StreamEventTypeFunctionCallArgumentsDone, ItemID: &itemID, OutputIndex: outputIndex,
		CallID: wire.CallID, Name: wire.Name, Namespace: wire.Namespace, Arguments: encoder.arguments[key].String(),
	})
}

func (encoder *canonicalStreamEncoder) hostedStatus(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	if encoder.items[key].Type != "image_generation_call" {
		// Responses exposes web-search lifecycle through output_item.added/done;
		// there is no separate web-search status event to synthesize.
		return nil
	}
	eventType := StreamEventTypeImageGenerationInProgress
	switch event.Delta.HostedStatus {
	case "generating":
		eventType = StreamEventTypeImageGenerationGenerating
	case "completed", "failed":
		eventType = StreamEventTypeImageGenerationCompleted
	}
	return emit(&StreamEvent{Type: eventType, ItemID: &itemID, OutputIndex: outputIndex})
}

func (encoder *canonicalStreamEncoder) mcpStatus(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	wireType := encoder.items[key].Type
	var eventType StreamEventType
	switch wireType {
	case "mcp_call":
		switch llm.MCPCallStatus(event.Delta.MCPStatus) {
		case llm.MCPCallStatusCalling, llm.MCPCallStatusInProgress:
			eventType = StreamEventTypeMCPCallInProgress
		case llm.MCPCallStatusCompleted:
			eventType = StreamEventTypeMCPCallCompleted
		case llm.MCPCallStatusFailed:
			eventType = StreamEventTypeMCPCallFailed
		case llm.MCPCallStatusIncomplete:
			return nil
		}
	case "mcp_list_tools":
		switch llm.MCPCallStatus(event.Delta.MCPStatus) {
		case llm.MCPCallStatusInProgress:
			eventType = StreamEventTypeMCPListToolsInProgress
		case llm.MCPCallStatusCompleted:
			eventType = StreamEventTypeMCPListToolsCompleted
		case llm.MCPCallStatusFailed:
			eventType = StreamEventTypeMCPListToolsFailed
		case llm.MCPCallStatusIncomplete:
			return nil
		}
	default:
		return fmt.Errorf("canonical MCP status targets Responses item type %q", wireType)
	}
	if eventType == "" {
		return fmt.Errorf("canonical MCP status %q has no Responses encoding", event.Delta.MCPStatus)
	}
	return emit(&StreamEvent{Type: eventType, ItemID: &itemID, OutputIndex: outputIndex})
}

func (encoder *canonicalStreamEncoder) doneItem(event llm.Event, emit func(*StreamEvent) error) error {
	key, itemID, outputIndex, err := encoder.eventCoordinates(event)
	if err != nil {
		return err
	}
	wire, ok := canonicalItemToResponses(event.Snapshot)
	if !ok {
		return fmt.Errorf("canonical item kind %q has no Responses item encoding", event.Snapshot.Kind)
	}
	if wire.ID == "" && event.Snapshot.ProtocolHints.SourceFormat != llm.APIFormatOpenAIResponse {
		wire.ID = itemID
	}
	ensureResponsesOutputAnnotations(&wire)
	if encoder.deferredAdded[key] {
		added := wire
		markResponsesWireItemInProgress(&added)
		if err := emit(&StreamEvent{Type: StreamEventTypeOutputItemAdded, OutputIndex: outputIndex, Item: &added}); err != nil {
			return err
		}
		delete(encoder.deferredAdded, key)
	}
	if !encoder.contentStarted[key] && hasProtocolFrame(event.ProtocolFrames, StreamEventTypeContentPartAdded) {
		encoder.contentStarted[key] = true
		contentIndex := intValue(event.ContentIndex)
		partType := "output_text"
		if len(event.Snapshot.Content) > 0 && event.Snapshot.Content[0].Kind == llm.ContentKindRefusal {
			partType = "refusal"
		}
		if err := emit(&StreamEvent{
			Type: StreamEventTypeContentPartAdded, ItemID: &itemID, OutputIndex: outputIndex,
			ContentIndex: &contentIndex, Part: &StreamEventContentPart{Type: partType},
		}); err != nil {
			return err
		}
	}
	if !encoder.reasonStarted[key] && hasProtocolFrame(event.ProtocolFrames, StreamEventTypeReasoningSummaryPartAdded) {
		encoder.reasonStarted[key] = true
		summaryIndex := intValue(event.ContentIndex)
		if err := emit(&StreamEvent{
			Type: StreamEventTypeReasoningSummaryPartAdded, ItemID: &itemID,
			OutputIndex: outputIndex, SummaryIndex: &summaryIndex,
			Part: &StreamEventContentPart{Type: "summary_text"},
		}); err != nil {
			return err
		}
	}
	if encoder.contentStarted[key] {
		contentIndex := intValue(event.ContentIndex)
		text := canonicalText(event.Snapshot.Content)
		doneType := StreamEventTypeOutputTextDone
		if len(event.Snapshot.Content) > 0 && event.Snapshot.Content[0].Kind == llm.ContentKindRefusal {
			doneType = StreamEventTypeRefusalDone
		}
		if err := emit(&StreamEvent{Type: doneType, ItemID: &itemID, OutputIndex: outputIndex, ContentIndex: &contentIndex, Text: text}); err != nil {
			return err
		}
		part := StreamEventContentPart{Type: "output_text", Text: text}
		if doneType == StreamEventTypeRefusalDone {
			part.Type = "refusal"
			part.Refusal = &text
		}
		if err := emit(&StreamEvent{Type: StreamEventTypeContentPartDone, ItemID: &itemID, OutputIndex: outputIndex, ContentIndex: &contentIndex, Part: &part}); err != nil {
			return err
		}
	}
	if encoder.reasonStarted[key] {
		summaryIndex := 0
		text := ""
		if event.Snapshot.Reasoning != nil {
			text = event.Snapshot.Reasoning.Content
		}
		if err := emit(&StreamEvent{Type: StreamEventTypeReasoningSummaryTextDone, ItemID: &itemID, OutputIndex: outputIndex, SummaryIndex: &summaryIndex, Text: text}); err != nil {
			return err
		}
		if err := emit(&StreamEvent{Type: StreamEventTypeReasoningSummaryPartDone, ItemID: &itemID, OutputIndex: outputIndex, SummaryIndex: &summaryIndex, Part: &StreamEventContentPart{Type: "summary_text", Text: text}}); err != nil {
			return err
		}
	}
	encoder.items[key] = wire
	encoder.outputByIndex[outputIndex] = wire
	if err := emit(&StreamEvent{Type: StreamEventTypeOutputItemDone, OutputIndex: outputIndex, Item: &wire}); err != nil {
		return err
	}
	if event.Snapshot.Kind == llm.ItemKindHostedCall && event.Snapshot.HostedCall != nil && event.Snapshot.HostedCall.Result != nil {
		result, ok := canonicalHostedResultToResponses(event.Snapshot.HostedCall.Invocation.Kind, event.Snapshot.HostedCall.Result)
		if !ok {
			return nil
		}
		if result.ID == "" {
			result.ID = generateItemID()
		}
		resultIndex := encoder.nextOutput
		encoder.nextOutput++
		added := result
		markResponsesWireItemInProgress(&added)
		if err := emit(&StreamEvent{Type: StreamEventTypeOutputItemAdded, OutputIndex: resultIndex, Item: &added}); err != nil {
			return err
		}
		encoder.outputByIndex[resultIndex] = result
		return emit(&StreamEvent{Type: StreamEventTypeOutputItemDone, OutputIndex: resultIndex, Item: &result})
	}
	return nil
}

func (encoder *canonicalStreamEncoder) eventCoordinates(event llm.Event) (string, string, int, error) {
	key := canonicalEventRefKey(event.ItemRef)
	itemID, exists := encoder.itemIDs[key]
	if !exists || itemID == "" {
		return "", "", 0, fmt.Errorf("canonical %s references unknown item %q", event.Kind, key)
	}
	outputIndex, exists := encoder.outputIndexes[key]
	if !exists {
		return "", "", 0, fmt.Errorf("canonical %s has no output index for %q", event.Kind, key)
	}
	return key, itemID, outputIndex, nil
}

func (encoder *canonicalStreamEncoder) outputIndex(key string, preferred *int) int {
	if existing, ok := encoder.outputIndexes[key]; ok {
		return existing
	}
	index := encoder.nextOutput
	if preferred != nil {
		index = *preferred
	}
	encoder.outputIndexes[key] = index
	if index >= encoder.nextOutput {
		encoder.nextOutput = index + 1
	}
	return index
}

func (encoder *canonicalStreamEncoder) output() []Item {
	if len(encoder.outputByIndex) == 0 {
		return []Item{}
	}
	indexes := make([]int, 0, len(encoder.outputByIndex))
	for index := range encoder.outputByIndex {
		indexes = append(indexes, index)
	}
	sort.Ints(indexes)
	output := make([]Item, 0, len(indexes))
	for _, index := range indexes {
		output = append(output, encoder.outputByIndex[index])
	}
	return output
}

func (source *responsesInboundStream) canonicalResponse(status string, output []Item) *Response {
	response := &Response{
		Object: "response", ID: source.responseID, Model: source.model,
		CreatedAt: source.createdAt, Status: &status, Output: output,
	}
	if source.usage != nil {
		response.Usage = ConvertLLMUsageToResponsesUsage(source.usage)
	}
	return response
}

func markResponsesWireItemInProgress(item *Item) {
	if item == nil || !responsesItemHasLifecycleStatus(item.Type) {
		return
	}
	status := "in_progress"
	item.Status = &status
	switch item.Type {
	case "function_call":
		item.Arguments = ""
	case "mcp_call":
		item.Arguments = ""
		item.Output = nil
		item.Error = nil
	case "mcp_list_tools":
		item.Tools = []MCPListedTool{}
		item.Error = nil
	case "custom_tool_call":
		item.Input = lo.ToPtr("")
	case "message":
		item.Content = &Input{Items: []Item{}}
	case "reasoning":
		item.Summary = []ReasoningSummary{}
	case "shell_call_output":
		item.RawOutput = json.RawMessage(`[]`)
	}
}

// responsesItemHasLifecycleStatus is deliberately closed. status is not a
// universal Responses item field: current Codex ResponseItem schemas exclude
// agent_message, compaction, context_compaction, and unknown future unions.
// Never synthesize a lifecycle field for those identities merely because the
// canonical stream state is in progress.
func responsesItemHasLifecycleStatus(itemType string) bool {
	switch itemType {
	case "message", "reasoning", "local_shell_call", "function_call", "tool_search_call",
		"tool_search_output", "custom_tool_call", "web_search_call", "image_generation_call",
		"mcp_call", "mcp_list_tools", "shell_call", "computer_call", "apply_patch_call",
		"shell_call_output", "computer_call_output", "apply_patch_call_output":
		return true
	default:
		return false
	}
}

func canonicalEventRefKey(ref llm.ItemRef) string {
	if ref.ItemID != "" {
		return "item:" + ref.ItemID
	}
	if ref.CallID != "" {
		return "call:" + ref.CallID
	}
	if ref.ToolIndex != nil && ref.ChoiceIndex != nil {
		return fmt.Sprintf("choice:%d/tool:%d", *ref.ChoiceIndex, *ref.ToolIndex)
	}
	if ref.OutputIndex != nil {
		return fmt.Sprintf("output:%d", *ref.OutputIndex)
	}
	if ref.ChoiceIndex != nil {
		return fmt.Sprintf("choice:%d", *ref.ChoiceIndex)
	}
	return ""
}

func responsesWireError(responseErr *llm.ResponseError) *Error {
	if responseErr == nil {
		return nil
	}
	return &Error{Type: responseErr.Detail.Type, Code: responseErr.Detail.Code, Message: responseErr.Detail.Message}
}

func intValue(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

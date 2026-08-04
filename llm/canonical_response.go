package llm

import (
	"encoding/json"
	"fmt"
	"sort"
)

// CanonicalResponseAccumulator materializes one validated canonical stream
// into the same Output representation used by a non-streaming response. It is
// request-owned and deliberately contains no locks: one stream iterator is
// the sole writer.
type CanonicalResponseAccumulator struct {
	machine            *StreamStateMachine
	items              map[string]Item
	order              map[string]int
	nextOrder          int
	usage              *Usage
	terminal           EventKind
	terminalReason     string
	id                 string
	model              string
	created            int64
	providerSeen       bool
	providerExtensions *ResponseProviderExtensions
}

func NewCanonicalResponseAccumulator() *CanonicalResponseAccumulator {
	return &CanonicalResponseAccumulator{
		machine: NewStreamStateMachine(),
		items:   make(map[string]Item),
		order:   make(map[string]int),
	}
}

// Observe consumes one unified response exactly once. For streaming
// responses, canonical Events are authoritative. For non-streaming responses,
// Output is already the authoritative materialized form.
func (accumulator *CanonicalResponseAccumulator) Observe(response *Response) error {
	if accumulator == nil || response == nil {
		return nil
	}
	// A protocol decoder may attach the canonical terminal lifecycle to its
	// transport [DONE] object. Ignore only a bare marker; otherwise the stream
	// would appear truncated whenever the provider omitted a separate usage
	// frame.
	if (response == DoneResponse || response.Object == "[DONE]") && len(response.Events) == 0 {
		return nil
	}
	if response.ID != "" {
		accumulator.id = response.ID
		accumulator.providerSeen = true
	}
	if response.Model != "" {
		accumulator.model = response.Model
	}
	if response.Created != 0 {
		accumulator.created = response.Created
	}
	if response.Usage != nil {
		accumulator.usage = cloneCanonicalUsage(response.Usage)
	}
	if response.TerminalReason != "" {
		accumulator.terminalReason = response.TerminalReason
	}
	if response.ProviderExtensions != nil {
		accumulator.providerExtensions = CloneResponseProviderExtensions(response.ProviderExtensions)
	}

	if len(response.Events) == 0 {
		if len(response.Output) > 0 {
			accumulator.replaceOutput(response.Output)
		}
		if response.Error != nil {
			accumulator.terminal = EventKindResponseFailed
		} else if len(response.Output) > 0 {
			accumulator.terminal = EventKindResponseCompleted
		}
		return nil
	}

	for index := range response.Events {
		event := response.Events[index]
		if len(event.ResponseSourceResidual) > 0 {
			accumulator.providerExtensions = &ResponseProviderExtensions{
				OpenAIResponses: &OpenAIResponsesResponseExtensions{
					ResidualFields: append(json.RawMessage(nil), event.ResponseSourceResidual...),
				},
			}
		}
		if err := accumulator.machine.Apply(event); err != nil {
			return fmt.Errorf("accumulate canonical response: %w", err)
		}
		switch event.Kind {
		case EventKindItemAdded, EventKindItemDone:
			key, err := event.ItemRef.StableKey()
			if err != nil {
				return fmt.Errorf("accumulate canonical item: %w", err)
			}
			if _, exists := accumulator.order[key]; !exists {
				ordinal := accumulator.nextOrder
				if event.ItemRef.OutputIndex != nil {
					ordinal = *event.ItemRef.OutputIndex
				}
				accumulator.order[key] = ordinal
				if ordinal >= accumulator.nextOrder {
					accumulator.nextOrder = ordinal + 1
				}
			}
			if event.Snapshot != nil {
				accumulator.items[key] = CloneCanonicalItem(*event.Snapshot)
			}
		case EventKindUsage:
			if event.Usage != nil {
				accumulator.usage = cloneCanonicalUsage(event.Usage)
			}
		case EventKindResponseCompleted, EventKindResponseFailed,
			EventKindResponseIncomplete, EventKindResponseCancelled:
			accumulator.terminal = event.Kind
			accumulator.terminalReason = event.TerminalReason
		}
	}
	return nil
}

func (accumulator *CanonicalResponseAccumulator) replaceOutput(output []Item) {
	accumulator.items = make(map[string]Item, len(output))
	accumulator.order = make(map[string]int, len(output))
	accumulator.nextOrder = len(output)
	for index := range output {
		key := fmt.Sprintf("output:%d", index)
		accumulator.items[key] = CloneCanonicalItem(output[index])
		accumulator.order[key] = index
	}
}

func (accumulator *CanonicalResponseAccumulator) TerminalKind() EventKind {
	if accumulator == nil {
		return ""
	}
	return accumulator.terminal
}

func (accumulator *CanonicalResponseAccumulator) IsTerminal() bool {
	return accumulator != nil && accumulator.terminal != ""
}

// Snapshot returns an isolated non-streaming canonical response. Callers may
// safely persist or mutate it without retaining stream decoder buffers.
func (accumulator *CanonicalResponseAccumulator) Snapshot() *Response {
	if accumulator == nil {
		return nil
	}
	type orderedItem struct {
		key     string
		ordinal int
	}
	ordered := make([]orderedItem, 0, len(accumulator.items))
	for key := range accumulator.items {
		ordered = append(ordered, orderedItem{key: key, ordinal: accumulator.order[key]})
	}
	sort.SliceStable(ordered, func(i, j int) bool {
		if ordered[i].ordinal == ordered[j].ordinal {
			return ordered[i].key < ordered[j].key
		}
		return ordered[i].ordinal < ordered[j].ordinal
	})
	output := make([]Item, 0, len(ordered))
	for _, entry := range ordered {
		output = append(output, CloneCanonicalItem(accumulator.items[entry.key]))
	}
	response := &Response{
		ID: accumulator.id, Model: accumulator.model, Created: accumulator.created,
		Output: output, Usage: cloneCanonicalUsage(accumulator.usage), Status: responseStatusFromTerminal(accumulator.terminal),
		TerminalReason:     accumulator.terminalReason,
		ProviderExtensions: CloneResponseProviderExtensions(accumulator.providerExtensions),
	}
	return response
}

func responseStatusFromTerminal(kind EventKind) ResponseStatus {
	switch kind {
	case EventKindResponseCompleted:
		return ResponseStatusCompleted
	case EventKindResponseIncomplete:
		return ResponseStatusIncomplete
	case EventKindResponseCancelled:
		return ResponseStatusCancelled
	case EventKindResponseFailed:
		return ResponseStatusFailed
	default:
		return ""
	}
}

// CloneCanonicalItem returns a deep attempt-safe copy without JSON
// serialization. It reuses Request.Clone's canonical graph copier.
func CloneCanonicalItem(item Item) Item {
	request := (&Request{Input: []Item{item}}).Clone()
	return request.Input[0]
}

func CloneCanonicalItems(items []Item) []Item {
	if len(items) == 0 {
		return nil
	}
	return (&Request{Input: items}).Clone().Input
}

// CanonicalEventsFromResponse renders a materialized response as one legal
// canonical stream. Gateway emulators use it after consuming private internal
// rounds so client protocol encoders receive only the public lifecycle.
func CanonicalEventsFromResponse(response *Response) ([]Event, error) {
	if response == nil {
		return nil, fmt.Errorf("canonical response is nil")
	}
	events := make([]Event, 0, 3+len(response.Output)*4)
	sequence := uint64(0)
	add := func(event Event) {
		event.Sequence = sequence
		sequence++
		events = append(events, event)
	}
	var responseResidual json.RawMessage
	if response.ProviderExtensions != nil && response.ProviderExtensions.OpenAIResponses != nil {
		responseResidual = append(json.RawMessage(nil), response.ProviderExtensions.OpenAIResponses.ResidualFields...)
	}
	add(Event{Kind: EventKindResponseStarted, ResponseSourceResidual: append(json.RawMessage(nil), responseResidual...)})
	add(Event{Kind: EventKindResponseInProgress, ResponseSourceResidual: append(json.RawMessage(nil), responseResidual...)})
	for index := range response.Output {
		item := CloneCanonicalItem(response.Output[index])
		outputIndex := index
		ref := ItemRef{OutputIndex: &outputIndex, ItemID: item.ID}
		if item.ToolCall != nil {
			ref.CallID = item.ToolCall.CallID
		}
		added := CloneCanonicalItem(item)
		markCanonicalItemInProgress(&added)
		add(Event{Kind: EventKindItemAdded, ItemRef: ref, Snapshot: &added})
		switch item.Kind {
		case ItemKindMessage:
			for contentIndex := range item.Content {
				block := item.Content[contentIndex]
				indexCopy := contentIndex
				switch block.Kind {
				case ContentKindText:
					if block.Text != "" {
						add(Event{Kind: EventKindTextDelta, ItemRef: ref, ContentIndex: &indexCopy, Delta: Delta{Text: block.Text}})
					}
				case ContentKindRefusal:
					if block.Text != "" {
						add(Event{Kind: EventKindRefusalDelta, ItemRef: ref, ContentIndex: &indexCopy, Delta: Delta{Text: block.Text}})
					}
				}
			}
		case ItemKindReasoning:
			if item.Reasoning != nil && item.Reasoning.Content != "" {
				add(Event{Kind: EventKindReasoningDelta, ItemRef: ref, Delta: Delta{Text: item.Reasoning.Content}})
			}
		case ItemKindToolCall:
			if item.ToolCall != nil {
				delta := Delta{ArgumentsJSON: string(item.ToolCall.ArgumentsJSON), InputText: item.ToolCall.InputText}
				if delta.ArgumentsJSON == "" && item.ToolCall.ArgumentsText != "" {
					delta.ArgumentsJSON = item.ToolCall.ArgumentsText
				}
				if delta.ArgumentsJSON != "" || delta.InputText != "" {
					add(Event{Kind: EventKindToolInputDelta, ItemRef: ref, Delta: delta})
				}
				add(Event{Kind: EventKindToolInputDone, ItemRef: ref})
			}
		case ItemKindHostedCall:
			status := "completed"
			if item.Status == ItemStatusFailed {
				status = "failed"
			}
			add(Event{Kind: EventKindHostedStatus, ItemRef: ref, Delta: Delta{HostedStatus: status}})
		case ItemKindMCPListTools:
			add(Event{Kind: EventKindMCPStatus, ItemRef: ref, Delta: Delta{MCPStatus: terminalMCPStatus(item.Status)}})
		case ItemKindMCPCall:
			if item.MCPCall != nil {
				arguments := string(item.MCPCall.ArgumentsJSON)
				if arguments == "" {
					arguments = item.MCPCall.ArgumentsText
				}
				if arguments != "" {
					add(Event{Kind: EventKindToolInputDelta, ItemRef: ref, Delta: Delta{ArgumentsJSON: arguments}})
				}
				add(Event{Kind: EventKindToolInputDone, ItemRef: ref})
				add(Event{Kind: EventKindMCPStatus, ItemRef: ref, Delta: Delta{MCPStatus: terminalMCPStatus(item.Status)}})
			}
		}
		add(Event{Kind: EventKindItemDone, ItemRef: ref, Snapshot: &item})
	}
	if response.Usage != nil {
		add(Event{Kind: EventKindUsage, Usage: cloneCanonicalUsage(response.Usage)})
	}
	terminal := Event{
		Kind: EventKindResponseCompleted, TerminalReason: response.TerminalReason,
		ResponseSourceResidual: append(json.RawMessage(nil), responseResidual...),
	}
	switch response.Status {
	case ResponseStatusIncomplete:
		terminal.Kind = EventKindResponseIncomplete
	case ResponseStatusCancelled:
		terminal.Kind = EventKindResponseCancelled
	case ResponseStatusFailed:
		terminal.Kind = EventKindResponseFailed
		terminal.Error = response.Error
	}
	if terminal.Kind == EventKindResponseFailed && terminal.Error == nil {
		terminal.Error = &ResponseError{Detail: ErrorDetail{Code: "gateway_emulation_failed", Message: "gateway emulation failed"}}
	}
	add(terminal)
	machine := NewStreamStateMachine()
	for index := range events {
		if err := machine.Apply(events[index]); err != nil {
			return nil, fmt.Errorf("render canonical response event %d: %w", index, err)
		}
	}
	return events, nil
}

func cloneCanonicalUsage(source *Usage) *Usage {
	if source == nil {
		return nil
	}
	clone := *source
	if source.PromptTokensDetails != nil {
		details := *source.PromptTokensDetails
		clone.PromptTokensDetails = &details
	}
	if source.CompletionTokensDetails != nil {
		details := *source.CompletionTokensDetails
		clone.CompletionTokensDetails = &details
	}
	if source.ServerToolUsage != nil {
		details := *source.ServerToolUsage
		clone.ServerToolUsage = &details
	}
	clone.PromptModalityTokenDetails = append([]ModalityTokenCount(nil), source.PromptModalityTokenDetails...)
	clone.CompletionModalityTokenDetails = append([]ModalityTokenCount(nil), source.CompletionModalityTokenDetails...)
	return &clone
}

func markCanonicalItemInProgress(item *Item) {
	if item == nil {
		return
	}
	item.Status = ItemStatusInProgress
	if item.ToolCall != nil {
		item.ToolCall.Status = ToolCallStatusInProgress
	}
	if item.HostedCall != nil {
		item.HostedCall.Invocation.Status = ToolCallStatusInProgress
		item.HostedCall.Result = nil
	}
	if item.MCPCall != nil {
		item.MCPCall.Status = MCPCallStatusInProgress
		item.MCPCall.Output = ""
		item.MCPCall.Error = ""
	}
}

func terminalMCPStatus(status ItemStatus) string {
	switch status {
	case ItemStatusFailed:
		return string(MCPCallStatusFailed)
	case ItemStatusIncomplete:
		return string(MCPCallStatusIncomplete)
	default:
		return string(MCPCallStatusCompleted)
	}
}

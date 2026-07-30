package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
)

type anthropicCanonicalDecoder struct {
	machine        *llm.StreamStateMachine
	nextSequence   uint64
	blocks         map[int]*anthropicCanonicalBlock
	hostedByCallID map[string]*anthropicCanonicalBlock
	usage          *llm.Usage
	stopReason     string
}

type anthropicCanonicalBlock struct {
	ref       llm.ItemRef
	snapshot  *llm.Item
	arguments string
	done      bool
	resultFor *anthropicCanonicalBlock
}

func newAnthropicCanonicalDecoder() *anthropicCanonicalDecoder {
	return &anthropicCanonicalDecoder{
		machine: llm.NewStreamStateMachine(), blocks: make(map[int]*anthropicCanonicalBlock),
		hostedByCallID: make(map[string]*anthropicCanonicalBlock),
	}
}

func (decoder *anthropicCanonicalDecoder) decode(wire *StreamEvent, platformType PlatformType) ([]llm.Event, error) {
	if wire == nil {
		return nil, fmt.Errorf("nil Anthropic stream event")
	}
	var events []llm.Event
	emit := decoder.emitter("Anthropic "+wire.Type, &events)
	if wire.Type != "message_start" && wire.Type != "ping" && decoder.machine.ResponseState() == llm.ResponseStreamStateInit {
		// Some Anthropic-compatible providers begin directly with a content
		// block. The omitted message lifecycle is unambiguous, so synthesize it.
		if err := emit(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
			return nil, err
		}
		if err := emit(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
			return nil, err
		}
	}

	switch wire.Type {
	case "message_start":
		if err := emit(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
			return nil, err
		}
		if err := emit(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
			return nil, err
		}
		if wire.Message != nil && wire.Message.Usage != nil {
			decoder.usage = convertToLlmUsage(wire.Message.Usage, platformType)
		}

	case "content_block_start":
		if wire.Index == nil || wire.ContentBlock == nil {
			return nil, fmt.Errorf("Anthropic content_block_start requires index and content_block")
		}
		index := int(*wire.Index)
		block, startEvent, statusEvent, err := decoder.startBlock(index, wire.ContentBlock)
		if err != nil {
			return nil, err
		}
		decoder.blocks[index] = block
		if startEvent != nil {
			if err := emit(*startEvent); err != nil {
				return nil, err
			}
		}
		if statusEvent != nil {
			if err := emit(*statusEvent); err != nil {
				return nil, err
			}
		}

	case "content_block_delta":
		if wire.Index == nil || wire.Delta == nil || wire.Delta.Type == nil {
			return nil, fmt.Errorf("Anthropic content_block_delta requires index and typed delta")
		}
		index := int(*wire.Index)
		block := decoder.blocks[index]
		if block == nil {
			return nil, fmt.Errorf("Anthropic content_block_delta references unknown block %d", index)
		}
		switch *wire.Delta.Type {
		case "text_delta":
			text := lo.FromPtr(wire.Delta.Text)
			appendAnthropicText(block.snapshot, text)
			if err := emit(llm.Event{Kind: llm.EventKindTextDelta, ItemRef: block.ref, ContentIndex: lo.ToPtr(0), Delta: llm.Delta{Text: text}}); err != nil {
				return nil, err
			}
		case "citations_delta":
			if wire.Delta.Citation == nil {
				return nil, fmt.Errorf("Anthropic citations_delta has no citation")
			}
			appendAnthropicCitation(block.snapshot, wire.Delta.Citation)
		case "thinking_delta":
			text := lo.FromPtr(wire.Delta.Thinking)
			if block.snapshot.Reasoning == nil {
				return nil, fmt.Errorf("Anthropic thinking_delta targets non-reasoning block %d", index)
			}
			block.snapshot.Reasoning.Content += text
			if err := emit(llm.Event{Kind: llm.EventKindReasoningDelta, ItemRef: block.ref, Delta: llm.Delta{Text: text}}); err != nil {
				return nil, err
			}
		case "signature_delta":
			if block.snapshot.Reasoning == nil {
				return nil, fmt.Errorf("Anthropic signature_delta targets non-reasoning block %d", index)
			}
			block.snapshot.Reasoning.Signature = lo.FromPtr(wire.Delta.Signature)
		case "input_json_delta":
			fragment := lo.FromPtr(wire.Delta.PartialJSON)
			block.arguments += fragment
			if block.snapshot.Kind == llm.ItemKindHostedCall {
				return events, nil
			}
			if fragment == "" {
				return events, nil
			}
			if err := emit(llm.Event{Kind: llm.EventKindToolInputDelta, ItemRef: block.ref, Delta: llm.Delta{ArgumentsJSON: fragment}}); err != nil {
				return nil, err
			}
		default:
			return nil, fmt.Errorf("unsupported Anthropic content delta type %q", *wire.Delta.Type)
		}

	case "content_block_stop":
		if wire.Index == nil {
			return nil, fmt.Errorf("Anthropic content_block_stop requires index")
		}
		block := decoder.blocks[int(*wire.Index)]
		if block == nil {
			return nil, fmt.Errorf("Anthropic content_block_stop references unknown block %d", *wire.Index)
		}
		if block.resultFor != nil {
			if err := decoder.finishHosted(block.resultFor, llm.ItemStatusCompleted, emit); err != nil {
				return nil, err
			}
			block.done = true
		} else if block.snapshot.Kind == llm.ItemKindHostedCall {
			if err := emit(llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: block.ref, Delta: llm.Delta{HostedStatus: "in_progress"}}); err != nil {
				return nil, err
			}
		} else {
			if err := decoder.finishBlock(block, llm.ItemStatusCompleted, emit); err != nil {
				return nil, err
			}
		}

	case "message_delta":
		if wire.Usage != nil {
			usage := convertToLlmUsage(wire.Usage, platformType)
			if decoder.usage != nil && usage.PromptTokens == 0 {
				usage.PromptTokens = decoder.usage.PromptTokens
				usage.PromptTokensDetails = decoder.usage.PromptTokensDetails
			}
			usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
			decoder.usage = usage
		}
		if wire.Delta != nil && wire.Delta.StopReason != nil {
			decoder.stopReason = *wire.Delta.StopReason
		}

	case "message_stop":
		status := anthropicTerminalItemStatus(decoder.stopReason)
		for _, index := range sortedAnthropicBlockIndexes(decoder.blocks) {
			block := decoder.blocks[index]
			if block == nil || block.done || block.resultFor != nil {
				continue
			}
			if block.snapshot.Kind == llm.ItemKindHostedCall {
				if err := decoder.finishHosted(block, status, emit); err != nil {
					return nil, err
				}
			} else if err := decoder.finishBlock(block, status, emit); err != nil {
				return nil, err
			}
		}
		if decoder.usage != nil {
			if err := emit(llm.Event{Kind: llm.EventKindUsage, Usage: decoder.usage}); err != nil {
				return nil, err
			}
		}
		terminalReason := ""
		if anthropicStopReasonNeedsMetadata(decoder.stopReason) {
			terminalReason = decoder.stopReason
		}
		if err := emit(llm.Event{Kind: anthropicTerminalEvent(decoder.stopReason), TerminalReason: terminalReason}); err != nil {
			return nil, err
		}

	case "ping":
		// Known transport keepalive; it has no semantic event.
	default:
		return nil, fmt.Errorf("unsupported Anthropic stream event type %q", wire.Type)
	}
	return events, nil
}

func (decoder *anthropicCanonicalDecoder) incomplete() ([]llm.Event, error) {
	var events []llm.Event
	emit := decoder.emitter("Anthropic truncated stream", &events)
	for _, index := range sortedAnthropicBlockIndexes(decoder.blocks) {
		block := decoder.blocks[index]
		if block == nil || block.done || block.resultFor != nil {
			continue
		}
		if block.snapshot.Kind == llm.ItemKindHostedCall {
			if err := decoder.finishHosted(block, llm.ItemStatusIncomplete, emit); err != nil {
				return nil, err
			}
		} else if err := decoder.finishBlock(block, llm.ItemStatusIncomplete, emit); err != nil {
			return nil, err
		}
	}
	if decoder.usage != nil {
		if err := emit(llm.Event{Kind: llm.EventKindUsage, Usage: decoder.usage}); err != nil {
			return nil, err
		}
	}
	if err := emit(llm.Event{Kind: llm.EventKindResponseIncomplete}); err != nil {
		return nil, err
	}
	return events, nil
}

func (decoder *anthropicCanonicalDecoder) emitter(label string, events *[]llm.Event) func(llm.Event) error {
	return func(event llm.Event) error {
		event.Sequence = decoder.nextSequence
		decoder.nextSequence++
		if err := decoder.machine.Apply(event); err != nil {
			return fmt.Errorf("%s: %w", label, err)
		}
		*events = append(*events, event)
		return nil
	}
}

func (decoder *anthropicCanonicalDecoder) startBlock(index int, wire *MessageContentBlock) (*anthropicCanonicalBlock, *llm.Event, *llm.Event, error) {
	outputIndex := index
	ref := llm.ItemRef{OutputIndex: &outputIndex, ItemID: wire.ID}
	hints := anthropicHints(wire.Type, 0, index)
	status := llm.ItemStatusInProgress
	var item *llm.Item
	wireRaw, err := json.Marshal(wire)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("marshal Anthropic stream block %q: %w", wire.Type, err)
	}

	switch wire.Type {
	case "text":
		item = &llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Status: status, Content: []llm.ContentBlock{{Kind: llm.ContentKindText}}, ProtocolHints: hints}
	case "thinking", "redacted_thinking":
		item = &llm.Item{Kind: llm.ItemKindReasoning, Role: llm.RoleAssistant, Status: status, Reasoning: &llm.ReasoningItem{ID: wire.ID}, ProtocolHints: hints}
		if wire.Type == "redacted_thinking" {
			item.Reasoning.Signature = wire.Data
		}
	case "tool_use":
		if wire.Name == nil || wire.ID == "" {
			return nil, nil, nil, fmt.Errorf("Anthropic tool_use block requires id and name")
		}
		ref.CallID = wire.ID
		item = &llm.Item{
			Kind: llm.ItemKindToolCall, ID: wire.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, ID: wire.ID, CallID: wire.ID, LogicalName: *wire.Name,
				Status: llm.ToolCallStatusInProgress, Execution: llm.ExecutionOwnerClient,
			},
			ProtocolHints: hints,
		}
	default:
		if isAnthropicSpecialToolUseBlock(wire.Type) {
			callID := wire.ID
			if callID == "" {
				return nil, nil, nil, fmt.Errorf("Anthropic hosted tool block %q has no id", wire.Type)
			}
			name := stringPointerValue(wire.Name)
			if name == "" {
				name = wire.Type
			}
			kind := anthropicToolKind(wire.Type, name)
			ref.CallID = callID
			item = &llm.Item{
				Kind: llm.ItemKindHostedCall, ID: wire.ID, Role: llm.RoleAssistant, Status: status,
				HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{
					Kind: kind, ID: wire.ID, CallID: callID, LogicalName: name,
					Status: llm.ToolCallStatusInProgress, Execution: llm.ExecutionOwnerProvider,
					ProviderData: append(json.RawMessage(nil), wireRaw...),
				}},
				ProtocolHints: hints,
			}
		} else if isAnthropicSpecialToolResultBlock(wire.Type) {
			callID := stringPointerValue(wire.ToolUseID)
			hosted := decoder.hostedByCallID[callID]
			if hosted == nil {
				if callID == "" {
					return nil, nil, nil, fmt.Errorf("Anthropic hosted result %q has no tool_use_id", wire.Type)
				}
				kind := anthropicToolKind(wire.Type, "")
				name := strings.TrimSuffix(strings.TrimSuffix(wire.Type, "_tool_result"), "_result")
				if name == "" {
					name = string(kind)
				}
				resultItem, err := anthropicToolResultItem(wire, wireRaw, 0, index, kind, true)
				if err != nil {
					return nil, nil, nil, err
				}
				ref.CallID = callID
				item := &llm.Item{
					Kind: llm.ItemKindHostedCall, ID: callID, Role: llm.RoleAssistant, Status: llm.ItemStatusInProgress,
					HostedCall: &llm.HostedToolCall{
						Invocation: llm.ToolInvocation{
							Kind: kind, ID: callID, CallID: callID, LogicalName: name,
							Status: llm.ToolCallStatusInProgress, Execution: llm.ExecutionOwnerProvider,
						},
						Result: resultItem.ToolResult,
					},
					ProtocolHints: hints,
				}
				hosted = &anthropicCanonicalBlock{ref: ref, snapshot: item}
				hosted.resultFor = hosted
				decoder.hostedByCallID[callID] = hosted
				start := &llm.Event{Kind: llm.EventKindItemAdded, ItemRef: ref, Snapshot: cloneAnthropicStreamItem(item)}
				statusEvent := &llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: ref, Delta: llm.Delta{HostedStatus: "completed"}}
				return hosted, start, statusEvent, nil
			}
			resultItem, err := anthropicToolResultItem(wire, wireRaw, 0, index, hosted.snapshot.HostedCall.Invocation.Kind, true)
			if err != nil {
				return nil, nil, nil, err
			}
			hosted.snapshot.HostedCall.Result = resultItem.ToolResult
			block := &anthropicCanonicalBlock{ref: hosted.ref, snapshot: hosted.snapshot, resultFor: hosted}
			statusEvent := &llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: hosted.ref, Delta: llm.Delta{HostedStatus: "completed"}}
			return block, nil, statusEvent, nil
		} else {
			return nil, nil, nil, fmt.Errorf("unsupported Anthropic content block type %q", wire.Type)
		}
	}

	block := &anthropicCanonicalBlock{ref: ref, snapshot: item}
	if item.Kind == llm.ItemKindHostedCall {
		decoder.hostedByCallID[ref.CallID] = block
	}
	start := &llm.Event{Kind: llm.EventKindItemAdded, ItemRef: ref, Snapshot: cloneAnthropicStreamItem(item)}
	return block, start, nil, nil
}

func (decoder *anthropicCanonicalDecoder) finishBlock(block *anthropicCanonicalBlock, status llm.ItemStatus, emit func(llm.Event) error) error {
	if block == nil || block.done {
		return nil
	}
	if block.snapshot.Kind == llm.ItemKindToolCall {
		if err := emit(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: block.ref}); err != nil {
			return err
		}
		if json.Valid([]byte(block.arguments)) {
			block.snapshot.ToolCall.ArgumentsJSON = append(json.RawMessage(nil), block.arguments...)
		} else {
			block.snapshot.ToolCall.ArgumentsText = block.arguments
		}
		if status == llm.ItemStatusCompleted {
			block.snapshot.ToolCall.Status = llm.ToolCallStatusCompleted
		} else {
			block.snapshot.ToolCall.Status = llm.ToolCallStatusIncomplete
		}
	}
	block.snapshot.Status = status
	block.done = true
	return emit(llm.Event{Kind: llm.EventKindItemDone, ItemRef: block.ref, Snapshot: cloneAnthropicStreamItem(block.snapshot)})
}

func (decoder *anthropicCanonicalDecoder) finishHosted(block *anthropicCanonicalBlock, status llm.ItemStatus, emit func(llm.Event) error) error {
	if block == nil || block.done {
		return nil
	}
	if block.snapshot.HostedCall == nil {
		return fmt.Errorf("Anthropic hosted block has no hosted payload")
	}
	if len(block.snapshot.HostedCall.Invocation.ArgumentsJSON) == 0 && block.arguments != "" {
		if json.Valid([]byte(block.arguments)) {
			block.snapshot.HostedCall.Invocation.ArgumentsJSON = append(json.RawMessage(nil), block.arguments...)
		} else {
			block.snapshot.HostedCall.Invocation.ArgumentsText = block.arguments
		}
	}
	if status == llm.ItemStatusCompleted {
		block.snapshot.HostedCall.Invocation.Status = llm.ToolCallStatusCompleted
	} else {
		block.snapshot.HostedCall.Invocation.Status = llm.ToolCallStatusIncomplete
	}
	block.snapshot.Status = status
	if block.snapshot.HostedCall.Result == nil {
		if err := emit(llm.Event{Kind: llm.EventKindHostedStatus, ItemRef: block.ref, Delta: llm.Delta{HostedStatus: "failed"}}); err != nil {
			return err
		}
	}
	block.done = true
	return emit(llm.Event{Kind: llm.EventKindItemDone, ItemRef: block.ref, Snapshot: cloneAnthropicStreamItem(block.snapshot)})
}

func appendAnthropicText(item *llm.Item, text string) {
	if item != nil && len(item.Content) > 0 {
		item.Content[0].Text += text
	}
}

func appendAnthropicCitation(item *llm.Item, citation *TextCitation) {
	if item == nil || citation == nil {
		return
	}
	item.Content = append(item.Content, llm.ContentBlock{
		Kind: llm.ContentKindCitation,
		Citation: &llm.URLCitation{
			URL: citation.URL, Title: citation.Title, EncryptedIndex: citation.EncryptedIndex, CitedText: citation.CitedText,
		},
	})
}

func cloneAnthropicStreamItem(item *llm.Item) *llm.Item {
	if item == nil {
		return nil
	}
	clone := *item
	clone.Content = append([]llm.ContentBlock(nil), item.Content...)
	if item.ToolCall != nil {
		call := *item.ToolCall
		call.ArgumentsJSON = append(json.RawMessage(nil), item.ToolCall.ArgumentsJSON...)
		clone.ToolCall = &call
	}
	if item.HostedCall != nil {
		hosted := *item.HostedCall
		hosted.Invocation.ArgumentsJSON = append(json.RawMessage(nil), item.HostedCall.Invocation.ArgumentsJSON...)
		if item.HostedCall.Result != nil {
			result := *item.HostedCall.Result
			result.Content = append([]llm.ContentBlock(nil), item.HostedCall.Result.Content...)
			result.StructuredContent = append(json.RawMessage(nil), item.HostedCall.Result.StructuredContent...)
			result.ProviderData = append(json.RawMessage(nil), item.HostedCall.Result.ProviderData...)
			hosted.Result = &result
		}
		clone.HostedCall = &hosted
	}
	if item.Reasoning != nil {
		reasoning := *item.Reasoning
		clone.Reasoning = &reasoning
	}
	return &clone
}

func sortedAnthropicBlockIndexes(blocks map[int]*anthropicCanonicalBlock) []int {
	indexes := make([]int, 0, len(blocks))
	for index := range blocks {
		indexes = append(indexes, index)
	}
	for i := 1; i < len(indexes); i++ {
		for j := i; j > 0 && indexes[j] < indexes[j-1]; j-- {
			indexes[j], indexes[j-1] = indexes[j-1], indexes[j]
		}
	}
	return indexes
}

func anthropicTerminalEvent(stopReason string) llm.EventKind {
	switch stopReason {
	case "max_tokens", "pause_turn", "refusal":
		return llm.EventKindResponseIncomplete
	default:
		return llm.EventKindResponseCompleted
	}
}

func anthropicTerminalItemStatus(stopReason string) llm.ItemStatus {
	if anthropicTerminalEvent(stopReason) == llm.EventKindResponseCompleted {
		return llm.ItemStatusCompleted
	}
	return llm.ItemStatusIncomplete
}

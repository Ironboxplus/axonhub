package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
)

// anthropicResponseToCanonical decodes a complete Messages response into the
// same ordered Output model used by the Anthropic stream decoder. Hosted tool
// use/result blocks become one provider-owned lifecycle item.
func anthropicResponseToCanonical(message *Message) ([]llm.Item, error) {
	if message == nil {
		return nil, nil
	}
	status := llm.ItemStatusIncomplete
	if message.StopReason != nil {
		status = anthropicTerminalItemStatus(*message.StopReason)
	}
	output := make([]llm.Item, 0, len(message.Content))
	hostedByCallID := make(map[string]int)
	if len(message.Content) == 0 {
		output = append(output, llm.Item{
			Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Status: status,
			Content:       []llm.ContentBlock{{Kind: llm.ContentKindText}},
			ProtocolHints: anthropicHints("text", 0, 0),
		})
	}
	for blockIndex := range message.Content {
		block := message.Content[blockIndex]
		raw, err := json.Marshal(block)
		if err != nil {
			return nil, fmt.Errorf("marshal Anthropic response block %d: %w", blockIndex, err)
		}
		rawArray, err := json.Marshal([]json.RawMessage{raw})
		if err != nil {
			return nil, fmt.Errorf("marshal Anthropic response block envelope %d: %w", blockIndex, err)
		}
		content := MessageContent{MultipleContent: []MessageContentBlock{block}, Raw: rawArray}
		items, err := anthropicMessageToCanonical(&MessageParam{Role: message.Role, Content: content}, blockIndex, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("decode Anthropic response block %d: %w", blockIndex, err)
		}
		if len(items) == 0 && block.Type == "text" {
			items = append(items, anthropicMessageItem(llm.RoleAssistant, []llm.ContentBlock{{Kind: llm.ContentKindText}}, "text", blockIndex))
		}
		if len(items) != 1 {
			return nil, fmt.Errorf("Anthropic response block %d produced %d canonical items", blockIndex, len(items))
		}
		item := items[0]
		item.ProtocolHints = anthropicHints(block.Type, 0, blockIndex)

		switch {
		case isAnthropicSpecialToolUseBlock(block.Type):
			if item.HostedCall == nil {
				return nil, fmt.Errorf("Anthropic hosted tool block %d has no invocation", blockIndex)
			}
			invocation := item.HostedCall.Invocation
			hostedByCallID[invocation.CallID] = len(output)

		case isAnthropicSpecialToolResultBlock(block.Type):
			if item.HostedCall == nil || item.HostedCall.Result == nil {
				return nil, fmt.Errorf("Anthropic hosted result block %d has no result", blockIndex)
			}
			callID := item.HostedCall.Result.CallID
			if outputIndex, ok := hostedByCallID[callID]; ok {
				output[outputIndex].HostedCall.Result = item.HostedCall.Result
				continue
			}
			hostedByCallID[callID] = len(output)

		case block.Type == "tool_result":
			return nil, fmt.Errorf("Anthropic provider response contains client tool_result block %d", blockIndex)
		}

		markAnthropicCanonicalItemTerminal(&item, status)
		output = append(output, item)
	}
	for index := range output {
		markAnthropicCanonicalItemTerminal(&output[index], status)
	}
	if err := llm.ValidateItemStructure(output); err != nil {
		return nil, fmt.Errorf("invalid canonical Anthropic response: %w", err)
	}
	return output, nil
}

func markAnthropicCanonicalItemTerminal(item *llm.Item, status llm.ItemStatus) {
	if item == nil {
		return
	}
	item.Status = status
	callStatus := llm.ToolCallStatusIncomplete
	if status == llm.ItemStatusCompleted {
		callStatus = llm.ToolCallStatusCompleted
	}
	if item.ToolCall != nil {
		item.ToolCall.Status = callStatus
	}
	if item.HostedCall != nil {
		item.HostedCall.Invocation.Status = callStatus
	}
}

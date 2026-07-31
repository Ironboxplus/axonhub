package anthropic

import (
	"encoding/json"
	"fmt"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
)

// canonicalResponseToAnthropic serializes the ordered canonical output. It is
// the non-streaming counterpart of anthropicCanonicalEncoder; the legacy
// Choices view cannot represent provider-hosted lifecycle items.
func canonicalResponseToAnthropic(response *llm.Response) (*Message, bool, error) {
	if response == nil || len(response.Output) == 0 {
		return nil, false, nil
	}
	message := &Message{
		ID: response.ID, Type: "message", Role: "assistant", Model: response.Model,
	}
	for index := range response.Output {
		item := &response.Output[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			message.Content = append(message.Content, canonicalAnthropicContent(item.Content)...)
		case llm.ItemKindReasoning:
			if item.Reasoning == nil {
				return nil, true, fmt.Errorf("canonical output item %d reasoning payload is missing", index)
			}
			if item.ProtocolHints.SourceFormat == llm.APIFormatAnthropicMessage && item.ProtocolHints.SourceType == "redacted_thinking" {
				message.Content = append(message.Content, MessageContentBlock{Type: "redacted_thinking", Data: item.Reasoning.Signature})
				continue
			}
			thinking, signature := item.Reasoning.Content, item.Reasoning.Signature
			if signature == "" {
				signature = generateSignature()
			}
			message.Content = append(message.Content, MessageContentBlock{Type: "thinking", Thinking: &thinking, Signature: &signature})
		case llm.ItemKindToolCall:
			if item.ToolCall == nil || item.ToolCall.Kind != llm.ToolKindFunction {
				return nil, true, fmt.Errorf("canonical output item %d tool call has no Anthropic encoding", index)
			}
			message.Content = append(message.Content, canonicalAnthropicToolUse(item.ToolCall))
		case llm.ItemKindHostedCall:
			blocks := canonicalAnthropicHostedCall(item.HostedCall)
			if len(blocks) == 0 {
				return nil, true, fmt.Errorf("canonical output item %d hosted call has no Anthropic encoding", index)
			}
			message.Content = append(message.Content, blocks...)
		case llm.ItemKindMCPListTools:
			if item.MCPListTools == nil || item.MCPListTools.ServerLabel == "" {
				return nil, true, fmt.Errorf("canonical output item %d MCP discovery payload is missing", index)
			}
			// Anthropic's MCP response vocabulary exposes actual mcp_tool_use and
			// mcp_tool_result blocks, but has no response block for an internal
			// tools/list discovery operation. Discovery remains in the bounded
			// conversion observation; emitting a fabricated model tool call here
			// would misrepresent what the model invoked.
		case llm.ItemKindMCPCall:
			blocks, err := canonicalAnthropicMCPCall(item)
			if err != nil {
				return nil, true, fmt.Errorf("canonical output item %d MCP call has no Anthropic encoding: %w", index, err)
			}
			message.Content = append(message.Content, blocks...)
		case llm.ItemKindUnknown:
			if item.Unknown == nil || !json.Valid(item.Unknown.Raw) {
				return nil, true, fmt.Errorf("canonical output item %d unknown payload is invalid", index)
			}
			var block MessageContentBlock
			if json.Unmarshal(item.Unknown.Raw, &block) != nil || block.Type == "" {
				return nil, true, fmt.Errorf("canonical output item %d unknown payload has no Anthropic block encoding", index)
			}
			message.Content = append(message.Content, block)
		default:
			return nil, true, fmt.Errorf("canonical output item %d kind %q has no Anthropic encoding", index, item.Kind)
		}
	}
	message.StopReason = canonicalAnthropicStopReason(response, message.Content)
	if stopSequence, ok := response.TransformerMetadata[TransformerMetadataKeyAnthropicStopSequence].(string); ok {
		message.StopSequence = lo.ToPtr(stopSequence)
	}
	if response.Usage != nil {
		message.Usage = convertToAnthropicUsage(response.Usage)
	}
	return message, true, nil
}

func canonicalAnthropicStopReason(response *llm.Response, content []MessageContentBlock) *string {
	if response != nil {
		if response.TerminalReason != "" {
			return lo.ToPtr(response.TerminalReason)
		}
		if stopReason, ok := response.TransformerMetadata[TransformerMetadataKeyAnthropicStopReason].(string); ok && stopReason != "" {
			return lo.ToPtr(stopReason)
		}
		if len(response.Choices) > 0 && response.Choices[0].FinishReason != nil {
			switch *response.Choices[0].FinishReason {
			case "stop":
				return lo.ToPtr("end_turn")
			case "length":
				return lo.ToPtr("max_tokens")
			case "tool_calls":
				return lo.ToPtr("tool_use")
			default:
				return response.Choices[0].FinishReason
			}
		}
	}
	for index := range content {
		if content[index].Type == "tool_use" || content[index].Type == "mcp_tool_use" {
			return lo.ToPtr("tool_use")
		}
	}
	return lo.ToPtr("end_turn")
}

// canonicalAnthropicMCPCall projects the typed, completed MCP gateway item to
// Anthropic's paired server-side blocks. The opaque list/discovery step has no
// Anthropic wire equivalent and is deliberately kept in observability rather
// than rendered as a false model invocation.
func canonicalAnthropicMCPCall(item *llm.Item) ([]MessageContentBlock, error) {
	if item == nil || item.ID == "" || item.MCPCall == nil {
		return nil, fmt.Errorf("MCP call item id or payload is missing")
	}
	call := item.MCPCall
	if call.ServerLabel == "" || call.LogicalName == "" {
		return nil, fmt.Errorf("MCP server label or logical name is missing")
	}
	arguments := append(json.RawMessage(nil), call.ArgumentsJSON...)
	if len(arguments) == 0 && call.ArgumentsText != "" && json.Valid([]byte(call.ArgumentsText)) {
		arguments = json.RawMessage(call.ArgumentsText)
	}
	if len(arguments) == 0 || !json.Valid(arguments) {
		arguments = json.RawMessage(`{}`)
	}
	use := MessageContentBlock{
		Type: "mcp_tool_use", ID: item.ID, Name: stringPointerNonNil(call.LogicalName),
		ServerName: call.ServerLabel, Input: arguments,
	}

	resultContent := MessageContent{}
	isError := item.Status == llm.ItemStatusFailed || call.Status == llm.MCPCallStatusFailed || call.Error != ""
	switch {
	case call.Error != "":
		resultContent.Content = stringPointerNonNil(call.Error)
	case json.Valid([]byte(call.Output)) && len(call.Output) > 0 && (call.Output[0] == '{' || call.Output[0] == '['):
		resultContent.SetRaw(append(json.RawMessage(nil), call.Output...))
	case call.Output != "":
		resultContent.Content = stringPointerNonNil(call.Output)
	default:
		resultContent.Content = stringPointerNonNil("")
	}
	return []MessageContentBlock{
		use,
		{Type: "mcp_tool_result", ToolUseID: stringPointerNonNil(item.ID), Content: &resultContent, IsError: &isError},
	}, nil
}

package openai

import (
	"fmt"

	"github.com/looplj/axonhub/llm"
)

// canonicalResponseToChat makes ordered canonical Output authoritative on
// cross-protocol responses. Identity Chat responses keep the compatibility
// Choices view so provider-private fields are not normalized away.
func canonicalResponseToChat(response *llm.Response) (*Response, bool, error) {
	if response == nil || len(response.Output) == 0 {
		return nil, false, nil
	}
	identity := true
	for index := range response.Output {
		if response.Output[index].ProtocolHints.SourceFormat != llm.APIFormatOpenAIChatCompletion {
			identity = false
			break
		}
	}
	if identity {
		return nil, false, nil
	}

	wire := ResponseFromLLM(response)
	message := Message{Role: "assistant"}
	for index := range response.Output {
		item := &response.Output[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			content, annotations, refusal := canonicalChatContent(item.Content)
			mergeCanonicalChatContent(&message.Content, content)
			message.Annotations = append(message.Annotations, annotations...)
			message.Refusal += refusal
		case llm.ItemKindReasoning:
			if item.Reasoning == nil {
				return nil, true, fmt.Errorf("canonical output item %d reasoning payload is missing", index)
			}
			value := item.Reasoning.Content
			mergeCanonicalChatString(&message.ReasoningContent, &value)
		case llm.ItemKindToolCall:
			if item.ToolCall == nil || item.ToolCall.Kind != llm.ToolKindFunction {
				return nil, true, fmt.Errorf("canonical output item %d tool call has no Chat encoding", index)
			}
			call := item.ToolCall
			message.ToolCalls = append(message.ToolCalls, ToolCall{
				Index: len(message.ToolCalls), ID: call.CallID, Type: llm.ToolTypeFunction,
				Function: FunctionCall{Name: call.LogicalName, Arguments: canonicalChatArguments(call)},
			})
		default:
			return nil, true, fmt.Errorf("canonical output item %d kind %q has no Chat encoding", index, item.Kind)
		}
	}
	finishReason := canonicalChatFinishReason(response, len(message.ToolCalls) > 0)
	wire.Choices = []Choice{{Index: 0, Message: &message, FinishReason: &finishReason}}
	if wire.Object == "" {
		wire.Object = "chat.completion"
	}
	return wire, true, nil
}

func canonicalChatFinishReason(response *llm.Response, hasToolCalls bool) string {
	if response != nil {
		switch response.TerminalReason {
		case "tool_calls", "tool_use":
			return "tool_calls"
		case "max_tokens", "length":
			return "length"
		case "content_filter":
			return "content_filter"
		case "stop", "end_turn", "stop_sequence", "refusal":
			return "stop"
		case "":
		default:
			return response.TerminalReason
		}
	}
	if hasToolCalls {
		return "tool_calls"
	}
	return "stop"
}

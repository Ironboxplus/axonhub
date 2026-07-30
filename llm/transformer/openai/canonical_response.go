package openai

import (
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
)

// responseToCanonical decodes a complete Chat Completions response directly
// into the same ordered Output model produced by the streaming decoder.
func responseToCanonical(response *Response, rawBody []byte) ([]llm.Item, error) {
	if response == nil {
		return nil, nil
	}
	rawContent := parseRawResponseChoiceContent(rawBody)
	output := make([]llm.Item, 0, len(response.Choices))
	for choicePosition := range response.Choices {
		choice := &response.Choices[choicePosition]
		message := choice.Message
		if message == nil {
			message = choice.Delta
		}
		if message == nil {
			continue
		}
		items, err := chatMessageToCanonical(message, rawContentAt(rawContent, choicePosition), choice.Index)
		if err != nil {
			return nil, fmt.Errorf("decode Chat response choice %d: %w", choice.Index, err)
		}
		if len(items) == 0 {
			// An empty assistant answer is still a completed output item. Keeping
			// it canonical lets lifecycle persistence distinguish it from a
			// response that never reached a terminal provider message.
			items = append(items, llm.Item{
				Kind: llm.ItemKindMessage, Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{{Kind: llm.ContentKindText}},
				ProtocolHints: llm.ProtocolHints{
					SourceFormat: llm.APIFormatOpenAIChatCompletion,
					SourceType:   "message", Ordinal: choice.Index,
				},
			})
		}
		status := llm.ItemStatusCompleted
		if choice.FinishReason == nil {
			status = llm.ItemStatusIncomplete
		} else {
			status = terminalItemStatus(*choice.FinishReason)
		}
		for index := range items {
			markChatCanonicalItemTerminal(&items[index], status)
		}
		output = append(output, items...)
	}
	if err := llm.ValidateItemStructure(output); err != nil {
		return nil, fmt.Errorf("invalid canonical Chat response: %w", err)
	}
	return output, nil
}

func markChatCanonicalItemTerminal(item *llm.Item, status llm.ItemStatus) {
	if item == nil {
		return
	}
	item.Status = status
	if item.ToolCall != nil {
		if status == llm.ItemStatusCompleted {
			item.ToolCall.Status = llm.ToolCallStatusCompleted
		} else {
			item.ToolCall.Status = llm.ToolCallStatusIncomplete
		}
	}
}

func parseRawResponseChoiceContent(rawBody []byte) [][]json.RawMessage {
	var raw struct {
		Choices []struct {
			Message struct {
				Content json.RawMessage `json:"content"`
			} `json:"message"`
			Delta struct {
				Content json.RawMessage `json:"content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if len(rawBody) == 0 || json.Unmarshal(rawBody, &raw) != nil {
		return nil
	}
	content := make([][]json.RawMessage, len(raw.Choices))
	for index := range raw.Choices {
		value := raw.Choices[index].Message.Content
		if len(value) == 0 {
			value = raw.Choices[index].Delta.Content
		}
		_ = json.Unmarshal(value, &content[index])
	}
	return content
}

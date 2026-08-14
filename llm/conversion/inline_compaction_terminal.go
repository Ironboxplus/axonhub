package conversion

import (
	"errors"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// inlineCompactionContinuation admits only a complete safe plaintext provider
// answer into a gateway-owned checkpoint. It deliberately lives apart from
// provider execution: terminal validation is the response-side security gate.
func inlineCompactionContinuation(response *llm.Response) (llm.Item, error) {
	if response == nil {
		return llm.Item{}, &InlineCompactionError{Code: InlineCompactionUnsafeOutput, Err: errors.New("summary response is nil")}
	}
	if !inlineCompactionSummaryReachedSafeTerminal(response) {
		return llm.Item{}, &InlineCompactionError{Code: InlineCompactionUnsafeOutput, Err: errors.New("summary response did not complete safely")}
	}
	if len(response.Output) == 1 {
		item := response.Output[0]
		if item.Kind == llm.ItemKindMessage && item.Role == llm.RoleAssistant && !hasInlinePrivateItemData(item) && len(item.Content) == 1 && item.Content[0].Kind == llm.ContentKindText && strings.TrimSpace(item.Content[0].Text) != "" {
			return llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionContinuationText(item.Content[0].Text)}}, ProtocolHints: llm.ProtocolHints{SourceGroup: inlineCompactionContinuationSourceGroup}}, nil
		}
	}
	if len(response.Output) == 0 && len(response.Choices) == 1 && response.Choices[0].Message != nil {
		message := response.Choices[0].Message
		if len(message.ToolCalls) == 0 && message.ReasoningContent == nil && message.ReasoningSignature == nil && message.Reasoning == nil && len(message.ReasoningItems) == 0 && len(message.Content.MultipleContent) == 0 && message.Content.Content != nil && strings.TrimSpace(*message.Content.Content) != "" {
			return llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionContinuationText(*message.Content.Content)}}, ProtocolHints: llm.ProtocolHints{SourceGroup: inlineCompactionContinuationSourceGroup}}, nil
		}
	}
	return llm.Item{}, &InlineCompactionError{Code: InlineCompactionUnsafeOutput, Err: errors.New("summary response has no single safe plaintext message")}
}

// inlineCompactionSummaryReachedSafeTerminal rejects any partial provider
// completion before a durable checkpoint can be sealed.
func inlineCompactionSummaryReachedSafeTerminal(response *llm.Response) bool {
	if response == nil || response.Error != nil {
		return false
	}
	if response.Status != "" && response.Status != llm.ResponseStatusCompleted {
		return false
	}
	for index := range response.Events {
		switch response.Events[index].Kind {
		case llm.EventKindResponseFailed, llm.EventKindResponseIncomplete, llm.EventKindResponseCancelled, llm.EventKindError:
			return false
		}
	}
	for index := range response.Choices {
		finish := response.Choices[index].FinishReason
		if finish == nil {
			continue
		}
		switch *finish {
		case "length", "max_tokens", "content_filter", "error", "cancelled", "canceled":
			return false
		}
	}
	return true
}

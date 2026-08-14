package conversion

import (
	"errors"
	"strings"

	"github.com/looplj/axonhub/llm"
)

type inlineCompactionTerminalOutputKind uint8

const (
	inlineCompactionTerminalOutputReject inlineCompactionTerminalOutputKind = iota
	inlineCompactionTerminalOutputDrop
	inlineCompactionTerminalOutputMessage
)

type inlineCompactionTerminalOutputDecision struct {
	kind    inlineCompactionTerminalOutputKind
	message llm.Item
}

// projectInlineCompactionTerminalAssistantMessage applies the common closed
// message sanitizer after removing one Responses-specific non-text sidecar:
// url_citation annotation blocks materialized from an output_text arm. They
// are response metadata, never continuation plaintext. Every other non-text
// arm remains subject to the common closed sanitizer and therefore fails.
func projectInlineCompactionTerminalAssistantMessage(item llm.Item) (inlineCompactionKnownMessageProjection, error) {
	withoutAnnotations := item
	withoutAnnotations.Content = make([]llm.ContentBlock, 0, len(item.Content))
	var annotationSidecars uint32
	for index := range item.Content {
		block := item.Content[index]
		if block.Kind != llm.ContentKindCitation {
			withoutAnnotations.Content = append(withoutAnnotations.Content, block)
			continue
		}
		if block.Citation == nil || block.Citation.Type != "url_citation" || block.Text != "" || block.Image != nil || block.Audio != nil || block.Document != nil || len(block.UnknownRaw) > 0 {
			return inlineCompactionKnownMessageProjection{}, &inlineCompactionKnownMessageError{semantic: "inline_terminal_unsupported_annotation"}
		}
		annotationSidecars++
	}
	projection, err := projectInlineCompactionKnownMessage(withoutAnnotations)
	if err != nil {
		return inlineCompactionKnownMessageProjection{}, err
	}
	projection.sourceSidecars += annotationSidecars
	return projection, nil
}

// classifyInlineCompactionTerminalOutput is the response-side closed union
// gate. A summary provider may return private reasoning before its visible
// answer; reasoning is intentionally discarded instead of becoming durable
// continuation data. The only admissible visible arm is one reconstructed,
// plaintext assistant message.
func classifyInlineCompactionTerminalOutput(item llm.Item) inlineCompactionTerminalOutputDecision {
	switch item.Kind {
	case llm.ItemKindReasoning:
		if err := item.Validate(); err != nil {
			return inlineCompactionTerminalOutputDecision{kind: inlineCompactionTerminalOutputReject}
		}
		return inlineCompactionTerminalOutputDecision{kind: inlineCompactionTerminalOutputDrop}
	case llm.ItemKindMessage:
		projection, err := projectInlineCompactionTerminalAssistantMessage(item)
		if err != nil || projection.item.Role != llm.RoleAssistant || len(projection.item.Content) != 1 || projection.item.Content[0].Kind != llm.ContentKindText || strings.TrimSpace(projection.item.Content[0].Text) == "" {
			return inlineCompactionTerminalOutputDecision{kind: inlineCompactionTerminalOutputReject}
		}
		return inlineCompactionTerminalOutputDecision{kind: inlineCompactionTerminalOutputMessage, message: projection.item}
	default:
		return inlineCompactionTerminalOutputDecision{kind: inlineCompactionTerminalOutputReject}
	}
}

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
	if len(response.Output) > 0 {
		var message *llm.Item
		for index := range response.Output {
			decision := classifyInlineCompactionTerminalOutput(response.Output[index])
			switch decision.kind {
			case inlineCompactionTerminalOutputDrop:
				continue
			case inlineCompactionTerminalOutputMessage:
				if message != nil {
					return llm.Item{}, &InlineCompactionError{Code: InlineCompactionUnsafeOutput, Err: errors.New("summary response has multiple visible messages")}
				}
				candidate := decision.message
				message = &candidate
			default:
				return llm.Item{}, &InlineCompactionError{Code: InlineCompactionUnsafeOutput, Err: errors.New("summary response contains an unsafe output item")}
			}
		}
		if message != nil {
			return llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionContinuationText(message.Content[0].Text)}}, ProtocolHints: llm.ProtocolHints{SourceGroup: inlineCompactionContinuationSourceGroup}}, nil
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

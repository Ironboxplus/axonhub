package conversion

import (
	"errors"

	"github.com/looplj/axonhub/llm"
)

type inlineCompactionSummaryStats struct {
	Truncated      uint32
	TruncatedBytes uint32
}
type inlineCompactionSummaryCandidate struct {
	message *llm.Message
	drops   inlineCompactionDrops
}

type inlineCompactionHistoryProjector func(llm.Item, llm.APIFormat, *inlineCompactionDrops) (*llm.Message, error)

func (drops *inlineCompactionDrops) add(other inlineCompactionDrops) {
	drops.Reasoning += other.Reasoning
	drops.HostedProjected += other.HostedProjected
	drops.Private += other.Private
	drops.Structured += other.Structured
	drops.DocumentProjected += other.DocumentProjected
	drops.DocumentTruncated += other.DocumentTruncated
	drops.DocumentTruncatedBytes += other.DocumentTruncatedBytes
	drops.DocumentSidecars += other.DocumentSidecars
	drops.SourceSidecars += other.SourceSidecars
}
func (drops inlineCompactionDrops) projectionEmpty() bool {
	return drops.HostedProjected == 0 && drops.Private == 0 && drops.Structured == 0 && drops.DocumentProjected == 0 && drops.DocumentTruncated == 0 && drops.DocumentTruncatedBytes == 0 && drops.DocumentSidecars == 0 && drops.SourceSidecars == 0
}

// inlineCompactionSummaryMessages validates the complete history before it
// chooses bounded context. Newest-first selection is restored to chronological
// order; trusted system/developer messages receive priority without allowing
// an external item with sidecar removal to be partially represented.
func inlineCompactionSummaryMessages(history []llm.Item, target llm.APIFormat) ([]llm.Message, inlineCompactionDrops, inlineCompactionSummaryStats, error) {
	return inlineCompactionSummaryMessagesWithProjector(history, target, inlineCompactionSummaryMessage)
}

func inlineCompactionSummaryMessagesWithProjector(history []llm.Item, target llm.APIFormat, project inlineCompactionHistoryProjector) ([]llm.Message, inlineCompactionDrops, inlineCompactionSummaryStats, error) {
	projected := make([]inlineCompactionSummaryCandidate, len(history))
	var drops inlineCompactionDrops
	var stats inlineCompactionSummaryStats
	for index := range history {
		decision := classifyInlineCompactionItemForTarget(history[index], target)
		switch decision.kind {
		case inlineCompactionProjectionBlock:
			return nil, drops, stats, &InlineCompactionError{Code: inlineCompactionPlanError(decision).Code, Err: errors.New(decision.semantic)}
		case inlineCompactionProjectionDrop:
			if history[index].Kind == llm.ItemKindReasoning {
				drops.Reasoning++
			}
		case inlineCompactionProjectionSummarize:
			var delta inlineCompactionDrops
			message, err := project(history[index], target, &delta)
			if err != nil {
				return nil, drops, stats, err
			}
			projected[index] = inlineCompactionSummaryCandidate{message: message, drops: delta}
		}
	}
	selected := make([]*llm.Message, len(history))
	used := 0
	reservedExternal := 0
	for index := range history {
		candidateIndex := len(history) - 1 - index
		if projected[candidateIndex].message == nil || inlineCompactionTrustedMessage(history[candidateIndex]) {
			continue
		}
		reservedExternal = inlineCompactionSummaryMessageBudget(*projected[candidateIndex].message)
		if reservedExternal > maxInlineCompactionSummaryVisibleBytes {
			return nil, drops, stats, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("latest external summary item exceeds the aggregate budget")}
		}
		break
	}
	selectGroup := func(trustedOnly bool, limit int) {
		for index := len(history) - 1; index >= 0; index-- {
			candidate := projected[index]
			if candidate.message == nil || inlineCompactionTrustedMessage(history[index]) != trustedOnly {
				continue
			}
			message := candidate.message
			cost := inlineCompactionSummaryMessageBudget(*message)
			remaining := limit - used
			if cost <= remaining {
				selected[index] = message
				used += cost
				drops.add(candidate.drops)
				continue
			}
			truncated, kept, safe := truncateInlineCompactionSummaryMessage(*message, remaining)
			stats.Truncated++
			if cost > kept {
				stats.TruncatedBytes += uint32(cost - kept)
			}
			if !safe || truncated == nil || kept == 0 || !candidate.drops.projectionEmpty() {
				continue
			}
			selected[index] = truncated
			used += kept
		}
	}
	trustedLimit := maxInlineCompactionSummaryVisibleBytes
	if reservedExternal > 0 {
		trustedLimit -= reservedExternal
	}
	selectGroup(true, trustedLimit)
	selectGroup(false, maxInlineCompactionSummaryVisibleBytes)
	messages := make([]llm.Message, 0, len(history))
	for index := range selected {
		if selected[index] != nil {
			messages = append(messages, *selected[index])
		}
	}
	if len(messages) == 0 {
		return nil, drops, stats, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("inline compaction summary exceeds its safe input budget")}
	}
	return messages, drops, stats, nil
}

func inlineCompactionTrustedMessage(item llm.Item) bool {
	return item.Kind == llm.ItemKindMessage && (item.Role == llm.RoleSystem || item.Role == llm.RoleDeveloper)
}

func inlineCompactionSummaryCanonicalItems(messages []llm.Message) ([]llm.Item, error) {
	items := make([]llm.Item, 0, len(messages))
	for index := range messages {
		message := messages[index]
		role := llm.Role(message.Role)
		if role != llm.RoleSystem && role != llm.RoleDeveloper && role != llm.RoleUser && role != llm.RoleAssistant {
			return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary message has an unsupported role")}
		}
		content := make([]llm.ContentBlock, 0, len(message.Content.MultipleContent)+1)
		if message.Content.Content != nil {
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindText, Text: *message.Content.Content})
		}
		for partIndex := range message.Content.MultipleContent {
			part := message.Content.MultipleContent[partIndex]
			switch part.Type {
			case "text":
				if part.Text == nil {
					return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary text part is empty")}
				}
				content = append(content, llm.ContentBlock{Kind: llm.ContentKindText, Text: *part.Text})
			case "image_url":
				if part.ImageURL == nil {
					return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary image part is empty")}
				}
				image := *part.ImageURL
				content = append(content, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &image})
			case "document":
				if part.Document == nil {
					return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary document part is empty")}
				}
				document := *part.Document
				content = append(content, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: &document})
			default:
				return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary contains an unsupported part")}
			}
		}
		if len(content) == 0 {
			return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("safe summary message has no content")}
		}
		items = append(items, llm.Item{Kind: llm.ItemKindMessage, Role: role, Content: content})
	}
	return items, nil
}

func truncateInlineCompactionSummaryMessage(message llm.Message, budget int) (*llm.Message, int, bool) {
	if budget <= 0 {
		return nil, 0, false
	}
	if message.Content.Content != nil {
		text := boundedInlineTextToLimit(*message.Content.Content, budget)
		if text == "" {
			return nil, 0, false
		}
		return &llm.Message{Role: message.Role, Content: llm.MessageContent{Content: &text}}, len(text), true
	}
	parts := make([]llm.MessageContentPart, 0, len(message.Content.MultipleContent))
	used := 0
	for index := range message.Content.MultipleContent {
		part := message.Content.MultipleContent[index]
		remaining := budget - used
		if remaining <= 0 {
			continue
		}
		switch part.Type {
		case "text":
			if part.Text == nil {
				return nil, 0, false
			}
			text := boundedInlineTextToLimit(*part.Text, remaining)
			if text == "" {
				continue
			}
			parts = append(parts, llm.MessageContentPart{Type: "text", Text: &text})
			used += len(text)
		case "image_url":
			if part.ImageURL == nil {
				return nil, 0, false
			}
			cost := len(part.ImageURL.URL)
			if cost > remaining {
				continue
			}
			image := *part.ImageURL
			parts = append(parts, llm.MessageContentPart{Type: "image_url", ImageURL: &image})
			used += cost
		case "document":
			if part.Document == nil {
				return nil, 0, false
			}
			cost := inlineCompactionDocumentProjectionBudget(part.Document)
			if cost > remaining {
				continue
			}
			document := *part.Document
			parts = append(parts, llm.MessageContentPart{Type: "document", Document: &document})
			used += cost
		default:
			return nil, 0, false
		}
	}
	if len(parts) == 0 {
		return nil, 0, false
	}
	return &llm.Message{Role: message.Role, Content: llm.MessageContent{MultipleContent: parts}}, used, true
}

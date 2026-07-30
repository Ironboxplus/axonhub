package conversion

import (
	"encoding/json"

	"github.com/looplj/axonhub/llm"
)

// projectCanonical creates the compatibility view consumed by the existing
// protocol encoders. Canonical remains authoritative; this projector is a
// migration boundary and contains no source-protocol parsing.
func projectCanonical(request *llm.Request, target llm.APIFormat) (*llm.Request, error) {
	if request == nil || len(request.Input) == 0 && len(request.ToolDefinitions) == 0 {
		return request, nil
	}
	// Native same-protocol paths already carry source-private sidecars and a
	// compatibility view built by that protocol's decoder. Until each encoder
	// consumes canonical directly, rebuilding that view would discard private
	// fields without gaining a conversion benefit.
	if request.APIFormat == target {
		return request, nil
	}
	projected := request.Clone()
	projected.Messages = canonicalItemsToMessages(projected.Input)
	projected.Tools = canonicalToolsToLegacy(projected.ToolDefinitions)
	projected.APIFormat = request.APIFormat
	return projected, nil
}

func canonicalItemsToMessages(items []llm.Item) []llm.Message {
	messages := make([]llm.Message, 0, len(items))
	for index := range items {
		item := &items[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			content, annotations := canonicalContentToLegacy(item.Content)
			messages = append(messages, llm.Message{
				ID: item.ID, Role: string(item.Role), Content: content, Annotations: annotations,
			})
		case llm.ItemKindReasoning:
			if item.Reasoning == nil {
				continue
			}
			reasoning := *item.Reasoning
			messages = append(messages, llm.Message{
				Role: string(item.Role), ReasoningItems: []llm.ReasoningItem{reasoning},
				ReasoningContent: stringPointer(reasoning.Content), ReasoningSignature: stringPointer(reasoning.Signature),
			})
		case llm.ItemKindToolCall:
			if item.ToolCall == nil {
				continue
			}
			call := item.ToolCall
			legacy := llm.ToolCall{ID: call.CallID, Type: llm.ToolTypeFunction}
			if call.Kind == llm.ToolKindCustom {
				legacy.Type = llm.ToolTypeResponsesCustomTool
				legacy.ResponseCustomToolCall = &llm.ResponseCustomToolCall{
					CallID: call.CallID, Name: call.LogicalName, Input: call.InputText,
				}
			} else {
				legacy.Function = llm.FunctionCall{
					Name: call.LogicalName, Namespace: call.Namespace, Arguments: canonicalArguments(call),
				}
			}
			messages = append(messages, llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{legacy}})
		case llm.ItemKindToolResult:
			if item.ToolResult == nil {
				continue
			}
			callID := item.ToolResult.CallID
			name := item.ToolResult.LogicalName
			isError := item.ToolResult.IsError
			messages = append(messages, llm.Message{
				Role: "tool", ToolCallID: &callID, ToolCallName: stringPointer(name), ToolCallIsError: &isError,
				Content: canonicalContentOnly(item.ToolResult.Content),
			})
		case llm.ItemKindCompaction:
			if item.Compaction == nil {
				continue
			}
			createdBy := item.Compaction.CreatedBy
			messages = append(messages, llm.Message{
				Role: "assistant",
				Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{
					Type: "compaction", Compact: &llm.CompactContent{
						ID: item.ID, EncryptedContent: item.Compaction.EncryptedContent, CreatedBy: stringPointer(createdBy),
					},
				}}},
			})
		}
	}
	return messages
}

func canonicalToolsToLegacy(definitions []llm.ToolDefinition) []llm.Tool {
	tools := make([]llm.Tool, 0, len(definitions))
	for index := range definitions {
		definition := &definitions[index]
		switch definition.Kind {
		case llm.ToolKindFunction:
			if definition.Function == nil {
				continue
			}
			tools = append(tools, llm.Tool{
				Type: llm.ToolTypeFunction,
				Function: llm.Function{
					Name: definition.LogicalName, Description: definition.Description,
					Parameters: append(json.RawMessage(nil), definition.Function.Parameters...), Strict: definition.Function.Strict,
				},
			})
		case llm.ToolKindCustom:
			custom := &llm.ResponseCustomTool{Name: definition.LogicalName, Description: definition.Description}
			if definition.Freeform != nil {
				custom.Format = &llm.ResponseCustomToolFormat{
					Type: definition.Freeform.Format, Syntax: definition.Freeform.Syntax, Definition: definition.Freeform.Definition,
				}
			}
			tools = append(tools, llm.Tool{Type: llm.ToolTypeResponsesCustomTool, ResponseCustomTool: custom})
		case llm.ToolKindWebSearch:
			webSearch := &llm.WebSearch{}
			if definition.Hosted != nil && definition.Hosted.WebSearch != nil {
				webSearch = cloneWebSearch(definition.Hosted.WebSearch)
			}
			tools = append(tools, llm.Tool{Type: llm.ToolTypeWebSearch, WebSearch: webSearch})
		case llm.ToolKindImageGeneration:
			tools = append(tools, llm.Tool{Type: llm.ToolTypeImageGeneration, ImageGeneration: &llm.ImageGeneration{}})
		}
	}
	return tools
}

func cloneWebSearch(source *llm.WebSearch) *llm.WebSearch {
	if source == nil {
		return &llm.WebSearch{}
	}
	clone := *source
	clone.AllowedDomains = append([]string(nil), source.AllowedDomains...)
	clone.BlockedDomains = append([]string(nil), source.BlockedDomains...)
	return &clone
}

func canonicalContentToLegacy(blocks []llm.ContentBlock) (llm.MessageContent, []llm.Annotation) {
	if len(blocks) == 1 && blocks[0].Kind == llm.ContentKindText {
		value := blocks[0].Text
		return llm.MessageContent{Content: &value}, nil
	}
	parts := make([]llm.MessageContentPart, 0, len(blocks))
	annotations := make([]llm.Annotation, 0)
	for index := range blocks {
		block := &blocks[index]
		switch block.Kind {
		case llm.ContentKindText, llm.ContentKindRefusal:
			value := block.Text
			parts = append(parts, llm.MessageContentPart{ID: block.ID, Type: "text", Text: &value})
		case llm.ContentKindImage:
			parts = append(parts, llm.MessageContentPart{ID: block.ID, Type: "image_url", ImageURL: block.Image})
		case llm.ContentKindAudio:
			parts = append(parts, llm.MessageContentPart{ID: block.ID, Type: "input_audio", InputAudio: block.Audio})
		case llm.ContentKindDocument:
			parts = append(parts, llm.MessageContentPart{ID: block.ID, Type: "document", Document: block.Document})
		case llm.ContentKindCitation:
			if block.Citation != nil {
				annotations = append(annotations, llm.Annotation{Type: "url_citation", URLCitation: block.Citation})
			}
		}
	}
	return llm.MessageContent{MultipleContent: parts}, annotations
}

func canonicalContentOnly(blocks []llm.ContentBlock) llm.MessageContent {
	content, _ := canonicalContentToLegacy(blocks)
	return content
}

func canonicalArguments(call *llm.ToolInvocation) string {
	if call == nil {
		return ""
	}
	if len(call.ArgumentsJSON) > 0 {
		return string(call.ArgumentsJSON)
	}
	return call.ArgumentsText
}

func stringPointer(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

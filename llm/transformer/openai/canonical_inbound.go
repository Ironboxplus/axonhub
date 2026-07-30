package openai

import (
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

// requestToCanonical decodes Chat Completions fields directly into the
// ordered canonical model. Request.Messages remains as the compatibility view
// for provider adapters that have not moved to canonical encoders.
func requestToCanonical(request *Request, rawBody []byte) ([]llm.Item, []llm.ToolDefinition, error) {
	if request == nil {
		return nil, nil, nil
	}
	rawContent := parseRawMessageContent(rawBody)
	items := make([]llm.Item, 0, len(request.Messages))
	for messageIndex := range request.Messages {
		messageItems, err := chatMessageToCanonical(&request.Messages[messageIndex], rawContentAt(rawContent, messageIndex), messageIndex)
		if err != nil {
			return nil, nil, err
		}
		items = append(items, messageItems...)
	}

	definitions := make([]llm.ToolDefinition, 0, len(request.Tools))
	for index := range request.Tools {
		tool := &request.Tools[index]
		kind := llm.ToolKindFunction
		if tool.Type != "" && tool.Type != "function" {
			kind = llm.ToolKindUnknownBehavioral
		}
		definitions = append(definitions, llm.ToolDefinition{
			Kind: kind, LogicalName: tool.Function.Name, Description: tool.Function.Description,
			Function: &llm.FunctionDefinition{
				Parameters: append(json.RawMessage(nil), tool.Function.Parameters...), Strict: tool.Function.Strict,
			},
			Execution: llm.ExecutionOwnerClient,
		})
	}
	return items, definitions, nil
}

func chatMessageToCanonical(message *Message, rawContent []json.RawMessage, ordinal int) ([]llm.Item, error) {
	if message == nil {
		return nil, nil
	}
	hints := llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIChatCompletion, SourceType: "message", Ordinal: ordinal}
	items := make([]llm.Item, 0, 2+len(message.ToolCalls))

	reasoning := message.ReasoningContent
	if reasoning == nil {
		reasoning = message.Reasoning
	}
	reasoningSignature := chatMessageThoughtSignature(message)
	if (reasoning != nil && *reasoning != "") || reasoningSignature != "" {
		content := ""
		if reasoning != nil {
			content = *reasoning
		}
		items = append(items, llm.Item{
			Kind: llm.ItemKindReasoning, Role: chatRole(message.Role),
			Reasoning: &llm.ReasoningItem{Content: content, Signature: reasoningSignature},
			ProtocolHints: llm.ProtocolHints{
				SourceFormat: hints.SourceFormat, SourceType: "reasoning", Ordinal: ordinal,
			},
		})
	}

	if message.Role == "tool" {
		if message.ToolCallID == nil || *message.ToolCallID == "" {
			return nil, fmt.Errorf("Chat tool message %d has no tool_call_id", ordinal)
		}
		content, err := chatContentBlocks(message.Content, rawContent)
		if err != nil {
			return nil, err
		}
		items = append(items, llm.Item{
			Kind: llm.ItemKindToolResult,
			ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindFunction, CallID: *message.ToolCallID, Content: content,
			},
			ProtocolHints: llm.ProtocolHints{
				SourceFormat: hints.SourceFormat, SourceType: "tool_result", Ordinal: ordinal,
			},
		})
		return items, nil
	}

	content, err := chatContentBlocks(message.Content, rawContent)
	if err != nil {
		return nil, err
	}
	if message.Refusal != "" {
		content = append(content, llm.ContentBlock{Kind: llm.ContentKindRefusal, Text: message.Refusal})
	}
	for index := range message.Annotations {
		annotation := &message.Annotations[index]
		if annotation.URLCitation != nil {
			content = append(content, llm.ContentBlock{
				Kind: llm.ContentKindCitation, StartIndex: annotation.StartIndex, EndIndex: annotation.EndIndex,
				Citation: &llm.URLCitation{
					URL: annotation.URLCitation.URL, Title: annotation.URLCitation.Title,
				},
			})
		}
	}
	if message.Audio != nil {
		raw, marshalErr := json.Marshal(message.Audio)
		if marshalErr != nil {
			return nil, fmt.Errorf("marshal Chat audio history at message %d: %w", ordinal, marshalErr)
		}
		content = append(content, llm.ContentBlock{Kind: llm.ContentKindUnknown, UnknownRaw: raw})
	}
	if len(content) > 0 {
		items = append(items, llm.Item{
			Kind: llm.ItemKindMessage, Role: chatRole(message.Role), Content: content, ProtocolHints: hints,
		})
	}

	for callIndex := range message.ToolCalls {
		call := &message.ToolCalls[callIndex]
		invocation := &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, ID: call.ID, CallID: call.ID, LogicalName: call.Function.Name,
			Execution: llm.ExecutionOwnerClient,
		}
		if json.Valid([]byte(call.Function.Arguments)) {
			invocation.ArgumentsJSON = append(json.RawMessage(nil), call.Function.Arguments...)
		} else {
			invocation.ArgumentsText = call.Function.Arguments
		}
		items = append(items, llm.Item{
			Kind: llm.ItemKindToolCall, ID: call.ID, Role: llm.RoleAssistant, ToolCall: invocation,
			ProtocolHints: llm.ProtocolHints{
				SourceFormat: hints.SourceFormat, SourceType: "function_call", Ordinal: ordinal,
			},
		})
	}
	return items, nil
}

func chatMessageThoughtSignature(message *Message) string {
	if message == nil {
		return ""
	}
	for index := range message.ToolCalls {
		call := &message.ToolCalls[index]
		extra := call.ExtraContent
		if extra == nil && call.ExtraFields != nil {
			extra = call.ExtraFields.ExtraContent
		}
		if extra != nil && extra.Google != nil && extra.Google.ThoughtSignature != "" {
			return extra.Google.ThoughtSignature
		}
	}
	return ""
}

func chatContentBlocks(content MessageContent, rawContent []json.RawMessage) ([]llm.ContentBlock, error) {
	if content.Content != nil {
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *content.Content}}, nil
	}
	blocks := make([]llm.ContentBlock, 0, len(content.MultipleContent))
	for index := range content.MultipleContent {
		part := &content.MultipleContent[index]
		switch part.Type {
		case "text":
			if part.Text != nil {
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindText, Text: *part.Text})
			}
		case "image_url":
			if part.ImageURL != nil {
				blocks = append(blocks, llm.ContentBlock{
					Kind:  llm.ContentKindImage,
					Image: &llm.ImageURL{URL: part.ImageURL.URL, Detail: part.ImageURL.Detail},
				})
			}
		case "input_audio":
			if part.InputAudio != nil {
				blocks = append(blocks, llm.ContentBlock{
					Kind:  llm.ContentKindAudio,
					Audio: &llm.InputAudio{Format: part.InputAudio.Format, Data: part.InputAudio.Data},
				})
			}
		case "file":
			if part.File != nil {
				document := &llm.DocumentURL{Filename: part.File.Filename}
				switch {
				case part.File.FileID != "":
					document.SourceType = llm.DocumentSourceFile
					document.FileID = part.File.FileID
				case part.File.FileData != "":
					document.SourceType = llm.DocumentSourceBase64
					document.Data = part.File.FileData
					if parsed := xurl.ParseDataURL(part.File.FileData); parsed != nil {
						document.Data = parsed.Data
						document.MIMEType = parsed.MediaType
					}
				}
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: document})
			}
		default:
			raw := rawContentItemAt(rawContent, index)
			if len(raw) == 0 {
				var err error
				raw, err = json.Marshal(part)
				if err != nil {
					return nil, fmt.Errorf("marshal unknown Chat content %q: %w", part.Type, err)
				}
			}
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindUnknown, UnknownRaw: append(json.RawMessage(nil), raw...)})
		}
	}
	return blocks, nil
}

func chatRole(role string) llm.Role {
	switch role {
	case "system":
		return llm.RoleSystem
	case "developer":
		return llm.RoleDeveloper
	case "assistant":
		return llm.RoleAssistant
	default:
		return llm.RoleUser
	}
}

func parseRawMessageContent(rawBody []byte) [][]json.RawMessage {
	var raw struct {
		Messages []struct {
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if len(rawBody) == 0 || json.Unmarshal(rawBody, &raw) != nil {
		return nil
	}
	content := make([][]json.RawMessage, len(raw.Messages))
	for index := range raw.Messages {
		_ = json.Unmarshal(raw.Messages[index].Content, &content[index])
	}
	return content
}

func rawContentAt(content [][]json.RawMessage, index int) []json.RawMessage {
	if index < 0 || index >= len(content) {
		return nil
	}
	return content[index]
}

func rawContentItemAt(content []json.RawMessage, index int) json.RawMessage {
	if index < 0 || index >= len(content) {
		return nil
	}
	return content[index]
}

package openai

import (
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

func validateCanonicalChatRequest(request *llm.Request) error {
	if request == nil || request.APIFormat == llm.APIFormatOpenAIChatCompletion || len(request.Input) == 0 && len(request.ToolDefinitions) == 0 {
		return nil
	}
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			return fmt.Errorf("canonical tool %d kind %q has no Chat encoding", index, definition.Kind)
		}
	}
	for index := range request.Input {
		item := &request.Input[index]
		switch item.Kind {
		case llm.ItemKindMessage, llm.ItemKindReasoning, llm.ItemKindToolDeclaration:
		case llm.ItemKindAgentMessage:
			if item.AgentMessage == nil {
				return fmt.Errorf("canonical item %d agent_message payload is missing", index)
			}
			if _, err := item.AgentMessage.LegacyInterAgentMessageJSON(); err != nil {
				return fmt.Errorf("canonical item %d agent_message has no Chat encoding: %w", index, err)
			}
		case llm.ItemKindToolCall:
			if item.ToolCall == nil || item.ToolCall.Kind != llm.ToolKindFunction {
				return fmt.Errorf("canonical item %d tool call has no Chat encoding", index)
			}
		case llm.ItemKindToolResult:
			if item.ToolResult == nil || item.ToolResult.Kind != llm.ToolKindFunction {
				return fmt.Errorf("canonical item %d tool result has no Chat encoding", index)
			}
		default:
			return fmt.Errorf("canonical item %d kind %q has no Chat encoding", index, item.Kind)
		}
	}
	return nil
}

func canonicalRequestMessages(request *llm.Request, reasoningField ReasoningField) ([]Message, bool) {
	// Identity routes keep the original wire-oriented compatibility view and
	// provider sidecars. Canonical encoding is the cross-protocol boundary; it
	// must not normalize private fields on a same-protocol pass-through.
	if request == nil || request.APIFormat == llm.APIFormatOpenAIChatCompletion || len(request.Input) == 0 {
		return nil, false
	}
	messages := make([]Message, 0, len(request.Input))
	for index := range request.Input {
		item := &request.Input[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			content, annotations, refusal := canonicalChatContent(item.Content)
			if content.Content == nil && len(content.MultipleContent) == 0 && refusal == "" {
				continue
			}
			messages = appendCanonicalChatMessage(messages, Message{
				Role: string(item.Role), Content: content, Refusal: refusal, Annotations: annotations,
			})
		case llm.ItemKindReasoning:
			if item.Reasoning == nil {
				continue
			}
			value := item.Reasoning.Content
			message := llm.Message{Role: "assistant", ReasoningContent: &value, Reasoning: &value}
			messages = appendCanonicalChatMessage(messages, MessageFromLLMWithConfig(message, reasoningField))
		case llm.ItemKindAgentMessage:
			if item.AgentMessage == nil {
				continue
			}
			envelope, err := item.AgentMessage.LegacyInterAgentMessageJSON()
			if err != nil {
				continue
			}
			// Codex detects the JSON envelope only when it is the whole assistant
			// message. Do not merge two typed inter-agent boundaries into one
			// multi-part Chat message.
			messages = append(messages, Message{
				Role: "assistant", Content: MessageContent{Content: &envelope},
			})
		case llm.ItemKindToolCall:
			if item.ToolCall == nil {
				continue
			}
			call := item.ToolCall
			messages = appendCanonicalChatMessage(messages, Message{
				Role: "assistant",
				ToolCalls: []ToolCall{{
					ID: call.CallID, Type: "function",
					Function: FunctionCall{Name: call.LogicalName, Arguments: canonicalChatArguments(call)},
				}},
			})
		case llm.ItemKindToolResult:
			if item.ToolResult == nil {
				continue
			}
			callID := item.ToolResult.CallID
			content, _, _ := canonicalChatContent(item.ToolResult.Content)
			messages = append(messages, Message{Role: "tool", ToolCallID: &callID, Content: content})
		}
	}
	return messages, true
}

func appendCanonicalChatMessage(messages []Message, next Message) []Message {
	if next.Role != "assistant" || len(messages) == 0 || messages[len(messages)-1].Role != "assistant" {
		return append(messages, next)
	}
	last := &messages[len(messages)-1]
	mergeCanonicalChatContent(&last.Content, next.Content)
	last.Refusal += next.Refusal
	last.Annotations = append(last.Annotations, next.Annotations...)
	last.ToolCalls = append(last.ToolCalls, next.ToolCalls...)
	mergeCanonicalChatString(&last.ReasoningContent, next.ReasoningContent)
	mergeCanonicalChatString(&last.Reasoning, next.Reasoning)
	return messages
}

func mergeCanonicalChatContent(destination *MessageContent, source MessageContent) {
	if destination == nil || source.Content == nil && len(source.MultipleContent) == 0 {
		return
	}
	if destination.Content == nil && len(destination.MultipleContent) == 0 {
		*destination = source
		return
	}
	parts := make([]MessageContentPart, 0, len(destination.MultipleContent)+len(source.MultipleContent)+2)
	if destination.Content != nil {
		value := *destination.Content
		parts = append(parts, MessageContentPart{Type: "text", Text: &value})
	} else {
		parts = append(parts, destination.MultipleContent...)
	}
	if source.Content != nil {
		value := *source.Content
		parts = append(parts, MessageContentPart{Type: "text", Text: &value})
	} else {
		parts = append(parts, source.MultipleContent...)
	}
	destination.Content = nil
	destination.MultipleContent = parts
}

func mergeCanonicalChatString(destination **string, source *string) {
	if source == nil || *source == "" {
		return
	}
	if *destination == nil || **destination == "" {
		value := *source
		*destination = &value
		return
	}
	value := **destination + *source
	*destination = &value
}

func canonicalRequestTools(request *llm.Request) ([]Tool, bool) {
	if request == nil || len(request.ToolDefinitions) == 0 {
		return nil, false
	}
	tools := make([]Tool, 0, len(request.ToolDefinitions))
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			continue
		}
		tools = append(tools, Tool{
			Type: "function",
			Function: Function{
				Name: definition.LogicalName, Description: definition.Description,
				Parameters: definition.Function.Parameters, Strict: definition.Function.Strict,
			},
		})
	}
	return tools, true
}

func canonicalChatContent(blocks []llm.ContentBlock) (MessageContent, []Annotation, string) {
	if len(blocks) == 1 && blocks[0].Kind == llm.ContentKindText {
		value := blocks[0].Text
		return MessageContent{Content: &value}, nil, ""
	}
	parts := make([]MessageContentPart, 0, len(blocks))
	annotations := make([]Annotation, 0)
	refusal := ""
	for index := range blocks {
		block := &blocks[index]
		switch block.Kind {
		case llm.ContentKindText:
			value := block.Text
			parts = append(parts, MessageContentPart{Type: "text", Text: &value})
		case llm.ContentKindRefusal:
			refusal += block.Text
		case llm.ContentKindImage:
			if block.Image != nil {
				parts = append(parts, MessageContentPart{Type: "image_url", ImageURL: &ImageURL{URL: block.Image.URL, Detail: block.Image.Detail}})
			}
		case llm.ContentKindAudio:
			if block.Audio != nil {
				parts = append(parts, MessageContentPart{Type: "input_audio", InputAudio: &InputAudio{Format: block.Audio.Format, Data: block.Audio.Data}})
			}
		case llm.ContentKindDocument:
			if block.Document == nil {
				continue
			}
			file := &File{Filename: block.Document.Filename}
			switch block.Document.SourceType {
			case llm.DocumentSourceBase64:
				file.FileData = xurl.BuildDataURL(block.Document.MIMEType, block.Document.Data, true)
			case llm.DocumentSourceFile:
				file.FileID = block.Document.FileID
			default:
				continue
			}
			parts = append(parts, MessageContentPart{Type: "file", File: file})
		case llm.ContentKindCitation:
			if block.Citation != nil {
				annotations = append(annotations, Annotation{
					Type: "url_citation", StartIndex: block.StartIndex, EndIndex: block.EndIndex,
					URLCitation: &URLCitation{URL: block.Citation.URL, Title: block.Citation.Title},
				})
			}
		}
	}
	return MessageContent{MultipleContent: parts}, annotations, refusal
}

func canonicalChatArguments(call *llm.ToolInvocation) string {
	if len(call.ArgumentsJSON) > 0 {
		return string(call.ArgumentsJSON)
	}
	return call.ArgumentsText
}

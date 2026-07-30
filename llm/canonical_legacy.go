package llm

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

// PopulateCanonicalFromLegacy creates the migration-time ordered view used by
// the three protocol decoders while their wire-specific direct item decoders
// are moved over incrementally. Other provider adapters may keep using the
// legacy fields without opting in.
func PopulateCanonicalFromLegacy(request *Request) error {
	if request == nil {
		return fmt.Errorf("canonical request is nil")
	}
	if len(request.ToolDefinitions) == 0 && len(request.Tools) > 0 {
		request.ToolDefinitions = LegacyToolsToDefinitions(request.Tools)
	}
	if len(request.Input) == 0 && len(request.Messages) > 0 {
		request.Input = LegacyMessagesToItems(request.Messages, request.APIFormat)
	}
	for index := range request.ToolDefinitions {
		if err := request.ToolDefinitions[index].Validate(); err != nil {
			return fmt.Errorf("tool definition %d: %w", index, err)
		}
	}
	return ValidateItemStructure(request.Input)
}

func LegacyToolsToDefinitions(tools []Tool) []ToolDefinition {
	definitions := make([]ToolDefinition, 0, len(tools))
	for index := range tools {
		tool := &tools[index]
		switch {
		case tool.Type == ToolTypeFunction || tool.Type == "":
			definitions = append(definitions, ToolDefinition{
				Kind: ToolKindFunction, LogicalName: tool.Function.Name, Description: tool.Function.Description,
				Function: &FunctionDefinition{
					Parameters: append(json.RawMessage(nil), tool.Function.Parameters...),
					Strict:     tool.Function.Strict,
				},
				Execution: ExecutionOwnerClient,
			})
		case tool.Type == ToolTypeResponsesCustomTool || tool.Type == "custom":
			definition := ToolDefinition{Kind: ToolKindCustom, Execution: ExecutionOwnerClient}
			if tool.ResponseCustomTool != nil {
				definition.LogicalName = tool.ResponseCustomTool.Name
				definition.Description = tool.ResponseCustomTool.Description
				definition.Freeform = &FreeformDefinition{}
				if tool.ResponseCustomTool.Format != nil {
					definition.Freeform.Format = tool.ResponseCustomTool.Format.Type
					definition.Freeform.Syntax = tool.ResponseCustomTool.Format.Syntax
					definition.Freeform.Definition = tool.ResponseCustomTool.Format.Definition
				}
			}
			definitions = append(definitions, definition)
		default:
			kind := legacyHostedToolKind(tool.Type)
			name := tool.Function.Name
			if name == "" {
				name = tool.Type
			}
			definitions = append(definitions, ToolDefinition{
				Kind: kind, LogicalName: name, Hosted: &HostedToolDefinition{Type: tool.Type}, Execution: ExecutionOwnerProvider,
			})
		}
	}
	return definitions
}

func LegacyMessagesToItems(messages []Message, source APIFormat) []Item {
	items := make([]Item, 0, len(messages))
	callKinds := make(map[string]ToolKind)
	callNames := make(map[string]string)
	for messageIndex := range messages {
		message := &messages[messageIndex]
		if message.Role == "tool" {
			callID := ""
			if message.ToolCallID != nil {
				callID = *message.ToolCallID
			}
			kind := callKinds[callID]
			if kind == "" {
				kind = ToolKindFunction
			}
			name := callNames[callID]
			if message.ToolCallName != nil && *message.ToolCallName != "" {
				name = *message.ToolCallName
			}
			items = append(items, Item{
				Kind: ItemKindToolResult,
				ToolResult: &ToolResult{
					Kind: kind, CallID: callID, LogicalName: name,
					Content: legacyMessageContent(message.Content), IsError: message.ToolCallIsError != nil && *message.ToolCallIsError,
				},
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
			continue
		}

		for reasoningIndex := range message.ReasoningItems {
			reasoning := message.ReasoningItems[reasoningIndex]
			items = append(items, Item{
				Kind: ItemKindReasoning, ID: reasoning.ID, Role: legacyRole(message.Role), Reasoning: &reasoning,
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
		}
		if len(message.ReasoningItems) == 0 && (message.ReasoningContent != nil || message.ReasoningSignature != nil) {
			reasoning := ReasoningItem{}
			if message.ReasoningContent != nil {
				reasoning.Content = *message.ReasoningContent
			}
			if message.ReasoningSignature != nil {
				reasoning.Signature = *message.ReasoningSignature
			}
			items = append(items, Item{
				Kind: ItemKindReasoning, Role: legacyRole(message.Role), Reasoning: &reasoning,
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
		}

		content := legacyMessageContent(message.Content)
		if message.Refusal != "" {
			content = append(content, ContentBlock{Kind: ContentKindRefusal, Text: message.Refusal})
		}
		if len(content) > 0 {
			items = append(items, Item{
				Kind: ItemKindMessage, ID: message.ID, Role: legacyRole(message.Role), Content: content,
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
		}

		for toolIndex := range message.ToolCalls {
			call := &message.ToolCalls[toolIndex]
			invocation := legacyToolInvocation(call)
			callKinds[invocation.CallID] = invocation.Kind
			callNames[invocation.CallID] = invocation.LogicalName
			items = append(items, Item{
				Kind: ItemKindToolCall, ID: call.ID, Role: RoleAssistant, ToolCall: &invocation,
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
		}
		for resultIndex := range message.InlineToolResults {
			result := &message.InlineToolResults[resultIndex]
			kind := callKinds[result.ToolCallID]
			if kind == "" {
				kind = ToolKindAnthropicServer
			}
			items = append(items, Item{
				Kind: ItemKindToolResult,
				ToolResult: &ToolResult{
					Kind: kind, CallID: result.ToolCallID, LogicalName: callNames[result.ToolCallID], IsError: result.IsError,
					Content: []ContentBlock{{Kind: ContentKindText, Text: result.Output}},
				},
				ProtocolHints: ProtocolHints{SourceFormat: source, Ordinal: messageIndex},
			})
		}
	}
	return items
}

func legacyToolInvocation(call *ToolCall) ToolInvocation {
	if call.ResponseCustomToolCall != nil || call.Type == ToolTypeResponsesCustomTool || call.Type == "custom" {
		custom := call.ResponseCustomToolCall
		if custom == nil {
			custom = &ResponseCustomToolCall{CallID: call.ID, Name: call.Function.Name, Input: call.Function.Arguments}
		}
		if custom.CallID == "" {
			custom.CallID = call.ID
		}
		return ToolInvocation{
			Kind: ToolKindCustom, ID: call.ID, CallID: custom.CallID, LogicalName: custom.Name,
			InputText: custom.Input, Execution: ExecutionOwnerClient,
		}
	}
	invocation := ToolInvocation{
		Kind: ToolKindFunction, ID: call.ID, CallID: call.ID, LogicalName: call.Function.Name,
		Namespace: call.Function.Namespace, Execution: ExecutionOwnerClient,
	}
	if json.Valid([]byte(call.Function.Arguments)) {
		invocation.ArgumentsJSON = append(json.RawMessage(nil), call.Function.Arguments...)
	} else {
		invocation.ArgumentsText = call.Function.Arguments
	}
	return invocation
}

func legacyMessageContent(content MessageContent) []ContentBlock {
	if content.Content != nil {
		return []ContentBlock{{Kind: ContentKindText, Text: *content.Content}}
	}
	blocks := make([]ContentBlock, 0, len(content.MultipleContent))
	for index := range content.MultipleContent {
		part := &content.MultipleContent[index]
		switch part.Type {
		case "text", "input_text", "output_text":
			if part.Text != nil {
				blocks = append(blocks, ContentBlock{Kind: ContentKindText, ID: part.ID, Text: *part.Text})
			}
		case "image", "image_url", "input_image":
			if part.ImageURL != nil {
				blocks = append(blocks, ContentBlock{Kind: ContentKindImage, ID: part.ID, Image: part.ImageURL})
			}
		case "input_audio", "audio":
			if part.InputAudio != nil {
				blocks = append(blocks, ContentBlock{Kind: ContentKindAudio, ID: part.ID, Audio: part.InputAudio})
			}
		case "document":
			if part.Document != nil {
				blocks = append(blocks, ContentBlock{Kind: ContentKindDocument, ID: part.ID, Document: part.Document})
			}
		case "file":
			if part.File != nil {
				document := &DocumentURL{Filename: part.File.Filename}
				switch {
				case part.File.FileData != "":
					document.SourceType = DocumentSourceBase64
					document.Data = part.File.FileData
					if parsed := xurl.ParseDataURL(part.File.FileData); parsed != nil {
						document.Data = parsed.Data
						document.MIMEType = parsed.MediaType
					}
				case part.File.FileID != "":
					document.SourceType = DocumentSourceFile
					document.FileID = part.File.FileID
				}
				blocks = append(blocks, ContentBlock{Kind: ContentKindDocument, ID: part.ID, Document: document})
			}
		}
	}
	return blocks
}

func legacyRole(role string) Role {
	switch strings.ToLower(role) {
	case "system":
		return RoleSystem
	case "developer":
		return RoleDeveloper
	case "assistant":
		return RoleAssistant
	default:
		return RoleUser
	}
}

func legacyHostedToolKind(toolType string) ToolKind {
	switch toolType {
	case ToolTypeImageGeneration:
		return ToolKindImageGeneration
	case ToolTypeWebSearch:
		return ToolKindWebSearch
	default:
		return ToolKindUnknownBehavioral
	}
}

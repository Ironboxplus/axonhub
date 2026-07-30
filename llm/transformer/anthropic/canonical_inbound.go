package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// anthropicRequestToCanonical decodes content blocks in wire order. The
// legacy Message projection remains for adapters that have not migrated, but
// it is deliberately not used to construct the authoritative item sequence.
func anthropicRequestToCanonical(request *MessageRequest) ([]llm.Item, []llm.ToolDefinition, error) {
	if request == nil {
		return nil, nil, nil
	}
	definitions, err := anthropicToolDefinitionsToCanonical(request.Tools)
	if err != nil {
		return nil, nil, err
	}
	toolKinds := make(map[string]llm.ToolKind, len(definitions))
	for index := range definitions {
		toolKinds[definitions[index].LogicalName] = definitions[index].Kind
	}
	callKinds := make(map[string]llm.ToolKind)
	items := make([]llm.Item, 0, len(request.Messages)+1)
	if request.System != nil {
		if request.System.Prompt != nil {
			items = append(items, anthropicMessageItem(llm.RoleSystem, []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *request.System.Prompt}}, "system", -1))
		} else {
			for index := range request.System.MultiplePrompts {
				prompt := &request.System.MultiplePrompts[index]
				items = append(items, anthropicMessageItem(llm.RoleSystem, []llm.ContentBlock{{Kind: llm.ContentKindText, Text: prompt.Text}}, "system", index-len(request.System.MultiplePrompts)))
			}
		}
	}

	for messageIndex := range request.Messages {
		messageItems, err := anthropicMessageToCanonical(&request.Messages[messageIndex], messageIndex, toolKinds, callKinds)
		if err != nil {
			return nil, nil, err
		}
		items = append(items, messageItems...)
	}

	return items, definitions, nil
}

func anthropicToolDefinitionsToCanonical(tools []Tool) ([]llm.ToolDefinition, error) {
	definitions := make([]llm.ToolDefinition, 0, len(tools))
	for index := range tools {
		tool := &tools[index]
		if tool.Type == "" || tool.Type == "custom" {
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindFunction, LogicalName: tool.Name, Description: tool.Description,
				Function:      &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), tool.InputSchema...), Strict: tool.Strict},
				Execution:     llm.ExecutionOwnerClient,
				DeferLoading:  cloneAnthropicBool(tool.DeferLoading),
				ProtocolHints: anthropicHints("tool_definition", -1, index),
			})
			continue
		}
		raw, err := json.Marshal(tool)
		if err != nil {
			return nil, fmt.Errorf("marshal Anthropic hosted tool %q: %w", tool.Type, err)
		}
		name := tool.Name
		if name == "" {
			name = tool.Type
		}
		kind := anthropicToolKind(tool.Type, tool.Name)
		definitions = append(definitions, llm.ToolDefinition{
			Kind: kind, LogicalName: name, Description: tool.Description,
			Hosted: &llm.HostedToolDefinition{
				Type: tool.Type, WebSearch: anthropicCanonicalWebSearch(tool), WebFetch: anthropicCanonicalWebFetch(tool), Configuration: raw,
			},
			Execution:     anthropicToolExecutionOwner(kind),
			ProtocolHints: anthropicHints(tool.Type, -1, index),
		})
	}
	return definitions, nil
}

func cloneAnthropicBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func anthropicCanonicalWebFetch(tool *Tool) *llm.WebFetch {
	if tool == nil || anthropicToolKind(tool.Type, tool.Name) != llm.ToolKindWebFetch {
		return nil
	}
	var citationsEnabled *bool
	if tool.Citations != nil {
		enabled := tool.Citations.Enabled
		citationsEnabled = &enabled
	}
	return &llm.WebFetch{
		MaxUses: tool.MaxUses, AllowedDomains: append([]string(nil), tool.AllowedDomains...),
		BlockedDomains: append([]string(nil), tool.BlockedDomains...), CitationsEnabled: citationsEnabled,
		MaxContentTokens: tool.MaxContentTokens, UseCache: tool.UseCache, ResponseInclusion: tool.ResponseInclusion,
	}
}

func anthropicCanonicalWebSearch(tool *Tool) *llm.WebSearch {
	if tool == nil || anthropicToolKind(tool.Type, tool.Name) != llm.ToolKindWebSearch {
		return nil
	}
	return &llm.WebSearch{
		MaxUses: tool.MaxUses, Strict: tool.Strict,
		AllowedDomains: append([]string(nil), tool.AllowedDomains...),
		BlockedDomains: append([]string(nil), tool.BlockedDomains...),
		UserLocation: llm.WebSearchToolUserLocation{
			Type: tool.UserLocation.Type, City: tool.UserLocation.City, Country: tool.UserLocation.Country,
			Region: tool.UserLocation.Region, Timezone: tool.UserLocation.Timezone,
		},
	}
}

func anthropicMessageToCanonical(message *MessageParam, ordinal int, toolKinds map[string]llm.ToolKind, callKinds map[string]llm.ToolKind) ([]llm.Item, error) {
	if message == nil {
		return nil, nil
	}
	role := anthropicRole(message.Role)
	if message.Content.Content != nil {
		return []llm.Item{anthropicMessageItem(role, []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *message.Content.Content}}, "text", ordinal)}, nil
	}

	rawBlocks := anthropicRawBlocks(message.Content)
	items := make([]llm.Item, 0, len(message.Content.MultipleContent))
	pending := make([]llm.ContentBlock, 0, len(message.Content.MultipleContent))
	flushContent := func(sourceType string) {
		if len(pending) == 0 {
			return
		}
		items = append(items, anthropicMessageItem(role, pending, sourceType, ordinal))
		pending = nil
	}

	for blockIndex := range message.Content.MultipleContent {
		block := &message.Content.MultipleContent[blockIndex]
		raw := rawJSONAt(rawBlocks, blockIndex)
		switch block.Type {
		case "text":
			if block.Text != nil {
				pending = append(pending, llm.ContentBlock{Kind: llm.ContentKindText, Text: *block.Text})
			}
			for citationIndex := range block.Citations {
				citation := &block.Citations[citationIndex]
				pending = append(pending, llm.ContentBlock{
					Kind: llm.ContentKindCitation,
					Citation: &llm.URLCitation{
						Type: citation.Type, URL: citation.URL, Title: citation.Title, EncryptedIndex: citation.EncryptedIndex, CitedText: citation.CitedText,
					},
				})
			}
		case "image":
			if part, ok := convertImageSourceToLLMImageURLPart(block.Source, block.CacheControl); ok && part.ImageURL != nil {
				pending = append(pending, llm.ContentBlock{Kind: llm.ContentKindImage, Image: part.ImageURL})
			}
		case "document":
			if document := anthropicDocumentToCanonical(block); document != nil {
				pending = append(pending, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: document})
			}
		case "thinking":
			flushContent("content")
			items = append(items, llm.Item{
				Kind: llm.ItemKindReasoning, Role: role,
				Reasoning:     &llm.ReasoningItem{Content: stringPointerValue(block.Thinking), Signature: stringPointerValue(block.Signature)},
				ProtocolHints: anthropicHints("thinking", ordinal, blockIndex),
			})
		case "redacted_thinking":
			flushContent("content")
			items = append(items, llm.Item{
				Kind: llm.ItemKindReasoning, Role: role,
				Reasoning:     &llm.ReasoningItem{Signature: block.Data},
				ProtocolHints: anthropicHints("redacted_thinking", ordinal, blockIndex),
			})
		case "tool_use":
			flushContent("content")
			kind := llm.ToolKindFunction
			if configured, ok := toolKinds[stringPointerValue(block.Name)]; ok {
				kind = configured
			}
			if block.ID != "" && kind != llm.ToolKindFunction {
				callKinds[block.ID] = kind
			}
			items = append(items, anthropicToolCallItem(block, nil, ordinal, blockIndex, kind, llm.ExecutionOwnerClient))
		case "tool_result":
			flushContent("content")
			kind := llm.ToolKindFunction
			if configured, ok := callKinds[stringPointerValue(block.ToolUseID)]; ok {
				kind = configured
			}
			result, err := anthropicToolResultItem(block, nil, ordinal, blockIndex, kind, false)
			if err != nil {
				return nil, err
			}
			items = append(items, result)
		default:
			flushContent("content")
			switch {
			case isAnthropicSpecialToolUseBlock(block.Type):
				items = append(items, anthropicToolCallItem(block, raw, ordinal, blockIndex, anthropicToolKind(block.Type, stringPointerValue(block.Name)), llm.ExecutionOwnerProvider))
			case isAnthropicSpecialToolResultBlock(block.Type):
				result, err := anthropicToolResultItem(block, raw, ordinal, blockIndex, anthropicToolKind(block.Type, ""), true)
				if err != nil {
					return nil, err
				}
				items = append(items, result)
			default:
				if len(raw) == 0 {
					var err error
					raw, err = json.Marshal(block)
					if err != nil {
						return nil, fmt.Errorf("marshal unknown Anthropic block %q: %w", block.Type, err)
					}
				}
				items = append(items, llm.Item{
					Kind:          llm.ItemKindUnknown,
					Unknown:       &llm.UnknownItem{Type: block.Type, Raw: append(json.RawMessage(nil), raw...), Behavioral: true},
					ProtocolHints: anthropicHints(block.Type, ordinal, blockIndex),
				})
			}
		}
	}
	flushContent("content")
	return mergeAnthropicHostedItems(items), nil
}

func anthropicDocumentToCanonical(block *MessageContentBlock) *llm.DocumentURL {
	if block == nil || block.Source == nil {
		return nil
	}
	source := block.Source
	document := &llm.DocumentURL{
		MIMEType: source.MediaType, Filename: block.Title, Title: block.Title, Context: block.Context,
		CacheControl: convertToLLMCacheControl(block.CacheControl),
	}
	if block.DocumentCitations != nil {
		enabled := block.DocumentCitations.Enabled
		document.CitationsEnabled = &enabled
	}
	switch source.Type {
	case "base64":
		document.SourceType = llm.DocumentSourceBase64
		document.Data = source.Data
	case "url":
		document.SourceType = llm.DocumentSourceURL
		document.URL = source.URL
	case "file":
		document.SourceType = llm.DocumentSourceFile
		document.FileID = source.FileID
	case "text":
		document.SourceType = llm.DocumentSourceText
		document.Data = source.Data
	case "content":
		document.SourceType = llm.DocumentSourceContent
		document.Content = append(json.RawMessage(nil), source.Content...)
	default:
		return nil
	}
	return document
}

func mergeAnthropicHostedItems(items []llm.Item) []llm.Item {
	if len(items) == 0 {
		return items
	}
	merged := make([]llm.Item, 0, len(items))
	hostedByCallID := make(map[string]int)
	for index := range items {
		item := items[index]
		if item.Kind == llm.ItemKindToolCall && item.ToolCall != nil && item.ToolCall.Execution == llm.ExecutionOwnerProvider {
			invocation := *item.ToolCall
			item.Kind = llm.ItemKindHostedCall
			item.ToolCall = nil
			item.HostedCall = &llm.HostedToolCall{Invocation: invocation}
			hostedByCallID[invocation.CallID] = len(merged)
			merged = append(merged, item)
			continue
		}
		if item.Kind == llm.ItemKindToolResult && item.ToolResult != nil && item.ToolResult.Execution == llm.ExecutionOwnerProvider {
			callID := item.ToolResult.CallID
			if hostedIndex, ok := hostedByCallID[callID]; ok {
				merged[hostedIndex].HostedCall.Result = item.ToolResult
				continue
			}
			logicalName := strings.TrimSuffix(item.ProtocolHints.SourceType, "_tool_result")
			if logicalName == "" {
				logicalName = string(item.ToolResult.Kind)
			}
			result := item.ToolResult
			item.Kind = llm.ItemKindHostedCall
			item.ID = callID
			item.ToolResult = nil
			item.HostedCall = &llm.HostedToolCall{
				Invocation: llm.ToolInvocation{
					Kind: result.Kind, ID: callID, CallID: callID, LogicalName: logicalName,
					Execution: llm.ExecutionOwnerProvider,
				},
				Result: result,
			}
			hostedByCallID[callID] = len(merged)
			merged = append(merged, item)
			continue
		}
		merged = append(merged, item)
	}
	return merged
}

func anthropicToolCallItem(block *MessageContentBlock, raw json.RawMessage, ordinal, blockIndex int, kind llm.ToolKind, owner llm.ExecutionOwner) llm.Item {
	callID := block.ID
	name := stringPointerValue(block.Name)
	if name == "" {
		name = block.Type
	}
	return llm.Item{
		Kind: llm.ItemKindToolCall, ID: block.ID, Role: llm.RoleAssistant,
		ToolCall: &llm.ToolInvocation{
			Kind: kind, ID: block.ID, CallID: callID, LogicalName: name,
			ArgumentsJSON: append(json.RawMessage(nil), block.Input...), Execution: owner,
			ProviderData: append(json.RawMessage(nil), raw...),
		},
		ProtocolHints: anthropicHints(block.Type, ordinal, blockIndex),
	}
}

func anthropicToolResultItem(block *MessageContentBlock, raw json.RawMessage, ordinal, blockIndex int, kind llm.ToolKind, preserveRawContent bool) (llm.Item, error) {
	callID := stringPointerValue(block.ToolUseID)
	content := make([]llm.ContentBlock, 0)
	var structuredContent json.RawMessage
	if block.Content != nil {
		if preserveRawContent {
			var err error
			content, err = anthropicHostedResultContent(block.Content)
			if err != nil {
				return llm.Item{}, fmt.Errorf("decode Anthropic %s content: %w", block.Type, err)
			}
			if block.Content.ObjectContent != nil {
				structuredContent, err = anthropicObjectContentRaw(block.Content)
				if err != nil {
					return llm.Item{}, fmt.Errorf("preserve Anthropic %s structured content: %w", block.Type, err)
				}
			}
		} else {
			var err error
			content, err = anthropicResultContent(block.Content)
			if err != nil {
				return llm.Item{}, err
			}
		}
	}
	isError := block.IsError != nil && *block.IsError
	status := llm.ToolResultStatusCompleted
	if isError {
		status = llm.ToolResultStatusFailed
	}
	execution := llm.ExecutionOwnerClient
	if preserveRawContent {
		execution = llm.ExecutionOwnerProvider
	}
	return llm.Item{
		Kind: llm.ItemKindToolResult,
		ToolResult: &llm.ToolResult{
			Kind: kind, CallID: callID, Content: content, Execution: execution, IsError: isError, Status: status,
			StructuredContent: structuredContent, ProviderData: append(json.RawMessage(nil), raw...),
		},
		ProtocolHints: anthropicHints(block.Type, ordinal, blockIndex),
	}, nil
}

func anthropicHostedResultContent(content *MessageContent) ([]llm.ContentBlock, error) {
	if content == nil {
		return nil, nil
	}
	if content.Content != nil {
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *content.Content}}, nil
	}
	if content.ObjectContent != nil {
		raw, err := anthropicObjectContentRaw(content)
		if err != nil {
			return nil, fmt.Errorf("marshal hosted result object %q: %w", content.ObjectContent.Type, err)
		}
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(raw)}}, nil
	}
	blocks := make([]llm.ContentBlock, 0, len(content.MultipleContent))
	rawBlocks := anthropicRawBlocks(*content)
	for index := range content.MultipleContent {
		block := &content.MultipleContent[index]
		switch block.Type {
		case "web_search_result":
			blocks = append(blocks, llm.ContentBlock{
				Kind: llm.ContentKindCitation,
				Citation: &llm.URLCitation{
					Type: "web_search_result_location", URL: block.URL, Title: block.Title, EncryptedIndex: block.EncryptedContent,
				},
			})
		case "text":
			if block.Text != nil {
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindText, Text: *block.Text})
			}
		case "image":
			if part, ok := convertImageSourceToLLMImageURLPart(block.Source, block.CacheControl); ok && part.ImageURL != nil {
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindImage, Image: part.ImageURL})
			}
		default:
			raw := rawJSONAt(rawBlocks, index)
			if len(raw) == 0 {
				var err error
				raw, err = json.Marshal(block)
				if err != nil {
					return nil, fmt.Errorf("marshal hosted result content %q: %w", block.Type, err)
				}
			}
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindUnknown, UnknownRaw: append(json.RawMessage(nil), raw...)})
		}
	}
	return blocks, nil
}

func anthropicObjectContentRaw(content *MessageContent) (json.RawMessage, error) {
	if content == nil || content.ObjectContent == nil {
		return nil, nil
	}
	raw := append(json.RawMessage(nil), content.Raw...)
	if len(raw) > 0 {
		return raw, nil
	}
	encoded, err := json.Marshal(content.ObjectContent)
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

func anthropicResultContent(content *MessageContent) ([]llm.ContentBlock, error) {
	if content == nil {
		return nil, nil
	}
	if content.Content != nil {
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *content.Content}}, nil
	}
	if content.ObjectContent != nil {
		raw := content.Raw
		if len(raw) == 0 {
			var err error
			raw, err = json.Marshal(content.ObjectContent)
			if err != nil {
				return nil, fmt.Errorf("marshal Anthropic tool-result object: %w", err)
			}
		}
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(raw)}}, nil
	}
	blocks := make([]llm.ContentBlock, 0, len(content.MultipleContent))
	rawBlocks := anthropicRawBlocks(*content)
	for index := range content.MultipleContent {
		block := &content.MultipleContent[index]
		switch block.Type {
		case "text":
			if block.Text != nil {
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindText, Text: *block.Text})
			}
		case "image":
			if part, ok := convertImageSourceToLLMImageURLPart(block.Source, block.CacheControl); ok && part.ImageURL != nil {
				blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindImage, Image: part.ImageURL})
			}
		default:
			raw := rawJSONAt(rawBlocks, index)
			if len(raw) == 0 {
				var err error
				raw, err = json.Marshal(block)
				if err != nil {
					return nil, fmt.Errorf("marshal unknown Anthropic tool-result content %q: %w", block.Type, err)
				}
			}
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindUnknown, UnknownRaw: append(json.RawMessage(nil), raw...)})
		}
	}
	return blocks, nil
}

func anthropicMessageItem(role llm.Role, content []llm.ContentBlock, sourceType string, ordinal int) llm.Item {
	return llm.Item{
		Kind: llm.ItemKindMessage, Role: role, Content: content,
		ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatAnthropicMessage, SourceType: sourceType, Ordinal: ordinal},
	}
}

func anthropicHints(sourceType string, messageIndex, blockIndex int) llm.ProtocolHints {
	return llm.ProtocolHints{
		SourceFormat: llm.APIFormatAnthropicMessage, SourceType: sourceType,
		Ordinal: messageIndex<<16 | blockIndex,
	}
}

func anthropicRole(role string) llm.Role {
	if role == "assistant" {
		return llm.RoleAssistant
	}
	return llm.RoleUser
}

func anthropicToolKind(value, name string) llm.ToolKind {
	lower := strings.ToLower(value + " " + name)
	switch {
	case strings.Contains(lower, "web_search"):
		return llm.ToolKindWebSearch
	case strings.Contains(lower, "web_fetch"):
		return llm.ToolKindWebFetch
	case strings.Contains(lower, "mcp"):
		return llm.ToolKindMCP
	case strings.Contains(lower, "tool_search"):
		return llm.ToolKindToolSearch
	case strings.Contains(lower, "code_execution"):
		return llm.ToolKindCodeExecution
	case strings.Contains(lower, "text_editor"):
		return llm.ToolKindApplyPatch
	case strings.Contains(lower, "bash"):
		return llm.ToolKindShell
	case strings.Contains(lower, "computer"):
		return llm.ToolKindComputer
	default:
		return llm.ToolKindUnknownBehavioral
	}
}

func anthropicToolExecutionOwner(kind llm.ToolKind) llm.ExecutionOwner {
	switch kind {
	case llm.ToolKindWebSearch, llm.ToolKindWebFetch, llm.ToolKindMCP,
		llm.ToolKindToolSearch, llm.ToolKindCodeExecution, llm.ToolKindAnthropicServer:
		return llm.ExecutionOwnerProvider
	default:
		return llm.ExecutionOwnerClient
	}
}

func anthropicRawBlocks(content MessageContent) []json.RawMessage {
	if len(content.Raw) == 0 {
		return nil
	}
	var blocks []json.RawMessage
	_ = json.Unmarshal(content.Raw, &blocks)
	return blocks
}

func rawJSONAt(items []json.RawMessage, index int) json.RawMessage {
	if index < 0 || index >= len(items) {
		return nil
	}
	return items[index]
}

func stringPointerValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

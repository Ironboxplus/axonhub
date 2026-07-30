package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

func validateCanonicalAnthropicRequest(request *llm.Request) error {
	if request == nil || request.APIFormat == llm.APIFormatAnthropicMessage || len(request.Input) == 0 && len(request.ToolDefinitions) == 0 {
		return nil
	}
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		switch definition.Kind {
		case llm.ToolKindFunction:
			if definition.Function == nil {
				return fmt.Errorf("canonical tool %d function payload is missing", index)
			}
		case llm.ToolKindWebSearch:
			if definition.Hosted == nil || definition.Hosted.WebSearch == nil {
				return fmt.Errorf("canonical tool %d web search payload is missing", index)
			}
		case llm.ToolKindWebFetch:
			if definition.Hosted == nil || definition.Hosted.WebFetch == nil {
				return fmt.Errorf("canonical tool %d web fetch payload is missing", index)
			}
		case llm.ToolKindCodeExecution:
			if definition.Hosted == nil {
				return fmt.Errorf("canonical tool %d code execution payload is missing", index)
			}
		case llm.ToolKindToolSearch:
			if definition.Hosted == nil ||
				!strings.HasPrefix(definition.Hosted.Type, "tool_search_tool_regex_") &&
					!strings.HasPrefix(definition.Hosted.Type, "tool_search_tool_bm25_") {
				return fmt.Errorf("canonical tool %d tool search payload is missing or unsupported", index)
			}
		default:
			return fmt.Errorf("canonical tool %d kind %q has no Anthropic encoding", index, definition.Kind)
		}
	}
	for index := range request.Input {
		item := &request.Input[index]
		switch item.Kind {
		case llm.ItemKindMessage, llm.ItemKindReasoning:
		case llm.ItemKindToolCall:
			if item.ToolCall == nil || item.ToolCall.Kind != llm.ToolKindFunction {
				return fmt.Errorf("canonical item %d tool call has no Anthropic encoding", index)
			}
		case llm.ItemKindToolResult:
			if item.ToolResult == nil || item.ToolResult.Kind != llm.ToolKindFunction {
				return fmt.Errorf("canonical item %d tool result has no Anthropic encoding", index)
			}
		case llm.ItemKindHostedCall:
			if len(canonicalAnthropicHostedCall(item.HostedCall)) == 0 {
				return fmt.Errorf("canonical item %d hosted call has no Anthropic encoding", index)
			}
		default:
			return fmt.Errorf("canonical item %d kind %q has no Anthropic encoding", index, item.Kind)
		}
	}
	return nil
}

func canonicalAnthropicRequest(request *llm.Request) (*SystemPrompt, []MessageParam, []Tool, bool) {
	// Same-protocol requests must preserve cache controls, original JSON and
	// Anthropic-only blocks through the native compatibility representation.
	// Canonical encoding is selected only for a real protocol conversion.
	if request == nil || request.APIFormat == llm.APIFormatAnthropicMessage || len(request.Input) == 0 && len(request.ToolDefinitions) == 0 {
		return nil, nil, nil, false
	}
	systemParts := make([]SystemPromptPart, 0)
	messages := make([]MessageParam, 0, len(request.Input))
	for index := range request.Input {
		item := &request.Input[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			blocks := canonicalAnthropicContent(item.Content)
			if len(blocks) == 0 {
				continue
			}
			if item.Role == llm.RoleSystem || item.Role == llm.RoleDeveloper {
				for blockIndex := range blocks {
					if blocks[blockIndex].Type == "text" && blocks[blockIndex].Text != nil {
						systemParts = append(systemParts, SystemPromptPart{Type: "text", Text: *blocks[blockIndex].Text})
					}
				}
				continue
			}
			messages = appendCanonicalAnthropicMessage(messages, string(item.Role), blocks)
		case llm.ItemKindReasoning:
			if item.Reasoning == nil {
				continue
			}
			content, signature := item.Reasoning.Content, item.Reasoning.Signature
			messages = appendCanonicalAnthropicMessage(messages, "assistant", []MessageContentBlock{{
				Type: "thinking", Thinking: &content, Signature: &signature,
			}})
		case llm.ItemKindToolCall:
			if item.ToolCall == nil {
				continue
			}
			block := canonicalAnthropicToolUse(item.ToolCall)
			messages = appendCanonicalAnthropicMessage(messages, "assistant", []MessageContentBlock{block})
		case llm.ItemKindToolResult:
			if item.ToolResult == nil {
				continue
			}
			block := canonicalAnthropicToolResult(item.ToolResult)
			messages = appendCanonicalAnthropicMessage(messages, "user", []MessageContentBlock{block})
		case llm.ItemKindHostedCall:
			blocks := canonicalAnthropicHostedCall(item.HostedCall)
			messages = appendCanonicalAnthropicMessage(messages, "assistant", blocks)
		}
	}

	var system *SystemPrompt
	if len(systemParts) == 1 {
		value := systemParts[0].Text
		system = &SystemPrompt{Prompt: &value}
	} else if len(systemParts) > 1 {
		system = &SystemPrompt{MultiplePrompts: systemParts}
	}
	return system, messages, canonicalAnthropicTools(request.ToolDefinitions), true
}

func canonicalAnthropicHostedCall(hosted *llm.HostedToolCall) []MessageContentBlock {
	if hosted == nil || hosted.Invocation.Execution != llm.ExecutionOwnerProvider {
		return nil
	}
	blocks := []MessageContentBlock{canonicalAnthropicHostedUseBlock(&hosted.Invocation)}
	if hosted.Result == nil {
		return blocks
	}
	blocks = append(blocks, canonicalAnthropicHostedResultBlock(&hosted.Invocation, hosted.Result))
	return blocks
}

func canonicalAnthropicHostedUseBlock(invocation *llm.ToolInvocation) MessageContentBlock {
	var block MessageContentBlock
	if invocation != nil && len(invocation.ProviderData) > 0 && json.Valid(invocation.ProviderData) {
		if json.Unmarshal(invocation.ProviderData, &block) != nil || !isAnthropicSpecialToolUseBlock(block.Type) {
			block = MessageContentBlock{}
		}
	}
	if block.Type == "" {
		block.Type = "server_tool_use"
	}
	if invocation == nil {
		return block
	}
	block.ID = invocation.CallID
	block.Name = stringPointerNonNil(invocation.LogicalName)
	block.Input = canonicalAnthropicArguments(invocation)
	return block
}

func canonicalAnthropicHostedResultBlock(invocation *llm.ToolInvocation, result *llm.ToolResult) MessageContentBlock {
	var block MessageContentBlock
	if result != nil && len(result.ProviderData) > 0 && json.Valid(result.ProviderData) {
		if json.Unmarshal(result.ProviderData, &block) != nil || !isAnthropicSpecialToolResultBlock(block.Type) {
			block = MessageContentBlock{}
		}
	}
	callID := ""
	if invocation != nil {
		callID = invocation.CallID
	}
	if callID == "" && result != nil {
		callID = result.CallID
	}
	if block.Type == "" {
		block.Type = anthropicHostedResultType(invocation)
	}
	block.ToolUseID = stringPointerNonNil(callID)
	if block.Content == nil && result != nil {
		content := canonicalAnthropicHostedResultContent(invocation, result)
		block.Content = &content
	}
	return block
}

func anthropicHostedResultType(invocation *llm.ToolInvocation) string {
	if invocation == nil {
		return "server_tool_result"
	}
	switch invocation.Kind {
	case llm.ToolKindWebSearch:
		return "web_search_tool_result"
	case llm.ToolKindWebFetch:
		return "web_fetch_tool_result"
	case llm.ToolKindMCP:
		return "mcp_tool_result"
	case llm.ToolKindToolSearch:
		return "tool_search_tool_result"
	case llm.ToolKindCodeExecution:
		name := strings.TrimSpace(invocation.LogicalName)
		if strings.HasSuffix(name, "_tool_result") {
			return name
		}
		if strings.Contains(name, "bash") {
			return "bash_code_execution_tool_result"
		}
		if strings.Contains(name, "text_editor") {
			return "text_editor_code_execution_tool_result"
		}
		return "code_execution_tool_result"
	default:
		name := strings.TrimSpace(invocation.LogicalName)
		if name == "" {
			name = string(invocation.Kind)
		}
		if strings.HasSuffix(name, "_tool_result") {
			return name
		}
		return name + "_tool_result"
	}
}

func canonicalAnthropicHostedResultContent(invocation *llm.ToolInvocation, result *llm.ToolResult) MessageContent {
	if result == nil {
		return MessageContent{MultipleContent: []MessageContentBlock{}}
	}
	if len(result.StructuredContent) > 0 && json.Valid(result.StructuredContent) {
		content := MessageContent{}
		content.SetRaw(append(json.RawMessage(nil), result.StructuredContent...))
		return content
	}
	if len(result.Content) == 1 && result.Content[0].Kind == llm.ContentKindUnknown && json.Valid(result.Content[0].UnknownRaw) {
		content := MessageContent{}
		content.SetRaw(append(json.RawMessage(nil), result.Content[0].UnknownRaw...))
		return content
	}
	if invocation != nil && invocation.Kind == llm.ToolKindWebFetch {
		return canonicalAnthropicWebFetchResultContent(invocation, result.Content)
	}
	if invocation != nil && invocation.Kind == llm.ToolKindCodeExecution {
		return canonicalAnthropicCodeExecutionResultContent(invocation, result.Content)
	}
	blocks := make([]MessageContentBlock, 0, len(result.Content))
	for index := range result.Content {
		part := &result.Content[index]
		switch part.Kind {
		case llm.ContentKindCitation:
			if part.Citation != nil {
				blocks = append(blocks, MessageContentBlock{
					Type: "web_search_result", URL: part.Citation.URL, Title: part.Citation.Title,
					EncryptedContent: part.Citation.EncryptedIndex,
				})
			}
		case llm.ContentKindText:
			text := part.Text
			blocks = append(blocks, MessageContentBlock{Type: "text", Text: &text})
		case llm.ContentKindImage:
			if converted := canonicalAnthropicContent([]llm.ContentBlock{*part}); len(converted) == 1 {
				blocks = append(blocks, converted[0])
			}
		}
	}
	return MessageContent{MultipleContent: blocks}
}

func canonicalAnthropicWebFetchResultContent(invocation *llm.ToolInvocation, content []llm.ContentBlock) MessageContent {
	url, text := "", ""
	if invocation != nil && len(invocation.ArgumentsJSON) > 0 {
		var input struct {
			URL string `json:"url"`
		}
		_ = json.Unmarshal(invocation.ArgumentsJSON, &input)
		url = input.URL
	}
	for index := range content {
		switch content[index].Kind {
		case llm.ContentKindText:
			text += content[index].Text
		case llm.ContentKindCitation:
			if content[index].Citation != nil && url == "" {
				url = content[index].Citation.URL
			}
		}
	}
	raw, _ := json.Marshal(map[string]any{
		"type": "web_fetch_result", "url": url,
		"content": map[string]any{
			"type":   "document",
			"source": map[string]any{"type": "text", "media_type": "text/plain", "data": text},
		},
	})
	result := MessageContent{}
	result.SetRaw(raw)
	return result
}

func canonicalAnthropicCodeExecutionResultContent(invocation *llm.ToolInvocation, content []llm.ContentBlock) MessageContent {
	stdout := ""
	for index := range content {
		if content[index].Kind == llm.ContentKindText {
			stdout += content[index].Text
		}
	}
	outerType := anthropicHostedResultType(invocation)
	innerType := strings.TrimSuffix(outerType, "_tool_result") + "_result"
	raw, _ := json.Marshal(map[string]any{
		"type": innerType, "stdout": stdout, "stderr": "", "return_code": 0, "content": []any{},
	})
	result := MessageContent{}
	result.SetRaw(raw)
	return result
}

func appendCanonicalAnthropicMessage(messages []MessageParam, role string, blocks []MessageContentBlock) []MessageParam {
	if len(blocks) == 0 {
		return messages
	}
	if len(messages) > 0 && messages[len(messages)-1].Role == role {
		last := &messages[len(messages)-1]
		if last.Content.Content != nil {
			value := *last.Content.Content
			last.Content.MultipleContent = append(last.Content.MultipleContent, MessageContentBlock{Type: "text", Text: &value})
			last.Content.Content = nil
		}
		last.Content.MultipleContent = append(last.Content.MultipleContent, blocks...)
		return messages
	}
	return append(messages, MessageParam{Role: role, Content: MessageContent{MultipleContent: blocks}})
}

func canonicalAnthropicContent(content []llm.ContentBlock) []MessageContentBlock {
	blocks := make([]MessageContentBlock, 0, len(content))
	for index := range content {
		block := &content[index]
		switch block.Kind {
		case llm.ContentKindText, llm.ContentKindRefusal:
			value := block.Text
			blocks = append(blocks, MessageContentBlock{Type: "text", Text: &value})
		case llm.ContentKindImage:
			if block.Image == nil {
				continue
			}
			if parsed := xurl.ParseDataURL(block.Image.URL); parsed != nil {
				blocks = append(blocks, MessageContentBlock{Type: "image", Source: &ImageSource{Type: "base64", MediaType: parsed.MediaType, Data: parsed.Data}})
			} else {
				blocks = append(blocks, MessageContentBlock{Type: "image", Source: &ImageSource{Type: "url", URL: block.Image.URL}})
			}
		case llm.ContentKindDocument:
			if block.Document == nil {
				continue
			}
			document := block.Document
			source := &ImageSource{Type: string(document.SourceType), MediaType: document.MIMEType}
			switch document.SourceType {
			case llm.DocumentSourceBase64, llm.DocumentSourceText:
				source.Data = document.Data
			case llm.DocumentSourceURL:
				source.URL = document.URL
			case llm.DocumentSourceFile:
				source.FileID = document.FileID
			case llm.DocumentSourceContent:
				source.Content = append(json.RawMessage(nil), document.Content...)
			case "":
				if document.URL == "" {
					continue
				}
				source.Type = "url"
				source.URL = document.URL
			default:
				continue
			}
			title := document.Title
			if title == "" {
				title = document.Filename
			}
			block := MessageContentBlock{
				Type: "document", Source: source, Title: title, Context: document.Context,
				CacheControl: convertToAnthropicCacheControl(document.CacheControl),
			}
			if document.CitationsEnabled != nil {
				block.DocumentCitations = &DocumentCitations{Enabled: *document.CitationsEnabled}
			}
			blocks = append(blocks, block)
		case llm.ContentKindCitation:
			if len(blocks) > 0 && block.Citation != nil {
				citationType := block.Citation.Type
				if citationType == "" {
					citationType = "web_search_result_location"
				}
				blocks[len(blocks)-1].Citations = append(blocks[len(blocks)-1].Citations, TextCitation{
					Type: citationType, URL: block.Citation.URL, Title: block.Citation.Title,
					EncryptedIndex: block.Citation.EncryptedIndex, CitedText: block.Citation.CitedText,
				})
			}
		}
	}
	return blocks
}

func canonicalAnthropicToolUse(call *llm.ToolInvocation) MessageContentBlock {
	if len(call.ProviderData) > 0 {
		var block MessageContentBlock
		if json.Unmarshal(call.ProviderData, &block) == nil && isAnthropicToolUseLike(block.Type) {
			block.ID = call.CallID
			block.Name = stringPointerNonNil(call.LogicalName)
			block.Input = canonicalAnthropicArguments(call)
			return block
		}
	}
	return MessageContentBlock{
		Type: "tool_use", ID: call.CallID, Name: stringPointerNonNil(call.LogicalName),
		Input: canonicalAnthropicArguments(call),
	}
}

func canonicalAnthropicToolResult(result *llm.ToolResult) MessageContentBlock {
	if len(result.ProviderData) > 0 {
		var block MessageContentBlock
		if json.Unmarshal(result.ProviderData, &block) == nil && isAnthropicToolResultLike(block.Type) {
			block.ToolUseID = stringPointerNonNil(result.CallID)
			return block
		}
	}
	content := MessageContent{MultipleContent: canonicalAnthropicContent(result.Content)}
	if len(result.Content) == 1 && result.Content[0].Kind == llm.ContentKindText {
		value := result.Content[0].Text
		content = MessageContent{Content: &value}
	}
	isError := result.IsError
	return MessageContentBlock{Type: "tool_result", ToolUseID: stringPointerNonNil(result.CallID), Content: &content, IsError: &isError}
}

func canonicalAnthropicTools(definitions []llm.ToolDefinition) []Tool {
	tools := make([]Tool, 0, len(definitions))
	for index := range definitions {
		definition := &definitions[index]
		switch definition.Kind {
		case llm.ToolKindFunction:
			if definition.Function != nil {
				tools = append(tools, Tool{
					Name: definition.LogicalName, Description: definition.Description,
					InputSchema: append(json.RawMessage(nil), definition.Function.Parameters...), Strict: definition.Function.Strict,
					DeferLoading: definition.DeferLoading,
				})
			}
		case llm.ToolKindWebSearch:
			tool := Tool{Type: ToolTypeWebSearch20250305, Name: WebSearchFunctionName}
			if definition.Hosted != nil && definition.Hosted.WebSearch != nil {
				webSearch := definition.Hosted.WebSearch
				tool.MaxUses, tool.Strict = webSearch.MaxUses, webSearch.Strict
				tool.AllowedDomains = append([]string(nil), webSearch.AllowedDomains...)
				tool.BlockedDomains = append([]string(nil), webSearch.BlockedDomains...)
				tool.UserLocation = WebSearchToolUserLocation{
					Type: webSearch.UserLocation.Type, City: webSearch.UserLocation.City, Country: webSearch.UserLocation.Country,
					Region: webSearch.UserLocation.Region, Timezone: webSearch.UserLocation.Timezone,
				}
			}
			tools = append(tools, tool)
		case llm.ToolKindWebFetch:
			if definition.Hosted == nil || definition.Hosted.WebFetch == nil {
				continue
			}
			toolType := definition.Hosted.Type
			if toolType == "" {
				toolType = "web_fetch_20260318"
			}
			tool := Tool{Type: toolType, Name: definition.LogicalName}
			if tool.Name == "" {
				tool.Name = "web_fetch"
			}
			webFetch := definition.Hosted.WebFetch
			tool.MaxUses = webFetch.MaxUses
			tool.AllowedDomains = append([]string(nil), webFetch.AllowedDomains...)
			tool.BlockedDomains = append([]string(nil), webFetch.BlockedDomains...)
			if webFetch.CitationsEnabled != nil {
				tool.Citations = &WebFetchCitations{Enabled: *webFetch.CitationsEnabled}
			}
			tool.MaxContentTokens = webFetch.MaxContentTokens
			tool.UseCache = webFetch.UseCache
			tool.ResponseInclusion = webFetch.ResponseInclusion
			tools = append(tools, tool)
		case llm.ToolKindCodeExecution:
			if definition.Hosted == nil {
				continue
			}
			var tool Tool
			if len(definition.Hosted.Configuration) > 0 && json.Valid(definition.Hosted.Configuration) {
				_ = json.Unmarshal(definition.Hosted.Configuration, &tool)
			}
			if !strings.HasPrefix(tool.Type, "code_execution_") {
				tool = Tool{Type: definition.Hosted.Type, Name: definition.LogicalName}
				if !strings.HasPrefix(tool.Type, "code_execution_") {
					tool.Type = "code_execution_20250825"
				}
			}
			if tool.Name == "" {
				tool.Name = "code_execution"
			}
			tools = append(tools, tool)
		case llm.ToolKindToolSearch:
			if definition.Hosted == nil {
				continue
			}
			var tool Tool
			if len(definition.Hosted.Configuration) > 0 && json.Valid(definition.Hosted.Configuration) {
				_ = json.Unmarshal(definition.Hosted.Configuration, &tool)
			}
			if !strings.HasPrefix(tool.Type, "tool_search_tool_regex_") && !strings.HasPrefix(tool.Type, "tool_search_tool_bm25_") {
				tool = Tool{Type: definition.Hosted.Type, Name: definition.LogicalName}
			}
			if tool.Name == "" {
				tool.Name = definition.LogicalName
			}
			tools = append(tools, tool)
		}
	}
	return tools
}

func canonicalAnthropicArguments(call *llm.ToolInvocation) json.RawMessage {
	if len(call.ArgumentsJSON) > 0 {
		return append(json.RawMessage(nil), call.ArgumentsJSON...)
	}
	if call.ArgumentsText != "" && json.Valid([]byte(call.ArgumentsText)) {
		return json.RawMessage(call.ArgumentsText)
	}
	return json.RawMessage(`{}`)
}

func stringPointerNonNil(value string) *string { return &value }

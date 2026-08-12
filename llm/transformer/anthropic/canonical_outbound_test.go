package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestCanonicalAnthropicRequestGroupsAdjacentTurnBlocksWithoutReordering(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIChatCompletion,
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "do both"}}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "working"}}},
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: "call-a", LogicalName: "a", ArgumentsJSON: json.RawMessage(`{"x":1}`)}},
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: "call-b", LogicalName: "b", ArgumentsJSON: json.RawMessage(`{"x":2}`)}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindFunction, CallID: "call-a", Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "one"}}}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindFunction, CallID: "call-b", Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "two"}}}},
		},
	}

	_, messages, _, ok := canonicalAnthropicRequest(request)
	if !ok || len(messages) != 3 {
		t.Fatalf("canonical Anthropic turns = %#v, ok=%v", messages, ok)
	}
	assistant := messages[1]
	if assistant.Role != "assistant" || len(assistant.Content.MultipleContent) != 3 ||
		assistant.Content.MultipleContent[0].Type != "text" || assistant.Content.MultipleContent[1].Type != "tool_use" ||
		assistant.Content.MultipleContent[1].ID != "call-a" || assistant.Content.MultipleContent[2].ID != "call-b" {
		t.Fatalf("assistant block order = %#v", assistant)
	}
	results := messages[2]
	if results.Role != "user" || len(results.Content.MultipleContent) != 2 ||
		results.Content.MultipleContent[0].ToolUseID == nil || *results.Content.MultipleContent[0].ToolUseID != "call-a" ||
		results.Content.MultipleContent[1].ToolUseID == nil || *results.Content.MultipleContent[1].ToolUseID != "call-b" {
		t.Fatalf("tool result block order = %#v", results)
	}
}

func TestCanonicalAnthropicRequestKeepsLegacyAgentMessagesAsAssistantTurnBoundaries(t *testing.T) {
	t.Parallel()
	request := &llm.Request{APIFormat: llm.APIFormatOpenAIResponse, Input: []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "before"}}},
		{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "first"}}}},
		{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "second"}}}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "after"}}},
	}}
	_, messages, _, ok := canonicalAnthropicRequest(request)
	if !ok || len(messages) != 4 || messages[0].Role != "user" || messages[1].Role != "assistant" || messages[2].Role != "assistant" || messages[3].Role != "user" ||
		len(messages[1].Content.MultipleContent) != 1 || len(messages[2].Content.MultipleContent) != 1 || messages[1].Content.MultipleContent[0].Text == nil || messages[2].Content.MultipleContent[0].Text == nil {
		t.Fatalf("legacy inter-agent turn boundaries=%#v ok=%v", messages, ok)
	}
	for index, want := range []string{"first", "second"} {
		var legacy struct {
			Content string `json:"content"`
			Trigger bool   `json:"trigger_turn"`
		}
		if err := json.Unmarshal([]byte(*messages[index+1].Content.MultipleContent[0].Text), &legacy); err != nil || legacy.Content != want || !legacy.Trigger {
			t.Fatalf("legacy inter-agent message %d=%q decoded=%#v err=%v", index, *messages[index+1].Content.MultipleContent[0].Text, legacy, err)
		}
	}
}

func TestCanonicalAnthropicRequestEncodesNativeProviderToolSearchDefinition(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "find a tool"}},
		}},
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindToolSearch, LogicalName: "tool_search_tool_regex", Execution: llm.ExecutionOwnerProvider,
			Hosted: &llm.HostedToolDefinition{
				Type:          "tool_search_tool_regex_20251119",
				Configuration: json.RawMessage(`{"type":"tool_search_tool_regex_20251119","name":"tool_search_tool_regex"}`),
			},
		}},
	}
	if err := validateCanonicalAnthropicRequest(request); err != nil {
		t.Fatalf("validate native tool search: %v", err)
	}
	_, _, tools, ok := canonicalAnthropicRequest(request)
	if !ok || len(tools) != 1 || tools[0].Type != "tool_search_tool_regex_20251119" || tools[0].Name != "tool_search_tool_regex" {
		t.Fatalf("canonical Anthropic tool search = %#v, ok=%v", tools, ok)
	}
}

func TestCanonicalAnthropicRequestPreservesProviderServerToolShapes(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{{
			Kind: llm.ItemKindHostedCall, Role: llm.RoleAssistant,
			HostedCall: &llm.HostedToolCall{
				Invocation: llm.ToolInvocation{
					Kind: llm.ToolKindWebFetch, ID: "fetch_1", CallID: "fetch_1", LogicalName: "web_fetch",
					ArgumentsJSON: json.RawMessage(`{"url":"https://example.com/article"}`), Execution: llm.ExecutionOwnerProvider,
					ProviderData: json.RawMessage(`{"type":"server_tool_use","id":"old","name":"web_fetch","input":{},"caller":{"type":"direct"}}`),
				},
				Result: &llm.ToolResult{
					Kind: llm.ToolKindWebFetch, CallID: "fetch_1", Status: llm.ToolResultStatusCompleted,
					ProviderData: json.RawMessage(`{"type":"web_fetch_tool_result","tool_use_id":"old","content":{"type":"web_fetch_result","url":"https://example.com/article","retrieved_at":"2026-07-30T00:00:00Z","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"Article content"}}}}`),
				},
			},
		}},
	}

	_, messages, _, ok := canonicalAnthropicRequest(request)
	if !ok || len(messages) != 1 || len(messages[0].Content.MultipleContent) != 2 {
		t.Fatalf("canonical Anthropic hosted turns = %#v, ok=%v", messages, ok)
	}
	use, result := messages[0].Content.MultipleContent[0], messages[0].Content.MultipleContent[1]
	if use.Type != "server_tool_use" || use.ID != "fetch_1" || use.Name == nil || *use.Name != "web_fetch" ||
		string(use.Input) != `{"url":"https://example.com/article"}` || string(use.Caller) != `{"type":"direct"}` {
		t.Fatalf("server tool use degraded: %#v", use)
	}
	if result.Type != "web_fetch_tool_result" || result.ToolUseID == nil || *result.ToolUseID != "fetch_1" || result.Content == nil {
		t.Fatalf("server tool result degraded: %#v", result)
	}
	encodedContent, err := json.Marshal(result.Content)
	if err != nil {
		t.Fatalf("marshal server tool content: %v", err)
	}
	var object map[string]any
	if err := json.Unmarshal(encodedContent, &object); err != nil || object["type"] != "web_fetch_result" {
		t.Fatalf("server tool object content degraded: %s err=%v", encodedContent, err)
	}
}

func TestCanonicalAnthropicRequestBuildsGatewayWebFetchLifecycle(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{{
			Kind: llm.ItemKindHostedCall, Role: llm.RoleAssistant,
			HostedCall: &llm.HostedToolCall{
				Invocation: llm.ToolInvocation{
					Kind: llm.ToolKindWebFetch, ID: "fetch_2", CallID: "fetch_2", LogicalName: "web_fetch",
					ArgumentsJSON: json.RawMessage(`{"url":"https://example.com/page"}`), Execution: llm.ExecutionOwnerProvider,
				},
				Result: &llm.ToolResult{
					Kind: llm.ToolKindWebFetch, CallID: "fetch_2", Status: llm.ToolResultStatusCompleted,
					Content: []llm.ContentBlock{
						{Kind: llm.ContentKindText, Text: "Page body"},
						{Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{URL: "https://example.com/page", Title: "Page"}},
					},
				},
			},
		}},
	}

	_, messages, _, ok := canonicalAnthropicRequest(request)
	if !ok || len(messages) != 1 || len(messages[0].Content.MultipleContent) != 2 {
		t.Fatalf("gateway web-fetch turns = %#v, ok=%v", messages, ok)
	}
	result := messages[0].Content.MultipleContent[1]
	encodedContent, err := json.Marshal(result.Content)
	if err != nil {
		t.Fatalf("marshal gateway web-fetch result: %v", err)
	}
	var object map[string]any
	if err := json.Unmarshal(encodedContent, &object); err != nil {
		t.Fatalf("decode gateway web-fetch result: %v", err)
	}
	if object["type"] != "web_fetch_result" || object["url"] != "https://example.com/page" {
		t.Fatalf("gateway web-fetch result shape = %s", encodedContent)
	}
	document, _ := object["content"].(map[string]any)
	source, _ := document["source"].(map[string]any)
	if document["type"] != "document" || source["type"] != "text" || source["data"] != "Page body" {
		t.Fatalf("gateway web-fetch document shape = %s", encodedContent)
	}
}

func TestCanonicalAnthropicToolsEncodeTypedWebFetchConfiguration(t *testing.T) {
	maxUses := int64(3)
	maxContentTokens := int64(8192)
	citationsEnabled := true
	useCache := false
	tools := canonicalAnthropicTools([]llm.ToolDefinition{{
		Kind: llm.ToolKindWebFetch, LogicalName: "web_fetch", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{
			Type: "web_fetch_20260318",
			WebFetch: &llm.WebFetch{
				MaxUses: &maxUses, AllowedDomains: []string{"example.com"}, BlockedDomains: []string{"blocked.example"},
				CitationsEnabled: &citationsEnabled, MaxContentTokens: &maxContentTokens, UseCache: &useCache,
				ResponseInclusion: "full",
			},
		},
	}})
	if len(tools) != 1 {
		t.Fatalf("web-fetch tool count = %d, want 1", len(tools))
	}
	tool := tools[0]
	if tool.Type != "web_fetch_20260318" || tool.Name != "web_fetch" || tool.MaxUses == nil || *tool.MaxUses != 3 ||
		len(tool.AllowedDomains) != 1 || tool.AllowedDomains[0] != "example.com" ||
		len(tool.BlockedDomains) != 1 || tool.BlockedDomains[0] != "blocked.example" ||
		tool.Citations == nil || !tool.Citations.Enabled || tool.MaxContentTokens == nil || *tool.MaxContentTokens != 8192 ||
		tool.UseCache == nil || *tool.UseCache || tool.ResponseInclusion != "full" {
		t.Fatalf("encoded Anthropic web-fetch tool degraded: %#v", tool)
	}
}

func TestCanonicalAnthropicToolsPreserveVersionedCodeExecutionConfiguration(t *testing.T) {
	t.Parallel()
	definitions := []llm.ToolDefinition{{
		Kind: llm.ToolKindCodeExecution, LogicalName: "code_execution", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{
			Type:          "code_execution_20250825",
			Configuration: json.RawMessage(`{"type":"code_execution_20250825","name":"code_execution","max_uses":3,"future_option":{"mode":"strict"}}`),
		},
	}}
	tools := canonicalAnthropicTools(definitions)
	if len(tools) != 1 || tools[0].Type != "code_execution_20250825" || tools[0].Name != "code_execution" {
		t.Fatalf("code execution definition degraded: %#v", tools)
	}
	raw, err := json.Marshal(tools[0])
	if err != nil {
		t.Fatalf("marshal code execution definition: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil || decoded["max_uses"] != float64(3) || decoded["future_option"] == nil {
		t.Fatalf("versioned code execution configuration degraded: raw=%s decoded=%#v err=%v", raw, decoded, err)
	}
}

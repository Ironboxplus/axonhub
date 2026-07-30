package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestAnthropicDeferredToolDefinitionSurvivesCanonicalCrossEncoding(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model","max_tokens":128,
		"messages":[{"role":"user","content":"find a calendar tool"}],
		"tools":[{"name":"create_event","description":"Create an event","defer_loading":true,"input_schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Anthropic deferred tool: %v", err)
	}
	if len(request.ToolDefinitions) != 1 || request.ToolDefinitions[0].DeferLoading == nil || !*request.ToolDefinitions[0].DeferLoading {
		t.Fatalf("Anthropic deferred metadata degraded: %#v", request.ToolDefinitions)
	}
	request.APIFormat = llm.APIFormatOpenAIResponse
	tools := canonicalAnthropicTools(request.ToolDefinitions)
	if len(tools) != 1 || tools[0].DeferLoading == nil || !*tools[0].DeferLoading {
		t.Fatalf("canonical Anthropic deferred encoding degraded: %#v", tools)
	}
}

func TestInboundPreservesAnthropicContentBlockOrderInCanonical(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"max_tokens":128,
		"system":[{"type":"text","text":"be precise"}],
		"messages":[
			{"role":"user","content":"start"},
			{"role":"assistant","content":[
				{"type":"text","text":"before"},
				{"type":"thinking","thinking":"think","signature":"opaque-signature"},
				{"type":"tool_use","id":"call_local","name":"lookup","input":{"city":"Singapore"}},
				{"type":"text","text":"between"},
				{"type":"server_tool_use","id":"call_web","name":"web_search","input":{"query":"weather"},"caller":{"type":"direct"}},
				{"type":"web_search_tool_result","tool_use_id":"call_web","content":[{"type":"web_search_result","url":"https://example.invalid","title":"Weather","encrypted_content":"opaque-result"}]},
				{"type":"text","text":"after"},
				{"type":"future_block","mode":"must_preserve"}
			]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_local","content":"sunny"}]}
		],
		"tools":[{"name":"lookup","description":"weather","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Anthropic request: %v", err)
	}

	wantKinds := []llm.ItemKind{
		llm.ItemKindMessage,
		llm.ItemKindMessage,
		llm.ItemKindMessage,
		llm.ItemKindReasoning,
		llm.ItemKindToolCall,
		llm.ItemKindMessage,
		llm.ItemKindHostedCall,
		llm.ItemKindMessage,
		llm.ItemKindUnknown,
		llm.ItemKindToolResult,
	}
	if len(request.Input) != len(wantKinds) {
		t.Fatalf("canonical item count = %d, want %d: %#v", len(request.Input), len(wantKinds), request.Input)
	}
	for index, want := range wantKinds {
		if request.Input[index].Kind != want {
			t.Fatalf("canonical item %d kind = %q, want %q: %#v", index, request.Input[index].Kind, want, request.Input[index])
		}
	}
	if request.Input[2].Content[0].Text != "before" || request.Input[5].Content[0].Text != "between" || request.Input[7].Content[0].Text != "after" {
		t.Fatalf("Anthropic text/tool interleaving degraded: %#v", request.Input)
	}
	if request.Input[3].Reasoning == nil || request.Input[3].Reasoning.Signature != "opaque-signature" {
		t.Fatalf("Anthropic thinking degraded: %#v", request.Input[3])
	}
	localCall := request.Input[4].ToolCall
	if localCall == nil || localCall.Kind != llm.ToolKindFunction || localCall.CallID != "call_local" || localCall.Execution != llm.ExecutionOwnerClient {
		t.Fatalf("Anthropic client tool call degraded: %#v", request.Input[4])
	}
	serverLifecycle := request.Input[6].HostedCall
	if serverLifecycle == nil || serverLifecycle.Invocation.Kind != llm.ToolKindWebSearch || serverLifecycle.Invocation.CallID != "call_web" ||
		serverLifecycle.Invocation.Execution != llm.ExecutionOwnerProvider || serverLifecycle.Result == nil {
		t.Fatalf("Anthropic server tool call degraded: %#v", request.Input[6])
	}
	serverResult := serverLifecycle.Result
	if serverResult.Kind != llm.ToolKindWebSearch || serverResult.CallID != "call_web" || len(serverResult.Content) != 1 ||
		serverResult.Content[0].Kind != llm.ContentKindCitation || serverResult.Content[0].Citation == nil ||
		serverResult.Content[0].Citation.URL != "https://example.invalid" || serverResult.Content[0].Citation.Title != "Weather" ||
		serverResult.Content[0].Citation.EncryptedIndex == nil || *serverResult.Content[0].Citation.EncryptedIndex != "opaque-result" {
		t.Fatalf("Anthropic server tool result degraded: %#v", serverResult)
	}
	unknown := request.Input[8].Unknown
	if unknown == nil || unknown.Type != "future_block" || !unknown.Behavioral {
		t.Fatalf("unknown Anthropic block degraded: %#v", request.Input[8])
	}
	var unknownRaw map[string]any
	if err := json.Unmarshal(unknown.Raw, &unknownRaw); err != nil || unknownRaw["mode"] != "must_preserve" {
		t.Fatalf("unknown Anthropic raw block degraded: raw=%s err=%v", unknown.Raw, err)
	}
	if len(request.ToolDefinitions) != 2 || request.ToolDefinitions[0].Kind != llm.ToolKindFunction || request.ToolDefinitions[1].Kind != llm.ToolKindWebSearch {
		t.Fatalf("Anthropic tool definitions degraded: %#v", request.ToolDefinitions)
	}
}

func TestInboundPreservesObjectShapedAnthropicServerToolResults(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"max_tokens":128,
		"messages":[{"role":"assistant","content":[
			{"type":"server_tool_use","id":"fetch_1","name":"web_fetch","input":{"url":"https://example.com/article"}},
			{"type":"web_fetch_tool_result","tool_use_id":"fetch_1","content":{"type":"web_fetch_result","url":"https://example.com/article","retrieved_at":"2026-07-30T00:00:00Z","content":{"type":"document","source":{"type":"text","media_type":"text/plain","data":"Article content"}}}},
			{"type":"server_tool_use","id":"bash_1","name":"bash_code_execution","input":{"command":"printf ok"}},
			{"type":"bash_code_execution_tool_result","tool_use_id":"bash_1","content":{"type":"bash_code_execution_result","stdout":"ok","stderr":"","return_code":0,"content":[]}}
		]}],
		"tools":[
			{"type":"web_fetch_20260318","name":"web_fetch","max_uses":2,"allowed_domains":["example.com"],"citations":{"enabled":true},"max_content_tokens":4096,"use_cache":false,"response_inclusion":"full"},
			{"type":"code_execution_20250825","name":"code_execution"}
		]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Anthropic object server-tool result: %v", err)
	}
	if len(request.Input) != 2 {
		t.Fatalf("hosted lifecycle count = %d, want 2: %#v", len(request.Input), request.Input)
	}
	want := []struct {
		kind       llm.ToolKind
		callID     string
		resultType string
	}{
		{llm.ToolKindWebFetch, "fetch_1", "web_fetch_result"},
		{llm.ToolKindCodeExecution, "bash_1", "bash_code_execution_result"},
	}
	for index, expected := range want {
		hosted := request.Input[index].HostedCall
		if hosted == nil || hosted.Invocation.Kind != expected.kind || hosted.Invocation.CallID != expected.callID || hosted.Result == nil {
			t.Fatalf("hosted lifecycle %d degraded: %#v", index, request.Input[index])
		}
		if len(hosted.Result.Content) != 1 || hosted.Result.Content[0].Kind != llm.ContentKindText {
			t.Fatalf("hosted result %d content degraded: %#v", index, hosted.Result.Content)
		}
		var raw map[string]any
		if err := json.Unmarshal(hosted.Result.StructuredContent, &raw); err != nil || raw["type"] != expected.resultType ||
			hosted.Result.Content[0].Text != string(hosted.Result.StructuredContent) {
			t.Fatalf("hosted result %d structured content = %s, err=%v", index, hosted.Result.StructuredContent, err)
		}
	}
	if len(request.ToolDefinitions) != 2 || request.ToolDefinitions[0].Kind != llm.ToolKindWebFetch ||
		request.ToolDefinitions[1].Kind != llm.ToolKindCodeExecution {
		t.Fatalf("Anthropic server tool definitions degraded: %#v", request.ToolDefinitions)
	}
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if definition.Execution != llm.ExecutionOwnerProvider || definition.Hosted == nil || !json.Valid(definition.Hosted.Configuration) {
			t.Fatalf("Anthropic server tool definition %d lost provider configuration: %#v", index, definition)
		}
	}
	webFetch := request.ToolDefinitions[0].Hosted.WebFetch
	if webFetch == nil || webFetch.MaxUses == nil || *webFetch.MaxUses != 2 ||
		len(webFetch.AllowedDomains) != 1 || webFetch.AllowedDomains[0] != "example.com" ||
		webFetch.CitationsEnabled == nil || !*webFetch.CitationsEnabled ||
		webFetch.MaxContentTokens == nil || *webFetch.MaxContentTokens != 4096 ||
		webFetch.UseCache == nil || *webFetch.UseCache || webFetch.ResponseInclusion != "full" {
		t.Fatalf("Anthropic web-fetch configuration degraded: %#v", webFetch)
	}
}

func TestInboundPreservesObjectToolResultAsJSONText(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"max_tokens":128,
		"messages":[
			{"role":"assistant","content":[{"type":"tool_use","id":"call_json","name":"inspect","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_json","content":{"foo":"bar","count":2}}]}
		],
		"tools":[{"name":"inspect","input_schema":{"type":"object"}}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Anthropic object tool result: %v", err)
	}
	if len(request.Input) != 2 || request.Input[1].ToolResult == nil || len(request.Input[1].ToolResult.Content) != 1 {
		t.Fatalf("canonical object tool result shape degraded: %#v", request.Input)
	}
	result := request.Input[1].ToolResult.Content[0]
	if result.Kind != llm.ContentKindText {
		t.Fatalf("object tool result kind = %q, want text", result.Kind)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(result.Text), &decoded); err != nil || decoded["foo"] != "bar" || decoded["count"] != float64(2) {
		t.Fatalf("object tool result JSON degraded: text=%s decoded=%#v err=%v", result.Text, decoded, err)
	}
}

func TestInboundClassifiesAnthropicClientToolFamiliesAndPreservesConfiguration(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model","max_tokens":128,
		"messages":[
			{"role":"assistant","content":[{"type":"tool_use","id":"bash_1","name":"bash","input":{"command":"pwd"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"bash_1","content":"ok"}]}
		],
		"tools":[
			{"type":"bash_20250124","name":"bash"},
			{"type":"text_editor_20250728","name":"str_replace_based_edit_tool"},
			{"type":"computer_20250124","name":"computer","display_width_px":1024,"display_height_px":768,"display_number":1},
			{"type":"code_execution_20250825","name":"code_execution"}
		]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Anthropic client tools: %v", err)
	}
	wantKinds := []llm.ToolKind{llm.ToolKindShell, llm.ToolKindApplyPatch, llm.ToolKindComputer, llm.ToolKindCodeExecution}
	for index, kind := range wantKinds {
		definition := request.ToolDefinitions[index]
		if definition.Kind != kind || definition.Hosted == nil || !json.Valid(definition.Hosted.Configuration) {
			t.Fatalf("definition %d = %#v, want kind %s with raw configuration", index, definition, kind)
		}
		wantOwner := llm.ExecutionOwnerClient
		if kind == llm.ToolKindCodeExecution {
			wantOwner = llm.ExecutionOwnerProvider
		}
		if definition.Execution != wantOwner {
			t.Fatalf("definition %d owner = %s, want %s", index, definition.Execution, wantOwner)
		}
	}
	var computer map[string]any
	if err := json.Unmarshal(request.ToolDefinitions[2].Hosted.Configuration, &computer); err != nil || computer["display_width_px"] != float64(1024) || computer["display_height_px"] != float64(768) {
		t.Fatalf("computer configuration degraded: raw=%s decoded=%#v err=%v", request.ToolDefinitions[2].Hosted.Configuration, computer, err)
	}
	if len(request.Input) != 2 || request.Input[0].ToolCall == nil || request.Input[0].ToolCall.Kind != llm.ToolKindShell ||
		request.Input[1].ToolResult == nil || request.Input[1].ToolResult.Kind != llm.ToolKindShell {
		t.Fatalf("client bash lifecycle was not typed: %#v", request.Input)
	}
}

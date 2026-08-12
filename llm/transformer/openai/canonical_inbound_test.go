package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestInboundBuildsCanonicalDirectlyFromChatMessages(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"messages":[
			{"role":"user","content":[{"type":"text","text":"before"},{"type":"future_content","mode":"must_preserve"}]},
			{"role":"assistant","reasoning_content":"think","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"not-json"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"sunny"}
		],
		"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/chat/completions",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Chat request: %v", err)
	}
	if err := llm.ValidateItems(request.Input); err != nil {
		t.Fatalf("validate canonical Chat lifecycle: %v\n%#v", err, request.Input)
	}
	wantKinds := []llm.ItemKind{llm.ItemKindMessage, llm.ItemKindReasoning, llm.ItemKindToolCall, llm.ItemKindToolResult}
	if len(request.Input) != len(wantKinds) {
		t.Fatalf("canonical item count = %d, want %d: %#v", len(request.Input), len(wantKinds), request.Input)
	}
	for index, want := range wantKinds {
		if request.Input[index].Kind != want {
			t.Fatalf("canonical item %d kind = %q, want %q", index, request.Input[index].Kind, want)
		}
	}
	message := request.Input[0]
	if len(message.Content) != 2 || message.Content[1].Kind != llm.ContentKindUnknown {
		t.Fatalf("unknown Chat content was dropped: %#v", message)
	}
	var unknown map[string]any
	if err := json.Unmarshal(message.Content[1].UnknownRaw, &unknown); err != nil || unknown["mode"] != "must_preserve" {
		t.Fatalf("unknown Chat content raw bytes degraded: raw=%s err=%v", message.Content[1].UnknownRaw, err)
	}
	call := request.Input[2]
	if call.ProtocolHints.SourceType != "function_call" || call.ToolCall == nil || call.ToolCall.ID != "call_1" ||
		call.ToolCall.CallID != "call_1" || call.ToolCall.ArgumentsText != "not-json" || len(call.ToolCall.ArgumentsJSON) != 0 {
		t.Fatalf("Chat function call degraded: %#v", call)
	}
}

func TestCanonicalChatOutboundGroupsAssistantContentReasoningAndParallelCalls(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "do both"}}},
			{Kind: llm.ItemKindReasoning, Role: llm.RoleAssistant, Reasoning: &llm.ReasoningItem{Content: "thinking"}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "working"}}},
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: "call-a", LogicalName: "a", ArgumentsJSON: json.RawMessage(`{"x":1}`)}},
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: "call-b", LogicalName: "b", ArgumentsJSON: json.RawMessage(`{"x":2}`)}},
		},
	}
	messages, ok := canonicalRequestMessages(request, ReasoningFieldContent)
	if !ok || len(messages) != 2 {
		t.Fatalf("canonical Chat messages = %#v, ok=%v", messages, ok)
	}
	assistant := messages[1]
	if assistant.Role != "assistant" || assistant.ReasoningContent == nil || *assistant.ReasoningContent != "thinking" ||
		len(assistant.ToolCalls) != 2 || assistant.ToolCalls[0].ID != "call-a" || assistant.ToolCalls[1].ID != "call-b" {
		t.Fatalf("grouped Chat assistant turn = %#v", assistant)
	}
	if assistant.Content.Content == nil || *assistant.Content.Content != "working" {
		t.Fatalf("grouped Chat assistant content = %#v", assistant.Content)
	}
}

func TestCanonicalChatRequestKeepsLegacyAgentMessagesAsAssistantTurnBoundaries(t *testing.T) {
	t.Parallel()
	request := &llm.Request{APIFormat: llm.APIFormatOpenAIResponse, Input: []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "before"}}},
		{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "first"}}}},
		{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "second"}}}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "after"}}},
	}}
	messages, ok := canonicalRequestMessages(request, ReasoningFieldContent)
	if !ok || len(messages) != 4 || messages[0].Role != "user" || messages[1].Role != "assistant" || messages[2].Role != "assistant" || messages[3].Role != "user" ||
		messages[1].Content.Content == nil || messages[2].Content.Content == nil {
		t.Fatalf("legacy inter-agent turn boundaries=%#v ok=%v", messages, ok)
	}
	for index, want := range []string{"first", "second"} {
		var legacy struct {
			Content string `json:"content"`
			Trigger bool   `json:"trigger_turn"`
		}
		if err := json.Unmarshal([]byte(*messages[index+1].Content.Content), &legacy); err != nil || legacy.Content != want || !legacy.Trigger {
			t.Fatalf("legacy inter-agent message %d=%q decoded=%#v err=%v", index, *messages[index+1].Content.Content, legacy, err)
		}
	}
}

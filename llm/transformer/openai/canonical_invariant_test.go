package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestChatOutboundRejectsUnplannedCanonicalItemInsteadOfDroppingIt(t *testing.T) {
	t.Parallel()
	outbound, err := NewOutboundTransformer("https://example.invalid/v1", "fixture-key")
	if err != nil {
		t.Fatalf("create outbound: %v", err)
	}
	_, err = outbound.TransformRequest(context.Background(), &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse, Model: "fixture-model",
		Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: stringPointerInvariant("run")}}},
		Input: []llm.Item{{Kind: llm.ItemKindMCPCall, MCPCall: &llm.MCPCall{
			ServerLabel: "docs", LogicalName: "search", ArgumentsJSON: json.RawMessage(`{"q":"x"}`),
		}}},
	})
	if err == nil || !strings.Contains(err.Error(), "canonical item") {
		t.Fatalf("unplanned canonical item was not rejected: %v", err)
	}
}

func TestChatInboundEncodesCanonicalToolCallWithoutLegacyChoices(t *testing.T) {
	t.Parallel()
	inbound := NewInboundTransformer()
	response, err := inbound.TransformResponse(context.Background(), &llm.Response{
		ID: "resp_canonical", Model: "fixture-model", TerminalReason: "tool_calls",
		Output: []llm.Item{{Kind: llm.ItemKindToolCall, Role: llm.RoleAssistant, Status: llm.ItemStatusCompleted, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, CallID: "call_1", LogicalName: "lookup", ArgumentsJSON: json.RawMessage(`{"q":"x"}`),
		}}},
	})
	if err != nil {
		t.Fatalf("encode canonical Chat response: %v", err)
	}
	var wire Response
	if err := json.Unmarshal(response.Body, &wire); err != nil {
		t.Fatalf("decode Chat wire: %v", err)
	}
	if len(wire.Choices) != 1 || wire.Choices[0].Message == nil || len(wire.Choices[0].Message.ToolCalls) != 1 ||
		wire.Choices[0].Message.ToolCalls[0].ID != "call_1" || wire.Choices[0].Message.ToolCalls[0].Function.Name != "lookup" {
		t.Fatalf("canonical tool call disappeared: %s", response.Body)
	}
}

func stringPointerInvariant(value string) *string { return &value }

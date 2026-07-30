package anthropic

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestAnthropicOutboundRejectsUnplannedCanonicalItemInsteadOfDroppingIt(t *testing.T) {
	t.Parallel()
	outbound, err := NewOutboundTransformer("https://example.invalid/v1", "fixture-key")
	if err != nil {
		t.Fatalf("create outbound: %v", err)
	}
	_, err = outbound.TransformRequest(context.Background(), &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse, Model: "fixture-model", MaxTokens: intPointerInvariant(64),
		Messages: []llm.Message{{Role: "user", Content: llm.MessageContent{Content: stringPointerInvariant("run")}}},
		Input: []llm.Item{{Kind: llm.ItemKindMCPCall, MCPCall: &llm.MCPCall{
			ServerLabel: "docs", LogicalName: "search", ArgumentsJSON: json.RawMessage(`{"q":"x"}`),
		}}},
	})
	if err == nil || !strings.Contains(err.Error(), "canonical item") {
		t.Fatalf("unplanned canonical item was not rejected: %v", err)
	}
}

func TestAnthropicInboundRejectsUnencodableCanonicalOutputInsteadOfDroppingIt(t *testing.T) {
	t.Parallel()
	inbound := NewInboundTransformer()
	_, err := inbound.TransformResponse(context.Background(), &llm.Response{
		ID: "resp_invalid", Model: "fixture-model",
		Output: []llm.Item{{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{
			Kind: llm.ToolKindFunction, CallID: "call_1", Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "ok"}},
		}}},
	})
	if err == nil || !strings.Contains(err.Error(), "canonical output") {
		t.Fatalf("unencodable canonical output was not rejected: %v", err)
	}
}

func intPointerInvariant(value int64) *int64      { return &value }
func stringPointerInvariant(value string) *string { return &value }

package emulation

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/mcp"
)

func TestInternalToolErrorsUseFailedStatuses(t *testing.T) {
	t.Parallel()
	call := &llm.ToolInvocation{Kind: llm.ToolKindCustom, CallID: "call_1", LogicalName: "custom"}

	for name, item := range map[string]llm.Item{
		"MCP":               toolResultItem(gatewayCall{item: llm.Item{ToolCall: call}}, "failed", true),
		"custom constraint": customConstraintResult(call),
	} {
		if item.Status != llm.ItemStatusFailed || item.ToolResult == nil || !item.ToolResult.IsError ||
			item.ToolResult.Status != llm.ToolResultStatusFailed {
			t.Errorf("%s error has inconsistent statuses: %#v", name, item)
		}
	}
}

func TestInvalidMCPDiscoveryArgumentsUseBoundedFailedResult(t *testing.T) {
	t.Parallel()
	registry := mcp.NewEmptyRegistry([]byte(strings.Repeat("discovery-failure-key-", 2)))
	call := gatewayCall{
		item:    llm.Item{ToolCall: &llm.ToolInvocation{CallID: "search_call", LogicalName: "search"}},
		binding: mcp.Binding{Discovery: true, ServerLabel: "inventory"},
	}
	execution := failedMCPResultExecution(registry, call, "MCP tool arguments are invalid JSON")
	if execution.err != nil || execution.result.ToolResult == nil || !execution.result.ToolResult.IsError ||
		execution.public.Kind != llm.ItemKindMCPListTools || execution.public.MCPListTools == nil ||
		execution.public.MCPListTools.Error != "MCP tool arguments are invalid JSON" {
		t.Fatalf("invalid discovery result = %#v", execution)
	}
}

func TestMCPCallResultSerializationFailureTerminates(t *testing.T) {
	t.Parallel()
	registry := mcp.NewEmptyRegistry([]byte(strings.Repeat("result-encoding-key-", 2)))
	call := gatewayCall{
		item: llm.Item{ToolCall: &llm.ToolInvocation{CallID: "encoding_call", LogicalName: "lookup"}},
		binding: mcp.Binding{
			ServerLabel: "inventory",
			Tool:        llm.MCPDiscoveredTool{Name: "lookup"},
		},
	}
	execution := mcpCallResultExecution(context.Background(), registry, call, json.RawMessage(`{"sku":"A-1"}`), mcp.CallToolResult{
		StructuredContent: json.RawMessage(`{`),
	})
	if execution.err == nil || execution.err.Error() != "encode MCP result" || execution.result.ToolResult != nil || execution.public.MCPCall != nil {
		t.Fatalf("serialization failure execution = %#v", execution)
	}
}

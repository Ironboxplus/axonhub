package emulation

import (
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

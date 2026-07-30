package emulation

import (
	"testing"

	"github.com/looplj/axonhub/llm"
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

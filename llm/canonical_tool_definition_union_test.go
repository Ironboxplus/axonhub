package llm

import "testing"

func TestToolDefinitionValidateRejectsKindPayloadMismatches(t *testing.T) {
	t.Parallel()

	tests := []ToolDefinition{
		{
			Kind: ToolKindFunction, LogicalName: "lookup", Execution: ExecutionOwnerClient,
			Hosted: &HostedToolDefinition{Type: "function"},
		},
		{
			Kind: ToolKindCustom, LogicalName: "exec", Execution: ExecutionOwnerClient,
			Function: &FunctionDefinition{},
		},
		{
			Kind: ToolKindMCP, LogicalName: "inventory", Execution: ExecutionOwnerProvider,
			Freeform: &FreeformDefinition{},
		},
		{
			Kind: ToolKindWebSearch, LogicalName: "web_search", Execution: ExecutionOwnerProvider,
			MCP: &MCPDefinition{ServerLabel: "inventory", ServerURL: "https://mcp.example.invalid"},
		},
		{
			Kind: ToolKindUnknownBehavioral, LogicalName: "future", Execution: ExecutionOwnerProvider,
			Hosted: &HostedToolDefinition{},
		},
	}
	for index := range tests {
		if err := tests[index].Validate(); err == nil {
			t.Fatalf("definition %d unexpectedly validated: %#v", index, tests[index])
		}
	}
}

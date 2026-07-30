package conversion

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestPlannerAccountsForCanonicalToolObjects(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{
			{Kind: llm.ToolKindFunction, LogicalName: "lookup", Function: &llm.FunctionDefinition{}, Execution: llm.ExecutionOwnerClient},
			{Kind: llm.ToolKindCustom, LogicalName: "apply_patch", Freeform: &llm.FreeformDefinition{}, Execution: llm.ExecutionOwnerClient},
		},
		Input: []llm.Item{
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindCustom, CallID: "call_1", LogicalName: "apply_patch"}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindCustom, CallID: "call_1"}},
		},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil {
		t.Fatalf("plan canonical tool objects: %v", err)
	}
	if len(plan.Actions) != 5 || plan.Summary.Native != 1 || plan.Summary.Lowered != 4 || !plan.Complete() {
		t.Fatalf("canonical conversion plan = %#v", plan)
	}
	for _, action := range plan.Actions {
		if action.Ref.Kind == "" || action.Strategy == "" || action.Kind == ActionUnknown {
			t.Fatalf("unplanned canonical action = %#v", action)
		}
	}
}

func TestDefaultProfilesDeclareNativeToolCapabilities(t *testing.T) {
	tests := []struct {
		format llm.APIFormat
		want   ToolCapabilitySet
	}{
		{llm.APIFormatOpenAIChatCompletion, CapabilityFunctionTool},
		{llm.APIFormatOpenAIResponse, CapabilityFunctionTool | CapabilityCustomTool | CapabilityWebSearchTool | CapabilityImageGenerationTool |
			CapabilityMCPTool | CapabilityLocalShellTool | CapabilityToolSearch | CapabilityFileSearchTool |
			CapabilityCodeExecutionTool | CapabilityComputerTool | CapabilityShellTool | CapabilityApplyPatchTool},
		{llm.APIFormatAnthropicMessage, CapabilityFunctionTool | CapabilityWebSearchTool | CapabilityWebFetchTool | CapabilityCodeExecutionTool | CapabilityToolSearch},
	}
	for _, test := range tests {
		profile, ok := ProfileFor(test.format)
		if !ok || profile.ID == "" || profile.NativeTools != test.want {
			t.Fatalf("profile %s = %#v, ok=%v", test.format, profile, ok)
		}
	}
}

func TestLowerRewritesEveryNamedClientToolChoiceWithDefinitionIdentity(t *testing.T) {
	t.Parallel()
	for _, kind := range []llm.ToolKind{llm.ToolKindComputer, llm.ToolKindApplyPatch, llm.ToolKindLocalShell, llm.ToolKindToolSearch} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			request := &llm.Request{
				APIFormat: llm.APIFormatOpenAIResponse,
				ToolDefinitions: []llm.ToolDefinition{{
					Kind: kind, LogicalName: string(kind), Execution: llm.ExecutionOwnerClient,
					Hosted: &llm.HostedToolDefinition{Type: string(kind), Configuration: json.RawMessage(`{"type":"` + string(kind) + `"}`)},
				}},
				ToolChoice: &llm.ToolChoice{NamedToolChoice: &llm.NamedToolChoice{Type: string(kind)}},
			}
			plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
			if err != nil || !plan.Complete() {
				t.Fatalf("plan %s tool choice: plan=%#v err=%v", kind, plan, err)
			}
			lowered, _, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower %s tool choice: %v", kind, err)
			}
			definitionName := lowered.ToolDefinitions[0].LogicalName
			choice := lowered.ToolChoice.NamedToolChoice
			if lowered.ToolDefinitions[0].Kind != llm.ToolKindFunction || choice == nil || choice.Type != llm.ToolTypeFunction ||
				choice.Function.Name == "" || choice.Function.Name != definitionName {
				t.Fatalf("lowered %s choice/definition identity diverged: definition=%#v choice=%#v", kind, lowered.ToolDefinitions[0], choice)
			}
		})
	}
}

func TestAllowedToolsProjectsToFilteredDefinitionsAndTargetMode(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{
			{Kind: llm.ToolKindFunction, LogicalName: "keep", Function: &llm.FunctionDefinition{}, Execution: llm.ExecutionOwnerClient},
			{Kind: llm.ToolKindFunction, LogicalName: "drop", Function: &llm.FunctionDefinition{}, Execution: llm.ExecutionOwnerClient},
			{Kind: llm.ToolKindCustom, LogicalName: "patch", Freeform: &llm.FreeformDefinition{}, Execution: llm.ExecutionOwnerClient},
		},
		ToolChoice: &llm.ToolChoice{AllowedTools: &llm.AllowedToolChoice{
			Mode:  "required",
			Tools: []llm.AllowedToolRef{{Type: "function", Name: "keep"}, {Type: "custom", Name: "patch"}},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil || !plan.Complete() || plan.Summary.Unknown != 0 {
		t.Fatalf("plan allowed_tools projection: plan=%#v err=%v", plan, err)
	}
	lowered, _, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower allowed_tools: %v", err)
	}
	if len(lowered.ToolDefinitions) != 2 || lowered.ToolDefinitions[0].LogicalName != "keep" ||
		lowered.ToolDefinitions[1].Kind != llm.ToolKindFunction || lowered.ToolDefinitions[1].LogicalName == "patch" {
		t.Fatalf("filtered/lowered definitions = %#v", lowered.ToolDefinitions)
	}
	if lowered.ToolChoice == nil || lowered.ToolChoice.ToolChoice == nil || *lowered.ToolChoice.ToolChoice != "required" ||
		lowered.ToolChoice.AllowedTools != nil {
		t.Fatalf("projected target tool choice = %#v", lowered.ToolChoice)
	}
	if len(request.ToolDefinitions) != 3 || request.ToolChoice.AllowedTools == nil {
		t.Fatalf("source request mutated: %#v", request)
	}
}

func TestAllowedToolsNarrowsMCPServerTools(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "docs", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{ServerLabel: "docs", AllowedTools: &llm.MCPToolFilter{ToolNames: []string{"search", "fetch"}}},
		}},
		ToolChoice: &llm.ToolChoice{AllowedTools: &llm.AllowedToolChoice{
			Mode: "auto", Tools: []llm.AllowedToolRef{{Type: "mcp", ServerLabel: "docs", Name: "search"}},
		}},
	}
	projected, err := ProjectAllowedTools(request)
	if err != nil {
		t.Fatalf("project MCP allowed_tools: %v", err)
	}
	if len(projected.ToolDefinitions) != 1 || projected.ToolDefinitions[0].MCP == nil ||
		projected.ToolDefinitions[0].MCP.AllowedTools == nil ||
		len(projected.ToolDefinitions[0].MCP.AllowedTools.ToolNames) != 1 ||
		projected.ToolDefinitions[0].MCP.AllowedTools.ToolNames[0] != "search" {
		t.Fatalf("projected MCP filter = %#v", projected.ToolDefinitions)
	}
}

func TestPlannerPreservesProviderHostedObjectsWithinSameProtocol(t *testing.T) {
	t.Parallel()
	const providerToolKind llm.ToolKind = "computer_20250124"
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: providerToolKind, LogicalName: "computer", Execution: llm.ExecutionOwnerProvider,
			Hosted: &llm.HostedToolDefinition{Type: string(providerToolKind)},
		}},
		Input: []llm.Item{{
			Kind: llm.ItemKindHostedCall,
			HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{
				Kind: providerToolKind, CallID: "computer_call_1", LogicalName: "computer",
				Execution: llm.ExecutionOwnerProvider,
			}},
		}},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil || !plan.Complete() || plan.Summary.Native != 2 || plan.Summary.Unknown != 0 {
		t.Fatalf("same-protocol hosted plan = %#v, err=%v", plan, err)
	}
	for _, action := range plan.Actions {
		if action.Kind != ActionNative || action.Strategy != StrategyNative || !action.Reversible {
			t.Fatalf("same-protocol hosted action = %#v", action)
		}
	}
}

func TestPlannerReversiblyLowersResponsesLocalShellToFunctionProtocols(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindLocalShell, LogicalName: "local_shell", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)},
		}},
		Input: []llm.Item{
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindLocalShell, CallID: "shell_1", LogicalName: "local_shell",
				ArgumentsJSON: json.RawMessage(`{"type":"exec","command":["pwd"]}`), Execution: llm.ExecutionOwnerClient,
			}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindLocalShell, CallID: "shell_1", LogicalName: "local_shell",
				Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "ok"}},
			}},
		},
	}
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		plan, err := NewPlanner().Plan(request, target)
		if err != nil || !plan.Complete() || plan.Summary.Lowered != 3 || plan.Summary.Unknown != 0 {
			t.Fatalf("local shell plan to %s = %#v, err=%v", target, plan, err)
		}
		for _, action := range plan.Actions {
			if action.Ref.Kind == ObjectContentBlock {
				continue
			}
			if action.Strategy != StrategyClientToolAsFunc || !action.Reversible {
				t.Fatalf("local shell action to %s = %#v", target, action)
			}
		}
	}
}

func TestPlannerReversiblyLowersEveryClientExecutedToolFamilyToFunction(t *testing.T) {
	t.Parallel()
	for _, kind := range []llm.ToolKind{
		llm.ToolKindComputer, llm.ToolKindShell, llm.ToolKindApplyPatch,
		llm.ToolKindCodeExecution, llm.ToolKindFileSearch,
	} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			request := &llm.Request{
				APIFormat: llm.APIFormatAnthropicMessage,
				ToolDefinitions: []llm.ToolDefinition{{
					Kind: kind, LogicalName: string(kind), Execution: llm.ExecutionOwnerClient,
					Hosted: &llm.HostedToolDefinition{Type: string(kind), Configuration: json.RawMessage(`{"type":"` + string(kind) + `"}`)},
				}},
				Input: []llm.Item{
					{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
						Kind: kind, CallID: "call_1", LogicalName: string(kind), ArgumentsJSON: json.RawMessage(`{"action":"run"}`), Execution: llm.ExecutionOwnerClient,
					}},
					{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{
						Kind: kind, CallID: "call_1", LogicalName: string(kind), Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "ok"}},
					}},
				},
			}
			plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
			if err != nil || !plan.Complete() || plan.Summary.Lowered != 3 {
				t.Fatalf("%s plan = %#v, err=%v", kind, plan, err)
			}
			lowered, session, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower %s: %v", kind, err)
			}
			if lowered.ToolDefinitions[0].Kind != llm.ToolKindFunction || lowered.ToolDefinitions[0].Function == nil ||
				lowered.ToolDefinitions[0].Hosted != nil || lowered.Input[0].ToolCall.Kind != llm.ToolKindFunction ||
				lowered.Input[1].ToolResult.Kind != llm.ToolKindFunction {
				t.Fatalf("lowered %s = %#v", kind, lowered)
			}
			response := &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, CallID: "call_2", LogicalName: lowered.ToolDefinitions[0].LogicalName,
				ArgumentsJSON: json.RawMessage(`{"action":"next"}`), Execution: llm.ExecutionOwnerClient,
			}}}}
			restored := RestoreResponse(response, session)
			if restored.Output[0].ToolCall.Kind != kind || restored.Output[0].ToolCall.LogicalName != string(kind) {
				t.Fatalf("restored %s = %#v", kind, restored.Output[0].ToolCall)
			}
		})
	}
}

func TestLowerDoesNotRewriteNativeDefinitionBecauseAnotherToolNeedsLowering(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{
			{
				Kind: llm.ToolKindCustom, LogicalName: "native_custom", Execution: llm.ExecutionOwnerClient,
				Freeform: &llm.FreeformDefinition{Format: "text"},
			},
			{
				Kind: llm.ToolKindShell, LogicalName: "bash", Execution: llm.ExecutionOwnerClient,
				Hosted: &llm.HostedToolDefinition{Type: "bash_20250124", Configuration: json.RawMessage(`{"type":"bash_20250124","name":"bash"}`)},
			},
		},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatalf("plan mixed native/lowered tools: %v", err)
	}
	lowered, _, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower mixed native/lowered tools: %v", err)
	}
	if lowered.ToolDefinitions[0].Kind != llm.ToolKindCustom || lowered.ToolDefinitions[0].Freeform == nil {
		t.Fatalf("native custom definition was rewritten: %#v", lowered.ToolDefinitions[0])
	}
	if lowered.ToolDefinitions[1].Kind != llm.ToolKindFunction || lowered.ToolDefinitions[1].Function == nil {
		t.Fatalf("client shell definition was not lowered: %#v", lowered.ToolDefinitions[1])
	}
}

func TestPlannerReversiblyLowersClientToolSearchAndDiscoveredTools(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindToolSearch, LogicalName: "tool_search", Description: "Search deferred tools",
			Execution: llm.ExecutionOwnerClient,
			Function:  &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)},
		}},
		Input: []llm.Item{
			{
				Kind: llm.ItemKindToolCall, ID: "tsc_1",
				ToolCall: &llm.ToolInvocation{
					Kind: llm.ToolKindToolSearch, CallID: "search_1", LogicalName: "tool_search",
					ArgumentsJSON: json.RawMessage(`{"query":"calendar"}`), Execution: llm.ExecutionOwnerClient,
				},
			},
			{
				Kind: llm.ItemKindToolResult, ID: "tso_1",
				ToolResult: &llm.ToolResult{
					Kind: llm.ToolKindToolSearch, CallID: "search_1", LogicalName: "tool_search",
					Execution: llm.ExecutionOwnerClient,
					DiscoveredTools: []llm.ToolDefinition{{
						Kind: llm.ToolKindFunction, LogicalName: "create_event", Execution: llm.ExecutionOwnerClient,
						Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"title":{"type":"string"}}}`)},
					}},
				},
			},
		},
	}
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		plan, err := NewPlanner().Plan(request, target)
		if err != nil || !plan.Complete() || plan.Summary.Unknown != 0 || plan.Summary.Lowered != 4 {
			t.Fatalf("tool search plan to %s = %#v, err=%v", target, plan, err)
		}
		lowered, session, err := Lower(request, plan)
		if err != nil {
			t.Fatalf("lower tool search to %s: %v", target, err)
		}
		if len(lowered.ToolDefinitions) != 2 || lowered.ToolDefinitions[0].Kind != llm.ToolKindFunction ||
			lowered.ToolDefinitions[1].LogicalName != "create_event" || lowered.Input[0].ToolCall.Kind != llm.ToolKindFunction ||
			lowered.Input[1].ToolResult.Kind != llm.ToolKindFunction || len(lowered.Input[1].ToolResult.DiscoveredTools) != 0 ||
			len(lowered.Input[1].ToolResult.Content) != 1 {
			t.Fatalf("lowered tool search to %s = %#v", target, lowered)
		}
		response := &llm.Response{Output: []llm.Item{{
			Kind: llm.ItemKindToolCall,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, CallID: "search_2", LogicalName: lowered.ToolDefinitions[0].LogicalName,
				ArgumentsJSON: json.RawMessage(`{"query":"mail"}`), Execution: llm.ExecutionOwnerClient,
			},
		}}}
		restored := RestoreResponse(response, session)
		if restored.Output[0].ToolCall.Kind != llm.ToolKindToolSearch || restored.Output[0].ToolCall.LogicalName != "tool_search" {
			t.Fatalf("restored tool search from %s = %#v", target, restored.Output[0].ToolCall)
		}
	}
}

func TestLowerActivatesOnlyDiscoveredDeferredDefinitionsForClientToolSearch(t *testing.T) {
	t.Parallel()
	deferred := true
	parameters := json.RawMessage(`{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}`)
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		target := target
		t.Run(string(target), func(t *testing.T) {
			t.Parallel()
			request := &llm.Request{
				APIFormat: llm.APIFormatOpenAIResponse,
				ToolDefinitions: []llm.ToolDefinition{
					{
						Kind: llm.ToolKindToolSearch, LogicalName: "tool_search", Description: "Search deferred tools",
						Execution: llm.ExecutionOwnerClient,
						Function:  &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)},
					},
					{
						Kind: llm.ToolKindFunction, LogicalName: "always_available", Execution: llm.ExecutionOwnerClient,
						Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)},
					},
					{
						Kind: llm.ToolKindFunction, LogicalName: "create_event", Execution: llm.ExecutionOwnerClient,
						Function: &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), parameters...)}, DeferLoading: &deferred,
					},
					{
						Kind: llm.ToolKindFunction, LogicalName: "delete_event", Execution: llm.ExecutionOwnerClient,
						Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}}}`)}, DeferLoading: &deferred,
					},
				},
				Input: []llm.Item{
					{
						Kind: llm.ItemKindToolCall, ID: "tsc_1",
						ToolCall: &llm.ToolInvocation{
							Kind: llm.ToolKindToolSearch, CallID: "search_1", LogicalName: "tool_search",
							ArgumentsJSON: json.RawMessage(`{"query":"calendar"}`), Execution: llm.ExecutionOwnerClient,
						},
					},
					{
						Kind: llm.ItemKindToolResult, ID: "tso_1",
						ToolResult: &llm.ToolResult{
							Kind: llm.ToolKindToolSearch, CallID: "search_1", LogicalName: "tool_search",
							Execution: llm.ExecutionOwnerClient,
							DiscoveredTools: []llm.ToolDefinition{{
								Kind: llm.ToolKindFunction, LogicalName: "create_event", Execution: llm.ExecutionOwnerClient,
								Function: &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), parameters...)}, DeferLoading: &deferred,
							}},
						},
					},
				},
			}

			plan, err := NewPlanner().Plan(request, target)
			if err != nil || !plan.Complete() {
				t.Fatalf("plan client tool search to %s: complete=%v err=%v plan=%#v", target, plan != nil && plan.Complete(), err, plan)
			}
			lowered, _, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower client tool search to %s: %v", target, err)
			}
			definitions := make(map[string]llm.ToolDefinition, len(lowered.ToolDefinitions))
			for _, definition := range lowered.ToolDefinitions {
				definitions[definition.LogicalName] = definition
			}
			if _, ok := definitions["delete_event"]; ok {
				t.Fatalf("undiscovered deferred tool leaked to %s: %#v", target, lowered.ToolDefinitions)
			}
			selected, ok := definitions["create_event"]
			if !ok || selected.DeferLoading != nil {
				t.Fatalf("discovered tool was not activated for %s: %#v", target, lowered.ToolDefinitions)
			}
			if _, ok := definitions["always_available"]; !ok {
				t.Fatalf("eager tool disappeared for %s: %#v", target, lowered.ToolDefinitions)
			}
			if len(definitions) != 3 {
				t.Fatalf("lowered definitions for %s = %#v, want search + eager + discovered", target, lowered.ToolDefinitions)
			}
			if request.ToolDefinitions[2].DeferLoading == nil || !*request.ToolDefinitions[2].DeferLoading ||
				request.ToolDefinitions[3].DeferLoading == nil || !*request.ToolDefinitions[3].DeferLoading {
				t.Fatalf("lowering mutated source deferred metadata: %#v", request.ToolDefinitions)
			}
		})
	}
}

func TestPlannerRejectsUnknownBehavioralCanonicalTool(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindUnknownBehavioral, LogicalName: "future_tool",
			Hosted: &llm.HostedToolDefinition{Type: "future_tool"}, Execution: llm.ExecutionOwnerProvider,
		}},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if !errors.Is(err, ErrIncompletePlan) || plan == nil || plan.Summary.Unknown != 1 || plan.Summary.Complete {
		t.Fatalf("unknown behavioral plan = %#v, err=%v", plan, err)
	}
}

func TestPlannerRoutesTypedMCPLifecycleThroughGatewayEmulation(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{{
			Kind: llm.ItemKindMCPCall, ID: "mcp_call_1",
			MCPCall: &llm.MCPCall{
				ServerLabel: "inventory", LogicalName: "lookup",
				ArgumentsJSON: json.RawMessage(`{"sku":"A-1"}`), Status: llm.MCPCallStatusCompleted,
			},
		}},
	}

	native, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || !native.Complete() {
		t.Fatalf("Responses MCP identity plan = %#v, err=%v", native, err)
	}
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		profile, _ := ProfileFor(target)
		profile.EmulatedTools |= CapabilityMCPTool
		plan, planErr := NewPlannerWithProfile(profile).Plan(request, target)
		if planErr != nil || plan == nil || plan.Summary.Emulated != 1 || plan.Summary.Unknown != 0 || !plan.Complete() {
			t.Fatalf("MCP plan to %s = %#v, err=%v", target, plan, planErr)
		}
	}
	withoutGateway, withoutGatewayErr := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if !errors.Is(withoutGatewayErr, ErrIncompletePlan) || withoutGateway == nil || withoutGateway.Summary.Unknown != 1 {
		t.Fatalf("MCP plan without declared gateway = %#v, err=%v", withoutGateway, withoutGatewayErr)
	}
}

func TestPlannerRoutesMCPDefinitionThroughGatewayEmulation(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{ServerLabel: "inventory", ServerURL: "https://mcp.example.test/rpc"},
		}},
	}
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		profile, _ := ProfileFor(target)
		profile.EmulatedTools |= CapabilityMCPTool
		plan, err := NewPlannerWithProfile(profile).Plan(request, target)
		if err != nil || !plan.Complete() || plan.Summary.Emulated != 1 || plan.Actions[0].Strategy != StrategyMCPGateway {
			t.Fatalf("MCP definition plan to %s = %#v, err=%v", target, plan, err)
		}
	}
	withoutGateway, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if !errors.Is(err, ErrIncompletePlan) || withoutGateway == nil || withoutGateway.Summary.Unknown != 1 || withoutGateway.Complete() {
		t.Fatalf("MCP definition without gateway = %#v, err=%v", withoutGateway, err)
	}
}

func TestPlannerLowersCrossProtocolReasoningAndRejectsOnlyUnknownContent(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindUnknown, UnknownRaw: json.RawMessage(`{"type":"future"}`)}}},
			{Kind: llm.ItemKindReasoning, Role: llm.RoleAssistant, Reasoning: &llm.ReasoningItem{Content: "private", Signature: "opaque"}},
		},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if !errors.Is(err, ErrIncompletePlan) {
		t.Fatalf("planner error = %v, want ErrIncompletePlan", err)
	}
	if plan.Summary.Native != 1 || plan.Summary.Lowered != 1 || plan.Summary.Unknown != 1 || plan.Summary.Complete {
		t.Fatalf("planner summary did not account for message/content/reasoning: %#v", plan.Summary)
	}
	if got := plan.Actions[len(plan.Actions)-1]; got.Kind != ActionLower || got.Strategy != StrategyReasoningProject {
		t.Fatalf("reasoning action = %#v, want reasoning projection", got)
	}
}

func TestPlannerKeepsResponsesPrivateSidecarsOnlyOnNativeResponsesTarget(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ProviderExtensions: &llm.ProviderExtensions{OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
			Request: &llm.OpenAIResponsesRequestExtensions{RawTools: []llm.OpenAIResponsesRawFragment{{
				Type: "future_hosted_tool", OriginalIndex: 0, Raw: []byte(`{"type":"future_hosted_tool"}`),
			}}},
		}},
	}

	native, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || native.Summary.Opaque != 1 || !native.Complete() {
		t.Fatalf("native sidecar plan = %#v, err=%v", native, err)
	}
	cross, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if !errors.Is(err, ErrIncompletePlan) || cross.Summary.Unknown != 1 || cross.Complete() {
		t.Fatalf("cross-protocol sidecar plan = %#v, err=%v", cross, err)
	}
}

func TestPlannerAllowsUnknownItemOnlyOnIdentityRoute(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		Input: []llm.Item{{
			Kind: llm.ItemKindUnknown,
			Unknown: &llm.UnknownItem{
				Type: "future_hosted_call", Raw: json.RawMessage(`{"type":"future_hosted_call","status":"in_progress"}`), Behavioral: true,
			},
		}},
	}

	identity, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || identity.Summary.Opaque != 1 || !identity.Complete() {
		t.Fatalf("identity opaque plan = %#v, err=%v", identity, err)
	}
	cross, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if !errors.Is(err, ErrIncompletePlan) || cross.Summary.Unknown != 1 || cross.Complete() {
		t.Fatalf("cross unknown plan = %#v, err=%v", cross, err)
	}
}

func TestPlannerRejectsInvalidFunctionArgumentsForAnthropic(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIChatCompletion,
		Input: []llm.Item{{
			Kind: llm.ItemKindToolCall,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, CallID: "call_invalid", LogicalName: "lookup", ArgumentsText: `{"city":`,
			},
		}},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if !errors.Is(err, ErrIncompletePlan) || plan.Summary.Unknown != 1 || plan.Complete() {
		t.Fatalf("invalid Anthropic input plan = %#v, err=%v", plan, err)
	}
	chat, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if err != nil || chat.Summary.Native != 1 || !chat.Complete() {
		t.Fatalf("Chat string arguments plan = %#v, err=%v", chat, err)
	}
}

func TestPlannerProjectsCrossProtocolCitationPublicState(t *testing.T) {
	encryptedIndex, citedText := "opaque-index", "quoted source"
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		Input: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{
				{Kind: llm.ContentKindText, Text: "answer"},
				{Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{
					URL: "https://example.com", EncryptedIndex: &encryptedIndex, CitedText: &citedText,
				}},
			},
		}},
	}
	cross, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if err != nil || cross.Summary.Lowered != 1 || cross.Summary.Unknown != 0 || !cross.Complete() {
		t.Fatalf("private citation cross plan = %#v, err=%v", cross, err)
	}
	identity, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil || !identity.Complete() {
		t.Fatalf("private citation identity plan = %#v, err=%v", identity, err)
	}
}

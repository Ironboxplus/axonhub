package conversion

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesLiteHeaderPlansOnlyNamespacedCustomAsFunction(t *testing.T) {
	t.Parallel()

	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		RawRequest: &httpclient.Request{Headers: http.Header{
			"X-OpenAI-Internal-Codex-Responses-Lite": []string{"true"},
		}},
		ToolDefinitions: []llm.ToolDefinition{
			{
				Kind: llm.ToolKindCustom, LogicalName: "top_level_exec", Execution: llm.ExecutionOwnerClient,
				Freeform: &llm.FreeformDefinition{Format: "grammar", Syntax: "lark", Definition: "start: source"},
			},
			{
				Kind: llm.ToolKindCustom, LogicalName: "agents__exec", Execution: llm.ExecutionOwnerClient,
				Freeform: &llm.FreeformDefinition{Format: "grammar", Syntax: "lark", Definition: "start: source", Namespace: "agents"},
			},
		},
	}
	if !llm.UsesResponsesLiteWireProfile(request) {
		t.Fatalf("Responses Lite raw header did not select the shared wire profile")
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() {
		t.Fatalf("plan Responses Lite namespace restriction = %#v, err=%v", plan, err)
	}
	var topLevel, namespaced *Action
	for index := range plan.Actions {
		action := &plan.Actions[index]
		if action.Ref.Kind != ObjectToolDefinition {
			continue
		}
		switch action.Ref.ToolIndex {
		case 0:
			topLevel = action
		case 1:
			namespaced = action
		}
	}
	if topLevel == nil || topLevel.Kind != ActionNative || topLevel.Strategy != StrategyNative {
		t.Fatalf("top-level custom action = %#v, want native", topLevel)
	}
	if namespaced == nil || namespaced.Kind != ActionLower || namespaced.Strategy != StrategyCustomAsFunction ||
		namespaced.Reason != ReasonTargetFunctionOnly || !namespaced.Reversible || plan.Summary.Lowered != 1 {
		t.Fatalf("namespaced custom lowering is not visible in plan summary: action=%#v plan=%#v", namespaced, plan)
	}
}

func TestResponsesLitePlannerHelpersCoverScopedCapabilityBranches(t *testing.T) {
	t.Parallel()

	base, ok := ProfileFor(llm.APIFormatOpenAIResponse)
	if !ok {
		t.Fatal("Responses capability profile is unavailable")
	}
	if got := effectiveRequestCapabilityProfile(base, nil); got != base {
		t.Fatalf("ordinary Responses profile changed without Lite signal: %#v", got)
	}
	chat, _ := ProfileFor(llm.APIFormatOpenAIChatCompletion)
	liteRequest := &llm.Request{TransformerMetadata: map[string]any{
		llm.ResponsesWireProfileMetadataKey: llm.ResponsesWireProfileLiteValue,
	}}
	if got := effectiveRequestCapabilityProfile(chat, liteRequest); got != chat {
		t.Fatalf("non-Responses target acquired a Lite restriction: %#v", got)
	}
	lite := effectiveRequestCapabilityProfile(base, liteRequest)
	if lite.NamespaceChildNativeTools != CapabilityFunctionTool || lite.NativeTools != base.NativeTools ||
		!strings.Contains(lite.ID, responsesLiteProfileSuffix) {
		t.Fatalf("Lite profile did not scope function-only restriction to namespace children: %#v", lite)
	}
	if repeated := effectiveRequestCapabilityProfile(lite, liteRequest); repeated.ID != lite.ID {
		t.Fatalf("Lite profile suffix was duplicated: %#v", repeated)
	}
	if got := namespaceChildCapabilityProfile(base, ""); got != base {
		t.Fatalf("empty namespace changed capability profile: %#v", got)
	}
	if got := namespaceChildCapabilityProfile(CapabilityProfile{NativeTools: base.NativeTools}, "agents"); got.NativeTools != base.NativeTools {
		t.Fatalf("unset namespace rule changed capability profile: %#v", got)
	}
	if got := namespaceChildCapabilityProfile(lite, "agents"); got.NativeTools != CapabilityFunctionTool {
		t.Fatalf("namespace capability = %b, want only function", got.NativeTools)
	}

	if action := actionForClientToolKind(llm.ToolKindCustom, base, ObjectRef{}); action.Kind != ActionNative {
		t.Fatalf("native top-level custom action = %#v", action)
	}
	functionOnly := CapabilityProfile{NativeTools: CapabilityFunctionTool}
	if action := actionForClientToolKind(llm.ToolKindCustom, functionOnly, ObjectRef{}); action.Kind != ActionLower || action.Strategy != StrategyCustomAsFunction {
		t.Fatalf("function-only custom action = %#v", action)
	}
	if action := actionForClientToolKind(llm.ToolKindShell, functionOnly, ObjectRef{}); action.Kind != ActionLower || action.Strategy != StrategyClientToolAsFunc {
		t.Fatalf("function-only client tool action = %#v", action)
	}
	if action := actionForClientToolKind(llm.ToolKindCustom, CapabilityProfile{}, ObjectRef{}); action.Kind != ActionUnknown {
		t.Fatalf("unsupported custom action = %#v", action)
	}

	result := &llm.ToolResult{CallID: "call_1"}
	if toolResultNamespace(nil, result) != "" || toolResultNamespace(&llm.Request{}, nil) != "" ||
		toolResultNamespace(&llm.Request{}, &llm.ToolResult{}) != "" ||
		toolResultNamespace(&llm.Request{}, result) != "" {
		t.Fatal("unrelated tool result acquired a namespace")
	}
	request := &llm.Request{Input: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
		CallID: "call_1", Namespace: "agents",
	}}}}
	if got := toolResultNamespace(request, result); got != "agents" {
		t.Fatalf("tool result namespace = %q, want agents", got)
	}
}

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

func TestPlannerUsesNativeResponsesMCPWhenFilterIsPortable(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{
				ServerLabel: "inventory", ServerURL: "https://mcp.example.test/rpc",
				AllowedTools: &llm.MCPToolFilter{ToolNames: []string{"lookup", "reserve"}},
			},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() || plan.Summary.Native != 1 || plan.Actions[0].Strategy != StrategyNative {
		t.Fatalf("portable MCP to Responses plan = %#v, err=%v", plan, err)
	}
}

func TestPlannerRoutesPortableResponsesMCPThroughGatewayWhenTargetProfileDoesNotSupportNativeMCP(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{
				ServerLabel: "inventory", ServerURL: "https://mcp.example.test/rpc",
				AllowedTools: &llm.MCPToolFilter{ToolNames: []string{"lookup"}},
			},
		}},
	}
	profile, _ := ProfileFor(llm.APIFormatOpenAIResponse)
	profile.NativeTools &^= CapabilityMCPTool
	profile.EmulatedTools |= CapabilityMCPTool
	plan, err := NewPlannerWithProfile(profile).Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() || plan.Summary.Emulated != 1 || plan.Actions[0].Strategy != StrategyMCPGateway {
		t.Fatalf("portable MCP to a Responses-compatible non-native provider = %#v, err=%v", plan, err)
	}
}

func TestPlannerRoutesSameProtocolHostedWebSearchThroughGatewayWhenProviderIsNotAdmitted(t *testing.T) {
	t.Parallel()
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
			Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{}},
		}},
		Input: []llm.Item{{
			Kind: llm.ItemKindHostedCall,
			HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{
				Kind: llm.ToolKindWebSearch, CallID: "search_1", LogicalName: "web_search",
				Execution: llm.ExecutionOwnerProvider,
			}},
		}},
	}
	profile, _ := ProfileFor(llm.APIFormatOpenAIResponse)
	profile.NativeTools &^= CapabilityWebSearchTool
	profile.EmulatedTools |= CapabilityWebSearchTool

	plan, err := NewPlannerWithProfile(profile).Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() || plan.Summary.Emulated != 2 || plan.Summary.Native != 0 {
		t.Fatalf("same-protocol non-native web-search plan = %#v, err=%v", plan, err)
	}
	for _, action := range plan.Actions {
		if action.Kind != ActionEmulate || action.Strategy != StrategyHostedGateway || !action.Reversible {
			t.Fatalf("same-protocol non-native web-search action = %#v", action)
		}
	}
}

func TestHostedPlannerBranchesFailClosedOrProjectExplicitly(t *testing.T) {
	t.Parallel()
	responsesProfile, _ := ProfileFor(llm.APIFormatOpenAIResponse)
	chatProfile, _ := ProfileFor(llm.APIFormatOpenAIChatCompletion)
	ref := ObjectRef{Kind: ObjectToolDefinition, ToolIndex: 0, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}

	if action := actionForHostedCall(llm.APIFormatOpenAIResponse, responsesProfile, nil, 0); action.Kind != ActionUnknown {
		t.Fatalf("nil hosted call action = %#v", action)
	}
	responsesWithoutSearch := responsesProfile
	responsesWithoutSearch.NativeTools &^= CapabilityWebSearchTool
	unsupportedCall := &llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{
		Kind: llm.ToolKindWebSearch, Execution: llm.ExecutionOwnerProvider,
	}}}
	if action := actionForHostedCall(llm.APIFormatOpenAIResponse, responsesWithoutSearch, unsupportedCall, 0); action.Kind != ActionUnknown {
		t.Fatalf("unadmitted hosted call action = %#v", action)
	}

	tests := []struct {
		name       string
		definition *llm.ToolDefinition
		wantKind   ActionKind
		want       StrategyID
	}{
		{name: "nil", wantKind: ActionUnknown, want: StrategyUnavailable},
		{name: "provider tool search unavailable", definition: &llm.ToolDefinition{
			Kind: llm.ToolKindToolSearch, Execution: llm.ExecutionOwnerProvider,
		}, wantKind: ActionUnknown, want: StrategyUnavailable},
		{name: "cross protocol web search missing typed hosted config", definition: &llm.ToolDefinition{
			Kind: llm.ToolKindWebSearch, Execution: llm.ExecutionOwnerProvider,
		}, wantKind: ActionUnknown, want: StrategyUnavailable},
		{name: "unknown client tool projects as function", definition: &llm.ToolDefinition{
			Kind: llm.ToolKind("future_client_tool"), Execution: llm.ExecutionOwnerClient,
		}, wantKind: ActionLower, want: StrategyClientToolAsFunc},
		{name: "provider non hosted definition falls through policy", definition: &llm.ToolDefinition{
			Kind: llm.ToolKindFunction, Execution: llm.ExecutionOwnerProvider,
		}, wantKind: ActionNative, want: StrategyNative},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			action := actionForToolDefinition(test.definition, llm.APIFormatOpenAIResponse, chatProfile, ref)
			if action.Kind != test.wantKind || action.Strategy != test.want {
				t.Fatalf("tool definition action = %#v", action)
			}
		})
	}
}

func TestProfileAdmitsToolCapabilitySeparatesProtocolIdentityFromProviderAdmission(t *testing.T) {
	t.Parallel()
	profile, _ := ProfileFor(llm.APIFormatOpenAIResponse)
	for _, test := range []struct {
		name       string
		source     llm.APIFormat
		capability ToolCapabilitySet
		mutate     func(*CapabilityProfile)
		want       bool
	}{
		{name: "known native", source: llm.APIFormatOpenAIResponse, capability: CapabilityWebSearchTool, want: true},
		{name: "known capability removed", source: llm.APIFormatOpenAIResponse, capability: CapabilityWebSearchTool, mutate: func(profile *CapabilityProfile) { profile.NativeTools &^= CapabilityWebSearchTool }},
		{name: "unknown same protocol", source: llm.APIFormatOpenAIResponse, capability: 0, want: true},
		{name: "unknown cross protocol", source: llm.APIFormatAnthropicMessage, capability: 0, want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := profile
			if test.mutate != nil {
				test.mutate(&candidate)
			}
			if got := profileAdmitsToolCapability(test.source, candidate, test.capability); got != test.want {
				t.Fatalf("profileAdmitsToolCapability() = %v, want %v", got, test.want)
			}
		})
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

func TestPlannerKeepsResponsesOpaqueCanonicalToolOnlyOnNativeResponsesTarget(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindUnknownBehavioral, LogicalName: "future_hosted_tool",
			Hosted:    &llm.HostedToolDefinition{Type: "future_hosted_tool", Configuration: []byte(`{"type":"future_hosted_tool"}`)},
			Execution: llm.ExecutionOwnerProvider,
		}},
	}

	native, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || native.Summary.Native != 1 || !native.Complete() {
		t.Fatalf("native sidecar plan = %#v, err=%v", native, err)
	}
	cross, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if !errors.Is(err, ErrIncompletePlan) || cross.Summary.Unknown != 1 || cross.Complete() {
		t.Fatalf("cross-protocol sidecar plan = %#v, err=%v", cross, err)
	}
}

func TestPlannerProjectsResponsesReasoningContextAsExplicitCrossProtocolDegradation(t *testing.T) {
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ProviderExtensions: &llm.ProviderExtensions{OpenAIResponses: &llm.OpenAIResponsesProviderExtensions{
			Request: &llm.OpenAIResponsesRequestExtensions{ReasoningContext: "all_turns"},
		}},
	}

	identity, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || len(identity.Actions) != 1 {
		t.Fatalf("identity reasoning context plan = %#v, err=%v", identity, err)
	}
	if action := identity.Actions[0]; action.Kind != ActionOpaque ||
		action.Strategy != StrategyOpaqueSidecar || action.Reason != ReasonSameProtocolOpaque || !action.Reversible {
		t.Fatalf("identity reasoning context action = %#v", action)
	}

	for _, target := range []llm.APIFormat{
		llm.APIFormatOpenAIChatCompletion,
		llm.APIFormatAnthropicMessage,
	} {
		cross, planErr := NewPlanner().Plan(request, target)
		if planErr != nil || len(cross.Actions) != 1 || !cross.Complete() || cross.Summary.Unknown != 0 {
			t.Fatalf("cross-protocol reasoning context plan for %s = %#v, err=%v", target, cross, planErr)
		}
		if action := cross.Actions[0]; action.Kind != ActionLower ||
			action.Strategy != StrategyReasoningProject || action.Reason != ReasonProtocolConstraint || action.Reversible {
			t.Fatalf("cross-protocol reasoning context action for %s = %#v", target, action)
		}
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

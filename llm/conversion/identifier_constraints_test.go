package conversion

import (
	"context"
	"encoding/json"
	"regexp"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"pgregory.net/rapid"
)

var portableIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func TestLowerNormalizesTargetIdentifiersWithoutBreakingToolAssociations(t *testing.T) {
	t.Parallel()
	invalidName := "read.file"
	collidingName := "read_file"
	invalidCallID := "call.with space:1"
	longCallID := "toolu_" + strings.Repeat("x", 80)
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{
			{Kind: llm.ToolKindFunction, LogicalName: invalidName, Execution: llm.ExecutionOwnerClient, Function: &llm.FunctionDefinition{}},
			{Kind: llm.ToolKindFunction, LogicalName: collidingName, Execution: llm.ExecutionOwnerClient, Function: &llm.FunctionDefinition{}},
		},
		ToolChoice: &llm.ToolChoice{NamedToolChoice: &llm.NamedToolChoice{
			Type: llm.ToolTypeFunction, Function: llm.ToolFunction{Name: invalidName},
		}},
		Input: []llm.Item{
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: invalidCallID, LogicalName: invalidName}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindFunction, CallID: invalidCallID, LogicalName: invalidName}},
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: longCallID, LogicalName: collidingName}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindFunction, CallID: longCallID, LogicalName: collidingName}},
		},
	}

	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil {
		t.Fatalf("plan identifier normalization: %v", err)
	}
	if !planHasStrategy(plan, StrategyIdentifierNormalize) {
		t.Fatalf("plan omitted identifier normalization: %#v", plan.Actions)
	}
	plan.Debug = llm.NewConversionDebugTrace(
		llm.WithConversionDebugTrace(context.Background(), []byte(t.Name()), 32),
		len(plan.Actions)+8,
	)
	lowered, session, err := lower(request, plan, true)
	if err != nil {
		t.Fatalf("lower identifier constraints: %v", err)
	}

	if request.ToolDefinitions[0].LogicalName != invalidName || request.Input[0].ToolCall.CallID != invalidCallID {
		t.Fatalf("source request mutated: %#v", request)
	}
	firstName := lowered.ToolDefinitions[0].LogicalName
	secondName := lowered.ToolDefinitions[1].LogicalName
	if !portableIdentifierPattern.MatchString(firstName) || !portableIdentifierPattern.MatchString(secondName) || firstName == secondName {
		t.Fatalf("target names are invalid or collided: %q %q", firstName, secondName)
	}
	if lowered.ToolChoice == nil || lowered.ToolChoice.NamedToolChoice == nil || lowered.ToolChoice.NamedToolChoice.Function.Name != firstName {
		t.Fatalf("named tool choice diverged from definition: %#v", lowered.ToolChoice)
	}
	if lowered.Input[0].ToolCall.LogicalName != firstName || lowered.Input[1].ToolResult.LogicalName != firstName {
		t.Fatalf("call/result name association diverged: %#v", lowered.Input[:2])
	}
	if lowered.Input[2].ToolCall.LogicalName != secondName || lowered.Input[3].ToolResult.LogicalName != secondName {
		t.Fatalf("second call/result name association diverged: %#v", lowered.Input[2:])
	}
	for _, pair := range [][2]string{
		{lowered.Input[0].ToolCall.CallID, lowered.Input[1].ToolResult.CallID},
		{lowered.Input[2].ToolCall.CallID, lowered.Input[3].ToolResult.CallID},
	} {
		if pair[0] != pair[1] || !portableIdentifierPattern.MatchString(pair[0]) {
			t.Fatalf("target call/result ID association is invalid: %#v", pair)
		}
	}
	if got := session.Summary().IdentifiersNormalized; got < 4 {
		t.Fatalf("normalized identifier count = %d, want at least 4", got)
	}
	trace := session.DebugTrace()
	if trace == nil || !debugTraceHasStrategy(trace, StrategyIdentifierNormalize) {
		t.Fatalf("identifier normalization missing from debug trace: %#v", trace)
	}
}

func TestIdentifierNormalizationProperties(t *testing.T) {
	t.Parallel()
	rapid.Check(t, func(t *rapid.T) {
		source := rapid.StringMatching(`.{1,256}`).Draw(t, "source")
		first := newIdentifierLedger(identifierRule{maxLength: 64, portable: true, fallback: "toolu"})
		second := newIdentifierLedger(identifierRule{maxLength: 64, portable: true, fallback: "toolu"})
		mapped, _ := first.mapValue(source)
		repeat, _ := first.mapValue(source)
		independent, _ := second.mapValue(source)
		if mapped != repeat || mapped != independent {
			t.Fatalf("mapping is not deterministic: %q %q %q", mapped, repeat, independent)
		}
		if !portableIdentifierPattern.MatchString(mapped) {
			t.Fatalf("mapping violates target constraints: source=%q mapped=%q", source, mapped)
		}
	})
}

func TestIdentifierLedgerResolvesAdversarialTargetCollision(t *testing.T) {
	t.Parallel()
	ledger := newIdentifierLedger(identifierRule{maxLength: 64, portable: true, fallback: "tool"})
	changed, _ := ledger.mapValue("read.file")
	unchanged, normalized := ledger.mapValue("already_valid")
	if normalized {
		t.Fatalf("valid identifier unexpectedly normalized: %q", unchanged)
	}
	validSource := changed
	validMapped, validChanged := ledger.mapValue(validSource)
	if !validChanged || changed == validMapped || !portableIdentifierPattern.MatchString(validMapped) {
		t.Fatalf("collision was not resolved: changed=%q valid=%q", changed, validMapped)
	}
}

func TestRestoreResponseNormalizesProviderCallIDForAnthropicClient(t *testing.T) {
	t.Parallel()
	request := &llm.Request{APIFormat: llm.APIFormatAnthropicMessage}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	_, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower: %v", err)
	}
	original := "call.with spaces:" + strings.Repeat("z", 80)
	response := &llm.Response{
		Output: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, CallID: original, LogicalName: "lookup",
		}}},
		Choices: []llm.Choice{{Index: 0, Message: &llm.Message{ToolCalls: []llm.ToolCall{{
			ID: original, Type: llm.ToolTypeFunction, Function: llm.FunctionCall{Name: "lookup"},
		}}}}},
	}

	RestoreResponse(response, session)
	canonicalID := response.Output[0].ToolCall.CallID
	legacyID := response.Choices[0].Message.ToolCalls[0].ID
	if canonicalID != legacyID || !portableIdentifierPattern.MatchString(canonicalID) || canonicalID == original {
		t.Fatalf("provider response ID was not consistently normalized: canonical=%q legacy=%q", canonicalID, legacyID)
	}
}

func TestIdentityNormalizesToolNamesWithoutRewritingOpaqueCallIDs(t *testing.T) {
	t.Parallel()
	formats := []llm.APIFormat{
		llm.APIFormatOpenAIChatCompletion,
		llm.APIFormatOpenAIResponse,
		llm.APIFormatAnthropicMessage,
	}
	for _, format := range formats {
		format := format
		t.Run(string(format), func(t *testing.T) {
			originalName := "read.file"
			originalCallID := "call.with spaces:" + strings.Repeat("x", 80)
			request := &llm.Request{
				APIFormat: format,
				ToolDefinitions: []llm.ToolDefinition{{
					Kind: llm.ToolKindFunction, LogicalName: originalName, Execution: llm.ExecutionOwnerClient,
					Function: &llm.FunctionDefinition{Parameters: []byte(`{"type":"object"}`)},
				}},
				ToolChoice: &llm.ToolChoice{NamedToolChoice: &llm.NamedToolChoice{
					Type: llm.ToolTypeFunction, Function: llm.ToolFunction{Name: originalName},
				}},
				Input: []llm.Item{
					{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: originalCallID, LogicalName: originalName}},
					{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Kind: llm.ToolKindFunction, CallID: originalCallID, LogicalName: originalName}},
				},
			}
			plan, err := NewPlanner().Plan(request, format)
			if err != nil {
				t.Fatalf("plan identity identifier normalization: %v", err)
			}
			if !planHasStrategy(plan, StrategyIdentifierNormalize) {
				t.Fatalf("identity plan omitted identifier normalization: %#v", plan.Actions)
			}
			lowered, session, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower identity identifiers: %v", err)
			}
			mappedName := lowered.ToolDefinitions[0].LogicalName
			mappedCallID := lowered.Input[0].ToolCall.CallID
			if mappedName == originalName || !portableIdentifierPattern.MatchString(mappedName) {
				t.Fatalf("identity tool name was not normalized: %q", mappedName)
			}
			if lowered.ToolChoice.NamedToolChoice.Function.Name != mappedName || lowered.Input[1].ToolResult.LogicalName != mappedName {
				t.Fatalf("identity tool references diverged: %#v", lowered)
			}
			if mappedCallID != originalCallID || lowered.Input[1].ToolResult.CallID != originalCallID {
				t.Fatalf("same-protocol call ID changed: %#v", lowered.Input)
			}

			response := &llm.Response{Output: []llm.Item{{
				Kind:     llm.ItemKindToolCall,
				ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, CallID: mappedCallID, LogicalName: mappedName},
			}}}
			RestoreResponse(response, session)
			if response.Output[0].ToolCall.LogicalName != originalName {
				t.Fatalf("identity tool name was not restored: got %q want %q", response.Output[0].ToolCall.LogicalName, originalName)
			}
			if response.Output[0].ToolCall.CallID != mappedCallID {
				t.Fatalf("new provider call ID was rewritten: got %q want %q", response.Output[0].ToolCall.CallID, mappedCallID)
			}
		})
	}
}

func TestResponsesIdentityRestoresNamespacedToolNamesWithoutGuessingCallIDCorrelation(t *testing.T) {
	t.Parallel()
	const namespace = "project"
	const sourceChild = "read.file"
	originalCallID := "history_call_" + strings.Repeat("x", 80)
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: namespace + "__" + sourceChild, Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{Namespace: namespace, Parameters: json.RawMessage(`{"type":"object"}`)},
		}},
		Input: []llm.Item{
			{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, Namespace: namespace, LogicalName: sourceChild, CallID: originalCallID,
			}},
			{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindFunction, CallID: originalCallID,
			}},
		},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatalf("plan namespaced identity route: %v", err)
	}
	lowered, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower namespaced identity route: %v", err)
	}
	mappedFullName := lowered.ToolDefinitions[0].LogicalName
	mappedChildName := strings.TrimPrefix(mappedFullName, namespace+"__")
	mappedCallID := lowered.Input[0].ToolCall.CallID
	if mappedChildName == sourceChild || mappedCallID != originalCallID {
		t.Fatalf("request identifiers were not normalized: name=%q call_id=%q", mappedChildName, mappedCallID)
	}
	if lowered.Input[0].ToolCall.LogicalName != mappedChildName {
		t.Fatalf("namespace definition/call diverged: definition=%q call=%q", mappedFullName, lowered.Input[0].ToolCall.LogicalName)
	}

	response := &llm.Response{Output: []llm.Item{{
		Kind: llm.ItemKindToolCall,
		ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, Namespace: namespace, LogicalName: mappedChildName,
			CallID: mappedCallID, ArgumentsJSON: json.RawMessage(`{"path":"README.md"}`),
		},
	}}}
	RestoreResponse(response, session)
	call := response.Output[0].ToolCall
	if call.LogicalName != sourceChild || call.Namespace != namespace {
		t.Fatalf("namespaced response identity = %s/%s, want %s/%s", call.Namespace, call.LogicalName, namespace, sourceChild)
	}
	if call.CallID != mappedCallID {
		t.Fatalf("provider-generated call ID collision was rewritten: got %q want %q", call.CallID, mappedCallID)
	}

	outputIndex := 0
	stream := &llm.Response{Events: []llm.Event{{
		Kind:    llm.EventKindItemAdded,
		ItemRef: llm.ItemRef{ItemID: "item_namespaced", CallID: mappedCallID, OutputIndex: &outputIndex},
		Snapshot: &llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, Namespace: namespace, LogicalName: mappedChildName, CallID: mappedCallID,
		}},
	}}}
	newStreamRestorer(session).restore(stream)
	streamCall := stream.Events[0].Snapshot.ToolCall
	if streamCall.LogicalName != sourceChild || streamCall.Namespace != namespace || streamCall.CallID != mappedCallID {
		t.Fatalf("stream namespaced identity was not restored safely: %#v", streamCall)
	}
}

func TestResponsesNamespaceUsesOneFlattenedIdentityForCrossProtocolTargets(t *testing.T) {
	t.Parallel()
	const namespace = "project"
	const sourceChild = "project__read.file"
	for _, target := range []llm.APIFormat{llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage} {
		target := target
		t.Run(string(target), func(t *testing.T) {
			request := &llm.Request{
				APIFormat: llm.APIFormatOpenAIResponse,
				ToolDefinitions: []llm.ToolDefinition{{
					Kind: llm.ToolKindFunction, LogicalName: namespace + "__" + sourceChild, Execution: llm.ExecutionOwnerClient,
					Function: &llm.FunctionDefinition{Namespace: namespace, Parameters: json.RawMessage(`{"type":"object"}`)},
				}},
				Input: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
					Kind: llm.ToolKindFunction, Namespace: namespace, LogicalName: sourceChild, CallID: "call_history",
				}}},
			}
			plan, err := NewPlanner().Plan(request, target)
			if err != nil {
				t.Fatalf("plan cross-protocol namespace route: %v", err)
			}
			lowered, session, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower cross-protocol namespace route: %v", err)
			}
			definitionName := lowered.ToolDefinitions[0].LogicalName
			callName := lowered.Input[0].ToolCall.LogicalName
			if definitionName != callName || definitionName == sourceChild || !portableIdentifierPattern.MatchString(definitionName) {
				t.Fatalf("cross-protocol definition/call identity diverged: definition=%q call=%q", definitionName, callName)
			}

			response := &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, LogicalName: definitionName, CallID: "provider_call",
			}}}}
			RestoreResponse(response, session)
			call := response.Output[0].ToolCall
			if call.LogicalName != sourceChild || call.Namespace != namespace || call.CallID != "provider_call" {
				t.Fatalf("cross-protocol response identity was not restored: %#v", call)
			}
		})
	}
}

func TestToolIdentityRegistryUsesNamespaceAwareWireKeys(t *testing.T) {
	t.Parallel()
	definitions := []*llm.ToolDefinition{
		nil,
		{Function: &llm.FunctionDefinition{Namespace: "function"}},
		{Freeform: &llm.FreeformDefinition{Namespace: "freeform"}},
		{Hosted: &llm.HostedToolDefinition{Namespace: "hosted"}},
		{},
	}
	wantNamespaces := []string{"", "function", "freeform", "hosted", ""}
	for index, definition := range definitions {
		if got := toolDefinitionNamespace(definition); got != wantNamespaces[index] {
			t.Fatalf("definition %d namespace = %q, want %q", index, got, wantNamespaces[index])
		}
	}

	identity := toolIdentity{SourceKind: llm.ToolKindFunction, SourceName: "read.file", SourceNamespace: "project"}
	var nilSession *Session
	nilSession.registerToolIdentity("ignored", "project", identity)
	if _, ok := nilSession.identity("ignored", "project"); ok {
		t.Fatal("nil session returned an identity")
	}
	session := &Session{}
	session.registerToolIdentity("", "project", identity)
	session.registerToolIdentity("project__read_file_hash", "project", identity)
	if got, ok := session.identity("read_file_hash", "project"); !ok || got != identity {
		t.Fatalf("namespace wire identity = %#v,%v", got, ok)
	}
	if got, ok := session.identity("project__read_file_hash", "missing"); !ok || got != identity {
		t.Fatalf("full-name identity fallback = %#v,%v", got, ok)
	}
	if _, ok := session.identity("missing", "project"); ok {
		t.Fatal("missing namespace identity unexpectedly resolved")
	}
}

func planHasStrategy(plan *Plan, strategy StrategyID) bool {
	if plan == nil {
		return false
	}
	for _, action := range plan.Actions {
		if action.Strategy == strategy {
			return true
		}
	}
	return false
}

func debugTraceHasStrategy(trace *llm.ConversionDebugTrace, strategy StrategyID) bool {
	if trace == nil {
		return false
	}
	for _, action := range trace.Actions {
		if action.Strategy == string(strategy) {
			return true
		}
	}
	return false
}

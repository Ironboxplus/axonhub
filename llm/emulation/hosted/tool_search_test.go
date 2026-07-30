package hosted

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestBuiltInRegexToolSearchActivatesOnlyMatchedDeferredDefinitions(t *testing.T) {
	registry := newToolSearchRegistry(t, "tool_search_tool_regex_20251119", nil)
	binding, ok := registry.BindingFor(llm.ToolKindToolSearch, "tool_search_tool_regex")
	require.True(t, ok)
	require.Len(t, registry.FunctionDefinitions(), 1)

	result, err := registry.Search(binding, llm.ToolInvocation{
		Kind: llm.ToolKindFunction, CallID: "search_1", LogicalName: binding.SyntheticName,
		ArgumentsJSON: json.RawMessage(`{"pattern":"^create_event$"}`),
	})
	require.NoError(t, err)
	require.Equal(t, llm.ExecutionOwnerGateway, result.Execution)
	require.Len(t, result.DiscoveredTools, 1)
	require.Equal(t, "create_event", result.DiscoveredTools[0].LogicalName)
	require.Nil(t, result.DiscoveredTools[0].DeferLoading)
	require.JSONEq(t, `{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"create_event"}]}`, string(result.StructuredContent))

	definitions := registry.FunctionDefinitions()
	require.Len(t, definitions, 2)
	require.Equal(t, 1, countToolDefinition(definitions, "create_event"))
	require.Equal(t, 0, countToolDefinition(definitions, "delete_event"))
}

func TestBuiltInToolSearchValidatesArgumentsAndReturnsEmptyReferences(t *testing.T) {
	regexRegistry := newToolSearchRegistry(t, "tool_search_tool_regex_20251119", nil)
	regexBinding, ok := regexRegistry.BindingFor(llm.ToolKindToolSearch, "tool_search_tool_regex")
	require.True(t, ok)
	_, err := regexRegistry.Search(regexBinding, llm.ToolInvocation{CallID: "search_1", ArgumentsJSON: json.RawMessage(`{"pattern":"["}`)})
	require.ErrorContains(t, err, "compile tool search pattern")
	_, err = regexRegistry.Search(regexBinding, llm.ToolInvocation{CallID: "search_2", ArgumentsJSON: mustToolSearchJSON(t, map[string]string{"pattern": strings.Repeat("x", maxToolSearchPattern+1)})})
	require.ErrorContains(t, err, "exceeds 200 characters")

	bm25Registry := newToolSearchRegistry(t, "tool_search_tool_bm25_20251119", nil)
	bm25Binding, ok := bm25Registry.BindingFor(llm.ToolKindToolSearch, "tool_search_tool_bm25")
	require.True(t, ok)
	result, err := bm25Registry.Search(bm25Binding, llm.ToolInvocation{CallID: "search_3", ArgumentsJSON: json.RawMessage(`{"query":"zirconium"}`)})
	require.NoError(t, err)
	require.Empty(t, result.DiscoveredTools)
	require.JSONEq(t, `{"type":"tool_search_tool_search_result","tool_references":[]}`, string(result.StructuredContent))
}

func TestBuiltInBM25ToolSearchConcurrentActivationDoesNotDuplicateDefinitions(t *testing.T) {
	registry := newToolSearchRegistry(t, "tool_search_tool_bm25_20251119", nil)
	binding, ok := registry.BindingFor(llm.ToolKindToolSearch, "tool_search_tool_bm25")
	require.True(t, ok)

	const searches = 64
	errorsByCall := make(chan error, searches)
	var wait sync.WaitGroup
	for index := 0; index < searches; index++ {
		wait.Add(1)
		go func() {
			defer wait.Done()
			_, err := registry.Search(binding, llm.ToolInvocation{
				CallID: "search_concurrent", ArgumentsJSON: json.RawMessage(`{"query":"create"}`),
			})
			errorsByCall <- err
		}()
	}
	wait.Wait()
	close(errorsByCall)
	for err := range errorsByCall {
		require.NoError(t, err)
	}
	definitions := registry.FunctionDefinitions()
	require.Equal(t, 1, countToolDefinition(definitions, "create_event"))
	require.Equal(t, 0, countToolDefinition(definitions, "delete_event"))
}

func TestBuiltInToolSearchRestoresActivatedDefinitionsFromAnthropicHistory(t *testing.T) {
	history := []llm.Item{{
		Kind: llm.ItemKindHostedCall,
		HostedCall: &llm.HostedToolCall{
			Invocation: llm.ToolInvocation{
				Kind: llm.ToolKindToolSearch, CallID: "search_previous", LogicalName: "tool_search_tool_bm25",
				Execution: llm.ExecutionOwnerProvider,
			},
			Result: &llm.ToolResult{
				Kind: llm.ToolKindToolSearch, CallID: "search_previous", Execution: llm.ExecutionOwnerProvider,
				StructuredContent: json.RawMessage(`{"type":"tool_search_tool_search_result","tool_references":[{"type":"tool_reference","tool_name":"create_event"}]}`),
			},
		},
	}}
	registry := newToolSearchRegistry(t, "tool_search_tool_bm25_20251119", history)
	definitions := registry.FunctionDefinitions()
	require.Equal(t, 1, countToolDefinition(definitions, "create_event"))
	require.Equal(t, 0, countToolDefinition(definitions, "delete_event"))
}

func newToolSearchRegistry(t *testing.T, typeName string, history []llm.Item) *Registry {
	t.Helper()
	logicalName := "tool_search_tool_bm25"
	if strings.Contains(typeName, "regex") {
		logicalName = "tool_search_tool_regex"
	}
	deferred := true
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{
			{
				Kind: llm.ToolKindToolSearch, LogicalName: logicalName, Execution: llm.ExecutionOwnerProvider,
				Hosted: &llm.HostedToolDefinition{Type: typeName, Configuration: mustToolSearchJSON(t, map[string]string{"type": typeName, "name": logicalName})},
			},
			{
				Kind: llm.ToolKindFunction, LogicalName: "create_event", Description: "Create a calendar event",
				DeferLoading: &deferred, Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}`)},
				Execution: llm.ExecutionOwnerClient,
			},
			{
				Kind: llm.ToolKindFunction, LogicalName: "delete_event", Description: "Delete a calendar event",
				DeferLoading: &deferred, Function: &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}`)},
				Execution: llm.ExecutionOwnerClient,
			},
		},
		Input: history,
	}
	registry, err := DiscoverRegistry(request, Config{SyntheticNameKey: []byte(strings.Repeat("tool-search-registry-key-", 2))}, func(definition llm.ToolDefinition) bool {
		return definition.Kind == llm.ToolKindToolSearch
	})
	require.NoError(t, err)
	return registry
}

func mustToolSearchJSON(t *testing.T, value any) json.RawMessage {
	t.Helper()
	encoded, err := json.Marshal(value)
	require.NoError(t, err)
	return encoded
}

func countToolDefinition(definitions []llm.ToolDefinition, name string) int {
	count := 0
	for index := range definitions {
		if definitions[index].LogicalName == name {
			count++
		}
	}
	return count
}

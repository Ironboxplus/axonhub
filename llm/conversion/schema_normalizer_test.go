package conversion

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestLowerNormalizesFunctionSchemasForTargetWithoutMutatingSource(t *testing.T) {
	t.Parallel()
	strict := true
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{
			{
				Kind: llm.ToolKindFunction, LogicalName: "empty", Execution: llm.ExecutionOwnerClient,
				Function: &llm.FunctionDefinition{},
			},
			{
				Kind: llm.ToolKindFunction, LogicalName: "lookup", Execution: llm.ExecutionOwnerClient,
				Function: &llm.FunctionDefinition{
					Parameters: json.RawMessage(`{"oneOf":[{"type":"object","properties":{"query":{"type":"string","nullable":true}},"required":["query"]},{"type":"object","properties":{"id":{"type":"integer"}}}],"additionalProperties":false}`),
					Strict:     &strict,
				},
			},
		},
	}
	original := request.Clone()

	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil {
		t.Fatalf("plan schema normalization: %v", err)
	}
	if !planHasStrategy(plan, StrategySchemaNormalize) {
		t.Fatalf("plan omitted schema normalization: %#v", plan.Actions)
	}
	plan.Debug = llm.NewConversionDebugTrace(
		llm.WithConversionDebugTrace(context.Background(), []byte(t.Name()), 32),
		len(plan.Actions)+8,
	)
	lowered, session, err := lower(request, plan, true)
	if err != nil {
		t.Fatalf("lower schema constraints: %v", err)
	}

	assertFunctionSchemaJSONEqual(t, request.ToolDefinitions[0].Function.Parameters, original.ToolDefinitions[0].Function.Parameters)
	assertFunctionSchemaJSONEqual(t, request.ToolDefinitions[1].Function.Parameters, original.ToolDefinitions[1].Function.Parameters)

	empty := decodeSchemaObject(t, lowered.ToolDefinitions[0].Function.Parameters)
	if empty["type"] != "object" {
		t.Fatalf("empty schema type = %#v, want object", empty["type"])
	}
	if properties, ok := empty["properties"].(map[string]any); !ok || len(properties) != 0 {
		t.Fatalf("empty schema properties = %#v, want empty object", empty["properties"])
	}
	if additional, ok := empty["additionalProperties"].(bool); !ok || additional {
		t.Fatalf("empty schema additionalProperties = %#v, want false", empty["additionalProperties"])
	}

	lookup := decodeSchemaObject(t, lowered.ToolDefinitions[1].Function.Parameters)
	if lookup["type"] != "object" || lookup["oneOf"] != nil {
		t.Fatalf("Anthropic root union was not normalized: %#v", lookup)
	}
	properties, ok := lookup["properties"].(map[string]any)
	if !ok || properties["query"] == nil || properties["id"] == nil {
		t.Fatalf("root union properties were not preserved: %#v", lookup["properties"])
	}
	query := properties["query"].(map[string]any)
	if query["nullable"] != true {
		t.Fatalf("nullable changed during normalization: %#v", query)
	}
	if additional, ok := lookup["additionalProperties"].(bool); !ok || additional {
		t.Fatalf("additionalProperties changed during normalization: %#v", lookup)
	}
	if lowered.ToolDefinitions[1].Function.Strict == nil || !*lowered.ToolDefinitions[1].Function.Strict {
		t.Fatalf("strict was not preserved: %#v", lowered.ToolDefinitions[1].Function.Strict)
	}
	if got := session.Summary().SchemasNormalized; got != 2 {
		t.Fatalf("normalized schema count = %d, want 2", got)
	}
	trace := session.DebugTrace()
	if trace == nil || !debugTraceHasStrategy(trace, StrategySchemaNormalize) {
		t.Fatalf("schema normalization missing from debug trace: %#v", trace)
	}
}

func TestSchemaNormalizerAddsNestedObjectPropertiesAndPreservesDialectKeywords(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","properties":{"nested":{"type":"object","additionalProperties":{"type":"object"}},"choice":{"type":["object","null"]}},"required":["nested"],"additionalProperties":true}`)
	normalized, changed := normalizeFunctionSchema(raw, llm.APIFormatOpenAIChatCompletion)
	if !changed {
		t.Fatal("schema requiring nested properties was not changed")
	}
	schema := decodeSchemaObject(t, normalized)
	if schema["$schema"] != "https://json-schema.org/draft/2020-12/schema" {
		t.Fatalf("schema dialect changed: %#v", schema["$schema"])
	}
	if additional, ok := schema["additionalProperties"].(bool); !ok || !additional {
		t.Fatalf("root additionalProperties changed: %#v", schema["additionalProperties"])
	}
	properties := schema["properties"].(map[string]any)
	nested := properties["nested"].(map[string]any)
	if nestedProperties, ok := nested["properties"].(map[string]any); !ok || len(nestedProperties) != 0 {
		t.Fatalf("nested properties missing: %#v", nested)
	}
	additionalSchema := nested["additionalProperties"].(map[string]any)
	if additionalProperties, ok := additionalSchema["properties"].(map[string]any); !ok || len(additionalProperties) != 0 {
		t.Fatalf("additionalProperties object schema was not normalized: %#v", additionalSchema)
	}
	choice := properties["choice"].(map[string]any)
	if choiceProperties, ok := choice["properties"].(map[string]any); !ok || len(choiceProperties) != 0 {
		t.Fatalf("nullable object union was not normalized: %#v", choice)
	}
}

func TestSchemaNormalizerLeavesCanonicalSchemaBytesUntouched(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}`)
	normalized, changed := normalizeFunctionSchema(raw, llm.APIFormatOpenAIResponse)
	if changed {
		t.Fatalf("canonical schema reported a rewrite: %s", normalized)
	}
	if string(normalized) != string(raw) {
		t.Fatalf("canonical schema bytes changed: got %s want %s", normalized, raw)
	}
}

func TestLowerLeavesSameProtocolSchemaUntouched(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"oneOf":[{"type":"object","properties":{"query":{"type":"string"}}},{"type":"object","properties":{"id":{"type":"integer"}}}]}`)
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: "lookup", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), raw...)},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatAnthropicMessage)
	if err != nil {
		t.Fatalf("plan identity route: %v", err)
	}
	lowered, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower identity route: %v", err)
	}
	if lowered != request {
		t.Fatal("identity route unexpectedly cloned the request")
	}
	if string(lowered.ToolDefinitions[0].Function.Parameters) != string(raw) {
		t.Fatalf("identity schema changed: got %s want %s", lowered.ToolDefinitions[0].Function.Parameters, raw)
	}
	if session.Summary().SchemasNormalized != 0 {
		t.Fatalf("identity route reported schema normalization: %#v", session.Summary())
	}
}

func TestSchemaNormalizerMergesAllOfRequiredButNotAlternativeRequired(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"allOf":[{"type":"object","properties":{"base":{"type":"string"}},"required":["base"]}],"oneOf":[{"type":"object","properties":{"left":{"type":"string"}},"required":["left"]},{"type":"object","properties":{"right":{"type":"string"}},"required":["right"]}]}`)
	normalized, changed := normalizeFunctionSchema(raw, llm.APIFormatAnthropicMessage)
	if !changed {
		t.Fatal("root unions were not normalized")
	}
	schema := decodeSchemaObject(t, normalized)
	required, ok := schema["required"].([]any)
	if !ok || len(required) != 1 || required[0] != "base" {
		t.Fatalf("required = %#v, want only allOf requirement", schema["required"])
	}
	properties := schema["properties"].(map[string]any)
	for _, name := range []string{"base", "left", "right"} {
		if properties[name] == nil {
			t.Fatalf("property %q missing after union normalization: %#v", name, properties)
		}
	}
}

func TestOpenAIStrictSchemaNormalizesOptionalFieldsAndRestoresArguments(t *testing.T) {
	t.Parallel()
	strict := true
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: "weather", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{
				Strict:     &strict,
				Parameters: json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"},"unit":{"type":"string"},"options":{"type":"object","properties":{"verbose":{"type":"boolean"}},"required":[],"additionalProperties":false}},"required":["location","options"],"additionalProperties":false}`),
			},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatalf("plan OpenAI strict normalization: %v", err)
	}
	lowered, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower OpenAI strict normalization: %v", err)
	}
	schema := decodeSchemaObject(t, lowered.ToolDefinitions[0].Function.Parameters)
	required := stringSet(schema["required"])
	if !required["location"] || !required["unit"] || !required["options"] {
		t.Fatalf("root strict required fields incomplete: %#v", schema["required"])
	}
	properties := schema["properties"].(map[string]any)
	unitType, ok := properties["unit"].(map[string]any)["type"].([]any)
	if !ok || !containsJSONValue(unitType, "string") || !containsJSONValue(unitType, "null") {
		t.Fatalf("optional unit was not made nullable: %#v", properties["unit"])
	}
	options := properties["options"].(map[string]any)
	if !stringSet(options["required"])["verbose"] || options["additionalProperties"] != false {
		t.Fatalf("nested strict object was not normalized: %#v", options)
	}
	verboseType := options["properties"].(map[string]any)["verbose"].(map[string]any)["type"].([]any)
	if !containsJSONValue(verboseType, "boolean") || !containsJSONValue(verboseType, "null") {
		t.Fatalf("nested optional field was not made nullable: %#v", verboseType)
	}

	response := &llm.Response{Output: []llm.Item{{
		Kind: llm.ItemKindToolCall,
		ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, CallID: "call_1", LogicalName: "weather",
			ArgumentsJSON: json.RawMessage(`{"location":"Paris","unit":null,"options":{"verbose":null}}`),
		},
	}}}
	RestoreResponse(response, session)
	var arguments map[string]any
	if err := json.Unmarshal(response.Output[0].ToolCall.ArgumentsJSON, &arguments); err != nil {
		t.Fatalf("decode restored arguments: %v", err)
	}
	if _, exists := arguments["unit"]; exists {
		t.Fatalf("synthetic optional null leaked to source arguments: %#v", arguments)
	}
	optionsArguments := arguments["options"].(map[string]any)
	if _, exists := optionsArguments["verbose"]; exists {
		t.Fatalf("nested synthetic optional null leaked to source arguments: %#v", arguments)
	}
	restoration := session.schemaRestorations["weather"]
	if restoration.originalHash == ([32]byte{}) || restoration.normalizedHash == ([32]byte{}) || restoration.originalHash == restoration.normalizedHash {
		t.Fatalf("schema sidecar hashes were not retained: %#v", restoration)
	}
}

func TestOpenAIStrictSchemaRestoresOptionalNullsInsideArrays(t *testing.T) {
	t.Parallel()
	strict := true
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: "batch", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{
				Strict:     &strict,
				Parameters: json.RawMessage(`{"type":"object","properties":{"users":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"},"nickname":{"type":"string"}},"required":["name"],"additionalProperties":false}}},"required":["users"],"additionalProperties":false}`),
			},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if err != nil {
		t.Fatalf("plan array strict schema: %v", err)
	}
	_, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower array strict schema: %v", err)
	}
	response := &llm.Response{Output: []llm.Item{{
		Kind: llm.ItemKindToolCall,
		ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, CallID: "call_array", LogicalName: "batch",
			ArgumentsJSON: json.RawMessage(`{"users":[{"name":"Ada","nickname":null},{"name":"Lin","nickname":"L"}]}`),
		},
	}}}
	RestoreResponse(response, session)
	var arguments struct {
		Users []map[string]any `json:"users"`
	}
	if err := json.Unmarshal(response.Output[0].ToolCall.ArgumentsJSON, &arguments); err != nil {
		t.Fatalf("decode array arguments: %v", err)
	}
	if _, exists := arguments.Users[0]["nickname"]; exists || arguments.Users[1]["nickname"] != "L" {
		t.Fatalf("array optional restoration failed: %#v", arguments.Users)
	}
}

func TestOpenAIStrictSchemaFallsBackToNonStrictForDynamicProperties(t *testing.T) {
	t.Parallel()
	strict := true
	request := &llm.Request{
		APIFormat: llm.APIFormatAnthropicMessage,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: "labels", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{
				Strict:     &strict,
				Parameters: json.RawMessage(`{"type":"object","properties":{"fixed":{"type":"string"}},"additionalProperties":{"type":"string"}}`),
			},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if err != nil {
		t.Fatalf("plan dynamic strict schema: %v", err)
	}
	lowered, _, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower dynamic strict schema: %v", err)
	}
	function := lowered.ToolDefinitions[0].Function
	if function.Strict == nil || *function.Strict {
		t.Fatalf("dynamic schema must use non-strict OpenAI mode: %#v", function.Strict)
	}
	schema := decodeSchemaObject(t, function.Parameters)
	if _, ok := schema["additionalProperties"].(map[string]any); !ok {
		t.Fatalf("dynamic additionalProperties schema was not preserved: %#v", schema)
	}
}

func TestSchemaNormalizerPreservesProtocolSpecificStrictDefaults(t *testing.T) {
	t.Parallel()
	compatible := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}`)
	dynamic := json.RawMessage(`{"type":"object","properties":{},"additionalProperties":{"type":"string"}}`)
	tests := []struct {
		name   string
		source llm.APIFormat
		target llm.APIFormat
		schema json.RawMessage
		want   bool
	}{
		{name: "Chat omitted to Responses is non-strict", source: llm.APIFormatOpenAIChatCompletion, target: llm.APIFormatOpenAIResponse, schema: compatible, want: false},
		{name: "Anthropic omitted to Responses is non-strict", source: llm.APIFormatAnthropicMessage, target: llm.APIFormatOpenAIResponse, schema: compatible, want: false},
		{name: "Responses omitted compatible to Chat becomes strict", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIChatCompletion, schema: compatible, want: true},
		{name: "Responses omitted compatible to Anthropic becomes strict", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatAnthropicMessage, schema: compatible, want: true},
		{name: "Responses omitted dynamic to Anthropic falls back", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatAnthropicMessage, schema: dynamic, want: false},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			request := &llm.Request{
				APIFormat: test.source,
				ToolDefinitions: []llm.ToolDefinition{{
					Kind: llm.ToolKindFunction, LogicalName: "lookup", Execution: llm.ExecutionOwnerClient,
					Function: &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), test.schema...)},
				}},
			}
			plan, err := NewPlanner().Plan(request, test.target)
			if err != nil {
				t.Fatalf("plan strict default: %v", err)
			}
			lowered, _, err := Lower(request, plan)
			if err != nil {
				t.Fatalf("lower strict default: %v", err)
			}
			strict := lowered.ToolDefinitions[0].Function.Strict
			if strict == nil || *strict != test.want {
				t.Fatalf("strict = %#v, want %v", strict, test.want)
			}
		})
	}
}

func decodeSchemaObject(t *testing.T, raw json.RawMessage) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatalf("decode schema %s: %v", raw, err)
	}
	return value
}

func assertFunctionSchemaJSONEqual(t *testing.T, got, want json.RawMessage) {
	t.Helper()
	if len(got) == 0 || len(want) == 0 {
		if len(got) != len(want) {
			t.Fatalf("schema presence changed: got %s want %s", got, want)
		}
		return
	}
	var gotValue, wantValue any
	if err := json.Unmarshal(got, &gotValue); err != nil {
		t.Fatalf("decode got schema: %v", err)
	}
	if err := json.Unmarshal(want, &wantValue); err != nil {
		t.Fatalf("decode want schema: %v", err)
	}
	gotJSON, _ := json.Marshal(gotValue)
	wantJSON, _ := json.Marshal(wantValue)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("schema changed: got %s want %s", gotJSON, wantJSON)
	}
}

func stringSet(value any) map[string]bool {
	result := make(map[string]bool)
	for _, item := range value.([]any) {
		if text, ok := item.(string); ok {
			result[text] = true
		}
	}
	return result
}

func containsJSONValue(values []any, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}

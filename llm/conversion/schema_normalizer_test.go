package conversion

import (
	"context"
	"encoding/json"
	"errors"
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
	result := normalizeFunctionSchemaDetailed(raw, nil, llm.APIFormatAnthropicMessage, llm.APIFormatOpenAIChatCompletion)
	normalized, changed := result.schema, result.changed
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

func TestLowerNormalizesImplicitRootUnionBranchesOnIdentityRoute(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"queries":{"type":"array","items":{"type":"string"}}},"anyOf":[{"required":["query"]},{"required":["queries"]}],"additionalProperties":true}`)
	request := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindFunction, LogicalName: "search", Execution: llm.ExecutionOwnerGateway,
			Function: &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), raw...)},
		}},
	}
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatalf("plan identity schema normalization: %v", err)
	}
	if !planHasStrategy(plan, StrategySchemaNormalize) {
		t.Fatalf("identity plan omitted schema normalization: %#v", plan.Actions)
	}
	lowered, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower identity schema normalization: %v", err)
	}
	if lowered == request {
		t.Fatal("schema normalization mutated the identity request in place")
	}
	if string(request.ToolDefinitions[0].Function.Parameters) != string(raw) {
		t.Fatalf("source schema changed: got %s want %s", request.ToolDefinitions[0].Function.Parameters, raw)
	}
	schema := decodeSchemaObject(t, lowered.ToolDefinitions[0].Function.Parameters)
	branches, ok := schema["anyOf"].([]any)
	if !ok || len(branches) != 2 {
		t.Fatalf("normalized anyOf = %#v, want two branches", schema["anyOf"])
	}
	for index, branch := range branches {
		branchSchema, ok := branch.(map[string]any)
		if !ok || branchSchema["type"] != "object" {
			t.Fatalf("normalized anyOf branch %d = %#v, want explicit object", index, branch)
		}
	}
	if got := session.Summary().SchemasNormalized; got != 1 {
		t.Fatalf("normalized schema count = %d, want 1", got)
	}
}

func TestNormalizeRootObjectUnionBranchesCoversEverySchemaForm(t *testing.T) {
	t.Parallel()
	root := map[string]any{
		"anyOf": []any{
			map[string]any{"required": []any{"query"}},
			map[string]any{"type": "object"},
			map[string]any{"type": []any{"string", "object"}},
			map[string]any{"type": "string"},
			true,
			false,
			"invalid-schema-branch",
		},
		"oneOf": []any{map[string]any{"required": []any{"id"}}},
	}
	if !normalizeRootObjectUnionBranches(root) {
		t.Fatal("mixed root unions were not normalized")
	}
	for _, keyword := range []string{"anyOf", "oneOf"} {
		for index, branch := range root[keyword].([]any) {
			schema, ok := branch.(map[string]any)
			if !ok || schema["type"] != "object" {
				t.Fatalf("%s branch %d = %#v, want explicit object", keyword, index, branch)
			}
		}
	}
	anyOf := root["anyOf"].([]any)
	if _, ok := anyOf[3].(map[string]any)["allOf"].([]any); !ok {
		t.Fatalf("explicit non-object branch was not preserved as an impossible object intersection: %#v", anyOf[3])
	}
	for _, index := range []int{5, 6} {
		if _, ok := anyOf[index].(map[string]any)["not"].(map[string]any); !ok {
			t.Fatalf("false or invalid branch %d was not made impossible: %#v", index, anyOf[index])
		}
	}
	if normalizeRootObjectUnionBranches(root) {
		t.Fatal("root union normalization is not idempotent")
	}
	if normalizeRootObjectUnionBranches(map[string]any{"anyOf": "invalid", "oneOf": map[string]any{}}) {
		t.Fatal("non-array union keywords were unexpectedly rewritten")
	}
}

func TestNormalizeIdentityFunctionSchemaCoversRootTypeContracts(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		raw            json.RawMessage
		root           map[string]any
		wantChanged    bool
		wantReversible bool
		wantWrapped    bool
	}{
		{name: "canonical object", raw: json.RawMessage(`{"type":"object"}`), root: map[string]any{"type": "object"}, wantReversible: true},
		{name: "explicit scalar", root: map[string]any{"type": "string"}, wantChanged: true, wantWrapped: true},
		{name: "object type union", root: map[string]any{"type": []any{"string", "object"}}, wantChanged: true, wantReversible: true},
		{name: "non-object type union", root: map[string]any{"type": []any{"string", "null"}}, wantChanged: true, wantWrapped: true},
		{name: "invalid type keyword", raw: json.RawMessage(`{"type":true}`), root: map[string]any{"type": true}},
		{name: "implicit object", root: map[string]any{"required": []any{"query"}}, wantChanged: true, wantReversible: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if test.name == "invalid type keyword" {
				result := normalizeFunctionSchemaDetailed(test.raw, nil, llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIResponse)
				if !result.invalid || result.changed || string(result.schema) != string(test.raw) {
					t.Fatalf("invalid identity schema = %#v, want unchanged fail-closed result", result)
				}
				return
			}
			result := normalizeIdentityFunctionSchema(test.raw, test.root)
			if result.changed != test.wantChanged || result.reversible != test.wantReversible {
				t.Fatalf("normalization = {changed:%v reversible:%v schema:%s}, want changed=%v reversible=%v", result.changed, result.reversible, result.schema, test.wantChanged, test.wantReversible)
			}
			if !test.wantChanged {
				if string(result.schema) != string(test.raw) {
					t.Fatalf("canonical schema bytes changed: got %s want %s", result.schema, test.raw)
				}
				return
			}
			schema := decodeSchemaObject(t, result.schema)
			if schema["type"] != "object" {
				t.Fatalf("normalized root type = %#v, want object", schema["type"])
			}
			_, wrapped := schema["allOf"].([]any)
			if wrapped != test.wantWrapped {
				t.Fatalf("normalized wrapper = %#v, want wrapped=%v", schema["allOf"], test.wantWrapped)
			}
		})
	}

	marshalFailure := normalizeIdentityFunctionSchema(nil, map[string]any{
		"required": []any{"query"}, "unencodable": make(chan struct{}),
	})
	if !marshalFailure.invalid || marshalFailure.changed {
		t.Fatalf("marshal failure = %#v, want fail-closed result", marshalFailure)
	}
	booleanUnion := normalizeIdentityFunctionSchema(
		json.RawMessage(`{"type":"object","anyOf":[true,false]}`),
		map[string]any{"type": "object", "anyOf": []any{true, false}},
	)
	if booleanUnion.invalid || !booleanUnion.changed {
		t.Fatalf("valid boolean union = %#v, want normalized result", booleanUnion)
	}
}

func TestIdentityRejectsInvalidJSONSchemasBeforeProviderDispatch(t *testing.T) {
	t.Parallel()
	formats := []llm.APIFormat{
		llm.APIFormatOpenAIChatCompletion,
		llm.APIFormatOpenAIResponse,
		llm.APIFormatAnthropicMessage,
	}
	invalidSchemas := []json.RawMessage{
		json.RawMessage(`[]`),
		json.RawMessage(`"invalid"`),
		json.RawMessage(`42`),
		json.RawMessage(`{"type":42}`),
		json.RawMessage(`{"type":"invalid"}`),
		json.RawMessage(`{"type":[]}`),
		json.RawMessage(`{"type":["object","object"]}`),
		json.RawMessage(`{"type":["object",42]}`),
		json.RawMessage(`{"type":"object","anyOf":["invalid",{"required":["query"]}]}`),
		json.RawMessage(`{"type":"object","anyOf":[{"type":42},{"required":["query"]}]}`),
		json.RawMessage(`{"type":"object","oneOf":{"required":["query"]}}`),
		json.RawMessage(`{"type":"object","properties":{"x":[]}}`),
		json.RawMessage(`{"type":"object","allOf":[]}`),
		json.RawMessage(`{"type":"object","required":[42]}`),
		json.RawMessage(`{"type":"object","required":["x","x"]}`),
		json.RawMessage(`{"type":"object","anyOf":[{"type":"object","oneOf":[]}]}`),
		json.RawMessage(`{"type":"object","properties":{"x":{"type":"invalid"}}}`),
		json.RawMessage(`{"type":"object","patternProperties":{"[":{"type":"string"}}}`),
		json.RawMessage(`{"type":"object","minProperties":-1}`),
		json.RawMessage(`{"type":"object","$ref":"https://schemas.example.invalid/tool.json"}`),
	}
	for _, source := range formats {
		source := source
		for _, target := range formats {
			target := target
			for index, raw := range invalidSchemas {
				raw := append(json.RawMessage(nil), raw...)
				t.Run(string(source)+"_to_"+string(target)+"/case_"+string(rune('a'+index)), func(t *testing.T) {
					request := &llm.Request{
						APIFormat: source,
						ToolDefinitions: []llm.ToolDefinition{{
							Kind: llm.ToolKindFunction, LogicalName: "invalid_schema", Execution: llm.ExecutionOwnerClient,
							Function: &llm.FunctionDefinition{Parameters: raw},
						}},
					}
					plan, err := NewPlanner().Plan(request, target)
					if !errors.Is(err, ErrIncompletePlan) {
						t.Fatalf("plan error = %v, want ErrIncompletePlan", err)
					}
					if plan == nil || plan.Complete() || plan.Summary.Unknown == 0 {
						t.Fatalf("invalid schema plan = %#v, want incomplete plan", plan)
					}
					if string(request.ToolDefinitions[0].Function.Parameters) != string(raw) {
						t.Fatalf("invalid source schema mutated: got %s want %s", request.ToolDefinitions[0].Function.Parameters, raw)
					}
				})
			}
		}
	}
}

func TestIdentityFunctionSchemaNeedsNormalization(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		raw  string
		want bool
	}{
		{name: "canonical compact", raw: `{"type":"object","properties":{"query":{"type":"string"}}}`, want: false},
		{name: "canonical whitespace", raw: `{ "properties": {}, "type" : "object" }`, want: false},
		{name: "root anyOf", raw: `{"type":"object","anyOf":[{"required":["query"]}]}`, want: true},
		{name: "root oneOf", raw: `{"type":"object","oneOf":[{"type":"object"}]}`, want: true},
		{name: "missing root type", raw: `{"properties":{}}`, want: true},
		{name: "root type union", raw: `{"type":["object","null"]}`, want: true},
		{name: "boolean schema", raw: `true`, want: true},
		{name: "non-object root", raw: `[]`, want: true},
		{name: "malformed", raw: `{`, want: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			schema, valid := parseValidFunctionSchema([]byte(test.raw))
			got := !valid || identityFunctionSchemaNeedsNormalization(schema)
			if got != test.want {
				t.Fatalf("identityFunctionSchemaNeedsNormalization(%s) = %v, want %v", test.raw, got, test.want)
			}
		})
	}
}

func TestFunctionSchemaStructuralValidatorRejectsInvalidNestedBranches(t *testing.T) {
	t.Parallel()
	invalidType := map[string]any{"type": "invalid"}
	tests := []struct {
		name   string
		schema any
	}{
		{name: "nil schema", schema: nil},
		{name: "empty enum", schema: map[string]any{"enum": []any{}}},
		{name: "duplicate enum", schema: map[string]any{"enum": []any{"x", "x"}}},
		{name: "duplicate dependent required", schema: map[string]any{"dependentRequired": map[string]any{"x": []any{"y", "y"}}}},
		{name: "non-positive multiple", schema: map[string]any{"multipleOf": float64(0)}},
		{name: "negative bound", schema: map[string]any{"minItems": float64(-1)}},
		{name: "invalid single child", schema: map[string]any{"items": invalidType}},
		{name: "invalid slice child", schema: map[string]any{"allOf": []any{invalidType}}},
		{name: "invalid map child", schema: map[string]any{"properties": map[string]any{"x": invalidType}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if validFunctionSchemaTree(test.schema) {
				t.Fatalf("invalid schema accepted: %#v", test.schema)
			}
		})
	}
	if validSchemaTypes("object", []string{"string"}) {
		t.Fatal("schema with both single and multiple types was accepted")
	}
	if validSchemaTypes("", []string{"object", "invalid"}) {
		t.Fatal("schema with an invalid member in a type array was accepted")
	}
	if validSchemaTypes("", []string{"object", "object"}) {
		t.Fatal("schema with duplicate members in a type array was accepted")
	}
	if !validSchemaTypes("", nil) {
		t.Fatal("schema without a type restriction was rejected")
	}
	if validSchemaTypes("", []string{}) {
		t.Fatal("schema with an empty type array was accepted")
	}
	if containsDuplicateJSONValues([]any{"x", "y"}) {
		t.Fatal("distinct enum values reported as duplicates")
	}
	if !containsDuplicateJSONValues([]any{map[string]any{"x": float64(1)}, map[string]any{"x": float64(1)}}) {
		t.Fatal("structurally equal enum values were not detected")
	}
}

func TestFunctionSchemaStructuralValidatorCoversStandardKeywordShapesAndReferences(t *testing.T) {
	t.Parallel()
	valid := json.RawMessage(`{
		"$schema":"https://json-schema.org/draft/2020-12/schema",
		"$id":"https://schemas.example.invalid/tool",
		"$anchor":"root",
		"$dynamicAnchor":"dynamic",
		"$ref":"#/$defs/value","$recursiveRef":"#/$defs/value","$recursiveAnchor":true,
		"$dynamicRef":"#dynamic",
		"$comment":"fixture",
		"title":"tool","description":"fixture","pattern":"^x+$",
		"contentEncoding":"base64","contentMediaType":"text/plain","format":"custom",
		"deprecated":false,"readOnly":false,"writeOnly":false,"uniqueItems":true,
		"multipleOf":1,"minimum":0,"maximum":10,"exclusiveMinimum":-1,"exclusiveMaximum":11,
		"minLength":0,"maxLength":10,"minItems":0,"maxItems":10,"minContains":0,"maxContains":10,"minProperties":0,"maxProperties":10,
		"enum":["x","y"],"examples":[{"x":"example"}],"required":["x"],"dependentRequired":{"x":["y"]},
		"additionalItems":true,"contains":true,"unevaluatedItems":true,"additionalProperties":true,
		"propertyNames":true,"unevaluatedProperties":true,"not":true,"if":true,"then":true,"else":true,"contentSchema":true,
		"items":[true],"prefixItems":[true],"allOf":[true],"anyOf":[true],"oneOf":[true],
		"$defs":{"value":true},"definitions":{"legacy":true},"properties":{"x":true},
		"patternProperties":{"^x$":true},"dependentSchemas":{"x":true},
		"dependencies":{"x":["y"],"z":true}
	}`)
	if _, ok := parseValidFunctionSchema(valid); !ok {
		t.Fatal("comprehensive valid JSON Schema was rejected")
	}

	invalid := []json.RawMessage{
		json.RawMessage(`{"title":42}`),
		json.RawMessage(`{"pattern":"["}`),
		json.RawMessage(`{"$recursiveRef":"https://example.invalid/schema"}`),
		json.RawMessage(`{"$recursiveAnchor":"yes"}`),
		json.RawMessage(`{"examples":"example"}`),
		json.RawMessage(`{"$anchor":""}`),
		json.RawMessage(`{"$anchor":"duplicate","allOf":[{"$anchor":"duplicate"}]}`),
		json.RawMessage(`{"$ref":"https://schemas.example.invalid/tool"}`),
		json.RawMessage(`{"deprecated":"false"}`),
		json.RawMessage(`{"minimum":"zero"}`),
		json.RawMessage(`{"multipleOf":0}`),
		json.RawMessage(`{"minLength":1.5}`),
		json.RawMessage(`{"enum":"x"}`),
		json.RawMessage(`{"required":"x"}`),
		json.RawMessage(`{"dependentRequired":[]}`),
		json.RawMessage(`{"dependentRequired":{"x":"y"}}`),
		json.RawMessage(`{"additionalProperties":[]}`),
		json.RawMessage(`{"items":[]}`),
		json.RawMessage(`{"items":[[]]}`),
		json.RawMessage(`{"items":[]}`),
		json.RawMessage(`{"allOf":{}}`),
		json.RawMessage(`{"allOf":[]}`),
		json.RawMessage(`{"allOf":[[]]}`),
		json.RawMessage(`{"properties":[]}`),
		json.RawMessage(`{"patternProperties":{"[":true}}`),
		json.RawMessage(`{"properties":{"x":[]}}`),
		json.RawMessage(`{"dependencies":[]}`),
		json.RawMessage(`{"dependencies":{"x":[]}}`),
		json.RawMessage(`{"dependencies":{"x":["y","y"]}}`),
		json.RawMessage(`{"dependencies":{"x":42}}`),
		json.RawMessage(`{"$ref":"#/$defs/missing","$defs":{"value":true}}`),
		json.RawMessage(`{"$dynamicRef":"#missing"}`),
	}
	for index, raw := range invalid {
		if parsed, ok := parseValidFunctionSchema(raw); ok || parsed != nil {
			t.Fatalf("invalid keyword shape %d accepted: %s", index, raw)
		}
	}
}

func TestFunctionSchemaStructuralValidatorHonorsDraft04ExclusiveBounds(t *testing.T) {
	t.Parallel()
	valid := json.RawMessage(`{
		"$schema":"http://json-schema.org/draft-04/schema#",
		"type":"object",
		"properties":{"count":{"type":"number","minimum":0,"exclusiveMinimum":true}}
	}`)
	if _, ok := parseValidFunctionSchema(valid); !ok {
		t.Fatal("valid Draft-04 boolean exclusiveMinimum was rejected")
	}
	invalidDraft04 := json.RawMessage(`{"$schema":"http://json-schema.org/draft-04/schema#","exclusiveMaximum":1}`)
	if _, ok := parseValidFunctionSchema(invalidDraft04); ok {
		t.Fatal("Draft-04 numeric exclusiveMaximum was accepted")
	}
	invalidModern := json.RawMessage(`{"$schema":"https://json-schema.org/draft/2020-12/schema","exclusiveMinimum":true}`)
	if _, ok := parseValidFunctionSchema(invalidModern); ok {
		t.Fatal("modern boolean exclusiveMinimum was accepted")
	}
	if isDraft04Schema(true) {
		t.Fatal("boolean schema was classified as Draft-04")
	}
}

func TestLocalSchemaReferenceResolutionCoversPointersAnchorsAndInvalidTargets(t *testing.T) {
	t.Parallel()
	root := map[string]any{
		"defs":   []any{map[string]any{"type": "object"}},
		"a/b":    map[string]any{"~key": true},
		"value":  "not-a-schema",
		"scalar": float64(1),
	}
	anchors := map[string]struct{}{"known": {}}
	tests := []struct {
		ref  string
		want bool
	}{
		{ref: "#", want: true},
		{ref: "#known", want: true},
		{ref: "#missing", want: false},
		{ref: "#/defs/0", want: true},
		{ref: "#/defs/not-an-index", want: false},
		{ref: "#/defs/-1", want: false},
		{ref: "#/defs/2", want: false},
		{ref: "#/missing", want: false},
		{ref: "#/scalar/child", want: false},
		{ref: "#/value", want: false},
		{ref: "#/a~1b/~0key", want: true},
	}
	for _, test := range tests {
		if got := localSchemaReferenceExists(root, test.ref, anchors); got != test.want {
			t.Fatalf("reference %q resolved=%v, want %v", test.ref, got, test.want)
		}
	}
}

func BenchmarkIdentityCanonicalFunctionSchema(b *testing.B) {
	raw := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"],"additionalProperties":false}`)
	b.ReportAllocs()
	for b.Loop() {
		_ = normalizeFunctionSchemaDetailed(raw, nil, llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIResponse)
	}
}

func TestLowerNormalizesIncompatibleSameProtocolSchemaWithoutFlattening(t *testing.T) {
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
	if !planHasStrategy(plan, StrategySchemaNormalize) {
		t.Fatalf("identity plan omitted schema normalization: %#v", plan.Actions)
	}
	lowered, session, err := Lower(request, plan)
	if err != nil {
		t.Fatalf("lower identity route: %v", err)
	}
	if lowered == request {
		t.Fatal("identity schema normalization mutated the request in place")
	}
	if string(request.ToolDefinitions[0].Function.Parameters) != string(raw) {
		t.Fatalf("source schema changed: got %s want %s", request.ToolDefinitions[0].Function.Parameters, raw)
	}
	schema := decodeSchemaObject(t, lowered.ToolDefinitions[0].Function.Parameters)
	if schema["type"] != "object" || schema["oneOf"] == nil {
		t.Fatalf("identity schema was not normalized in place: %#v", schema)
	}
	if session.Summary().SchemasNormalized != 1 {
		t.Fatalf("identity route reported schema normalization: %#v", session.Summary())
	}
}

func TestSchemaNormalizerMergesAllOfRequiredButNotAlternativeRequired(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"allOf":[{"type":"object","properties":{"base":{"type":"string"}},"required":["base"]}],"oneOf":[{"type":"object","properties":{"left":{"type":"string"}},"required":["left"]},{"type":"object","properties":{"right":{"type":"string"}},"required":["right"]}]}`)
	result := normalizeFunctionSchemaDetailed(raw, nil, llm.APIFormatOpenAIResponse, llm.APIFormatAnthropicMessage)
	normalized, changed := result.schema, result.changed
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
	restoration := session.schemaRestorations[toolWireIdentityKey{name: "weather"}]
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

package conversion

import (
	"bytes"
	"encoding/json"
	"slices"
	"sort"

	"github.com/looplj/axonhub/llm"
)

var emptyFunctionParameters = json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)

type schemaNormalization struct {
	schema         json.RawMessage
	strictOverride *bool
	optionalPaths  []schemaOptionalPath
	changed        bool
	reversible     bool
}

type schemaPathStep struct {
	property string
	array    bool
}

type schemaOptionalPath []schemaPathStep

// normalizeFunctionSchema applies only wire-compatibility rules shared by the
// three supported function-tool protocols. It is deliberately not a general
// JSON Schema validator: keywords it does not need to change remain intact.
func normalizeFunctionSchema(raw json.RawMessage, target llm.APIFormat) (json.RawMessage, bool) {
	result := normalizeFunctionSchemaDetailed(raw, nil, target, target)
	return result.schema, result.changed
}

func normalizeFunctionSchemaDetailed(raw json.RawMessage, strict *bool, source, target llm.APIFormat) schemaNormalization {
	if !isFunctionSchemaTarget(target) {
		return schemaNormalization{schema: raw, reversible: true}
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return normalizeEmptyFunctionSchema(strict, source, target, true)
	}

	var value any
	if json.Unmarshal(trimmed, &value) != nil {
		// Canonical validation normally makes this unreachable. Keeping a valid,
		// closed empty argument object here prevents a legacy projection from
		// emitting malformed JSON if it bypassed canonical validation.
		return normalizeEmptyFunctionSchema(strict, source, target, false)
	}
	root, ok := value.(map[string]any)
	if !ok {
		result := normalizeNonObjectRootSchema(value)
		var normalizedRoot map[string]any
		if json.Unmarshal(result.schema, &normalizedRoot) != nil {
			return result
		}
		effectiveStrict, strictOverride := conversionStrictMode(strict, source, target, normalizedRoot)
		result.strictOverride = strictOverride
		if strictOverride != nil {
			result.changed = true
		}
		if !effectiveStrict || !isOpenAIFormat(target) {
			return result
		}
		if schemaHasDynamicObjectProperties(normalizedRoot) {
			value := false
			result.strictOverride = &value
			result.reversible = false
			return result
		}
		normalizeOpenAIStrictSchema(normalizedRoot, nil, &result.optionalPaths, &result.changed)
		if normalized, err := json.Marshal(normalizedRoot); err == nil {
			result.schema = normalized
		}
		return result
	}

	changed := false
	normalizeSchemaNode(root, &changed)
	reversible := true
	if target == llm.APIFormatAnthropicMessage && flattenRootSchemaUnions(root) {
		changed = true
		reversible = false
	}
	if schemaType, ok := root["type"].(string); !ok || schemaType != "object" {
		root["type"] = "object"
		changed = true
		if schemaType != "" {
			reversible = false
		}
	}
	if ensureObjectProperties(root) {
		changed = true
	}
	strictOverride := (*bool)(nil)
	optionalPaths := []schemaOptionalPath(nil)
	effectiveStrict, strictOverride := conversionStrictMode(strict, source, target, root)
	if strictOverride != nil {
		changed = true
	}
	if effectiveStrict && isOpenAIFormat(target) {
		if schemaHasDynamicObjectProperties(root) {
			value := false
			strictOverride = &value
			changed = true
			reversible = false
		} else {
			normalizeOpenAIStrictSchema(root, nil, &optionalPaths, &changed)
		}
	}
	if !changed {
		return schemaNormalization{schema: raw, reversible: true}
	}
	normalized, err := json.Marshal(root)
	if err != nil {
		return schemaNormalization{
			schema: append(json.RawMessage(nil), emptyFunctionParameters...), changed: true, reversible: false,
		}
	}
	return schemaNormalization{
		schema: normalized, strictOverride: strictOverride, optionalPaths: optionalPaths,
		changed: true, reversible: reversible,
	}
}

func normalizeEmptyFunctionSchema(strict *bool, source, target llm.APIFormat, reversible bool) schemaNormalization {
	result := schemaNormalization{
		schema: append(json.RawMessage(nil), emptyFunctionParameters...), changed: true, reversible: reversible,
	}
	var root map[string]any
	_ = json.Unmarshal(result.schema, &root)
	_, result.strictOverride = conversionStrictMode(strict, source, target, root)
	return result
}

func conversionStrictMode(strict *bool, source, target llm.APIFormat, schema map[string]any) (bool, *bool) {
	if strict != nil {
		return *strict, nil
	}
	if source == llm.APIFormatOpenAIResponse && source != target {
		value := !schemaHasDynamicObjectProperties(schema)
		return value, &value
	}
	if target == llm.APIFormatOpenAIResponse && source != target {
		value := false
		return false, &value
	}
	return false, nil
}

func normalizeNonObjectRootSchema(value any) schemaNormalization {
	if allowed, ok := value.(bool); ok {
		if allowed {
			return schemaNormalization{schema: json.RawMessage(`{"type":"object","properties":{}}`), changed: true, reversible: true}
		}
		return schemaNormalization{schema: json.RawMessage(`{"type":"object","properties":{},"not":{}}`), changed: true, reversible: true}
	}
	return schemaNormalization{
		schema: append(json.RawMessage(nil), emptyFunctionParameters...), changed: true, reversible: false,
	}
}

func isFunctionSchemaTarget(target llm.APIFormat) bool {
	switch target {
	case llm.APIFormatOpenAIChatCompletion, llm.APIFormatOpenAIResponse, llm.APIFormatAnthropicMessage:
		return true
	default:
		return false
	}
}

func isOpenAIFormat(target llm.APIFormat) bool {
	return target == llm.APIFormatOpenAIChatCompletion || target == llm.APIFormatOpenAIResponse
}

func schemaHasDynamicObjectProperties(node any) bool {
	switch value := node.(type) {
	case map[string]any:
		if schemaAllowsObject(value) {
			additional, exists := value["additionalProperties"]
			if !exists || additional != false {
				return true
			}
		}
		for _, child := range value {
			if schemaHasDynamicObjectProperties(child) {
				return true
			}
		}
	case []any:
		for _, child := range value {
			if schemaHasDynamicObjectProperties(child) {
				return true
			}
		}
	}
	return false
}

func normalizeOpenAIStrictSchema(node any, path schemaOptionalPath, optionalPaths *[]schemaOptionalPath, changed *bool) {
	schema, ok := node.(map[string]any)
	if !ok {
		return
	}
	if schemaAllowsObject(schema) {
		properties, _ := schema["properties"].(map[string]any)
		requiredValues, _ := schema["required"].([]any)
		required := make(map[string]struct{}, len(requiredValues))
		for _, value := range requiredValues {
			if name, ok := value.(string); ok {
				required[name] = struct{}{}
			}
		}
		names := make([]string, 0, len(properties))
		for name := range properties {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			propertyPath := appendSchemaPath(path, schemaPathStep{property: name})
			propertySchema := properties[name]
			if _, exists := required[name]; !exists {
				propertySchema = makeSchemaNullable(propertySchema)
				properties[name] = propertySchema
				requiredValues = append(requiredValues, name)
				required[name] = struct{}{}
				*optionalPaths = append(*optionalPaths, propertyPath)
				*changed = true
			}
			normalizeOpenAIStrictSchema(propertySchema, propertyPath, optionalPaths, changed)
		}
		if len(names) > 0 && len(requiredValues) != len(schemaRequiredStrings(schema["required"])) {
			schema["required"] = requiredValues
		}
	}
	if items, exists := schema["items"]; exists {
		normalizeOpenAIStrictSchema(items, appendSchemaPath(path, schemaPathStep{array: true}), optionalPaths, changed)
	}
	for _, keyword := range []string{"allOf", "anyOf", "oneOf", "prefixItems"} {
		branches, _ := schema[keyword].([]any)
		for _, branch := range branches {
			normalizeOpenAIStrictSchema(branch, path, optionalPaths, changed)
		}
	}
}

func schemaRequiredStrings(value any) []string {
	values, _ := value.([]any)
	result := make([]string, 0, len(values))
	for _, item := range values {
		if text, ok := item.(string); ok {
			result = append(result, text)
		}
	}
	return result
}

func makeSchemaNullable(value any) any {
	schema, ok := value.(map[string]any)
	if !ok {
		return map[string]any{"anyOf": []any{value, map[string]any{"type": "null"}}}
	}
	if rawType, exists := schema["type"]; exists {
		switch schemaType := rawType.(type) {
		case string:
			if schemaType != "null" {
				schema["type"] = []any{schemaType, "null"}
			}
		case []any:
			if !slices.Contains(schemaType, any("null")) {
				schema["type"] = append(schemaType, "null")
			}
		}
		return schema
	}
	if branches, ok := schema["anyOf"].([]any); ok {
		for _, branch := range branches {
			if branchSchema, ok := branch.(map[string]any); ok && branchSchema["type"] == "null" {
				return schema
			}
		}
		schema["anyOf"] = append(branches, map[string]any{"type": "null"})
		return schema
	}
	original := make(map[string]any, len(schema))
	for key, child := range schema {
		original[key] = child
		delete(schema, key)
	}
	schema["anyOf"] = []any{original, map[string]any{"type": "null"}}
	return schema
}

func appendSchemaPath(path schemaOptionalPath, step schemaPathStep) schemaOptionalPath {
	result := make(schemaOptionalPath, len(path)+1)
	copy(result, path)
	result[len(path)] = step
	return result
}

func normalizeSchemaNode(node any, changed *bool) {
	switch value := node.(type) {
	case map[string]any:
		for _, child := range value {
			normalizeSchemaNode(child, changed)
		}
		if schemaAllowsObject(value) && ensureObjectProperties(value) {
			*changed = true
		}
	case []any:
		for _, child := range value {
			normalizeSchemaNode(child, changed)
		}
	}
}

func schemaAllowsObject(schema map[string]any) bool {
	typeValue, exists := schema["type"]
	if !exists {
		return false
	}
	switch value := typeValue.(type) {
	case string:
		return value == "object"
	case []any:
		for _, item := range value {
			if item == "object" {
				return true
			}
		}
	}
	return false
}

func ensureObjectProperties(schema map[string]any) bool {
	if properties, exists := schema["properties"]; exists {
		if _, ok := properties.(map[string]any); ok {
			return false
		}
	}
	schema["properties"] = map[string]any{}
	return true
}

func flattenRootSchemaUnions(root map[string]any) bool {
	changed := false
	properties, _ := root["properties"].(map[string]any)
	if properties == nil {
		properties = make(map[string]any)
	}
	for _, keyword := range []string{"anyOf", "oneOf", "allOf"} {
		raw, exists := root[keyword]
		if !exists {
			continue
		}
		delete(root, keyword)
		changed = true
		branches, ok := raw.([]any)
		if !ok {
			continue
		}
		for _, candidate := range branches {
			branch, ok := candidate.(map[string]any)
			if !ok || !schemaBranchCanBeObject(branch) {
				continue
			}
			if branchProperties, ok := branch["properties"].(map[string]any); ok {
				for name, property := range branchProperties {
					if _, occupied := properties[name]; !occupied {
						properties[name] = property
					}
				}
			}
			if keyword == "allOf" {
				mergeSchemaRequired(root, branch["required"])
			}
		}
	}
	if changed {
		root["properties"] = properties
	}
	return changed
}

func schemaBranchCanBeObject(branch map[string]any) bool {
	typeValue, exists := branch["type"]
	if !exists {
		return true
	}
	switch value := typeValue.(type) {
	case string:
		return value == "object"
	case []any:
		return slices.Contains(value, any("object"))
	default:
		return false
	}
}

func mergeSchemaRequired(root map[string]any, candidate any) {
	values, ok := candidate.([]any)
	if !ok || len(values) == 0 {
		return
	}
	required, _ := root["required"].([]any)
	seen := make(map[string]struct{}, len(required)+len(values))
	for _, value := range required {
		if name, ok := value.(string); ok {
			seen[name] = struct{}{}
		}
	}
	for _, value := range values {
		name, ok := value.(string)
		if !ok {
			continue
		}
		if _, exists := seen[name]; exists {
			continue
		}
		seen[name] = struct{}{}
		required = append(required, name)
	}
	if len(required) > 0 {
		root["required"] = required
	}
}

func appendSchemaConstraintActions(plan *Plan, request *llm.Request, target llm.APIFormat) {
	if plan == nil || request == nil || request.APIFormat == target || !isFunctionSchemaTarget(target) {
		return
	}
	appendDefinition := func(definition *llm.ToolDefinition, ref ObjectRef) {
		if definition == nil || definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			return
		}
		result := normalizeFunctionSchemaDetailed(definition.Function.Parameters, definition.Function.Strict, request.APIFormat, target)
		if !result.changed {
			return
		}
		plan.Actions = append(plan.Actions, Action{
			Ref: ref, Kind: ActionLower, Strategy: StrategySchemaNormalize,
			Reason: ReasonProtocolConstraint, Reversible: result.reversible,
		})
	}
	for index := range request.ToolDefinitions {
		appendDefinition(&request.ToolDefinitions[index], ObjectRef{
			Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
		})
	}
	if len(request.ToolDefinitions) == 0 {
		for index := range request.Tools {
			tool := &request.Tools[index]
			if tool.Type != llm.ToolTypeFunction {
				continue
			}
			result := normalizeFunctionSchemaDetailed(tool.Function.Parameters, tool.Function.Strict, request.APIFormat, target)
			if result.changed {
				plan.Actions = append(plan.Actions, Action{
					Ref:  ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
					Kind: ActionLower, Strategy: StrategySchemaNormalize, Reason: ReasonProtocolConstraint, Reversible: result.reversible,
				})
			}
		}
	}
	for itemIndex := range request.Input {
		result := request.Input[itemIndex].ToolResult
		if result == nil {
			continue
		}
		for discoveredIndex := range result.DiscoveredTools {
			appendDefinition(&result.DiscoveredTools[discoveredIndex], ObjectRef{
				Kind: ObjectToolDefinition, ToolIndex: discoveredIndex, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
			})
		}
	}
}

func normalizeRequestSchemas(request *llm.Request, session *Session, target llm.APIFormat) {
	if request == nil || session == nil || session.plan == nil || session.plan.Source == target || !isFunctionSchemaTarget(target) {
		return
	}
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			continue
		}
		ref := ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		normalizeCanonicalDefinitionSchema(definition, session, target, ref)
	}
	for itemIndex := range request.Input {
		result := request.Input[itemIndex].ToolResult
		if result == nil {
			continue
		}
		for discoveredIndex := range result.DiscoveredTools {
			ref := ObjectRef{Kind: ObjectToolDefinition, ToolIndex: discoveredIndex, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
			normalizeCanonicalDefinitionSchema(&result.DiscoveredTools[discoveredIndex], session, target, ref)
		}
	}
	// Keep the legacy compatibility projection coherent. Canonical definitions
	// are authoritative and account for the observation counter above.
	for index := range request.Tools {
		tool := &request.Tools[index]
		if tool.Type != llm.ToolTypeFunction {
			continue
		}
		original := append(json.RawMessage(nil), tool.Function.Parameters...)
		normalized := normalizeFunctionSchemaDetailed(original, tool.Function.Strict, session.plan.Source, target)
		if normalized.changed {
			tool.Function.Parameters = normalized.schema
			if normalized.strictOverride != nil {
				tool.Function.Strict = normalized.strictOverride
			}
			if len(request.ToolDefinitions) == 0 {
				session.registerSchemaRestoration(tool.Function.Name, normalized.optionalPaths, original, normalized.schema)
				session.recordSchemaNormalization(ObjectRef{
					Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
				}, normalized.reversible)
			}
		}
	}
}

func normalizeCanonicalDefinitionSchema(definition *llm.ToolDefinition, session *Session, target llm.APIFormat, ref ObjectRef) {
	if definition == nil || definition.Kind != llm.ToolKindFunction || definition.Function == nil {
		return
	}
	original := append(json.RawMessage(nil), definition.Function.Parameters...)
	normalized := normalizeFunctionSchemaDetailed(original, definition.Function.Strict, session.plan.Source, target)
	if !normalized.changed {
		return
	}
	definition.Function.Parameters = normalized.schema
	if normalized.strictOverride != nil {
		definition.Function.Strict = normalized.strictOverride
	}
	session.registerSchemaRestoration(definition.LogicalName, normalized.optionalPaths, original, normalized.schema)
	session.recordSchemaNormalization(ref, normalized.reversible)
}

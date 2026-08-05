package conversion

import (
	"bytes"
	"encoding/json"
	"math"
	"net/url"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/dlclark/regexp2/syntax"
	"github.com/looplj/axonhub/llm"
)

var (
	emptyFunctionParameters         = json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`)
	functionSchemaBaseURL           = url.URL{Scheme: "https", Host: "axon.invalid", Path: "/function-schema"}
	singleFunctionSubschemaKeywords = []string{
		"additionalItems", "contains", "unevaluatedItems", "additionalProperties", "propertyNames",
		"unevaluatedProperties", "not", "if", "then", "else", "contentSchema",
	}
	arrayFunctionSubschemaKeywords = []string{"prefixItems", "allOf", "anyOf", "oneOf"}
	mapFunctionSubschemaKeywords   = []string{"$defs", "definitions", "properties", "patternProperties", "dependentSchemas"}
)

type schemaNormalization struct {
	schema         json.RawMessage
	strictOverride *bool
	optionalPaths  []schemaOptionalPath
	changed        bool
	reversible     bool
	invalid        bool
}

type schemaPathStep struct {
	property string
	array    bool
}

type schemaOptionalPath []schemaPathStep

// normalizeFunctionSchema validates supported JSON Schema keyword structures
// before applying wire-compatibility rules shared by the three function-tool
// protocols. Unknown extension keywords remain intact, but malformed standard
// keywords fail closed instead of being widened or forwarded to a provider.
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
	value, valid := parseValidFunctionSchema(trimmed)
	if !valid {
		return schemaNormalization{schema: raw, reversible: true, invalid: true}
	}
	if source == target && !identityFunctionSchemaNeedsNormalization(value) {
		return schemaNormalization{schema: raw, reversible: true}
	}
	root, ok := value.(map[string]any)
	if !ok {
		if source == target {
			if _, booleanSchema := value.(bool); !booleanSchema {
				return schemaNormalization{schema: raw, reversible: true, invalid: true}
			}
		}
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
	if source == target {
		return normalizeIdentityFunctionSchema(raw, root)
	}

	changed := false
	normalizeSchemaNode(root, &changed)
	reversible := true
	if source != target && target == llm.APIFormatAnthropicMessage && flattenRootSchemaUnions(root) {
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
	if normalizeRootObjectUnionBranches(root) {
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

func identityFunctionSchemaNeedsNormalization(value any) bool {
	root, ok := value.(map[string]any)
	if !ok {
		return true
	}
	schemaType, ok := root["type"].(string)
	return !ok || schemaType != "object" || root["anyOf"] != nil || root["oneOf"] != nil
}

func normalizeIdentityFunctionSchema(raw json.RawMessage, root map[string]any) schemaNormalization {
	typeValue := root["type"]
	changed := false
	reversible := true
	switch value := typeValue.(type) {
	case string:
		if value != "object" {
			root = map[string]any{"type": "object", "allOf": []any{root}}
			changed = true
			reversible = false
		}
	case []any:
		if slices.Contains(value, any("object")) {
			root["type"] = "object"
		} else {
			root = map[string]any{"type": "object", "allOf": []any{root}}
			reversible = false
		}
		changed = true
	default:
		// parseValidFunctionSchema already rejected every explicit non-string
		// and non-array type, so this is the implicit-object case only.
		root["type"] = "object"
		changed = true
	}
	if normalizeRootObjectUnionBranches(root) {
		changed = true
	}
	if !changed {
		return schemaNormalization{schema: raw, reversible: true}
	}
	normalized, err := json.Marshal(root)
	if err != nil {
		return schemaNormalization{schema: raw, reversible: true, invalid: true}
	}
	return schemaNormalization{schema: normalized, changed: true, reversible: reversible}
}

type schemaValidationState struct {
	draft04             bool
	hasReferenceKeyword bool
}

type schemaResource struct {
	root    any
	anchors map[string]any
}

type schemaReference struct {
	base *url.URL
	ref  *url.URL
}

type schemaReferenceIndex struct {
	resources map[string]*schemaResource
	refs      []schemaReference
}

func parseValidFunctionSchema(raw []byte) (any, bool) {
	var schema any
	if json.Unmarshal(raw, &schema) != nil {
		return nil, false
	}
	state := &schemaValidationState{draft04: isDraft04Schema(schema)}
	if !validFunctionSchemaNode(schema, state) {
		return nil, false
	}
	if state.hasReferenceKeyword && !validateFunctionSchemaReferences(schema) {
		return nil, false
	}
	return schema, true
}

func validFunctionSchemaTree(schema any) bool {
	return validFunctionSchemaNode(schema, nil)
}

func validFunctionSchemaNode(schema any, state *schemaValidationState) bool {
	if _, ok := schema.(bool); ok {
		return true
	}
	object, ok := schema.(map[string]any)
	if !ok {
		return false
	}
	typeValue, hasType := object["type"]
	if !validSchemaTypeValue(typeValue, hasType) {
		return false
	}
	for _, keyword := range []string{
		"$schema", "$id", "$anchor", "$dynamicAnchor", "$ref", "$dynamicRef", "$recursiveRef", "$comment",
		"title", "description", "pattern", "contentEncoding", "contentMediaType", "format",
	} {
		value, exists := object[keyword]
		if !exists {
			continue
		}
		text, ok := value.(string)
		if !ok {
			return false
		}
		if state != nil {
			switch keyword {
			case "$id", "$anchor", "$dynamicAnchor", "$ref", "$dynamicRef", "$recursiveRef":
				state.hasReferenceKeyword = true
			}
		}
		if keyword == "pattern" {
			if !validJSONSchemaPattern(text) {
				return false
			}
		}
		if keyword == "$schema" {
			identifier, validURI := parseSchemaURIReference(text)
			if !validURI || !identifier.IsAbs() {
				return false
			}
		}
		if keyword == "$anchor" || keyword == "$dynamicAnchor" {
			if !validSchemaAnchor(text) {
				return false
			}
		}
	}
	for _, keyword := range []string{"deprecated", "readOnly", "writeOnly", "uniqueItems", "$recursiveAnchor"} {
		if value, exists := object[keyword]; exists {
			if _, ok := value.(bool); !ok {
				return false
			}
		}
	}
	for _, keyword := range []string{"multipleOf", "minimum", "maximum"} {
		value, exists := object[keyword]
		if !exists {
			continue
		}
		number, ok := value.(float64)
		if !ok || math.IsNaN(number) || math.IsInf(number, 0) || keyword == "multipleOf" && number <= 0 {
			return false
		}
	}
	for _, keyword := range []string{"exclusiveMinimum", "exclusiveMaximum"} {
		value, exists := object[keyword]
		if !exists {
			continue
		}
		if state != nil && state.draft04 {
			if _, ok := value.(bool); !ok {
				return false
			}
			continue
		}
		number, ok := value.(float64)
		if !ok || math.IsNaN(number) || math.IsInf(number, 0) {
			return false
		}
	}
	for _, keyword := range []string{
		"minLength", "maxLength", "minItems", "maxItems", "minContains", "maxContains", "minProperties", "maxProperties",
	} {
		if value, exists := object[keyword]; exists && !nonNegativeJSONInteger(value) {
			return false
		}
	}
	if value, exists := object["enum"]; exists {
		values, ok := value.([]any)
		if !ok || len(values) == 0 || containsDuplicateJSONValues(values) {
			return false
		}
	}
	if value, exists := object["examples"]; exists {
		if _, ok := value.([]any); !ok {
			return false
		}
	}
	if value, exists := object["required"]; exists {
		required, ok := stringArray(value)
		if !ok || hasDuplicateStrings(required) {
			return false
		}
	}
	if value, exists := object["dependentRequired"]; exists {
		dependencies, ok := value.(map[string]any)
		if !ok {
			return false
		}
		for _, raw := range dependencies {
			required, ok := stringArray(raw)
			if !ok || hasDuplicateStrings(required) {
				return false
			}
		}
	}
	for _, keyword := range singleFunctionSubschemaKeywords {
		if child, exists := object[keyword]; exists && !validFunctionSchemaNode(child, state) {
			return false
		}
	}
	if items, exists := object["items"]; exists {
		if array, ok := items.([]any); ok {
			if len(array) == 0 {
				return false
			}
			for _, child := range array {
				if !validFunctionSchemaNode(child, state) {
					return false
				}
			}
		} else if !validFunctionSchemaNode(items, state) {
			return false
		}
	}
	for _, keyword := range arrayFunctionSubschemaKeywords {
		value, exists := object[keyword]
		if !exists {
			continue
		}
		children, ok := value.([]any)
		if !ok || len(children) == 0 {
			return false
		}
		for _, child := range children {
			if !validFunctionSchemaNode(child, state) {
				return false
			}
		}
	}
	for _, keyword := range mapFunctionSubschemaKeywords {
		value, exists := object[keyword]
		if !exists {
			continue
		}
		children, ok := value.(map[string]any)
		if !ok {
			return false
		}
		for name, child := range children {
			if keyword == "patternProperties" {
				if !validJSONSchemaPattern(name) {
					return false
				}
			}
			if !validFunctionSchemaNode(child, state) {
				return false
			}
		}
	}
	if value, exists := object["dependencies"]; exists {
		dependencies, ok := value.(map[string]any)
		if !ok {
			return false
		}
		for _, dependency := range dependencies {
			if required, ok := stringArray(dependency); ok {
				if len(required) == 0 || hasDuplicateStrings(required) {
					return false
				}
				continue
			}
			if !validFunctionSchemaNode(dependency, state) {
				return false
			}
		}
	}
	return true
}

func isDraft04Schema(schema any) bool {
	object, ok := schema.(map[string]any)
	if !ok {
		return false
	}
	identifier, ok := object["$schema"].(string)
	return ok && strings.Contains(strings.ToLower(identifier), "draft-04")
}

func validSchemaTypeValue(value any, exists bool) bool {
	if !exists {
		return true
	}
	if single, ok := value.(string); ok {
		return validSchemaTypes(single, nil)
	}
	values, ok := stringArray(value)
	return ok && validSchemaTypes("", values)
}

func validSchemaTypes(single string, multiple []string) bool {
	valid := func(candidate string) bool {
		switch candidate {
		case "null", "boolean", "object", "array", "number", "string", "integer":
			return true
		default:
			return false
		}
	}
	if single != "" {
		return multiple == nil && valid(single)
	}
	if multiple == nil {
		return true
	}
	if len(multiple) == 0 {
		return false
	}
	seen := make(map[string]struct{}, len(multiple))
	for _, candidate := range multiple {
		if !valid(candidate) {
			return false
		}
		if _, duplicate := seen[candidate]; duplicate {
			return false
		}
		seen[candidate] = struct{}{}
	}
	return true
}

func hasDuplicateStrings(values []string) bool {
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if _, duplicate := seen[value]; duplicate {
			return true
		}
		seen[value] = struct{}{}
	}
	return false
}

func containsDuplicateJSONValues(values []any) bool {
	for index := range values {
		for other := 0; other < index; other++ {
			if reflect.DeepEqual(values[index], values[other]) {
				return true
			}
		}
	}
	return false
}

func stringArray(value any) ([]string, bool) {
	items, ok := value.([]any)
	if !ok {
		return nil, false
	}
	result := make([]string, len(items))
	for index, item := range items {
		text, ok := item.(string)
		if !ok {
			return nil, false
		}
		result[index] = text
	}
	return result, true
}

func nonNegativeJSONInteger(value any) bool {
	number, ok := value.(float64)
	return ok && number >= 0 && number == math.Trunc(number)
}

func validJSONSchemaPattern(pattern string) bool {
	_, err := syntax.Parse(pattern, syntax.ECMAScript|syntax.Unicode)
	return err == nil
}

func validSchemaAnchor(anchor string) bool {
	if anchor == "" || (!isASCIILetter(anchor[0]) && anchor[0] != '_') {
		return false
	}
	for index := 1; index < len(anchor); index++ {
		char := anchor[index]
		if !isASCIILetter(char) && (char < '0' || char > '9') && char != '-' && char != '.' && char != '_' {
			return false
		}
	}
	return true
}

func isASCIILetter(char byte) bool {
	return char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z'
}

func validateFunctionSchemaReferences(root any) bool {
	base := functionSchemaBaseURL
	index := &schemaReferenceIndex{resources: make(map[string]*schemaResource)}
	if !indexFunctionSchemaResources(root, &base, index) {
		return false
	}
	for _, ref := range index.refs {
		if !localSchemaReferenceExists(index, ref) {
			return false
		}
	}
	return true
}

func indexFunctionSchemaResources(schema any, inheritedBase *url.URL, index *schemaReferenceIndex) bool {
	if _, ok := schema.(bool); ok {
		if len(index.resources) == 0 {
			index.resources[schemaResourceKey(inheritedBase)] = &schemaResource{root: schema}
		}
		return true
	}
	object, ok := schema.(map[string]any)
	if !ok {
		return false
	}
	base := inheritedBase
	establishesResource := len(index.resources) == 0
	if rawID, exists := object["$id"]; exists {
		identifier, ok := rawID.(string)
		if !ok {
			return false
		}
		parsedID, ok := parseSchemaURIReference(identifier)
		if !ok || parsedID.Fragment != "" {
			return false
		}
		resolved := inheritedBase.ResolveReference(parsedID)
		if !resolved.IsAbs() {
			return false
		}
		base = resolved
		establishesResource = true
	}
	resourceKey := schemaResourceKey(base)
	resource := index.resources[resourceKey]
	if establishesResource {
		if resource != nil {
			return false
		}
		resource = &schemaResource{root: schema}
		index.resources[resourceKey] = resource
	} else if resource == nil {
		return false
	}
	for _, keyword := range []string{"$anchor", "$dynamicAnchor"} {
		anchor, exists := object[keyword].(string)
		if !exists {
			continue
		}
		if resource.anchors == nil {
			resource.anchors = make(map[string]any)
		}
		if _, duplicate := resource.anchors[anchor]; duplicate {
			return false
		}
		resource.anchors[anchor] = schema
	}
	for _, keyword := range []string{"$ref", "$dynamicRef", "$recursiveRef"} {
		rawRef, exists := object[keyword].(string)
		if !exists {
			continue
		}
		parsedRef, ok := parseSchemaURIReference(rawRef)
		if !ok {
			return false
		}
		index.refs = append(index.refs, schemaReference{base: base, ref: parsedRef})
	}
	return forEachFunctionSubschema(object, func(child any) bool {
		return indexFunctionSchemaResources(child, base, index)
	})
}

func forEachFunctionSubschema(object map[string]any, visit func(any) bool) bool {
	for _, keyword := range singleFunctionSubschemaKeywords {
		if child, exists := object[keyword]; exists && !visit(child) {
			return false
		}
	}
	if items, exists := object["items"]; exists {
		if children, ok := items.([]any); ok {
			for _, child := range children {
				if !visit(child) {
					return false
				}
			}
		} else if !visit(items) {
			return false
		}
	}
	for _, keyword := range arrayFunctionSubschemaKeywords {
		children, exists := object[keyword].([]any)
		if !exists {
			continue
		}
		for _, child := range children {
			if !visit(child) {
				return false
			}
		}
	}
	for _, keyword := range mapFunctionSubschemaKeywords {
		children, exists := object[keyword].(map[string]any)
		if !exists {
			continue
		}
		for _, child := range children {
			if !visit(child) {
				return false
			}
		}
	}
	if dependencies, exists := object["dependencies"].(map[string]any); exists {
		for _, dependency := range dependencies {
			if _, requiredNames := stringArray(dependency); requiredNames {
				continue
			}
			if !visit(dependency) {
				return false
			}
		}
	}
	return true
}

func parseSchemaURIReference(raw string) (*url.URL, bool) {
	for index := 0; index < len(raw); index++ {
		char := raw[index]
		alphaNumeric := (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') || (char >= '0' && char <= '9')
		if !alphaNumeric && !strings.ContainsRune("-._~:/?#[]@!$&'()*+,;=%", rune(char)) {
			return nil, false
		}
	}
	parsed, err := url.Parse(raw)
	return parsed, err == nil
}

func schemaResourceKey(uri *url.URL) string {
	resourceURI := *uri
	resourceURI.Fragment = ""
	resourceURI.RawFragment = ""
	return resourceURI.String()
}

func localSchemaReferenceExists(index *schemaReferenceIndex, reference schemaReference) bool {
	resolved := reference.base.ResolveReference(reference.ref)
	resource := index.resources[schemaResourceKey(resolved)]
	if resource == nil {
		return false
	}
	fragment := resolved.Fragment
	var target any
	if fragment == "" {
		target = resource.root
	} else if !strings.HasPrefix(fragment, "/") {
		var ok bool
		target, ok = resource.anchors[fragment]
		if !ok {
			return false
		}
	} else {
		var ok bool
		target, ok = dereferenceSchemaJSONPointer(resource.root, fragment)
		if !ok {
			return false
		}
	}
	return validFunctionSchemaTree(target)
}

func dereferenceSchemaJSONPointer(root any, fragment string) (any, bool) {
	current := root
	for _, rawSegment := range strings.Split(strings.TrimPrefix(fragment, "/"), "/") {
		segment, ok := decodeJSONPointerToken(rawSegment)
		if !ok {
			return nil, false
		}
		switch value := current.(type) {
		case map[string]any:
			current, ok = value[segment]
			if !ok {
				return nil, false
			}
		case []any:
			arrayIndex, ok := parseJSONPointerArrayIndex(segment)
			if !ok || arrayIndex >= len(value) {
				return nil, false
			}
			current = value[arrayIndex]
		default:
			return nil, false
		}
	}
	return current, true
}

func decodeJSONPointerToken(raw string) (string, bool) {
	if !strings.Contains(raw, "~") {
		return raw, true
	}
	var decoded strings.Builder
	decoded.Grow(len(raw))
	for index := 0; index < len(raw); index++ {
		if raw[index] != '~' {
			decoded.WriteByte(raw[index])
			continue
		}
		if index+1 >= len(raw) {
			return "", false
		}
		index++
		switch raw[index] {
		case '0':
			decoded.WriteByte('~')
		case '1':
			decoded.WriteByte('/')
		default:
			return "", false
		}
	}
	return decoded.String(), true
}

func parseJSONPointerArrayIndex(token string) (int, bool) {
	if token == "0" {
		return 0, true
	}
	if token == "" || token[0] < '1' || token[0] > '9' {
		return 0, false
	}
	for index := 1; index < len(token); index++ {
		if token[index] < '0' || token[index] > '9' {
			return 0, false
		}
	}
	value, err := strconv.Atoi(token)
	return value, err == nil
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

// normalizeRootObjectUnionBranches makes the object contract explicit on
// every root anyOf/oneOf branch. Function arguments are object-valued on all
// three supported protocols, so intersecting each alternative with object is
// semantics-preserving while satisfying providers that validate union
// branches independently instead of inheriting the root type.
func normalizeRootObjectUnionBranches(root map[string]any) bool {
	changed := false
	for _, keyword := range []string{"anyOf", "oneOf"} {
		branches, ok := root[keyword].([]any)
		if !ok {
			continue
		}
		for index := range branches {
			normalized, branchChanged := normalizeRootObjectUnionBranch(branches[index])
			if !branchChanged {
				continue
			}
			branches[index] = normalized
			changed = true
		}
	}
	return changed
}

func normalizeRootObjectUnionBranch(branch any) (any, bool) {
	schema, ok := branch.(map[string]any)
	if !ok {
		if allowed, booleanSchema := branch.(bool); booleanSchema && allowed {
			return map[string]any{"type": "object"}, true
		}
		return map[string]any{"type": "object", "not": map[string]any{}}, true
	}
	typeValue, hasType := schema["type"]
	if !hasType {
		schema["type"] = "object"
		return schema, true
	}
	switch value := typeValue.(type) {
	case string:
		if value == "object" {
			return schema, false
		}
	case []any:
		if slices.Contains(value, any("object")) {
			schema["type"] = "object"
			return schema, true
		}
	}
	return map[string]any{
		"type": "object", "allOf": []any{schema},
	}, true
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
	if plan == nil || request == nil || !isFunctionSchemaTarget(target) {
		return
	}
	appendDefinition := func(definition *llm.ToolDefinition, ref ObjectRef) {
		if definition == nil || definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			return
		}
		result := normalizeFunctionSchemaDetailed(definition.Function.Parameters, definition.Function.Strict, request.APIFormat, target)
		if result.invalid {
			plan.Actions = append(plan.Actions, Action{
				Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable,
				Reason: ReasonProtocolConstraint, Reversible: false,
			})
			return
		}
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
			if result.invalid {
				plan.Actions = append(plan.Actions, Action{
					Ref:  ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
					Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonProtocolConstraint,
				})
			} else if result.changed {
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
	if request == nil || session == nil || session.plan == nil || !isFunctionSchemaTarget(target) {
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
				session.registerSchemaRestoration(tool.Function.Name, "", normalized.optionalPaths, original, normalized.schema)
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
	session.registerSchemaRestoration(definition.LogicalName, definition.Function.Namespace, normalized.optionalPaths, original, normalized.schema)
	session.recordSchemaNormalization(ref, normalized.reversible)
}

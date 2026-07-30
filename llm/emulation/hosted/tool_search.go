package hosted

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/looplj/axonhub/llm"
)

const (
	maxToolSearchPattern = 200
	maxToolSearchQuery   = 500
	maxToolSearchResults = 5
)

var (
	regexToolSearchParameters = json.RawMessage(`{"type":"object","properties":{"pattern":{"type":"string","maxLength":200}},"required":["pattern"],"additionalProperties":false}`)
	bm25ToolSearchParameters  = json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","maxLength":500}},"required":["query"],"additionalProperties":false}`)
)

type toolReference struct {
	Type     string `json:"type"`
	ToolName string `json:"tool_name"`
}

type toolSearchResult struct {
	Type           string          `json:"type"`
	ToolReferences []toolReference `json:"tool_references"`
}

func toolSearchFunction(definition llm.ToolDefinition) (llm.FunctionDefinition, error) {
	mode, err := toolSearchMode(definition)
	if err != nil {
		return llm.FunctionDefinition{}, err
	}
	parameters := bm25ToolSearchParameters
	if mode == "regex" {
		parameters = regexToolSearchParameters
	}
	return llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), parameters...)}, nil
}

func toolSearchMode(definition llm.ToolDefinition) (string, error) {
	if definition.Kind != llm.ToolKindToolSearch || definition.Hosted == nil {
		return "", errors.New("tool search definition is missing hosted configuration")
	}
	typeName := strings.ToLower(strings.TrimSpace(definition.Hosted.Type))
	if typeName == "" && len(definition.Hosted.Configuration) > 0 {
		var configuration struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(definition.Hosted.Configuration, &configuration) == nil {
			typeName = strings.ToLower(configuration.Type)
		}
	}
	switch {
	case strings.Contains(typeName, "regex"):
		return "regex", nil
	case strings.Contains(typeName, "bm25"):
		return "bm25", nil
	default:
		return "", fmt.Errorf("unsupported tool search type %q", definition.Hosted.Type)
	}
}

// Search executes Anthropic's provider-owned tool search against the deferred
// client-tool catalog retained by this request-local registry. Matches are
// activated atomically so the next provider round sees exactly the discovered
// definitions while concurrent calls cannot introduce duplicates.
func (registry *Registry) Search(binding Binding, invocation llm.ToolInvocation) (*llm.ToolResult, error) {
	if registry == nil {
		return nil, errors.New("tool search registry is missing")
	}
	registry.mu.RLock()
	current, ok := registry.bindings[binding.SyntheticName]
	catalog := make([]llm.ToolDefinition, 0, len(registry.catalog))
	for _, definition := range registry.catalog {
		catalog = append(catalog, cloneToolDefinition(definition))
	}
	registry.mu.RUnlock()
	if !ok || current.Definition.Kind != llm.ToolKindToolSearch || current.Executor != nil {
		return nil, errors.New("binding is not a built-in tool search")
	}

	arguments, err := invocationArguments(invocation)
	if err != nil {
		return nil, err
	}
	mode, err := toolSearchMode(current.Definition)
	if err != nil {
		return nil, err
	}
	var selected []llm.ToolDefinition
	switch mode {
	case "regex":
		selected, err = searchToolsRegex(catalog, arguments)
	case "bm25":
		selected, err = searchToolsBM25(catalog, arguments)
	}
	if err != nil {
		return nil, err
	}
	for index := range selected {
		selected[index].DeferLoading = nil
	}
	registry.activate(selected)

	references := make([]toolReference, 0, len(selected))
	for index := range selected {
		references = append(references, toolReference{Type: "tool_reference", ToolName: selected[index].LogicalName})
	}
	structured, err := json.Marshal(toolSearchResult{
		Type: "tool_search_tool_search_result", ToolReferences: references,
	})
	if err != nil {
		return nil, errors.New("encode tool search result")
	}
	return &llm.ToolResult{
		Kind: llm.ToolKindToolSearch, CallID: invocation.CallID, LogicalName: current.Definition.LogicalName,
		Content:           []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(structured)}},
		StructuredContent: structured, DiscoveredTools: cloneToolDefinitions(selected),
		Execution: llm.ExecutionOwnerGateway, Status: llm.ToolResultStatusCompleted,
	}, nil
}

func invocationArguments(invocation llm.ToolInvocation) (json.RawMessage, error) {
	if len(invocation.ArgumentsJSON) > 0 {
		if !json.Valid(invocation.ArgumentsJSON) {
			return nil, errors.New("tool search arguments are invalid JSON")
		}
		return invocation.ArgumentsJSON, nil
	}
	if strings.TrimSpace(invocation.ArgumentsText) == "" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid([]byte(invocation.ArgumentsText)) {
		return nil, errors.New("tool search arguments are invalid JSON")
	}
	return json.RawMessage(invocation.ArgumentsText), nil
}

func searchToolsRegex(catalog []llm.ToolDefinition, arguments json.RawMessage) ([]llm.ToolDefinition, error) {
	var input struct {
		Pattern string `json:"pattern"`
	}
	if err := json.Unmarshal(arguments, &input); err != nil {
		return nil, fmt.Errorf("decode tool search pattern: %w", err)
	}
	if input.Pattern == "" {
		return nil, errors.New("tool search pattern is required")
	}
	if utf8.RuneCountInString(input.Pattern) > maxToolSearchPattern {
		return nil, fmt.Errorf("tool search pattern exceeds %d characters", maxToolSearchPattern)
	}
	expression, err := regexp.Compile("(?i:" + input.Pattern + ")")
	if err != nil {
		return nil, fmt.Errorf("compile tool search pattern: %w", err)
	}
	selected := make([]llm.ToolDefinition, 0, maxToolSearchResults)
	sort.SliceStable(catalog, func(i, j int) bool { return catalog[i].LogicalName < catalog[j].LogicalName })
	for index := range catalog {
		if matchesToolRegex(expression, catalog[index]) {
			selected = append(selected, catalog[index])
			if len(selected) == maxToolSearchResults {
				break
			}
		}
	}
	return selected, nil
}

func matchesToolRegex(expression *regexp.Regexp, definition llm.ToolDefinition) bool {
	if expression.MatchString(definition.LogicalName) || expression.MatchString(definition.Description) {
		return true
	}
	if definition.Function != nil && expression.Match(definition.Function.Parameters) {
		return true
	}
	if definition.Freeform != nil {
		return expression.MatchString(definition.Freeform.Format) || expression.MatchString(definition.Freeform.Syntax) ||
			expression.MatchString(definition.Freeform.Definition)
	}
	return false
}

func searchToolsBM25(catalog []llm.ToolDefinition, arguments json.RawMessage) ([]llm.ToolDefinition, error) {
	var input struct {
		Query string `json:"query"`
	}
	if err := json.Unmarshal(arguments, &input); err != nil {
		return nil, fmt.Errorf("decode tool search query: %w", err)
	}
	if strings.TrimSpace(input.Query) == "" {
		return nil, errors.New("tool search query is required")
	}
	if utf8.RuneCountInString(input.Query) > maxToolSearchQuery {
		return nil, fmt.Errorf("tool search query exceeds %d characters", maxToolSearchQuery)
	}
	queryTerms := uniqueTerms(tokenizeToolText(input.Query))
	if len(queryTerms) == 0 || len(catalog) == 0 {
		return nil, nil
	}
	type document struct {
		definition llm.ToolDefinition
		terms      []string
		frequency  map[string]int
		score      float64
	}
	documents := make([]document, 0, len(catalog))
	documentFrequency := make(map[string]int, len(queryTerms))
	totalLength := 0
	for index := range catalog {
		terms := tokenizeToolText(searchableToolText(catalog[index]))
		frequency := make(map[string]int, len(terms))
		for _, term := range terms {
			frequency[term]++
		}
		for _, term := range queryTerms {
			if frequency[term] > 0 {
				documentFrequency[term]++
			}
		}
		totalLength += len(terms)
		documents = append(documents, document{definition: catalog[index], terms: terms, frequency: frequency})
	}
	averageLength := float64(totalLength) / float64(len(documents))
	if averageLength == 0 {
		averageLength = 1
	}
	const k1, b = 1.2, 0.75
	for index := range documents {
		documentLength := float64(len(documents[index].terms))
		for _, term := range queryTerms {
			frequency := float64(documents[index].frequency[term])
			if frequency == 0 {
				continue
			}
			df := float64(documentFrequency[term])
			idf := math.Log(1 + (float64(len(documents))-df+0.5)/(df+0.5))
			denominator := frequency + k1*(1-b+b*documentLength/averageLength)
			documents[index].score += idf * frequency * (k1 + 1) / denominator
		}
	}
	sort.SliceStable(documents, func(i, j int) bool {
		if documents[i].score == documents[j].score {
			return documents[i].definition.LogicalName < documents[j].definition.LogicalName
		}
		return documents[i].score > documents[j].score
	})
	selected := make([]llm.ToolDefinition, 0, min(maxToolSearchResults, len(documents)))
	for index := range documents {
		if documents[index].score <= 0 || len(selected) == maxToolSearchResults {
			break
		}
		selected = append(selected, documents[index].definition)
	}
	return selected, nil
}

func searchableToolText(definition llm.ToolDefinition) string {
	parts := []string{definition.LogicalName, definition.Description}
	if definition.Function != nil {
		parts = append(parts, string(definition.Function.Parameters))
	}
	if definition.Freeform != nil {
		parts = append(parts, definition.Freeform.Format, definition.Freeform.Syntax, definition.Freeform.Definition)
	}
	return strings.Join(parts, " ")
}

func tokenizeToolText(value string) []string {
	return strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

func uniqueTerms(terms []string) []string {
	seen := make(map[string]struct{}, len(terms))
	result := make([]string, 0, len(terms))
	for _, term := range terms {
		if _, exists := seen[term]; exists {
			continue
		}
		seen[term] = struct{}{}
		result = append(result, term)
	}
	return result
}

func (registry *Registry) activate(definitions []llm.ToolDefinition) {
	registry.mu.Lock()
	defer registry.mu.Unlock()
	for index := range definitions {
		definition := cloneToolDefinition(definitions[index])
		key := catalogKey(definition.Kind, definition.LogicalName)
		if _, cataloged := registry.catalog[key]; !cataloged {
			continue
		}
		if _, active := registry.active[key]; active {
			continue
		}
		definition.DeferLoading = nil
		registry.active[key] = struct{}{}
		registry.definitions = append(registry.definitions, definition)
	}
	sort.SliceStable(registry.definitions, func(i, j int) bool {
		return registry.definitions[i].LogicalName < registry.definitions[j].LogicalName
	})
}

func (registry *Registry) activateFromHistory(items []llm.Item) {
	for index := range items {
		item := &items[index]
		if item.Kind != llm.ItemKindHostedCall || item.HostedCall == nil ||
			item.HostedCall.Invocation.Kind != llm.ToolKindToolSearch || item.HostedCall.Result == nil {
			continue
		}
		result := item.HostedCall.Result
		if len(result.DiscoveredTools) > 0 {
			registry.activate(result.DiscoveredTools)
			continue
		}
		var decoded toolSearchResult
		if json.Unmarshal(result.StructuredContent, &decoded) != nil {
			continue
		}
		registry.mu.RLock()
		definitions := make([]llm.ToolDefinition, 0, len(decoded.ToolReferences))
		for _, reference := range decoded.ToolReferences {
			for _, definition := range registry.catalog {
				if definition.LogicalName == reference.ToolName {
					definitions = append(definitions, cloneToolDefinition(definition))
					break
				}
			}
		}
		registry.mu.RUnlock()
		registry.activate(definitions)
	}
}

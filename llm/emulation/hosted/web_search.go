package hosted

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/looplj/axonhub/llm"
)

const defaultWebSearchResponseLimit int64 = 2 << 20

type EndpointPolicy func(*url.URL) error

type WebSearchExecutorConfig struct {
	Endpoint         string
	Client           *http.Client
	Headers          http.Header
	EndpointPolicy   EndpointPolicy
	MaxResponseBytes int64
}

// WebSearchExecutor calls a configured typed search service. Its transport
// contract is deliberately smaller than any LLM protocol: one query request
// and text/citation output. Credentials remain in executor configuration and
// never enter canonical state or conversion traces.
type WebSearchExecutor struct {
	endpoint *url.URL
	client   *http.Client
	headers  http.Header
	limit    int64
}

type WebSearchInput struct {
	Query          string                        `json:"query,omitempty"`
	Queries        []string                      `json:"queries,omitempty"`
	AllowedDomains []string                      `json:"allowed_domains,omitempty"`
	BlockedDomains []string                      `json:"blocked_domains,omitempty"`
	UserLocation   llm.WebSearchToolUserLocation `json:"user_location,omitempty"`
}

type WebSearchSource struct {
	URL   string `json:"url"`
	Title string `json:"title,omitempty"`
}

type WebSearchOutput struct {
	Text    string            `json:"text,omitempty"`
	Sources []WebSearchSource `json:"sources,omitempty"`
}

func NewWebSearchExecutor(config WebSearchExecutorConfig) (*WebSearchExecutor, error) {
	endpoint, client, err := policyHTTPClient("web search executor", config.Endpoint, config.Client, config.EndpointPolicy)
	if err != nil {
		return nil, err
	}
	limit := config.MaxResponseBytes
	if limit <= 0 {
		limit = defaultWebSearchResponseLimit
	}
	headers := config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	return &WebSearchExecutor{endpoint: endpoint, client: client, headers: headers, limit: limit}, nil
}

func (*WebSearchExecutor) Kind() llm.ToolKind { return llm.ToolKindWebSearch }

func (*WebSearchExecutor) Function(definition llm.ToolDefinition) (llm.FunctionDefinition, error) {
	if definition.Kind != llm.ToolKindWebSearch {
		return llm.FunctionDefinition{}, errors.New("web search executor received another tool kind")
	}
	return llm.FunctionDefinition{Parameters: json.RawMessage(`{
		"type":"object",
		"properties":{
			"query":{"type":"string"},
			"queries":{"type":"array","items":{"type":"string"}}
		},
		"anyOf":[{"required":["query"]},{"required":["queries"]}],
		"additionalProperties":true
	}`)}, nil
}

func (executor *WebSearchExecutor) Execute(ctx context.Context, definition llm.ToolDefinition, invocation llm.ToolInvocation) (*llm.ToolResult, error) {
	if executor == nil || executor.endpoint == nil {
		return nil, errors.New("web search executor is not initialized")
	}
	input, err := webSearchInput(definition, invocation)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode web search request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, executor.endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create web search request: %w", err)
	}
	request.Header = executor.headers.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := executor.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("execute web search request: %w", err)
	}
	defer response.Body.Close()
	limited := io.LimitReader(response.Body, executor.limit+1)
	responseBody, err := io.ReadAll(limited)
	if err != nil {
		return nil, fmt.Errorf("read web search response: %w", err)
	}
	if int64(len(responseBody)) > executor.limit {
		return nil, errors.New("web search response exceeds limit")
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("web search service returned HTTP %d", response.StatusCode)
	}
	var output WebSearchOutput
	if err := json.Unmarshal(responseBody, &output); err != nil {
		return nil, fmt.Errorf("decode web search response: %w", err)
	}
	result := &llm.ToolResult{
		Kind: llm.ToolKindWebSearch, CallID: invocation.CallID, LogicalName: definition.LogicalName,
		Status: llm.ToolResultStatusCompleted,
	}
	if output.Text != "" {
		result.Content = append(result.Content, llm.ContentBlock{Kind: llm.ContentKindText, Text: output.Text})
	}
	for index := range output.Sources {
		source := output.Sources[index]
		if strings.TrimSpace(source.URL) == "" {
			continue
		}
		result.Content = append(result.Content, llm.ContentBlock{
			Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{URL: source.URL, Title: source.Title},
		})
	}
	return result, nil
}

func webSearchInput(definition llm.ToolDefinition, invocation llm.ToolInvocation) (WebSearchInput, error) {
	input := WebSearchInput{}
	arguments := invocation.ArgumentsJSON
	if len(arguments) == 0 && invocation.ArgumentsText != "" {
		arguments = json.RawMessage(invocation.ArgumentsText)
	}
	if len(arguments) == 0 || !json.Valid(arguments) {
		return input, errors.New("web search invocation requires JSON arguments")
	}
	var wrapped struct {
		Type    string   `json:"type"`
		Query   string   `json:"query"`
		Queries []string `json:"queries"`
	}
	if err := json.Unmarshal(arguments, &wrapped); err != nil {
		return input, errors.New("decode web search arguments")
	}
	input.Query = strings.TrimSpace(wrapped.Query)
	for _, query := range wrapped.Queries {
		if query = strings.TrimSpace(query); query != "" {
			input.Queries = append(input.Queries, query)
		}
	}
	if input.Query == "" && len(input.Queries) == 0 {
		return input, errors.New("web search invocation requires query or queries")
	}
	if definition.Hosted != nil && definition.Hosted.WebSearch != nil {
		webSearch := definition.Hosted.WebSearch
		input.AllowedDomains = append([]string(nil), webSearch.AllowedDomains...)
		input.BlockedDomains = append([]string(nil), webSearch.BlockedDomains...)
		input.UserLocation = webSearch.UserLocation
	}
	if _, err := normalizeDomainRules(input.AllowedDomains); err != nil {
		return input, fmt.Errorf("invalid web search allowed domains: %w", err)
	}
	if _, err := normalizeDomainRules(input.BlockedDomains); err != nil {
		return input, fmt.Errorf("invalid web search blocked domains: %w", err)
	}
	return input, nil
}

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

const defaultWebFetchResponseLimit int64 = 8 << 20

type WebFetchExecutorConfig struct {
	Endpoint         string
	Client           *http.Client
	Headers          http.Header
	EndpointPolicy   EndpointPolicy
	MaxResponseBytes int64
}

// WebFetchExecutor delegates provider-hosted URL retrieval to a deployment
// service. The service boundary is intentionally protocol-neutral: it returns
// extracted text and source metadata, while the source encoder reconstructs
// the client's native hosted-tool lifecycle.
type WebFetchExecutor struct {
	endpoint *url.URL
	client   *http.Client
	headers  http.Header
	limit    int64
}

type WebFetchInput struct {
	URL               string   `json:"url"`
	AllowedDomains    []string `json:"allowed_domains,omitempty"`
	BlockedDomains    []string `json:"blocked_domains,omitempty"`
	CitationsEnabled  *bool    `json:"citations_enabled,omitempty"`
	MaxContentTokens  *int64   `json:"max_content_tokens,omitempty"`
	UseCache          *bool    `json:"use_cache,omitempty"`
	ResponseInclusion string   `json:"response_inclusion,omitempty"`
}

type WebFetchOutput struct {
	URL         string `json:"url,omitempty"`
	Title       string `json:"title,omitempty"`
	RetrievedAt string `json:"retrieved_at,omitempty"`
	Text        string `json:"text,omitempty"`
}

func NewWebFetchExecutor(config WebFetchExecutorConfig) (*WebFetchExecutor, error) {
	endpoint, err := url.Parse(config.Endpoint)
	if err != nil || endpoint.Scheme == "" || endpoint.Host == "" {
		return nil, errors.New("web fetch executor requires an absolute endpoint")
	}
	if config.EndpointPolicy == nil {
		return nil, errors.New("web fetch executor requires an endpoint policy")
	}
	if err := config.EndpointPolicy(endpoint); err != nil {
		return nil, fmt.Errorf("web fetch executor endpoint rejected: %w", err)
	}
	client := config.Client
	if client == nil {
		client = http.DefaultClient
	}
	limit := config.MaxResponseBytes
	if limit <= 0 {
		limit = defaultWebFetchResponseLimit
	}
	return &WebFetchExecutor{endpoint: endpoint, client: client, headers: config.Headers.Clone(), limit: limit}, nil
}

func (*WebFetchExecutor) Kind() llm.ToolKind { return llm.ToolKindWebFetch }

func (*WebFetchExecutor) Function(definition llm.ToolDefinition) (llm.FunctionDefinition, error) {
	if definition.Kind != llm.ToolKindWebFetch {
		return llm.FunctionDefinition{}, errors.New("web fetch executor received another tool kind")
	}
	return llm.FunctionDefinition{Parameters: json.RawMessage(`{
		"type":"object",
		"properties":{"url":{"type":"string","format":"uri"}},
		"required":["url"],
		"additionalProperties":false
	}`)}, nil
}

func (executor *WebFetchExecutor) Execute(ctx context.Context, definition llm.ToolDefinition, invocation llm.ToolInvocation) (*llm.ToolResult, error) {
	if executor == nil || executor.endpoint == nil {
		return nil, errors.New("web fetch executor is not initialized")
	}
	if definition.Kind != llm.ToolKindWebFetch || invocation.Kind != llm.ToolKindWebFetch {
		return nil, errors.New("web fetch executor received another tool kind")
	}
	input, err := webFetchInput(definition, invocation)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("encode web fetch request: %w", err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, executor.endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create web fetch request: %w", err)
	}
	request.Header = executor.headers.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := executor.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("execute web fetch request: %w", err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, executor.limit+1))
	if err != nil {
		return nil, fmt.Errorf("read web fetch response: %w", err)
	}
	if int64(len(responseBody)) > executor.limit {
		return nil, errors.New("web fetch response exceeds limit")
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("web fetch service returned HTTP %d", response.StatusCode)
	}
	var output WebFetchOutput
	if err := json.Unmarshal(responseBody, &output); err != nil {
		return nil, fmt.Errorf("decode web fetch response: %w", err)
	}
	if strings.TrimSpace(output.URL) == "" {
		output.URL = input.URL
	}
	result := &llm.ToolResult{
		Kind: llm.ToolKindWebFetch, CallID: invocation.CallID, LogicalName: definition.LogicalName,
		Status: llm.ToolResultStatusCompleted,
	}
	if output.Text != "" {
		result.Content = append(result.Content, llm.ContentBlock{Kind: llm.ContentKindText, Text: output.Text})
	}
	if output.URL != "" {
		result.Content = append(result.Content, llm.ContentBlock{
			Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{URL: output.URL, Title: output.Title},
		})
	}
	return result, nil
}

func webFetchInput(definition llm.ToolDefinition, invocation llm.ToolInvocation) (WebFetchInput, error) {
	arguments := invocation.ArgumentsJSON
	if len(arguments) == 0 && invocation.ArgumentsText != "" {
		arguments = json.RawMessage(invocation.ArgumentsText)
	}
	if len(arguments) == 0 || !json.Valid(arguments) {
		return WebFetchInput{}, errors.New("web fetch invocation requires JSON arguments")
	}
	var input WebFetchInput
	if err := json.Unmarshal(arguments, &input); err != nil {
		return WebFetchInput{}, errors.New("decode web fetch arguments")
	}
	input.URL = strings.TrimSpace(input.URL)
	parsed, err := url.Parse(input.URL)
	if err != nil || parsed.Scheme != "http" && parsed.Scheme != "https" || parsed.Host == "" || parsed.User != nil {
		return WebFetchInput{}, errors.New("web fetch invocation requires an absolute HTTP URL")
	}
	if definition.Hosted != nil && definition.Hosted.WebFetch != nil {
		webFetch := definition.Hosted.WebFetch
		input.AllowedDomains = append([]string(nil), webFetch.AllowedDomains...)
		input.BlockedDomains = append([]string(nil), webFetch.BlockedDomains...)
		input.CitationsEnabled = webFetch.CitationsEnabled
		input.MaxContentTokens = webFetch.MaxContentTokens
		input.UseCache = webFetch.UseCache
		input.ResponseInclusion = webFetch.ResponseInclusion
	}
	return input, nil
}

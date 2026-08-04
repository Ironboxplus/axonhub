package hosted

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"testing"

	"github.com/looplj/axonhub/llm"
)

type recordingWebSearchService struct {
	input WebSearchInput
}

type failingWebSearchService struct{}

type staticWebSearchService struct {
	output WebSearchOutput
}

func (service *staticWebSearchService) Search(context.Context, WebSearchInput) (WebSearchOutput, error) {
	return service.output, nil
}

func (*failingWebSearchService) Search(context.Context, WebSearchInput) (WebSearchOutput, error) {
	return WebSearchOutput{}, errors.New("search unavailable")
}

func (service *recordingWebSearchService) Search(_ context.Context, input WebSearchInput) (WebSearchOutput, error) {
	service.input = input
	return WebSearchOutput{
		Text:    "gateway search result",
		Sources: []WebSearchSource{{URL: "https://example.test/result", Title: "result"}},
	}, nil
}

func TestWebSearchExecutorUsesTypedInProcessService(t *testing.T) {
	t.Parallel()
	service := &recordingWebSearchService{}
	executor, err := NewWebSearchExecutor(WebSearchExecutorConfig{Service: service})
	if err != nil {
		t.Fatalf("create typed web-search executor: %v", err)
	}
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{
			AllowedDomains: []string{"example.test"}, BlockedDomains: []string{"private.example.test"},
		}},
	}
	result, err := executor.Execute(context.Background(), definition, llm.ToolInvocation{
		Kind: llm.ToolKindWebSearch, CallID: "search_1", LogicalName: "web_search",
		ArgumentsJSON: json.RawMessage(`{"query":"axon hosted gateway"}`),
		Execution:     llm.ExecutionOwnerProvider,
	})
	if err != nil {
		t.Fatalf("execute typed web-search service: %v", err)
	}
	if service.input.Query != "axon hosted gateway" || len(service.input.AllowedDomains) != 1 ||
		service.input.AllowedDomains[0] != "example.test" || len(service.input.BlockedDomains) != 1 ||
		service.input.BlockedDomains[0] != "private.example.test" {
		t.Fatalf("typed service input = %#v", service.input)
	}
	if result == nil || result.CallID != "search_1" || result.Status != llm.ToolResultStatusCompleted ||
		len(result.Content) != 2 || result.Content[0].Text != "gateway search result" ||
		result.Content[1].Citation == nil || result.Content[1].Citation.URL != "https://example.test/result" {
		t.Fatalf("typed service result = %#v", result)
	}
}

func TestWebSearchTypedServiceRejectsAmbiguousConfigAndPropagatesSafeErrors(t *testing.T) {
	t.Parallel()
	if _, err := NewWebSearchExecutor(WebSearchExecutorConfig{}); err == nil {
		t.Fatal("web-search executor without service or HTTP endpoint was accepted")
	}
	if _, err := NewWebSearchExecutor(WebSearchExecutorConfig{
		Service: &recordingWebSearchService{}, Endpoint: "https://example.test/search",
	}); err == nil {
		t.Fatal("typed service plus HTTP endpoint was accepted")
	}
	if _, err := NewWebSearchExecutor(WebSearchExecutorConfig{
		Service: &recordingWebSearchService{}, Headers: http.Header{"X-Test": []string{"value"}},
	}); err == nil {
		t.Fatal("typed service plus HTTP headers was accepted")
	}
	if _, err := (*WebSearchExecutor)(nil).Execute(context.Background(), llm.ToolDefinition{}, llm.ToolInvocation{}); err == nil {
		t.Fatal("nil executor was accepted")
	}
	executor, err := NewWebSearchExecutor(WebSearchExecutorConfig{Service: &failingWebSearchService{}})
	if err != nil {
		t.Fatalf("create failing typed service: %v", err)
	}
	if executor.Kind() != llm.ToolKindWebSearch {
		t.Fatalf("executor kind = %q", executor.Kind())
	}
	if _, err := executor.Function(llm.ToolDefinition{Kind: llm.ToolKindFunction}); err == nil {
		t.Fatal("web-search executor accepted another tool kind")
	}
	if _, err := executor.Function(llm.ToolDefinition{Kind: llm.ToolKindWebSearch}); err != nil {
		t.Fatalf("web-search function contract: %v", err)
	}
	if _, err := executor.Execute(context.Background(), llm.ToolDefinition{Kind: llm.ToolKindWebSearch}, llm.ToolInvocation{}); err == nil {
		t.Fatal("missing web-search arguments were accepted")
	}
	if _, err := executor.Execute(context.Background(), llm.ToolDefinition{Kind: llm.ToolKindWebSearch}, llm.ToolInvocation{
		ArgumentsJSON: json.RawMessage(`{"query":"axon"}`),
	}); err == nil || err.Error() != "execute web search service: search unavailable" {
		t.Fatalf("typed service error = %v", err)
	}
}

func TestWebSearchToolResultSkipsEmptyOutputAndInvalidSources(t *testing.T) {
	t.Parallel()
	result := webSearchToolResult(llm.ToolDefinition{LogicalName: "web_search"}, llm.ToolInvocation{CallID: "search_2"}, WebSearchOutput{
		Sources: []WebSearchSource{{}, {URL: "https://example.test/kept", Title: "kept"}},
	})
	if result == nil || result.CallID != "search_2" || len(result.Content) != 1 || result.Content[0].Citation == nil ||
		result.Content[0].Citation.URL != "https://example.test/kept" {
		t.Fatalf("filtered web-search result = %#v", result)
	}
}

func TestWebSearchTypedServiceEnforcesBoundedOutput(t *testing.T) {
	t.Parallel()
	definition := llm.ToolDefinition{Kind: llm.ToolKindWebSearch, LogicalName: "web_search"}
	invocation := llm.ToolInvocation{
		Kind: llm.ToolKindWebSearch, CallID: "search_bounded",
		ArgumentsJSON: json.RawMessage(`{"query":"axon"}`),
	}
	for name, output := range map[string]WebSearchOutput{
		"bytes":        {Text: "response exceeds configured bytes"},
		"source_bytes": {Text: "a", Sources: []WebSearchSource{{URL: "12345678"}}},
		"sources":      {Sources: make([]WebSearchSource, maxWebSearchSources+1)},
	} {
		t.Run(name, func(t *testing.T) {
			executor, err := NewWebSearchExecutor(WebSearchExecutorConfig{
				Service: &staticWebSearchService{output: output}, MaxResponseBytes: 8,
			})
			if err != nil {
				t.Fatalf("create bounded typed service: %v", err)
			}
			if _, err := executor.Execute(context.Background(), definition, invocation); err == nil {
				t.Fatalf("unbounded %s output was accepted", name)
			}
		})
	}
	if err := validateWebSearchOutput(WebSearchOutput{Text: "ok"}, 8); err != nil {
		t.Fatalf("bounded output rejected: %v", err)
	}
}

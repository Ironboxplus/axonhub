package hosted

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestWebFetchExecutorOverRealHTTP(t *testing.T) {
	t.Parallel()
	const credential = "fetch-fixture-secret"
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost || request.Header.Get("X-Fetch-Key") != credential {
			http.Error(writer, "unauthorized", http.StatusUnauthorized)
			return
		}
		var input WebFetchInput
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil || input.URL != "https://docs.example.com/axon" ||
			len(input.AllowedDomains) != 1 || input.AllowedDomains[0] != "docs.example.com" ||
			len(input.BlockedDomains) != 1 || input.BlockedDomains[0] != "private.example.com" ||
			input.CitationsEnabled == nil || !*input.CitationsEnabled || input.MaxContentTokens == nil || *input.MaxContentTokens != 4096 ||
			input.UseCache == nil || *input.UseCache || input.ResponseInclusion != "all" {
			http.Error(writer, "invalid input", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"url":"https://docs.example.com/axon","title":"Axon docs","retrieved_at":"2026-07-30T00:00:00Z","text":"canonical body"}`)
	}))
	t.Cleanup(server.Close)
	allowed, err := url.Parse(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewWebFetchExecutor(WebFetchExecutorConfig{
		Endpoint: server.URL, Client: server.Client(), Headers: http.Header{"X-Fetch-Key": []string{credential}},
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowed.Scheme || candidate.Host != allowed.Host {
				return fmt.Errorf("endpoint is not allowlisted")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("create web fetch executor: %v", err)
	}
	citationsEnabled, maxContentTokens, useCache := true, int64(4096), false
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindWebFetch, LogicalName: "web_fetch",
		Hosted: &llm.HostedToolDefinition{Type: "web_fetch_20260318", WebFetch: &llm.WebFetch{
			AllowedDomains: []string{"docs.example.com"}, BlockedDomains: []string{"private.example.com"},
			CitationsEnabled: &citationsEnabled, MaxContentTokens: &maxContentTokens, UseCache: &useCache,
			ResponseInclusion: "all",
		}}, Execution: llm.ExecutionOwnerProvider,
	}
	function, err := executor.Function(definition)
	if err != nil {
		t.Fatalf("lower web fetch definition: %v", err)
	}
	if !json.Valid(function.Parameters) {
		t.Fatalf("web fetch function schema is invalid: %s", function.Parameters)
	}
	result, err := executor.Execute(context.Background(), definition, llm.ToolInvocation{
		Kind: llm.ToolKindWebFetch, CallID: "fetch_1", LogicalName: "web_fetch",
		ArgumentsJSON: json.RawMessage(`{"url":"https://docs.example.com/axon"}`),
	})
	if err != nil {
		t.Fatalf("execute web fetch: %v", err)
	}
	if result.Kind != llm.ToolKindWebFetch || result.CallID != "fetch_1" || result.Status != llm.ToolResultStatusCompleted {
		t.Fatalf("web fetch result identity degraded: %#v", result)
	}
	if len(result.Content) != 2 || result.Content[0].Kind != llm.ContentKindText || result.Content[0].Text != "canonical body" ||
		result.Content[1].Kind != llm.ContentKindCitation || result.Content[1].Citation == nil ||
		result.Content[1].Citation.URL != "https://docs.example.com/axon" || result.Content[1].Citation.Title != "Axon docs" {
		t.Fatalf("web fetch result content degraded: %#v", result.Content)
	}
}

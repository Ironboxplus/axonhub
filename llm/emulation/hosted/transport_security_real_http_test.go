package hosted

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestHostedExecutorsRecheckEndpointPolicyBeforeRedirect(t *testing.T) {
	t.Parallel()

	var blockedHits atomic.Int64
	blocked := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		blockedHits.Add(1)
	}))
	t.Cleanup(blocked.Close)
	allowed := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		http.Redirect(writer, request, blocked.URL, http.StatusTemporaryRedirect)
	}))
	t.Cleanup(allowed.Close)
	allowedURL, err := url.Parse(allowed.URL)
	if err != nil {
		t.Fatal(err)
	}
	policy := func(candidate *url.URL) error {
		if candidate.Scheme != allowedURL.Scheme || candidate.Host != allowedURL.Host {
			return errors.New("endpoint is not allowlisted")
		}
		return nil
	}

	tests := []struct {
		name    string
		execute func(context.Context) error
	}{
		{
			name: "generic hosted executor",
			execute: func(ctx context.Context) error {
				executor, createErr := NewHTTPExecutor(HTTPExecutorConfig{
					Kind: llm.ToolKindShell, Endpoint: allowed.URL, Client: allowed.Client(),
					Headers: http.Header{"X-Hosted-Secret": []string{"must-not-leak"}}, EndpointPolicy: policy,
				})
				if createErr != nil {
					return createErr
				}
				definition := llm.ToolDefinition{
					Kind: llm.ToolKindShell, LogicalName: "shell", Execution: llm.ExecutionOwnerProvider,
					Hosted: &llm.HostedToolDefinition{Type: "shell", Configuration: json.RawMessage(`{"type":"shell"}`)},
				}
				_, executeErr := executor.Execute(ctx, definition, llm.ToolInvocation{
					Kind: llm.ToolKindShell, CallID: "call_1", LogicalName: "shell",
					ArgumentsJSON: json.RawMessage(`{"commands":["pwd"]}`),
				})
				return executeErr
			},
		},
		{
			name: "web fetch executor",
			execute: func(ctx context.Context) error {
				executor, createErr := NewWebFetchExecutor(WebFetchExecutorConfig{
					Endpoint: allowed.URL, Client: allowed.Client(),
					Headers: http.Header{"X-Hosted-Secret": []string{"must-not-leak"}}, EndpointPolicy: policy,
				})
				if createErr != nil {
					return createErr
				}
				definition := llm.ToolDefinition{Kind: llm.ToolKindWebFetch, LogicalName: "web_fetch"}
				_, executeErr := executor.Execute(ctx, definition, llm.ToolInvocation{
					Kind: llm.ToolKindWebFetch, CallID: "call_1", LogicalName: "web_fetch",
					ArgumentsJSON: json.RawMessage(`{"url":"https://docs.example.com/axon"}`),
				})
				return executeErr
			},
		},
		{
			name: "web search executor",
			execute: func(ctx context.Context) error {
				executor, createErr := NewWebSearchExecutor(WebSearchExecutorConfig{
					Endpoint: allowed.URL, Client: allowed.Client(),
					Headers: http.Header{"X-Hosted-Secret": []string{"must-not-leak"}}, EndpointPolicy: policy,
				})
				if createErr != nil {
					return createErr
				}
				definition := llm.ToolDefinition{Kind: llm.ToolKindWebSearch, LogicalName: "web_search"}
				_, executeErr := executor.Execute(ctx, definition, llm.ToolInvocation{
					Kind: llm.ToolKindWebSearch, CallID: "call_1", LogicalName: "web_search",
					ArgumentsJSON: json.RawMessage(`{"query":"Axon"}`),
				})
				return executeErr
			},
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			err := test.execute(context.Background())
			if err == nil || !strings.Contains(err.Error(), "redirect rejected") {
				t.Fatalf("redirect policy was not enforced: %v", err)
			}
		})
	}
	if blockedHits.Load() != 0 {
		t.Fatalf("blocked redirect target received %d requests", blockedHits.Load())
	}
}

func TestWebFetchEnforcesDomainFiltersBeforeServiceCall(t *testing.T) {
	t.Parallel()

	var serviceHits atomic.Int64
	service := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		serviceHits.Add(1)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"text":"ok"}`))
	}))
	t.Cleanup(service.Close)
	executor, err := NewWebFetchExecutor(WebFetchExecutorConfig{
		Endpoint: service.URL, Client: service.Client(), EndpointPolicy: func(*url.URL) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		target  string
		allowed []string
		blocked []string
		wantErr bool
	}{
		{name: "allowed exact", target: "https://docs.example.com/axon", allowed: []string{"docs.example.com"}},
		{name: "allowed subdomain", target: "https://api.docs.example.com/axon", allowed: []string{"docs.example.com"}},
		{name: "allowed and unrelated blocked", target: "https://docs.example.com/axon", allowed: []string{"docs.example.com"}, blocked: []string{"private.example.com"}},
		{name: "lookalike is not subdomain", target: "https://docs.example.com.evil.invalid/axon", allowed: []string{"docs.example.com"}, wantErr: true},
		{name: "blocked exact", target: "https://private.example.com/axon", blocked: []string{"private.example.com"}, wantErr: true},
		{name: "blocked subdomain", target: "https://api.private.example.com/axon", blocked: []string{"private.example.com"}, wantErr: true},
		{name: "wildcard rule rejected", target: "https://docs.example.com/axon", allowed: []string{"*.example.com"}, wantErr: true},
		{name: "path rule rejected", target: "https://docs.example.com/axon", allowed: []string{"docs.example.com/axon"}, wantErr: true},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			before := serviceHits.Load()
			definition := llm.ToolDefinition{
				Kind: llm.ToolKindWebFetch, LogicalName: "web_fetch", Execution: llm.ExecutionOwnerProvider,
				Hosted: &llm.HostedToolDefinition{WebFetch: &llm.WebFetch{
					AllowedDomains: test.allowed, BlockedDomains: test.blocked,
				}},
			}
			_, executeErr := executor.Execute(context.Background(), definition, llm.ToolInvocation{
				Kind: llm.ToolKindWebFetch, CallID: "fetch_1", LogicalName: "web_fetch",
				ArgumentsJSON: json.RawMessage(`{"url":` + string(mustJSON(t, test.target)) + `}`),
			})
			if test.wantErr {
				if executeErr == nil {
					t.Fatal("expected domain filter error")
				}
				if serviceHits.Load() != before {
					t.Fatal("rejected target reached the web fetch service")
				}
				return
			}
			if executeErr != nil {
				t.Fatalf("allowed target failed: %v", executeErr)
			}
			if serviceHits.Load() != before+1 {
				t.Fatal("allowed target did not reach the web fetch service")
			}
		})
	}
}

func TestWebSearchForwardsBlockedDomainContractOverRealHTTP(t *testing.T) {
	t.Parallel()

	service := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var input WebSearchInput
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
			http.Error(writer, "invalid request", http.StatusBadRequest)
			return
		}
		if len(input.AllowedDomains) != 1 || input.AllowedDomains[0] != "docs.example.com" ||
			len(input.BlockedDomains) != 1 || input.BlockedDomains[0] != "private.example.com" {
			http.Error(writer, "domain filters lost", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"text":"ok"}`))
	}))
	t.Cleanup(service.Close)
	executor, err := NewWebSearchExecutor(WebSearchExecutorConfig{
		Endpoint: service.URL, Client: service.Client(), EndpointPolicy: func(*url.URL) error { return nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{WebSearch: &llm.WebSearch{
			AllowedDomains: []string{"docs.example.com"}, BlockedDomains: []string{"private.example.com"},
		}},
	}
	if _, err := executor.Execute(context.Background(), definition, llm.ToolInvocation{
		Kind: llm.ToolKindWebSearch, CallID: "search_1", LogicalName: "web_search",
		ArgumentsJSON: json.RawMessage(`{"query":"Axon"}`),
	}); err != nil {
		t.Fatalf("web search domain contract failed: %v", err)
	}
}

func mustJSON(t *testing.T, value string) []byte {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return encoded
}

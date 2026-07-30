package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestRegistryDiscoversFiltersBindsAndCallsOverRealHTTP(t *testing.T) {
	t.Parallel()
	const sessionID = "registry-session"
	const authorization = "Bearer registry-private-token"
	var methodsMu sync.Mutex
	var methods []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != authorization {
			http.Error(w, "missing authorization", http.StatusUnauthorized)
			return
		}
		if r.Method == http.MethodDelete {
			methodsMu.Lock()
			methods = append(methods, "session/delete")
			methodsMu.Unlock()
			w.WriteHeader(http.StatusNoContent)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var envelope rpcEnvelope
		if json.Unmarshal(body, &envelope) != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		methodsMu.Lock()
		methods = append(methods, envelope.Method)
		methodsMu.Unlock()
		switch envelope.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Mcp-Session-Id", sessionID)
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"registry-fixture","version":"1"}}}`, envelope.ID, ProtocolVersion)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"lookup","description":"read inventory","inputSchema":{"type":"object","properties":{"sku":{"type":"string"}}},"annotations":{"readOnlyHint":true}},{"name":"reserve","description":"write inventory","inputSchema":{"type":"object"},"annotations":{"readOnlyHint":false}}]}}`, envelope.ID)
		case "tools/call":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"7"}],"structuredContent":{"available":7}}}`, envelope.ID)
		default:
			http.Error(w, "unexpected method", http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)

	allowed, err := url.Parse(server.URL)
	require.NoError(t, err)
	readOnly := true
	request := &llm.Request{
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{
				ServerLabel: "inventory", ServerURL: server.URL,
				AllowedTools:    &llm.MCPToolFilter{ReadOnly: &readOnly},
				RequireApproval: &llm.MCPApprovalPolicy{Mode: llm.MCPApprovalNever},
			},
		}},
		ToolExecutionSecrets: &llm.ToolExecutionSecrets{MCP: map[string]llm.MCPConnectionSecrets{
			"inventory": {Authorization: authorization},
		}},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	registry, err := DiscoverRegistry(ctx, request, RegistryConfig{
		SyntheticNameKey: []byte(strings.Repeat("k", 32)),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowed.Scheme || candidate.Host != allowed.Host {
				return fmt.Errorf("not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)
	defer func() { require.NoError(t, registry.Close(ctx)) }()

	definitions := registry.FunctionDefinitions()
	require.Len(t, definitions, 1)
	require.Equal(t, llm.ExecutionOwnerGateway, definitions[0].Execution)
	require.True(t, strings.HasPrefix(definitions[0].LogicalName, syntheticNamePrefix))
	require.NotContains(t, definitions[0].LogicalName, "inventory")
	require.NotContains(t, definitions[0].LogicalName, "lookup")
	require.JSONEq(t, `{"type":"object","properties":{"sku":{"type":"string"}}}`, string(definitions[0].Function.Parameters))

	binding, ok := registry.Binding(definitions[0].LogicalName)
	require.True(t, ok)
	require.Equal(t, "inventory", binding.ServerLabel)
	require.Equal(t, "lookup", binding.Tool.Name)
	require.False(t, binding.ApprovalRequired(true))
	require.Len(t, registry.ListItems(), 1)
	require.Len(t, registry.ListItems()[0].MCPListTools.Tools, 1)

	result, err := registry.Call(ctx, binding, json.RawMessage(`{"sku":"A-1"}`))
	require.NoError(t, err)
	require.Len(t, result.Content, 1)
	require.Equal(t, "7", result.Content[0].Text)
	require.JSONEq(t, `{"available":7}`, string(result.StructuredContent))

	require.NoError(t, registry.Close(ctx))
	methodsMu.Lock()
	require.Equal(t, []string{"initialize", "notifications/initialized", "tools/list", "tools/call", "session/delete"}, methods)
	methodsMu.Unlock()
}

func TestSyntheticRegistryNamesAreStableKeyedAndOpaque(t *testing.T) {
	key := []byte(strings.Repeat("s", 32))
	generated := syntheticToolName(key, "inventory", "lookup")
	require.True(t, strings.HasPrefix(generated, syntheticNamePrefix))
	require.NotEqual(t, syntheticToolName(key, "inventory", "reserve"), generated)
	require.NotEqual(t, syntheticToolName([]byte(strings.Repeat("t", 32)), "inventory", "lookup"), generated)
}

func TestRegistrySelectsEndpointPolicyFromDefinitionProvenanceOverRealHTTP(t *testing.T) {
	t.Parallel()
	var requestsMu sync.Mutex
	var requests int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestsMu.Lock()
		requests++
		requestsMu.Unlock()
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var envelope rpcEnvelope
		if json.Unmarshal(body, &envelope) != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		switch envelope.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"connector-fixture","version":"1"}}}`, envelope.ID, ProtocolVersion)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"tools":[]}}`, envelope.ID)
		default:
			http.Error(w, "unexpected method", http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)

	connectorRequest := &llm.Request{ToolDefinitions: []llm.ToolDefinition{{
		Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
		MCP: &llm.MCPDefinition{ServerLabel: "inventory", ConnectorID: "deployment-inventory"},
	}}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	config := RegistryConfig{
		SyntheticNameKey: []byte(strings.Repeat("p", 32)),
		EndpointResolver: func(_ context.Context, definition llm.MCPDefinition) (string, error) {
			if definition.ConnectorID != "deployment-inventory" {
				return "", fmt.Errorf("connector unavailable")
			}
			return server.URL, nil
		},
		EndpointPolicyFactory: func(definition llm.MCPDefinition, endpoint string) (EndpointPolicy, error) {
			if definition.ConnectorID != "" {
				return func(candidate *url.URL) error {
					if candidate.String() != endpoint {
						return fmt.Errorf("connector endpoint mismatch")
					}
					return nil
				}, nil
			}
			return func(*url.URL) error { return fmt.Errorf("direct private endpoint denied") }, nil
		},
	}
	registry, err := DiscoverRegistry(ctx, connectorRequest, config)
	require.NoError(t, err)
	require.NoError(t, registry.Close(ctx))

	requestsMu.Lock()
	connectorRequests := requests
	requestsMu.Unlock()
	require.Equal(t, 3, connectorRequests, "connector must use the real initialize/initialized/list exchange")

	directRequest := connectorRequest.Clone()
	directRequest.ToolDefinitions[0].MCP.ConnectorID = ""
	directRequest.ToolDefinitions[0].MCP.ServerURL = server.URL
	_, err = DiscoverRegistry(ctx, directRequest, config)
	require.ErrorContains(t, err, "direct private endpoint denied")
	requestsMu.Lock()
	require.Equal(t, connectorRequests, requests, "direct URL must be rejected before it reaches the trusted connector host")
	requestsMu.Unlock()
}

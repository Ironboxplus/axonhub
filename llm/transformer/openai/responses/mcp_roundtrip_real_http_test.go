package responses_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestResponsesMCPDefinitionAndSecretsRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	const mcpAuthorization = "Bearer private-mcp-token"
	const mcpHeaderSecret = "private-header-value"
	requestBody := []byte(`{
		"model":"fixture-model",
		"input":"use the remote tools",
		"tools":[
			{"type":"function","name":"before","parameters":{"type":"object"}},
			{
				"type":"mcp",
				"server_label":"inventory",
				"server_description":"Inventory service",
				"server_url":"https://mcp.example.invalid/rpc",
				"authorization":"Bearer private-mcp-token",
				"headers":{"X-MCP-Secret":"private-header-value"},
				"defer_loading":true,
				"allowed_callers":["direct"],
				"allowed_tools":["lookup","reserve"],
				"require_approval":"never"
			},
			{"type":"custom","name":"after"}
		]
	}`)

	inbound := responses.NewInboundTransformer()
	decoded, err := inbound.TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
	})
	require.NoError(t, err)
	require.Len(t, decoded.ToolDefinitions, 3)
	require.NotNil(t, decoded.ToolDefinitions[1].MCP)
	require.Equal(t, "inventory", decoded.ToolDefinitions[1].MCP.ServerLabel)
	require.NotNil(t, decoded.ToolExecutionSecrets)
	require.Equal(t, mcpAuthorization, decoded.ToolExecutionSecrets.MCP["inventory"].Authorization)
	require.Equal(t, mcpHeaderSecret, decoded.ToolExecutionSecrets.MCP["inventory"].Headers["X-MCP-Secret"])

	serialized, err := json.Marshal(decoded)
	require.NoError(t, err)
	require.NotContains(t, string(serialized), mcpAuthorization)
	require.NotContains(t, string(serialized), mcpHeaderSecret)
	if decoded.ProviderExtensions != nil && decoded.ProviderExtensions.OpenAIResponses != nil &&
		decoded.ProviderExtensions.OpenAIResponses.Request != nil {
		for _, fragment := range decoded.ProviderExtensions.OpenAIResponses.Request.RawTools {
			require.NotEqual(t, "mcp", fragment.Type, "typed MCP must not survive in the raw secret-bearing sidecar")
		}
	}

	clone := decoded.Clone()
	cloneSecret := clone.ToolExecutionSecrets.MCP["inventory"]
	cloneSecret.Headers["X-MCP-Secret"] = "mutated-clone"
	clone.ToolExecutionSecrets.MCP["inventory"] = cloneSecret
	require.Equal(t, mcpHeaderSecret, decoded.ToolExecutionSecrets.MCP["inventory"].Headers["X-MCP-Secret"])

	received := make(chan []byte, 1)
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read provider request", http.StatusBadRequest)
			return
		}
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_mcp_native","object":"response","created_at":1785380000,
			"model":"fixture-model","status":"completed",
			"output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],
			"usage":{"input_tokens":8,"output_tokens":1,"total_tokens":9}
		}`)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).
		Pipeline(inbound, conversion.NewOutbound(outbound)).
		Process(ctx, &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Response)
	require.Contains(t, string(result.Response.Body), "resp_mcp_native")

	var providerRequest struct {
		Tools []struct {
			Type              string            `json:"type"`
			Name              string            `json:"name"`
			ServerLabel       string            `json:"server_label"`
			Authorization     string            `json:"authorization"`
			Headers           map[string]string `json:"headers"`
			AllowedTools      json.RawMessage   `json:"allowed_tools"`
			RequireApproval   json.RawMessage   `json:"require_approval"`
			ServerURL         string            `json:"server_url"`
			ServerDescription string            `json:"server_description"`
			AllowedCallers    []string          `json:"allowed_callers"`
			DeferLoading      *bool             `json:"defer_loading"`
		} `json:"tools"`
	}
	select {
	case body := <-received:
		require.NoError(t, json.Unmarshal(body, &providerRequest), "provider body: %s", body)
	case <-ctx.Done():
		t.Fatal("provider did not receive request")
	}
	require.Len(t, providerRequest.Tools, 3)
	require.Equal(t, []string{"function", "mcp", "custom"}, []string{
		providerRequest.Tools[0].Type, providerRequest.Tools[1].Type, providerRequest.Tools[2].Type,
	})
	mcp := providerRequest.Tools[1]
	require.Equal(t, "inventory", mcp.ServerLabel)
	require.Equal(t, "https://mcp.example.invalid/rpc", mcp.ServerURL)
	require.Equal(t, mcpAuthorization, mcp.Authorization)
	require.Equal(t, mcpHeaderSecret, mcp.Headers["X-MCP-Secret"])
	require.NotNil(t, mcp.DeferLoading)
	require.True(t, *mcp.DeferLoading)
	require.Equal(t, []string{"direct"}, mcp.AllowedCallers)
	require.JSONEq(t, `["lookup","reserve"]`, string(mcp.AllowedTools))
	require.JSONEq(t, `"never"`, string(mcp.RequireApproval))
}

func TestResponsesMCPReadOnlyFilterRequiresGatewayBeforeProviderHTTP(t *testing.T) {
	t.Parallel()
	inbound := responses.NewInboundTransformer()
	decoded, err := inbound.TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","input":"inspect inventory",
			"tools":[{
				"type":"mcp","server_label":"inventory","server_url":"https://mcp.example.invalid/rpc",
				"allowed_tools":{"tool_names":["lookup"],"read_only":true},"require_approval":"never"
			}]
		}`),
	})
	require.NoError(t, err)

	outbound, err := responses.NewOutboundTransformer("https://example.invalid", "fixture-key")
	require.NoError(t, err)
	_, err = outbound.TransformRequest(context.Background(), decoded)
	require.ErrorContains(t, err, "read_only requires MCP gateway projection")
}

package responses_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type canonicalMutationInbound struct {
	transformer.Inbound
}

func (inbound canonicalMutationInbound) TransformRequest(ctx context.Context, request *httpclient.Request) (*llm.Request, error) {
	canonical, err := inbound.Inbound.TransformRequest(ctx, request)
	if err != nil {
		return nil, err
	}
	for index := range canonical.ToolDefinitions {
		definition := &canonical.ToolDefinitions[index]
		if definition.LogicalName != "collaboration__send_message" || definition.Function == nil {
			continue
		}
		definition.Description = "Send a policy-reviewed message"
		definition.Function.Parameters = json.RawMessage(`{"type":"object","properties":{"recipient":{"type":"string"}},"required":["recipient"],"additionalProperties":false}`)
		strict := true
		definition.Function.Strict = &strict
	}
	return canonical, nil
}

type canonicalToolRemovalInbound struct {
	transformer.Inbound
}

func (inbound canonicalToolRemovalInbound) TransformRequest(ctx context.Context, request *httpclient.Request) (*llm.Request, error) {
	canonical, err := inbound.Inbound.TransformRequest(ctx, request)
	if err != nil {
		return nil, err
	}
	canonical.Tools = nil
	canonical.ToolDefinitions = nil
	return canonical, nil
}

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

func TestResponsesMixedNamespaceIdentityPreservesRawDefinitionOverRealHTTP(t *testing.T) {
	t.Parallel()
	requestBody := []byte(`{
		"model":"fixture-model",
		"input":"use the namespace",
		"tools":[
			{
				"type":"namespace",
				"name":"collaboration",
				"tools":[
					{"type":"function","name":"send_message","description":"Send a message","parameters":{"type":"object","properties":{"target":{"type":"string"}}}},
					{"type":"custom","name":"apply_patch","description":"Apply a patch","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/s"}},
					{"type":"future_tool","name":"handoff","future_field":{"enabled":true}}
				]
			},
			{"type":"function","name":"after","parameters":{"type":"object"}}
		]
	}`)

	received := make(chan []byte, 1)
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read provider request", http.StatusBadRequest)
			return
		}
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_namespace_identity","object":"response","created_at":1785380000,
			"model":"fixture-model","status":"completed",
			"output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],
			"usage":{"input_tokens":8,"output_tokens":1,"total_tokens":9}
		}`)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(ctx, &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Response)
	require.Contains(t, string(result.Response.Body), "resp_namespace_identity")

	var source, wire struct {
		Tools []json.RawMessage `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(requestBody, &source))
	select {
	case body := <-received:
		require.NoError(t, json.Unmarshal(body, &wire), "provider body: %s", body)
	case <-ctx.Done():
		t.Fatal("provider did not receive identity request")
	}
	require.Len(t, source.Tools, 2)
	require.Len(t, wire.Tools, 2)
	require.JSONEq(t, string(source.Tools[0]), string(wire.Tools[0]))
	require.JSONEq(t, string(source.Tools[1]), string(wire.Tools[1]))
}

func TestResponsesMixedNamespaceCanonicalChildMutationWinsOverRawSidecarOverRealHTTP(t *testing.T) {
	t.Parallel()
	requestBody := []byte(`{
		"model":"fixture-model",
		"input":"use the namespace",
		"tools":[{
			"type":"namespace",
			"name":"collaboration",
			"future_namespace_field":{"mode":"opaque"},
			"tools":[
				{"type":"function","name":"send_message","description":"Send a message","parameters":{"type":"object","properties":{"target":{"type":"string"}}}},
				{"type":"future_tool","name":"handoff","future_field":{"enabled":true}}
			]
		}]
	}`)

	received := make(chan []byte, 1)
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		received <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_namespace_mutation","object":"response","created_at":1785380000,
			"model":"fixture-model","status":"completed",
			"output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],
			"usage":{"input_tokens":8,"output_tokens":1,"total_tokens":9}
		}`)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).
		Pipeline(
			canonicalMutationInbound{Inbound: responses.NewInboundTransformer()},
			conversion.NewOutbound(outbound),
		).
		Process(ctx, &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Response)

	var wire struct {
		Tools []json.RawMessage `json:"tools"`
	}
	select {
	case body := <-received:
		require.NoError(t, json.Unmarshal(body, &wire), "provider body: %s", body)
	case <-ctx.Done():
		t.Fatal("provider did not receive canonical child mutation request")
	}
	require.Len(t, wire.Tools, 1)
	var namespace struct {
		Type                 string            `json:"type"`
		Name                 string            `json:"name"`
		Tools                []json.RawMessage `json:"tools"`
		FutureNamespaceField struct {
			Mode string `json:"mode"`
		} `json:"future_namespace_field"`
	}
	require.NoError(t, json.Unmarshal(wire.Tools[0], &namespace))
	require.Equal(t, "namespace", namespace.Type)
	require.Equal(t, "collaboration", namespace.Name)
	require.Equal(t, "opaque", namespace.FutureNamespaceField.Mode)
	require.Len(t, namespace.Tools, 2, "canonical mutation must not discard the opaque unknown child")

	var child responses.Tool
	require.NoError(t, json.Unmarshal(namespace.Tools[0], &child))
	require.Equal(t, "function", child.Type)
	require.Equal(t, "send_message", child.Name)
	require.Equal(t, "Send a policy-reviewed message", child.Description)
	require.NotNil(t, child.Strict)
	require.True(t, *child.Strict)
	properties, ok := child.Parameters["properties"].(map[string]any)
	require.True(t, ok)
	require.Contains(t, properties, "recipient")
	require.NotContains(t, properties, "target")

	var unknown struct {
		Type        string `json:"type"`
		Name        string `json:"name"`
		FutureField struct {
			Enabled bool `json:"enabled"`
		} `json:"future_field"`
	}
	require.NoError(t, json.Unmarshal(namespace.Tools[1], &unknown))
	require.Equal(t, "future_tool", unknown.Type)
	require.Equal(t, "handoff", unknown.Name)
	require.True(t, unknown.FutureField.Enabled)
}

func TestResponsesMixedNamespaceCrossProtocolRejectsBeforeProviderHTTP(t *testing.T) {
	t.Parallel()
	var providerCalls atomic.Uint64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		providerCalls.Add(1)
		http.Error(w, "cross-protocol request must not be dispatched", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)

	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)

	_, err = pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{
				"model":"fixture-model","input":"use the namespace",
				"tools":[{"type":"namespace","name":"collaboration","tools":[
					{"type":"function","name":"send_message","parameters":{"type":"object"}},
					{"type":"future_tool","name":"handoff","future_field":{"enabled":true}}
				]}]
			}`),
		})
	require.Error(t, err)
	require.True(t, errors.Is(err, conversion.ErrIncompletePlan), "error = %v", err)
	require.Zero(t, providerCalls.Load(), "incomplete cross-protocol plan reached provider")
}

func TestResponsesMixedNamespaceToChatRejectsBeforeProviderHTTP(t *testing.T) {
	t.Parallel()
	var providerCalls atomic.Uint64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		providerCalls.Add(1)
		http.Error(w, "cross-protocol request must not be dispatched", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)

	outbound, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)

	_, err = pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{
				"model":"fixture-model","input":"use the namespace",
				"tools":[{"type":"namespace","name":"collaboration","tools":[
					{"type":"function","name":"send_message","parameters":{"type":"object"}},
					{"type":"future_tool","name":"handoff","future_field":{"enabled":true}}
				]}]
			}`),
		})
	require.Error(t, err)
	require.True(t, errors.Is(err, conversion.ErrIncompletePlan), "error = %v", err)
	require.Zero(t, providerCalls.Load(), "incomplete Responses-to-Chat plan reached provider")
}

func TestResponsesDeletedOpaqueToolIsNotReplayedOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerCalls atomic.Uint64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		providerCalls.Add(1)
		body, err := io.ReadAll(request.Body)
		require.NoError(t, err)
		var payload struct {
			Tools []json.RawMessage `json:"tools"`
		}
		require.NoError(t, json.Unmarshal(body, &payload))
		require.Empty(t, payload.Tools, "deleted canonical tools were replayed from stale raw state")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"resp_deleted_tool","object":"response","created_at":1,"model":"fixture-model","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)

	result, err := pipeline.NewFactory(executor).
		Pipeline(
			canonicalToolRemovalInbound{Inbound: responses.NewInboundTransformer()},
			conversion.NewOutbound(outbound),
		).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{
				"model":"fixture-model","input":"use the namespace",
				"tools":[{"type":"namespace","name":"collaboration","tools":[
					{"type":"function","name":"send_message","parameters":{"type":"object"}},
					{"type":"future_tool","name":"handoff","future_field":{"enabled":true}}
				]}]
			}`),
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.EqualValues(t, 1, providerCalls.Load())
}

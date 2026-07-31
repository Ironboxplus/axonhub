package emulation_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/emulation/mcp"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestControllerRunsResponsesMCPThroughChatTargetOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer private-mcp-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
	var providerIssueMu sync.Mutex
	var providerIssues []string
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
				Content    any    `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Type != "function" {
			providerIssueMu.Lock()
			providerIssues = append(providerIssues, fmt.Sprintf("round %d tools: %s", round, body))
			providerIssueMu.Unlock()
			http.Error(w, "tools", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Function.Name
			if !strings.HasPrefix(syntheticName, "axon_mcp_") || strings.Contains(syntheticName, "inventory") || strings.Contains(syntheticName, "lookup") {
				providerIssueMu.Lock()
				providerIssues = append(providerIssues, "synthetic name is not opaque")
				providerIssueMu.Unlock()
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-round-1","object":"chat.completion","created":1785380000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"checking inventory","tool_calls":[{"id":"provider_call_1","type":"function","function":{"name":%q,"arguments":"{\"sku\":\"A-1\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":1}}}`, syntheticName)
			return
		}
		foundResult := false
		for _, message := range payload.Messages {
			if message.Role == "tool" && message.ToolCallID == "provider_call_1" {
				encoded, _ := json.Marshal(message.Content)
				foundResult = strings.Contains(string(encoded), "available") && strings.Contains(string(encoded), "7")
			}
		}
		if !foundResult {
			providerIssueMu.Lock()
			providerIssues = append(providerIssues, "second round has no MCP result")
			providerIssueMu.Unlock()
			http.Error(w, "missing result", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-round-2","object":"chat.completion","created":1785380001,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"7 units available"},"finish_reason":"stop"}],"usage":{"prompt_tokens":12,"completion_tokens":4,"total_tokens":16,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("controller-key-", 3)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)

	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"check stock",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never"}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.NotNil(t, result.Response)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	providerIssueMu.Lock()
	require.Empty(t, providerIssues)
	providerIssueMu.Unlock()

	wire := string(result.Response.Body)
	require.NotContains(t, wire, syntheticName)
	require.NotContains(t, wire, authorization)
	var responseBody struct {
		Status string `json:"status"`
		Output []struct {
			Type        string `json:"type"`
			ServerLabel string `json:"server_label"`
			Name        string `json:"name"`
			Output      string `json:"output"`
			Content     []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			TotalTokens       int64 `json:"total_tokens"`
			InputTokenDetails struct {
				CachedTokens int64 `json:"cached_tokens"`
			} `json:"input_tokens_details"`
			OutputTokenDetails struct {
				ReasoningTokens int64 `json:"reasoning_tokens"`
			} `json:"output_tokens_details"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Equal(t, "completed", responseBody.Status)
	require.Len(t, responseBody.Output, 4)
	require.Equal(t, "mcp_list_tools", responseBody.Output[0].Type)
	require.Equal(t, "message", responseBody.Output[1].Type)
	require.Equal(t, "checking inventory", responseBody.Output[1].Content[0].Text)
	require.Equal(t, "mcp_call", responseBody.Output[2].Type)
	require.Equal(t, "inventory", responseBody.Output[2].ServerLabel)
	require.Equal(t, "lookup", responseBody.Output[2].Name)
	require.Contains(t, responseBody.Output[2].Output, `"available":7`)
	require.Equal(t, "message", responseBody.Output[3].Type)
	require.Equal(t, "7 units available", responseBody.Output[3].Content[0].Text)
	require.EqualValues(t, 29, responseBody.Usage.TotalTokens)
	require.EqualValues(t, 5, responseBody.Usage.InputTokenDetails.CachedTokens)
	require.EqualValues(t, 3, responseBody.Usage.OutputTokenDetails.ReasoningTokens)

	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 2, emulationObservation.InternalRounds)
	require.EqualValues(t, 1, emulationObservation.ToolCalls)
}

func TestControllerRunsPortableResponsesMCPThroughForcedGatewayOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer responses-portable-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
	var providerIssues []string
	var providerIssuesMu sync.Mutex
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tools"`
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Type != "function" ||
			strings.Contains(string(body), `"type":"mcp"`) || strings.Contains(string(body), `"read_only"`) {
			providerIssuesMu.Lock()
			providerIssues = append(providerIssues, fmt.Sprintf("round %d provider body was not a lowered Responses request: %s", round, body))
			providerIssuesMu.Unlock()
			http.Error(writer, "unexpected request", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Name
			if !strings.HasPrefix(syntheticName, "axon_mcp_") {
				providerIssuesMu.Lock()
				providerIssues = append(providerIssues, "first round did not use an opaque MCP function")
				providerIssuesMu.Unlock()
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(writer, `{"id":"resp_readonly_1","object":"response","created_at":1785382000,"model":"fixture-model","status":"completed","output":[{"id":"item_call_1","type":"function_call","call_id":"provider_call_1","name":%q,"arguments":"{\"sku\":\"A-1\"}","status":"completed"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`, syntheticName)
			return
		}
		var inputItems []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Output string `json:"output"`
		}
		foundResult := json.Unmarshal(payload.Input, &inputItems) == nil
		if foundResult {
			foundResult = false
			for _, item := range inputItems {
				if item.Type == "function_call_output" && item.CallID == "provider_call_1" && strings.Contains(item.Output, `"available":7`) {
					foundResult = true
					break
				}
			}
		}
		if round != 2 || !foundResult {
			providerIssuesMu.Lock()
			providerIssues = append(providerIssues, fmt.Sprintf("round %d missing lowered MCP result: %s", round, body))
			providerIssuesMu.Unlock()
			http.Error(writer, "missing function result", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_readonly_2","object":"response","created_at":1785382001,"model":"fixture-model","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"7 units available","annotations":[]}]}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("responses-portable-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		ForceMCPGateway: true,
		MaxRounds:       4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"check inventory",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,
			"allowed_tools":["lookup"],"require_approval":"never"}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.NotNil(t, result.Response)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	providerIssuesMu.Lock()
	require.Empty(t, providerIssues)
	providerIssuesMu.Unlock()

	wire := string(result.Response.Body)
	require.Contains(t, wire, `"type":"mcp_list_tools"`)
	require.Contains(t, wire, `"type":"mcp_call"`)
	require.Contains(t, wire, `"server_label":"inventory"`)
	require.Contains(t, wire, "7 units available")
	require.NotContains(t, wire, authorization)
	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 2, emulationObservation.InternalRounds)
	require.EqualValues(t, 1, emulationObservation.ToolCalls)
}

// TestControllerRunsAnthropicMCPServersThroughResponsesGatewayOverRealHTTP
// exercises the exact client shape used by Anthropic mcp_servers callers. The
// two HTTP servers are real protocol peers: one is a Streamable HTTP MCP
// server, and the other is a Responses-form upstream. No tool executor is
// stubbed inside the controller.
func TestControllerRunsAnthropicMCPServersThroughResponsesGatewayOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer anthropic-mcp-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
	var providerIssues []string
	var providerIssuesMu sync.Mutex
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tools"`
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Type != "function" ||
			strings.Contains(string(body), `"type":"mcp"`) || strings.Contains(string(body), authorization) {
			providerIssuesMu.Lock()
			providerIssues = append(providerIssues, fmt.Sprintf("round %d provider body was not an isolated Anthropic MCP gateway request: %s", round, body))
			providerIssuesMu.Unlock()
			http.Error(writer, "unexpected request", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Name
			writer.Header().Set("Content-Type", "application/json")
			argumentsJSON := `{"sku":"A-1"}`
			_, _ = fmt.Fprintf(writer, `{"id":"resp_anthropic_mcp_1","object":"response","created_at":1785382000,"model":"fixture-model","status":"completed","output":[{"id":"item_call_1","type":"function_call","call_id":"provider_call_1","name":%q,"arguments":%q,"status":"completed"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`, syntheticName, argumentsJSON)
			return
		}
		var inputItems []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Output string `json:"output"`
		}
		foundResult := json.Unmarshal(payload.Input, &inputItems) == nil
		if foundResult {
			foundResult = false
			for _, item := range inputItems {
				if item.Type == "function_call_output" && item.CallID == "provider_call_1" && strings.Contains(item.Output, `"available":7`) {
					foundResult = true
					break
				}
			}
		}
		if round != 2 || !foundResult {
			providerIssuesMu.Lock()
			providerIssues = append(providerIssues, fmt.Sprintf("round %d missing Anthropic MCP result: %s", round, body))
			providerIssuesMu.Unlock()
			http.Error(writer, "missing function result", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_anthropic_mcp_2","object":"response","created_at":1785382001,"model":"fixture-model","status":"completed","output":[{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"7 units available","annotations":[]}]}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("anthropic-mcp-server-key-", 3)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		ForceMCPGateway: true,
		MaxRounds:       4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).Pipeline(
		anthropic.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","max_tokens":128,"messages":[{"role":"user","content":"check inventory"}],
			"mcp_servers":[{"name":"inventory","url":%q,"authorization_token":%q}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.NotNil(t, result.Response)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	providerIssuesMu.Lock()
	require.Empty(t, providerIssues)
	providerIssuesMu.Unlock()

	wire := string(result.Response.Body)
	require.Contains(t, wire, "7 units available")
	require.NotContains(t, wire, syntheticName)
	require.NotContains(t, wire, authorization)
	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 2, emulationObservation.InternalRounds)
	require.EqualValues(t, 1, emulationObservation.ToolCalls)
}

func TestControllerPausesAndResumesMCPApprovalOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer approval-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		syntheticName := payload.Tools[0].Function.Name
		if !strings.HasPrefix(syntheticName, "axon_mcp_") {
			http.Error(w, "synthetic tool", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-approval-1","object":"chat.completion","created":1785380100,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"provider_approval_call","type":"function","function":{"name":%q,"arguments":"{\"sku\":\"A-1\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, syntheticName)
			return
		}
		foundResult := false
		for _, message := range payload.Messages {
			if message.Role != "tool" {
				continue
			}
			encoded, _ := json.Marshal(message.Content)
			if strings.Contains(string(encoded), "available") && strings.Contains(string(encoded), "7") {
				foundResult = true
			}
		}
		if !foundResult {
			http.Error(w, "approval result missing", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl-approval-2","object":"chat.completion","created":1785380101,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"approved: 7 units available"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":3,"total_tokens":10}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("approval-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	gateway := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first, err := gateway.Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"check stock",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"always"}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, providerRounds.Load())
	require.Zero(t, mcpCalls.Load(), "MCP must not execute before client approval")

	var firstBody struct {
		Output []json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(first.Response.Body, &firstBody), string(first.Response.Body))
	require.Len(t, firstBody.Output, 2)
	var approval struct {
		ID        string `json:"id"`
		Type      string `json:"type"`
		Arguments string `json:"arguments"`
	}
	require.NoError(t, json.Unmarshal(firstBody.Output[1], &approval))
	require.Equal(t, "mcp_approval_request", approval.Type)
	require.NotEmpty(t, approval.ID)
	require.JSONEq(t, `{"sku":"A-1"}`, approval.Arguments)
	require.NotContains(t, string(first.Response.Body), authorization)

	decision, err := json.Marshal(map[string]any{
		"type": "mcp_approval_response", "id": "client_decision_1",
		"approval_request_id": approval.ID, "approve": true,
	})
	require.NoError(t, err)
	history := append(append([]json.RawMessage(nil), firstBody.Output...), decision)
	secondPayload, err := json.Marshal(map[string]any{
		"model": "fixture-model", "input": history,
		"tools": []map[string]any{{
			"type": "mcp", "server_label": "inventory", "server_url": mcpServer.URL,
			"authorization": authorization, "require_approval": "always",
		}},
	})
	require.NoError(t, err)
	second, err := gateway.Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: secondPayload,
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	require.NotContains(t, string(second.Response.Body), authorization)

	var secondBody struct {
		Output []struct {
			Type              string `json:"type"`
			ApprovalRequestID string `json:"approval_request_id"`
			Output            string `json:"output"`
			Content           []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			TotalTokens int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(second.Response.Body, &secondBody), string(second.Response.Body))
	require.Len(t, secondBody.Output, 2)
	require.Equal(t, "mcp_call", secondBody.Output[0].Type)
	require.Equal(t, approval.ID, secondBody.Output[0].ApprovalRequestID)
	require.Contains(t, secondBody.Output[0].Output, `"available":7`)
	require.Equal(t, "message", secondBody.Output[1].Type)
	require.Equal(t, "approved: 7 units available", secondBody.Output[1].Content[0].Text)
	require.EqualValues(t, 10, secondBody.Usage.TotalTokens)
}

func TestControllerPreservesInvalidMCPArgumentsAndDoesNotCallServer(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer invalid-arguments-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-invalid-1","object":"chat.completion","created":1785380200,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"provider_invalid_call","type":"function","function":{"name":%q,"arguments":"{not-json"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`, payload.Tools[0].Function.Name)
			return
		}
		foundFailure := false
		for _, message := range payload.Messages {
			if message.Role != "tool" {
				continue
			}
			encoded, _ := json.Marshal(message.Content)
			foundFailure = strings.Contains(string(encoded), "invalid JSON")
		}
		if !foundFailure {
			http.Error(w, "invalid-argument result missing", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl-invalid-2","object":"chat.completion","created":1785380201,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"the MCP arguments were invalid"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("invalid-argument-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"check invalid arguments",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never"}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.Zero(t, mcpCalls.Load(), "invalid arguments must never reach the MCP server")
	wire := string(result.Response.Body)
	require.NotContains(t, wire, authorization)
	require.NotContains(t, wire, `"arguments":"{}"`)

	var responseBody struct {
		Output []struct {
			ID        string `json:"id"`
			Type      string `json:"type"`
			Status    string `json:"status"`
			Arguments string `json:"arguments"`
			Error     string `json:"error"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Len(t, responseBody.Output, 3)
	require.Equal(t, "mcp_call", responseBody.Output[1].Type)
	require.True(t, strings.HasPrefix(responseBody.Output[1].ID, "mcp_"))
	require.NotContains(t, responseBody.Output[1].ID, "axon_mcp_")
	require.Equal(t, "failed", responseBody.Output[1].Status)
	require.Equal(t, "{not-json", responseBody.Output[1].Arguments)
	require.Contains(t, responseBody.Output[1].Error, "invalid JSON")
	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 1, emulationObservation.Failures)
}

func TestControllerPreservesMixedClientAndMCPCallBatchAcrossResume(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer mixed-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
				Content    any    `json:"content"`
				ToolCalls  []struct {
					ID       string `json:"id"`
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 2 {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		syntheticName := ""
		for _, tool := range payload.Tools {
			if strings.HasPrefix(tool.Function.Name, "axon_mcp_") {
				syntheticName = tool.Function.Name
			}
		}
		if syntheticName == "" {
			http.Error(w, "synthetic tool missing", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-mixed-1","object":"chat.completion","created":1785380300,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"client_call_1","type":"function","function":{"name":"client_lookup","arguments":"{\"key\":\"B-2\"}"}},{"id":"provider_mcp_call_1","type":"function","function":{"name":%q,"arguments":"{\"sku\":\"A-1\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, syntheticName)
			return
		}

		assistantCalls := []string(nil)
		toolResultIDs := []string(nil)
		for _, message := range payload.Messages {
			if message.Role == "assistant" && len(message.ToolCalls) > 0 {
				for _, call := range message.ToolCalls {
					assistantCalls = append(assistantCalls, call.Function.Name)
				}
			}
			if message.Role == "tool" {
				toolResultIDs = append(toolResultIDs, message.ToolCallID)
			}
		}
		if len(assistantCalls) != 2 || assistantCalls[0] != "client_lookup" || assistantCalls[1] != syntheticName {
			http.Error(w, "parallel call batch was not preserved", http.StatusBadRequest)
			return
		}
		if len(toolResultIDs) != 2 || toolResultIDs[0] != "client_call_1" || toolResultIDs[1] == "client_call_1" {
			http.Error(w, "tool result order was not preserved", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl-mixed-2","object":"chat.completion","created":1785380301,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"both results received"},"finish_reason":"stop"}],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("mixed-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	gateway := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	)
	toolJSON := fmt.Sprintf(`[
		{"type":"function","name":"client_lookup","description":"client owned","parameters":{"type":"object","properties":{"key":{"type":"string"}}}},
		{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never"}
	]`, mcpServer.URL, authorization)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	first, err := gateway.Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    []byte(fmt.Sprintf(`{"model":"fixture-model","input":"run both","tools":%s}`, toolJSON)),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	var firstBody struct {
		Output []json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(first.Response.Body, &firstBody), string(first.Response.Body))
	require.Len(t, firstBody.Output, 3)
	var firstTypes []string
	for _, raw := range firstBody.Output {
		var item struct {
			Type string `json:"type"`
		}
		require.NoError(t, json.Unmarshal(raw, &item))
		firstTypes = append(firstTypes, item.Type)
	}
	require.Equal(t, []string{"mcp_list_tools", "function_call", "mcp_call"}, firstTypes)

	clientResult := json.RawMessage(`{"type":"function_call_output","call_id":"client_call_1","output":"client value"}`)
	history := append(append([]json.RawMessage(nil), firstBody.Output...), clientResult)
	secondPayload, err := json.Marshal(map[string]any{
		"model": "fixture-model", "input": history,
		"tools": []map[string]any{
			{"type": "function", "name": "client_lookup", "description": "client owned", "parameters": map[string]any{"type": "object", "properties": map[string]any{"key": map[string]any{"type": "string"}}}},
			{"type": "mcp", "server_label": "inventory", "server_url": mcpServer.URL, "authorization": authorization, "require_approval": "never"},
		},
	})
	require.NoError(t, err)
	second, err := gateway.Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: secondPayload,
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load(), "hydrated MCP history must not execute twice")
	require.NotContains(t, string(second.Response.Body), authorization)
	var secondBody struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(second.Response.Body, &secondBody), string(second.Response.Body))
	require.Len(t, secondBody.Output, 1)
	require.Equal(t, "message", secondBody.Output[0].Type)
	require.Equal(t, "both results received", secondBody.Output[0].Content[0].Text)
}

func TestControllerStreamsTwoRealFragmentedChatRoundsAsOneResponsesLifecycle(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer streaming-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	firstFinalDeltaWritten := make(chan struct{})
	releaseFinalRound := make(chan struct{})
	providerErrors := make(chan error, 2)
	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			providerErrors <- readErr
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || !payload.Stream || len(payload.Tools) != 1 {
			providerErrors <- fmt.Errorf("unexpected streaming request: %s", body)
			http.Error(w, "request", http.StatusBadRequest)
			return
		}
		syntheticName := payload.Tools[0].Function.Name
		if !strings.HasPrefix(syntheticName, "axon_mcp_") {
			providerErrors <- errors.New("synthetic MCP tool is missing")
			http.Error(w, "tool", http.StatusBadRequest)
			return
		}
		if round == 2 {
			foundResult := false
			for _, message := range payload.Messages {
				if message.Role != "tool" {
					continue
				}
				encoded, _ := json.Marshal(message.Content)
				foundResult = foundResult || strings.Contains(string(encoded), "available")
			}
			if !foundResult {
				providerErrors <- errors.New("second streaming round has no MCP result")
				http.Error(w, "result", http.StatusBadRequest)
				return
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			providerErrors <- errors.New("response writer has no flusher")
			return
		}
		writeFrame := func(value any) bool {
			encoded, marshalErr := json.Marshal(value)
			if marshalErr != nil {
				providerErrors <- marshalErr
				return false
			}
			frame := append(append([]byte("data: "), encoded...), []byte("\n\n")...)
			for _, value := range frame {
				if _, writeErr := w.Write([]byte{value}); writeErr != nil {
					providerErrors <- writeErr
					return false
				}
				flusher.Flush()
			}
			return true
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": fmt.Sprintf("chatcmpl-stream-%d", round), "object": "chat.completion.chunk",
				"created": 1785380400 + round, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		if round == 1 {
			if !writeFrame(chunk(map[string]any{"role": "assistant", "content": "checking inventory"}, nil)) ||
				!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "id": "stream_mcp_call", "type": "function",
					"function": map[string]any{"name": syntheticName, "arguments": `{"sku":"`},
				}}}, nil)) ||
				!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "function": map[string]any{"arguments": `A-1"}`},
				}}}, nil)) ||
				!writeFrame(chunk(map[string]any{}, "tool_calls")) ||
				!writeFrame(map[string]any{
					"id": "chatcmpl-stream-1", "object": "chat.completion.chunk", "model": "fixture-model",
					"choices": []any{}, "usage": map[string]any{"prompt_tokens": 6, "completion_tokens": 3, "total_tokens": 9},
				}) {
				return
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			flusher.Flush()
			providerErrors <- nil
			return
		}
		if !writeFrame(chunk(map[string]any{"role": "assistant", "content": "7 units"}, nil)) {
			return
		}
		close(firstFinalDeltaWritten)
		<-releaseFinalRound
		if !writeFrame(chunk(map[string]any{"content": " available"}, nil)) ||
			!writeFrame(chunk(map[string]any{}, "stop")) ||
			!writeFrame(map[string]any{
				"id": "chatcmpl-stream-2", "object": "chat.completion.chunk", "model": "fixture-model",
				"choices": []any{}, "usage": map[string]any{"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
			}) {
			return
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
		providerErrors <- nil
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("stream-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","stream":true,"input":"stream stock",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never"}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)

	type streamCapture struct {
		types []string
		wire  strings.Builder
		err   error
	}
	finalDeltaSeen := make(chan struct{})
	captured := make(chan streamCapture, 1)
	go func() {
		capture := streamCapture{}
		seenFinal := false
		for result.EventStream.Next() {
			event := result.EventStream.Current()
			if event == nil {
				continue
			}
			capture.types = append(capture.types, event.Type)
			capture.wire.Write(event.Data)
			if !seenFinal && event.Type == "response.output_text.delta" && strings.Contains(string(event.Data), "7 units") {
				seenFinal = true
				close(finalDeltaSeen)
			}
		}
		capture.err = result.EventStream.Err()
		captured <- capture
	}()

	select {
	case <-firstFinalDeltaWritten:
	case <-time.After(5 * time.Second):
		t.Fatal("provider did not write the first final-round delta")
	}
	select {
	case <-finalDeltaSeen:
		// The client observed final-round text while the provider was still
		// blocked before its terminal event: this proves incremental forwarding.
	case <-time.After(2 * time.Second):
		close(releaseFinalRound)
		t.Fatal("final-round text was buffered until provider completion")
	}
	close(releaseFinalRound)
	capture := <-captured
	require.NoError(t, capture.err)
	require.NoError(t, <-providerErrors)
	require.NoError(t, <-providerErrors)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	wire := capture.wire.String()
	require.Contains(t, wire, "checking inventory")
	require.Contains(t, wire, "7 units")
	require.Contains(t, wire, " available")
	require.Contains(t, wire, "mcp_call")
	require.NotContains(t, wire, "axon_mcp_")
	require.NotContains(t, wire, authorization)
	require.Equal(t, 1, countString(capture.types, "response.created"))
	require.Equal(t, 1, countString(capture.types, "response.completed"))
	require.Contains(t, capture.types, "response.mcp_call.completed")
	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 2, emulationObservation.InternalRounds)
	require.EqualValues(t, 1, emulationObservation.ToolCalls)
}

func TestControllerLoadsDeferredMCPToolThroughInternalSearchOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer deferred-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var searchName, lookupName string
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Function struct {
					Name        string `json:"name"`
					Description string `json:"description"`
				} `json:"function"`
			} `json:"tools"`
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch round {
		case 1:
			if len(payload.Tools) != 1 || !strings.Contains(payload.Tools[0].Function.Description, "Search and load") {
				http.Error(w, "deferred tool was eagerly exposed", http.StatusBadRequest)
				return
			}
			searchName = payload.Tools[0].Function.Name
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-deferred-1","object":"chat.completion","created":1785380500,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"search_call_1","type":"function","function":{"name":%q,"arguments":"{\"query\":\"stock inventory\",\"limit\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, searchName)
		case 2:
			if len(payload.Tools) != 2 {
				http.Error(w, "selected deferred tool was not loaded", http.StatusBadRequest)
				return
			}
			for _, tool := range payload.Tools {
				if strings.Contains(tool.Function.Description, "look up stock") {
					lookupName = tool.Function.Name
				}
			}
			if lookupName == "" || lookupName == searchName {
				http.Error(w, "lookup binding missing", http.StatusBadRequest)
				return
			}
			foundSearchResult := false
			for _, message := range payload.Messages {
				if message.Role != "tool" {
					continue
				}
				encoded, _ := json.Marshal(message.Content)
				foundSearchResult = foundSearchResult || strings.Contains(string(encoded), "lookup")
			}
			if !foundSearchResult {
				http.Error(w, "tool search result missing", http.StatusBadRequest)
				return
			}
			_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-deferred-2","object":"chat.completion","created":1785380501,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"deferred_lookup_call","type":"function","function":{"name":%q,"arguments":"{\"sku\":\"A-1\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}`, lookupName)
		case 3:
			foundMCPResult := false
			for _, message := range payload.Messages {
				if message.Role != "tool" {
					continue
				}
				encoded, _ := json.Marshal(message.Content)
				foundMCPResult = foundMCPResult || strings.Contains(string(encoded), "available")
			}
			if !foundMCPResult {
				http.Error(w, "MCP result missing", http.StatusBadRequest)
				return
			}
			_, _ = io.WriteString(w, `{"id":"chatcmpl-deferred-3","object":"chat.completion","created":1785380502,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"deferred lookup completed"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`)
		default:
			http.Error(w, "too many rounds", http.StatusBadRequest)
		}
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("deferred-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 5, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"find and use the inventory tool",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never","defer_loading":true}]
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.EqualValues(t, 3, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	wire := string(result.Response.Body)
	require.NotContains(t, wire, searchName)
	require.NotContains(t, wire, lookupName)
	require.NotContains(t, wire, authorization)
	var responseBody struct {
		Output []struct {
			Type        string `json:"type"`
			ServerLabel string `json:"server_label"`
			Tools       []struct {
				Name string `json:"name"`
			} `json:"tools"`
			Name    string `json:"name"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			TotalTokens int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Len(t, responseBody.Output, 3)
	require.Equal(t, "mcp_list_tools", responseBody.Output[0].Type)
	require.Equal(t, "inventory", responseBody.Output[0].ServerLabel)
	require.Len(t, responseBody.Output[0].Tools, 1)
	require.Equal(t, "lookup", responseBody.Output[0].Tools[0].Name)
	require.Equal(t, "mcp_call", responseBody.Output[1].Type)
	require.Equal(t, "lookup", responseBody.Output[1].Name)
	require.Equal(t, "message", responseBody.Output[2].Type)
	require.Equal(t, "deferred lookup completed", responseBody.Output[2].Content[0].Text)
	require.EqualValues(t, 23, responseBody.Usage.TotalTokens)
	emulationObservation := observer.emulation()
	require.NotNil(t, emulationObservation)
	require.EqualValues(t, 3, emulationObservation.InternalRounds)
	require.EqualValues(t, 2, emulationObservation.ToolCalls)
}

func TestControllerRunsResponsesMCPThroughAnthropicTargetOverRealHTTP(t *testing.T) {
	t.Parallel()
	const authorization = "Bearer anthropic-mcp-private-token"
	var mcpCalls atomic.Int64
	mcpServer := newControllerMCPServer(t, authorization, &mcpCalls)
	allowedMCP, err := url.Parse(mcpServer.URL)
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(r.Body)
		if readErr != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
			ToolChoice *struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tool_choice"`
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if round == 1 {
			syntheticName = payload.Tools[0].Name
			if !strings.HasPrefix(syntheticName, "axon_mcp_") {
				http.Error(w, "synthetic tool", http.StatusBadRequest)
				return
			}
			if payload.ToolChoice == nil || payload.ToolChoice.Type != "tool" || payload.ToolChoice.Name != syntheticName {
				http.Error(w, "required MCP tool was not specialized", http.StatusBadRequest)
				return
			}
			_, _ = fmt.Fprintf(w, `{"id":"msg_mcp_round_1","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"anthropic_mcp_call","name":%q,"input":{"sku":"A-1"}}],"stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":2}}`, syntheticName)
			return
		}
		if payload.ToolChoice != nil {
			http.Error(w, "completed MCP call kept forcing another tool", http.StatusBadRequest)
			return
		}
		foundResult := false
		for _, message := range payload.Messages {
			if message.Role == "user" && strings.Contains(string(message.Content), "available") {
				foundResult = true
			}
		}
		if !foundResult {
			http.Error(w, "MCP result missing", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(w, `{"id":"msg_mcp_round_2","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"Anthropic received 7 units"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":3}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		MCP: mcp.RegistryConfig{
			SyntheticNameKey: []byte(strings.Repeat("anthropic-controller-key-", 2)),
			EndpointPolicy: func(candidate *url.URL) error {
				if candidate.Scheme != allowedMCP.Scheme || candidate.Host != allowedMCP.Host {
					return fmt.Errorf("MCP endpoint is not allowlisted")
				}
				return nil
			},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","input":"check through Anthropic",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":%q,"authorization":%q,"require_approval":"never"}],
			"tool_choice":"required"
		}`, mcpServer.URL, authorization)),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, mcpCalls.Load())
	wire := string(result.Response.Body)
	require.NotContains(t, wire, syntheticName)
	require.NotContains(t, wire, authorization)
	var responseBody struct {
		Output []struct {
			Type    string `json:"type"`
			Name    string `json:"name"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			TotalTokens int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Len(t, responseBody.Output, 3)
	require.Equal(t, "mcp_list_tools", responseBody.Output[0].Type)
	require.Equal(t, "mcp_call", responseBody.Output[1].Type)
	require.Equal(t, "lookup", responseBody.Output[1].Name)
	require.Equal(t, "message", responseBody.Output[2].Type)
	require.Equal(t, "Anthropic received 7 units", responseBody.Output[2].Content[0].Text)
	require.EqualValues(t, 17, responseBody.Usage.TotalTokens)
}

func TestControllerEmulatesAnthropicServerToolSearchThenExposesDiscoveredCallOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRounds atomic.Int64
	var searchName string
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Function struct {
					Name        string `json:"name"`
					Description string `json:"description"`
				} `json:"function"`
			} `json:"tools"`
			Messages json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		toolNames := make([]string, 0, len(payload.Tools))
		for _, tool := range payload.Tools {
			toolNames = append(toolNames, tool.Function.Name)
			if strings.HasPrefix(tool.Function.Name, "axh_") {
				searchName = tool.Function.Name
			}
		}
		w.Header().Set("Content-Type", "application/json")
		switch round {
		case 1:
			if searchName == "" || len(toolNames) != 2 || !slices.Contains(toolNames, "always_available") ||
				slices.Contains(toolNames, "create_event") || slices.Contains(toolNames, "delete_event") {
				http.Error(w, fmt.Sprintf("initial catalog leaked: %v", toolNames), http.StatusBadRequest)
				return
			}
			_, _ = fmt.Fprintf(w, `{"id":"chat_anthropic_search_1","object":"chat.completion","created":1785429300,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"search_call_1","type":"function","function":{"name":%q,"arguments":"{\"query\":\"create\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, searchName)
		case 2:
			if len(toolNames) != 3 || !slices.Contains(toolNames, searchName) || !slices.Contains(toolNames, "always_available") ||
				!slices.Contains(toolNames, "create_event") || slices.Contains(toolNames, "delete_event") ||
				!strings.Contains(string(payload.Messages), "create_event") {
				http.Error(w, fmt.Sprintf("discovered catalog missing: tools=%v messages=%s", toolNames, payload.Messages), http.StatusBadRequest)
				return
			}
			_, _ = io.WriteString(w, `{"id":"chat_anthropic_search_2","object":"chat.completion","created":1785429301,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"create_call_1","type":"function","function":{"name":"create_event","arguments":"{\"title\":\"Architecture review\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}`)
		default:
			http.Error(w, "too many rounds", http.StatusBadRequest)
		}
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted:    hosted.Config{SyntheticNameKey: []byte(strings.Repeat("anthropic-tool-search-key-", 2))},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "fixture-provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).Pipeline(
		anthropic.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	).Process(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","max_tokens":256,
			"messages":[{"role":"user","content":"create a calendar event"}],
			"tools":[
				{"type":"tool_search_tool_bm25_20251119","name":"tool_search_tool_bm25"},
				{"name":"always_available","description":"Always available","input_schema":{"type":"object"}},
				{"name":"create_event","description":"Create a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"title":{"type":"string","description":"Event title"}},"required":["title"]}},
				{"name":"delete_event","description":"Delete a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}
			]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	wire := string(result.Response.Body)
	require.NotContains(t, wire, searchName)
	var response struct {
		Content []struct {
			Type      string `json:"type"`
			ID        string `json:"id"`
			Name      string `json:"name"`
			ToolUseID string `json:"tool_use_id"`
			Content   struct {
				Type           string `json:"type"`
				ToolReferences []struct {
					Type     string `json:"type"`
					ToolName string `json:"tool_name"`
				} `json:"tool_references"`
			} `json:"content"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &response), wire)
	require.Len(t, response.Content, 3, wire)
	require.Equal(t, "server_tool_use", response.Content[0].Type)
	require.Equal(t, "tool_search_tool_bm25", response.Content[0].Name)
	require.Equal(t, "tool_search_tool_result", response.Content[1].Type)
	require.Equal(t, "search_call_1", response.Content[1].ToolUseID)
	require.Equal(t, "tool_search_tool_search_result", response.Content[1].Content.Type)
	require.Len(t, response.Content[1].Content.ToolReferences, 1)
	require.Equal(t, "tool_reference", response.Content[1].Content.ToolReferences[0].Type)
	require.Equal(t, "create_event", response.Content[1].Content.ToolReferences[0].ToolName)
	require.Equal(t, "tool_use", response.Content[2].Type)
	require.Equal(t, "create_event", response.Content[2].Name)
	require.Equal(t, "create_call_1", response.Content[2].ID)
	require.Equal(t, "tool_use", response.StopReason)
}

func countString(values []string, wanted string) int {
	count := 0
	for _, value := range values {
		if value == wanted {
			count++
		}
	}
	return count
}

func newControllerMCPServer(t *testing.T, authorization string, calls *atomic.Int64) *httptest.Server {
	t.Helper()
	const sessionID = "controller-mcp-session"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != authorization {
			http.Error(w, "authorization", http.StatusUnauthorized)
			return
		}
		if r.Method == http.MethodDelete {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var envelope struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if json.Unmarshal(body, &envelope) != nil {
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		switch envelope.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Mcp-Session-Id", sessionID)
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"inventory","version":"1"}}}`, envelope.ID, mcp.ProtocolVersion)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"lookup","description":"look up stock","inputSchema":{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]},"annotations":{"readOnlyHint":true}}]}}`, envelope.ID)
		case "tools/call":
			calls.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"7"}],"structuredContent":{"available":7}}}`, envelope.ID)
		default:
			http.Error(w, "method", http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

type controllerObservationRecorder struct {
	mu     sync.Mutex
	events []pipeline.Observation
}

func (recorder *controllerObservationRecorder) Observe(_ context.Context, event pipeline.Observation) {
	recorder.mu.Lock()
	recorder.events = append(recorder.events, event)
	recorder.mu.Unlock()
}

func (recorder *controllerObservationRecorder) emulation() *llm.EmulationTraceSummary {
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	for index := range recorder.events {
		if recorder.events[index].Stage == pipeline.StageConversionEmulation {
			return recorder.events[index].Emulation
		}
	}
	return nil
}

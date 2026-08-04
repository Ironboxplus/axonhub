package emulation_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestControllerRunsResponsesWebSearchThroughChatTargetOverRealHTTP(t *testing.T) {
	t.Parallel()
	observer := &controllerObservationRecorder{}
	const searchSecret = "private-search-secret"
	var searchCalls atomic.Int64
	searchServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		searchCalls.Add(1)
		if request.Method != http.MethodPost || request.Header.Get("X-Search-Key") != searchSecret {
			http.Error(writer, "unauthorized", http.StatusUnauthorized)
			return
		}
		var input hosted.WebSearchInput
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil || input.Query != "weather Singapore" ||
			len(input.AllowedDomains) != 1 || input.AllowedDomains[0] != "weather.example" {
			http.Error(writer, "bad search input", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"text":"Singapore is sunny","sources":[{"url":"https://weather.example/singapore","title":"Singapore weather"}]}`)
	}))
	t.Cleanup(searchServer.Close)
	allowedSearch, err := url.Parse(searchServer.URL)
	require.NoError(t, err)
	webExecutor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{
		Endpoint: searchServer.URL, Client: searchServer.Client(), Headers: http.Header{"X-Search-Key": []string{searchSecret}},
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedSearch.Scheme || candidate.Host != allowedSearch.Host {
				return fmt.Errorf("search endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		if bytes.Contains(body, []byte(searchSecret)) || bytes.Contains(body, []byte("weather.example/singapore")) && round == 1 {
			http.Error(writer, "private executor data leaked", http.StatusBadRequest)
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
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Type != "function" {
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Function.Name
			if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, "web") || strings.Contains(syntheticName, "search") {
				http.Error(writer, "hosted name is not opaque", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(writer, `{"id":"chat-search-1","object":"chat.completion","created":1785381000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"search_call_1","type":"function","function":{"name":%q,"arguments":"{\"query\":\"weather Singapore\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`, syntheticName)
			return
		}
		foundResult := false
		for _, message := range payload.Messages {
			if message.Role == "tool" && message.ToolCallID == "search_call_1" {
				encoded, _ := json.Marshal(message.Content)
				foundResult = strings.Contains(string(encoded), "Singapore is sunny")
			}
		}
		if !foundResult {
			http.Error(writer, "search result missing", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat-search-2","object":"chat.completion","created":1785381001,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"It is sunny in Singapore."},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("hosted-controller-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: webExecutor},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","input":"check the weather",
			"tools":[{"type":"web_search","filters":{"allowed_domains":["weather.example"]}}]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, searchCalls.Load())
	require.EqualValues(t, 1, hostedExecutionCount(observer.emulation(), llm.ToolKindWebSearch))
	require.EqualValues(t, 1, totalHostedExecutions(observer.emulation()))
	wire := string(result.Response.Body)
	require.NotContains(t, wire, syntheticName)
	require.NotContains(t, wire, searchSecret)
	var responseBody struct {
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Action struct {
				Query   string `json:"query"`
				Sources []struct {
					URL string `json:"url"`
				} `json:"sources"`
			} `json:"action"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
		Usage struct {
			TotalTokens int64 `json:"total_tokens"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Len(t, responseBody.Output, 2)
	require.Equal(t, "web_search_call", responseBody.Output[0].Type)
	require.Equal(t, "search_call_1", responseBody.Output[0].CallID)
	require.Equal(t, "weather Singapore", responseBody.Output[0].Action.Query)
	require.Len(t, responseBody.Output[0].Action.Sources, 1)
	require.Equal(t, "https://weather.example/singapore", responseBody.Output[0].Action.Sources[0].URL)
	require.Equal(t, "message", responseBody.Output[1].Type)
	require.Equal(t, "It is sunny in Singapore.", responseBody.Output[1].Content[0].Text)
	require.EqualValues(t, 18, responseBody.Usage.TotalTokens)
}

func TestControllerForcesResponsesWebSearchThroughGatewayOverRealHTTP(t *testing.T) {
	t.Parallel()
	var searchCalls atomic.Int64
	searchServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		searchCalls.Add(1)
		var input hosted.WebSearchInput
		if request.Method != http.MethodPost || json.NewDecoder(request.Body).Decode(&input) != nil || input.Query != "same protocol search" {
			http.Error(writer, "bad search request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"text":"same protocol gateway result","sources":[{"url":"https://example.test/evidence","title":"evidence"}]}`)
	}))
	t.Cleanup(searchServer.Close)
	searchEndpoint, err := url.Parse(searchServer.URL)
	require.NoError(t, err)
	webExecutor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{
		Endpoint: searchServer.URL, Client: searchServer.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != searchEndpoint.Scheme || candidate.Host != searchEndpoint.Host {
				return fmt.Errorf("search endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	var syntheticName string
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
			bytes.Contains(body, []byte(`"type":"web_search"`)) {
			http.Error(writer, "native hosted tool reached unadmitted provider", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Name
			if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, "search") {
				http.Error(writer, "hosted function name is not opaque", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(writer, `{"id":"resp_forced_search_1","object":"response","created_at":1785382000,"model":"fixture-model","status":"completed","output":[{"id":"item_search_1","type":"function_call","call_id":"search_call_1","name":%q,"arguments":"{\"query\":\"same protocol search\"}","status":"completed"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`, syntheticName)
			return
		}
		var inputItems []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Output any    `json:"output"`
		}
		foundResult := json.Unmarshal(payload.Input, &inputItems) == nil
		if foundResult {
			foundResult = false
			for _, item := range inputItems {
				encodedOutput, _ := json.Marshal(item.Output)
				if item.Type == "function_call_output" && item.CallID == "search_call_1" &&
					bytes.Contains(encodedOutput, []byte("same protocol gateway result")) {
					foundResult = true
					break
				}
			}
		}
		if round != 2 || !foundResult {
			http.Error(writer, "gateway result missing from continuation", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_forced_search_2","object":"response","created_at":1785382001,"model":"fixture-model","status":"completed","output":[{"id":"msg_forced_search","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"search finished","annotations":[]}]}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("forced-same-protocol-hosted-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: webExecutor},
		},
		ForceHostedGateway: conversion.CapabilityWebSearchTool,
		MaxRounds:          4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","input":"search","tools":[{"type":"web_search"}]}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, searchCalls.Load())
	wire := string(result.Response.Body)
	require.Contains(t, wire, `"type":"web_search_call"`)
	require.Contains(t, wire, "https://example.test/evidence")
	require.Contains(t, wire, "search finished")
	require.NotContains(t, wire, syntheticName)
}

type hostedSafeFailure struct{}

func (hostedSafeFailure) Error() string { return "Authorization: Bearer private-hosted-secret" }

func (hostedSafeFailure) SafeDiagnostic() llm.ErrorDiagnostic {
	return llm.ErrorDiagnostic{
		Component: "web_search", Code: "search_unavailable", Message: "Web search is unavailable", StatusCode: http.StatusBadGateway,
	}
}

type failingWebSearchService struct{ calls *atomic.Int64 }

func (service failingWebSearchService) Search(context.Context, hosted.WebSearchInput) (hosted.WebSearchOutput, error) {
	service.calls.Add(1)
	return hosted.WebSearchOutput{}, hostedSafeFailure{}
}

func TestControllerStopsAfterHostedExecutorFailureOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerCalls atomic.Int64
	var searchCalls atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		providerCalls.Add(1)
		var payload struct {
			Tools []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tools"`
		}
		if json.NewDecoder(request.Body).Decode(&payload) != nil || len(payload.Tools) != 1 ||
			payload.Tools[0].Type != "function" || !strings.HasPrefix(payload.Tools[0].Name, "axh_") {
			http.Error(writer, "native hosted tool reached provider", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(writer, `{"id":"resp_failed_search","object":"response","created_at":1785382002,"model":"fixture-model","status":"completed","output":[{"id":"item_failed_search","type":"function_call","call_id":"failed_search_call","name":%q,"arguments":"{\"query\":\"same protocol search\"}","status":"completed"}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`, payload.Tools[0].Name)
	}))
	t.Cleanup(provider.Close)

	webExecutor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{
		Service: failingWebSearchService{calls: &searchCalls},
	})
	require.NoError(t, err)
	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("failed-same-protocol-hosted-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: webExecutor},
		},
		ForceHostedGateway: conversion.CapabilityWebSearchTool,
		MaxRounds:          4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err = pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","input":"search","tools":[{"type":"web_search"}]}`),
	})
	require.Error(t, err)
	require.EqualValues(t, 1, providerCalls.Load(), "a local hosted failure must not trigger another provider round")
	require.EqualValues(t, 1, searchCalls.Load())
	require.NotContains(t, err.Error(), "private-hosted-secret")
	require.Equal(t, &llm.ErrorDiagnostic{
		Component: "web_search", Code: "search_unavailable", Message: "Web search is unavailable", StatusCode: http.StatusBadGateway,
	}, llm.ErrorDiagnosticFrom(err))
}

func TestControllerForcesResponsesWebSearchThroughGatewayStreamOverRealHTTP(t *testing.T) {
	t.Parallel()
	var searchCalls atomic.Int64
	searchServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		searchCalls.Add(1)
		var input hosted.WebSearchInput
		if request.Method != http.MethodPost || json.NewDecoder(request.Body).Decode(&input) != nil || input.Query != "stream same protocol search" {
			http.Error(writer, "bad search request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"text":"stream same protocol result","sources":[{"url":"https://example.test/stream-evidence","title":"stream evidence"}]}`)
	}))
	t.Cleanup(searchServer.Close)
	searchEndpoint, err := url.Parse(searchServer.URL)
	require.NoError(t, err)
	webExecutor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{
		Endpoint: searchServer.URL, Client: searchServer.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != searchEndpoint.Scheme || candidate.Host != searchEndpoint.Host {
				return fmt.Errorf("search endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	providerErrors := make(chan error, 2)
	var syntheticName string
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			providerErrors <- readErr
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Type string `json:"type"`
				Name string `json:"name"`
			} `json:"tools"`
			Input json.RawMessage `json:"input"`
		}
		if json.Unmarshal(body, &payload) != nil || !payload.Stream || len(payload.Tools) != 1 || payload.Tools[0].Type != "function" ||
			bytes.Contains(body, []byte(`"type":"web_search"`)) {
			providerErrors <- fmt.Errorf("unexpected same-protocol streaming request: %s", body)
			http.Error(writer, "request", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Name
			if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, "search") {
				providerErrors <- fmt.Errorf("hosted name is not opaque: %s", syntheticName)
				http.Error(writer, "tool", http.StatusBadRequest)
				return
			}
		} else if round == 2 {
			if !bytes.Contains(payload.Input, []byte(`"type":"function_call_output"`)) ||
				!bytes.Contains(payload.Input, []byte("stream same protocol result")) {
				providerErrors <- fmt.Errorf("gateway result missing from continuation: %s", payload.Input)
				http.Error(writer, "result", http.StatusBadRequest)
				return
			}
		} else {
			providerErrors <- fmt.Errorf("unexpected provider round %d", round)
			http.Error(writer, "round", http.StatusBadRequest)
			return
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			providerErrors <- errors.New("response writer has no flusher")
			return
		}
		writeFrame := func(eventType string, value map[string]any) bool {
			encoded, marshalErr := json.Marshal(value)
			if marshalErr != nil {
				providerErrors <- marshalErr
				return false
			}
			if _, writeErr := fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", eventType, encoded); writeErr != nil {
				providerErrors <- writeErr
				return false
			}
			flusher.Flush()
			return true
		}
		responseID := fmt.Sprintf("resp_same_stream_%d", round)
		if !writeFrame("response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": responseID, "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}},
		}) {
			return
		}
		if round == 1 {
			arguments := `{"query":"stream same protocol search"}`
			if !writeFrame("response.output_item.added", map[string]any{
				"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
				"item": map[string]any{"id": "item_same_stream_search", "type": "function_call", "status": "in_progress", "call_id": "same_stream_search_call", "name": syntheticName, "arguments": ""},
			}) || !writeFrame("response.function_call_arguments.delta", map[string]any{
				"type": "response.function_call_arguments.delta", "sequence_number": 2, "output_index": 0,
				"item_id": "item_same_stream_search", "delta": arguments,
			}) || !writeFrame("response.function_call_arguments.done", map[string]any{
				"type": "response.function_call_arguments.done", "sequence_number": 3, "output_index": 0,
				"item_id": "item_same_stream_search", "call_id": "same_stream_search_call", "name": syntheticName, "arguments": arguments,
			}) || !writeFrame("response.output_item.done", map[string]any{
				"type": "response.output_item.done", "sequence_number": 4, "output_index": 0,
				"item": map[string]any{"id": "item_same_stream_search", "type": "function_call", "status": "completed", "call_id": "same_stream_search_call", "name": syntheticName, "arguments": arguments},
			}) {
				return
			}
		} else {
			if !writeFrame("response.output_item.added", map[string]any{
				"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
				"item": map[string]any{"id": "msg_same_stream", "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}},
			}) || !writeFrame("response.output_text.delta", map[string]any{
				"type": "response.output_text.delta", "sequence_number": 2, "output_index": 0,
				"item_id": "msg_same_stream", "content_index": 0, "delta": "stream search finished",
			}) || !writeFrame("response.output_text.done", map[string]any{
				"type": "response.output_text.done", "sequence_number": 3, "output_index": 0,
				"item_id": "msg_same_stream", "content_index": 0, "text": "stream search finished",
			}) || !writeFrame("response.output_item.done", map[string]any{
				"type": "response.output_item.done", "sequence_number": 4, "output_index": 0,
				"item": map[string]any{"id": "msg_same_stream", "type": "message", "role": "assistant", "status": "completed", "content": []any{map[string]any{"type": "output_text", "text": "stream search finished", "annotations": []any{}}}},
			}) {
				return
			}
		}
		if !writeFrame("response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 5,
			"response": map[string]any{"id": responseID, "object": "response", "model": "fixture-model", "status": "completed", "output": []any{}, "usage": map[string]any{"input_tokens": 4, "output_tokens": 2, "total_tokens": 6}},
		}) {
			return
		}
		providerErrors <- nil
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("forced-same-protocol-stream-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: webExecutor},
		},
		ForceHostedGateway: conversion.CapabilityWebSearchTool,
		MaxRounds:          4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","stream":true,"input":"search","tools":[{"type":"web_search"}]}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	var eventTypes []string
	var wire strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event != nil {
			eventTypes = append(eventTypes, event.Type)
			wire.Write(event.Data)
		}
	}
	require.NoError(t, result.EventStream.Err())
	require.NoError(t, <-providerErrors)
	require.NoError(t, <-providerErrors)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, searchCalls.Load())
	encoded := wire.String()
	require.Contains(t, encoded, `"type":"web_search_call"`)
	require.Contains(t, encoded, "https://example.test/stream-evidence")
	require.Contains(t, encoded, "stream search finished")
	require.NotContains(t, encoded, syntheticName)
	require.NotContains(t, encoded, "response.function_call_arguments")
	require.Equal(t, 1, countString(eventTypes, "response.created"))
	require.Equal(t, 1, countString(eventTypes, "response.completed"))
}

func TestControllerStreamsResponsesWebSearchThroughChatTargetOverRealHTTP(t *testing.T) {
	t.Parallel()
	const searchSecret = "private-stream-search-secret"
	var searchCalls atomic.Int64
	searchServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		searchCalls.Add(1)
		if request.Method != http.MethodPost || request.Header.Get("X-Search-Key") != searchSecret {
			http.Error(writer, "unauthorized", http.StatusUnauthorized)
			return
		}
		var input hosted.WebSearchInput
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil || input.Query != "weather Singapore" ||
			len(input.AllowedDomains) != 1 || input.AllowedDomains[0] != "weather.example" {
			http.Error(writer, "bad search input", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"text":"Singapore is sunny","sources":[{"url":"https://weather.example/singapore","title":"Singapore weather"}]}`)
	}))
	t.Cleanup(searchServer.Close)
	allowedSearch, err := url.Parse(searchServer.URL)
	require.NoError(t, err)
	webExecutor, err := hosted.NewWebSearchExecutor(hosted.WebSearchExecutorConfig{
		Endpoint: searchServer.URL, Client: searchServer.Client(), Headers: http.Header{"X-Search-Key": []string{searchSecret}},
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedSearch.Scheme || candidate.Host != allowedSearch.Host {
				return fmt.Errorf("search endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	providerErrors := make(chan error, 2)
	var syntheticName string
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			providerErrors <- readErr
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		if bytes.Contains(body, []byte(searchSecret)) {
			providerErrors <- fmt.Errorf("private executor credential leaked")
			http.Error(writer, "private data", http.StatusBadRequest)
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
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
				Content    any    `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil || !payload.Stream || len(payload.Tools) != 1 {
			providerErrors <- fmt.Errorf("unexpected streaming request: %s", body)
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		if round == 1 {
			syntheticName = payload.Tools[0].Function.Name
			if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, "web") || strings.Contains(syntheticName, "search") {
				providerErrors <- fmt.Errorf("hosted name is not opaque: %s", syntheticName)
				http.Error(writer, "tool", http.StatusBadRequest)
				return
			}
		} else {
			foundResult := false
			for _, message := range payload.Messages {
				if message.Role == "tool" && message.ToolCallID == "stream_search_call_1" {
					encoded, _ := json.Marshal(message.Content)
					foundResult = strings.Contains(string(encoded), "Singapore is sunny")
				}
			}
			if !foundResult {
				providerErrors <- fmt.Errorf("search result missing from second round")
				http.Error(writer, "result", http.StatusBadRequest)
				return
			}
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			providerErrors <- fmt.Errorf("response writer has no flusher")
			return
		}
		writeFrame := func(value any) bool {
			encoded, marshalErr := json.Marshal(value)
			if marshalErr != nil {
				providerErrors <- marshalErr
				return false
			}
			if _, writeErr := fmt.Fprintf(writer, "data: %s\n\n", encoded); writeErr != nil {
				providerErrors <- writeErr
				return false
			}
			flusher.Flush()
			return true
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": fmt.Sprintf("chatcmpl-hosted-stream-%d", round), "object": "chat.completion.chunk",
				"created": 1785381100 + round, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		if round == 1 {
			if !writeFrame(chunk(map[string]any{"role": "assistant"}, nil)) ||
				!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "id": "stream_search_call_1", "type": "function",
					"function": map[string]any{"name": syntheticName, "arguments": `{"query":"weather`},
				}}}, nil)) ||
				!writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
					"index": 0, "function": map[string]any{"arguments": ` Singapore"}`},
				}}}, nil)) ||
				!writeFrame(chunk(map[string]any{}, "tool_calls")) ||
				!writeFrame(map[string]any{
					"id": "chatcmpl-hosted-stream-1", "object": "chat.completion.chunk", "model": "fixture-model",
					"choices": []any{}, "usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
				}) {
				return
			}
		} else {
			if !writeFrame(chunk(map[string]any{"role": "assistant", "content": "It is sunny in Singapore."}, nil)) ||
				!writeFrame(chunk(map[string]any{}, "stop")) ||
				!writeFrame(map[string]any{
					"id": "chatcmpl-hosted-stream-2", "object": "chat.completion.chunk", "model": "fixture-model",
					"choices": []any{}, "usage": map[string]any{"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
				}) {
				return
			}
		}
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
		providerErrors <- nil
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("hosted-stream-controller-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: webExecutor},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"check the weather",
			"tools":[
				{"type":"web_search","filters":{"allowed_domains":["weather.example"]}},
				{"type":"function","name":"disallowed_stream_decoy","parameters":{"type":"object"}}
			],
			"tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":"web_search"}]}
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	var eventTypes []string
	var wire strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		eventTypes = append(eventTypes, event.Type)
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.NoError(t, <-providerErrors)
	require.NoError(t, <-providerErrors)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, searchCalls.Load())
	encoded := wire.String()
	require.Contains(t, encoded, `"type":"web_search_call"`)
	require.Contains(t, encoded, `"query":"weather Singapore"`)
	require.Contains(t, encoded, "https://weather.example/singapore")
	require.Contains(t, encoded, "It is sunny in Singapore.")
	require.Contains(t, encoded, `"total_tokens":18`)
	require.NotContains(t, encoded, syntheticName)
	require.NotContains(t, encoded, searchSecret)
	require.NotContains(t, encoded, "response.function_call_arguments")
	require.Equal(t, 1, countString(eventTypes, "response.created"))
	require.Equal(t, 1, countString(eventTypes, "response.completed"))
}

func TestControllerTurnsInterruptedHostedProviderStreamIntoResponsesIncompleteLifecycle(t *testing.T) {
	t.Parallel()
	var hostedCalls atomic.Int64
	executor := &neverCalledHostedExecutor{calls: &hostedCalls}
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if json.Unmarshal(body, &payload) != nil || !payload.Stream || len(payload.Tools) != 1 ||
			!strings.HasPrefix(payload.Tools[0].Function.Name, "axh_") {
			http.Error(writer, "request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher := writer.(http.Flusher)
		writeFrame := func(value any) {
			encoded, _ := json.Marshal(value)
			_, _ = fmt.Fprintf(writer, "data: %s\n\n", encoded)
			flusher.Flush()
		}
		chunk := func(delta map[string]any) map[string]any {
			return map[string]any{
				"id": "chatcmpl-interrupted-hosted", "object": "chat.completion.chunk",
				"created": 1785381200, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": nil}},
			}
		}
		writeFrame(chunk(map[string]any{"role": "assistant", "content": "partial visible text"}))
		writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "interrupted_search_call", "type": "function",
			"function": map[string]any{"name": payload.Tools[0].Function.Name, "arguments": `{"query":"weather`},
		}}}))
		// Deliberately close without a finish_reason, terminal chunk, or [DONE].
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("interrupted-hosted-stream-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: executor},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	httpExecutor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(httpExecutor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(httpExecutor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"check the weather",
			"tools":[{"type":"web_search"}]
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	var eventTypes []string
	var wire strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		eventTypes = append(eventTypes, event.Type)
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.Zero(t, hostedCalls.Load())
	encoded := wire.String()
	require.Contains(t, encoded, "partial visible text")
	require.Contains(t, encoded, `"type":"web_search_call"`)
	require.Contains(t, encoded, `"call_id":"interrupted_search_call"`)
	require.Equal(t, 1, countString(eventTypes, "response.incomplete"))
	require.Zero(t, countString(eventTypes, "response.failed"))
	require.Zero(t, countString(eventTypes, "response.completed"))
	require.NotContains(t, encoded, "axh_")
}

func TestClosingResponsesStreamCancelsInFlightHostedExecutor(t *testing.T) {
	t.Parallel()
	started := make(chan struct{})
	cancelled := make(chan struct{})
	executor := &blockingHostedExecutor{started: started, cancelled: cancelled}
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Stream bool `json:"stream"`
			Tools  []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if json.Unmarshal(body, &payload) != nil || !payload.Stream || len(payload.Tools) != 1 {
			http.Error(writer, "request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		flusher := writer.(http.Flusher)
		writeFrame := func(value any) {
			encoded, _ := json.Marshal(value)
			_, _ = fmt.Fprintf(writer, "data: %s\n\n", encoded)
			flusher.Flush()
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": "chatcmpl-cancel-hosted", "object": "chat.completion.chunk",
				"created": 1785381300, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		writeFrame(chunk(map[string]any{"role": "assistant"}, nil))
		writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "cancelled_search_call", "type": "function",
			"function": map[string]any{"name": payload.Tools[0].Function.Name, "arguments": `{"query":"weather"}`},
		}}}, nil))
		writeFrame(chunk(map[string]any{}, "tool_calls"))
		writeFrame(map[string]any{
			"id": "chatcmpl-cancel-hosted", "object": "chat.completion.chunk", "model": "fixture-model",
			"choices": []any{}, "usage": map[string]any{"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
		})
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("cancel-hosted-stream-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: executor},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	httpExecutor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(httpExecutor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(httpExecutor).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound), pipeline.WithToolLoopController(controller),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"check the weather",
			"tools":[{"type":"web_search"}]
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	closeResult := make(chan error, 1)
	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				closeResult <- fmt.Errorf("stream close goroutine panicked: %v", recovered)
			}
		}()
		<-started
		closeResult <- result.EventStream.Close()
	}()
	for result.EventStream.Next() {
		// Current advances the source encoder's queued-event cursor. Pull until
		// Close interrupts the in-flight executor and the stream ends.
		_ = result.EventStream.Current()
	}
	require.NoError(t, <-closeResult)
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("hosted executor did not observe stream cancellation")
	}
}

type neverCalledHostedExecutor struct {
	calls *atomic.Int64
}

type blockingHostedExecutor struct {
	started   chan<- struct{}
	cancelled chan<- struct{}
}

func (*blockingHostedExecutor) Kind() llm.ToolKind { return llm.ToolKindWebSearch }

func (*blockingHostedExecutor) Function(llm.ToolDefinition) (llm.FunctionDefinition, error) {
	return llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)}, nil
}

func (executor *blockingHostedExecutor) Execute(ctx context.Context, _ llm.ToolDefinition, _ llm.ToolInvocation) (*llm.ToolResult, error) {
	close(executor.started)
	<-ctx.Done()
	close(executor.cancelled)
	return nil, ctx.Err()
}

func (*neverCalledHostedExecutor) Kind() llm.ToolKind { return llm.ToolKindWebSearch }

func (*neverCalledHostedExecutor) Function(llm.ToolDefinition) (llm.FunctionDefinition, error) {
	return llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)}, nil
}

func (executor *neverCalledHostedExecutor) Execute(context.Context, llm.ToolDefinition, llm.ToolInvocation) (*llm.ToolResult, error) {
	executor.calls.Add(1)
	return nil, fmt.Errorf("interrupted hosted executor must not run")
}

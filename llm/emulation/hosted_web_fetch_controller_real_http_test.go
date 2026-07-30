package emulation_test

import (
	"bytes"
	"context"
	"encoding/json"
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
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

func TestControllerRunsAnthropicWebFetchThroughChatTargetOverRealHTTP(t *testing.T) {
	t.Parallel()
	observer := &controllerObservationRecorder{}
	const fetchSecret = "private-fetch-secret"
	var fetchCalls atomic.Int64
	fetchServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		fetchCalls.Add(1)
		if request.Method != http.MethodPost || request.Header.Get("X-Fetch-Key") != fetchSecret {
			http.Error(writer, "unauthorized", http.StatusUnauthorized)
			return
		}
		var input hosted.WebFetchInput
		if err := json.NewDecoder(request.Body).Decode(&input); err != nil || input.URL != "https://docs.example/axon" ||
			len(input.AllowedDomains) != 1 || input.AllowedDomains[0] != "docs.example" ||
			input.MaxContentTokens == nil || *input.MaxContentTokens != 2048 {
			http.Error(writer, "bad fetch input", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"url":"https://docs.example/axon","title":"Axon docs","text":"canonical fetch body"}`)
	}))
	t.Cleanup(fetchServer.Close)
	allowedFetch, err := url.Parse(fetchServer.URL)
	require.NoError(t, err)
	fetchExecutor, err := hosted.NewWebFetchExecutor(hosted.WebFetchExecutorConfig{
		Endpoint: fetchServer.URL, Client: fetchServer.Client(), Headers: http.Header{"X-Fetch-Key": []string{fetchSecret}},
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedFetch.Scheme || candidate.Host != allowedFetch.Host {
				return fmt.Errorf("fetch endpoint is not allowlisted")
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
		if readErr != nil || bytes.Contains(body, []byte(fetchSecret)) {
			http.Error(writer, "invalid provider request", http.StatusBadRequest)
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
			if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, "fetch") {
				http.Error(writer, "hosted name is not opaque", http.StatusBadRequest)
				return
			}
			writer.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(writer, `{"id":"chat-fetch-1","object":"chat.completion","created":1785381000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"fetch_call_1","type":"function","function":{"name":%q,"arguments":"{\"url\":\"https://docs.example/axon\"}"}}]},"finish_reason":"tool_calls"}]}`, syntheticName)
			return
		}
		foundResult := false
		for _, message := range payload.Messages {
			if message.Role == "tool" && message.ToolCallID == "fetch_call_1" {
				encoded, _ := json.Marshal(message.Content)
				foundResult = strings.Contains(string(encoded), "canonical fetch body")
			}
		}
		if !foundResult {
			http.Error(writer, "fetch result missing", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"chat-fetch-2","object":"chat.completion","created":1785381001,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"The document was fetched."},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("hosted-controller-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebFetch: fetchExecutor},
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
		anthropic.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","max_tokens":256,"messages":[{"role":"user","content":"fetch the Axon docs"}],
			"tools":[{"type":"web_fetch_20260318","name":"web_fetch","allowed_domains":["docs.example"],"max_content_tokens":2048}]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 1, fetchCalls.Load())
	require.EqualValues(t, 1, hostedExecutionCount(observer.emulation(), llm.ToolKindWebFetch))
	require.EqualValues(t, 1, totalHostedExecutions(observer.emulation()))
	require.NotContains(t, string(result.Response.Body), syntheticName)
	require.NotContains(t, string(result.Response.Body), fetchSecret)
	var responseBody struct {
		Content []json.RawMessage `json:"content"`
		Usage   struct {
			ServerToolUse struct {
				WebFetchRequests int64 `json:"web_fetch_requests"`
			} `json:"server_tool_use"`
		} `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), string(result.Response.Body))
	require.Len(t, responseBody.Content, 3)
	require.JSONEq(t, `{"type":"server_tool_use","id":"fetch_call_1","name":"web_fetch","input":{"url":"https://docs.example/axon"}}`, string(responseBody.Content[0]))
	require.Contains(t, string(responseBody.Content[1]), `"type":"web_fetch_tool_result"`)
	require.Contains(t, string(responseBody.Content[1]), `"canonical fetch body"`)
	require.JSONEq(t, `{"type":"text","text":"The document was fetched."}`, string(responseBody.Content[2]))
	require.EqualValues(t, 1, responseBody.Usage.ServerToolUse.WebFetchRequests)
}

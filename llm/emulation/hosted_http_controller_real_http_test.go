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
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestControllerRunsResponsesHostedHTTPFamiliesThroughChatOverRealHTTP(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name          string
		kind          llm.ToolKind
		tool          string
		arguments     string
		result        string
		executorBody  string
		wireTypes     []string
		resultNeedle  string
		wireNeedles   []string
		providerOwned bool
	}{
		{
			name: "file_search", kind: llm.ToolKindFileSearch,
			tool:      `{"type":"file_search","vector_store_ids":["vs_1"]}`,
			arguments: `{"queries":["canonical conversion"]}`,
			result:    `[{"file_id":"file_1","filename":"design.md","score":0.9,"text":"FILE_MARKER"}]`,
			wireTypes: []string{"file_search_call", "message"}, resultNeedle: "FILE_MARKER",
		},
		{
			name: "code_interpreter", kind: llm.ToolKindCodeInterpreter,
			tool:      `{"type":"code_interpreter","container":{"type":"auto"}}`,
			arguments: `{"code":"print(42)"}`,
			result:    `[{"type":"logs","logs":"CODE_MARKER\\n"}]`,
			wireTypes: []string{"code_interpreter_call", "message"}, resultNeedle: "CODE_MARKER",
		},
		{
			name: "shell", kind: llm.ToolKindShell,
			tool:      `{"type":"shell"}`,
			arguments: `{"commands":["printf SHELL_MARKER"],"timeout_ms":30000}`,
			result:    `[{"stdout":"SHELL_MARKER","stderr":"","outcome":{"type":"exit","exit_code":0}}]`,
			wireTypes: []string{"shell_call", "shell_call_output", "message"}, resultNeedle: "SHELL_MARKER",
		},
		{
			name: "image_generation", kind: llm.ToolKindImageGeneration,
			tool:      `{"type":"image_generation","output_format":"png"}`,
			arguments: `{"prompt":"draw marker","action":"generate"}`,
			result:    `{"data_base64":"SU1BR0VfTUFSS0VS","output_format":"png"}`,
			wireTypes: []string{"image_generation_call", "message"}, resultNeedle: "SU1BR0VfTUFSS0VS",
		},
		{
			name: "computer", kind: llm.ToolKindComputer,
			tool:      `{"type":"computer_use_preview","display_width":1024,"display_height":768,"environment":"browser"}`,
			arguments: `{"type":"click","x":12,"y":34,"button":"left"}`,
			executorBody: `{"status":"completed","result":{"type":"computer_screenshot","image_url":"data:image/png;base64,Q09NUFVURVJfTUFSS0VS"},` +
				`"pending_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}],` +
				`"acknowledged_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}]}`,
			wireTypes: []string{"computer_call", "computer_call_output", "message"}, resultNeedle: "Q09NUFVURVJfTUFSS0VS",
			wireNeedles:   []string{`"pending_safety_checks":[{"id":"safe_1"`, `"acknowledged_safety_checks":[{"id":"safe_1"`},
			providerOwned: true,
		},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			observer := &controllerObservationRecorder{}
			const executorSecret = "hosted-http-controller-secret"
			var executorCalls atomic.Int64
			executorServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				executorCalls.Add(1)
				if request.Method != http.MethodPost || request.Header.Get("X-Executor-Key") != executorSecret {
					http.Error(writer, "unauthorized", http.StatusUnauthorized)
					return
				}
				var input hosted.HTTPExecutorRequest
				if err := json.NewDecoder(request.Body).Decode(&input); err != nil || input.Kind != test.kind ||
					input.LogicalName != test.name || string(input.Arguments) != test.arguments {
					http.Error(writer, "bad executor input", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				if test.executorBody != "" {
					_, _ = io.WriteString(writer, test.executorBody)
					return
				}
				_, _ = fmt.Fprintf(writer, `{"result":%s}`, test.result)
			}))
			t.Cleanup(executorServer.Close)
			allowedEndpoint, err := url.Parse(executorServer.URL)
			require.NoError(t, err)
			httpExecutor, err := hosted.NewHTTPExecutor(hosted.HTTPExecutorConfig{
				Kind: test.kind, Endpoint: executorServer.URL, Client: executorServer.Client(),
				Headers: http.Header{"X-Executor-Key": []string{executorSecret}},
				EndpointPolicy: func(candidate *url.URL) error {
					if candidate.Scheme != allowedEndpoint.Scheme || candidate.Host != allowedEndpoint.Host {
						return fmt.Errorf("executor endpoint is not allowlisted")
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
				if readErr != nil || bytes.Contains(body, []byte(executorSecret)) || bytes.Contains(body, []byte(test.resultNeedle)) && round == 1 {
					http.Error(writer, "executor private state leaked", http.StatusBadRequest)
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
					http.Error(writer, "decode provider request", http.StatusBadRequest)
					return
				}
				if round == 1 {
					syntheticName = payload.Tools[0].Function.Name
					if !strings.HasPrefix(syntheticName, "axh_") || strings.Contains(syntheticName, test.name) {
						http.Error(writer, "hosted name is not opaque", http.StatusBadRequest)
						return
					}
					writer.Header().Set("Content-Type", "application/json")
					_, _ = fmt.Fprintf(writer, `{"id":"chat-hosted-1","object":"chat.completion","created":1785381000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"hosted_call_1","type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}]}`, syntheticName, test.arguments)
					return
				}
				foundResult := false
				for _, message := range payload.Messages {
					if message.Role == "tool" && message.ToolCallID == "hosted_call_1" {
						encoded, _ := json.Marshal(message.Content)
						foundResult = strings.Contains(string(encoded), test.resultNeedle)
					}
				}
				if !foundResult {
					http.Error(writer, "hosted result missing", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"id":"chat-hosted-2","object":"chat.completion","created":1785381001,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"hosted complete"},"finish_reason":"stop"}]}`)
			}))
			t.Cleanup(provider.Close)

			controller, err := emulation.NewController(emulation.ControllerConfig{
				Hosted: hosted.Config{
					SyntheticNameKey: []byte(strings.Repeat("hosted-http-controller-key-", 2)),
					Executors:        map[llm.ToolKind]hosted.Executor{test.kind: httpExecutor},
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
			requestBody := []byte(fmt.Sprintf(`{"model":"fixture-model","input":"run hosted tool","tools":[%s,{"type":"function","name":"disallowed_decoy","parameters":{"type":"object"}}],"tool_choice":{"type":"allowed_tools","mode":"required","tools":[{"type":%q}]}}`, test.tool, test.name))
			decoded, err := responses.NewInboundTransformer().TransformRequest(ctx, &httpclient.Request{
				Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
			})
			require.NoError(t, err)
			plan, err := controller.Preflight(decoded, llm.APIFormatOpenAIChatCompletion)
			require.NoError(t, err)
			require.True(t, plan.Complete(), "%#v", plan)
			options := []pipeline.Option{pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer)}
			if test.providerOwned {
				options = append(options, pipeline.WithMiddlewares(pipeline.OnLlmRequest("canonical-provider-owned-computer", func(_ context.Context, request *llm.Request) (*llm.Request, error) {
					for index := range request.ToolDefinitions {
						if request.ToolDefinitions[index].Kind == test.kind {
							request.ToolDefinitions[index].Execution = llm.ExecutionOwnerProvider
						}
					}
					return request, nil
				})))
			}
			result, err := pipeline.NewFactory(executor).Pipeline(
				responses.NewInboundTransformer(), conversion.NewOutbound(outbound), options...,
			).Process(ctx, &httpclient.Request{
				Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
				Body: requestBody,
			})
			require.NoError(t, err)
			require.EqualValues(t, 2, providerRounds.Load())
			require.EqualValues(t, 1, executorCalls.Load())
			emulationSummary := observer.emulation()
			require.NotNil(t, emulationSummary)
			require.EqualValues(t, 1, hostedExecutionCount(emulationSummary, test.kind), "%#v", emulationSummary)
			require.EqualValues(t, 1, totalHostedExecutions(emulationSummary), "%#v", emulationSummary)
			wire := string(result.Response.Body)
			require.NotContains(t, wire, syntheticName)
			require.NotContains(t, wire, executorSecret)
			var responseBody struct {
				Output []struct {
					Type string `json:"type"`
				} `json:"output"`
			}
			require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
			require.Len(t, responseBody.Output, len(test.wireTypes))
			for index, want := range test.wireTypes {
				require.Equal(t, want, responseBody.Output[index].Type, wire)
			}
			require.Contains(t, wire, test.resultNeedle)
			for _, needle := range test.wireNeedles {
				require.Contains(t, wire, needle)
			}
		})
	}
}

func TestHostedExecutionInfrastructureFailureStopsBeforeAnotherProviderRoundOverRealHTTP(t *testing.T) {
	t.Parallel()
	var executorCalls atomic.Int64
	executorServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		executorCalls.Add(1)
		http.Error(writer, "private executor diagnostic", http.StatusServiceUnavailable)
	}))
	t.Cleanup(executorServer.Close)
	allowedEndpoint, err := url.Parse(executorServer.URL)
	require.NoError(t, err)
	codeExecutor, err := hosted.NewHTTPExecutor(hosted.HTTPExecutorConfig{
		Kind: llm.ToolKindCodeInterpreter, Endpoint: executorServer.URL, Client: executorServer.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedEndpoint.Scheme || candidate.Host != allowedEndpoint.Host {
				return fmt.Errorf("executor endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil || bytes.Contains(body, []byte("private executor diagnostic")) {
			http.Error(writer, "private executor response leaked", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Type != "function" {
			http.Error(writer, "decode provider request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = fmt.Fprintf(writer, `{"id":"chat-hosted-failure-1","object":"chat.completion","created":1785381000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"hosted_failure_1","type":"function","function":{"name":%q,"arguments":"{\"code\":\"panic()\"}"}}]},"finish_reason":"tool_calls"}]}`, payload.Tools[0].Function.Name)
			return
		}
		http.Error(writer, "local executor failure must not trigger another provider round", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("hosted-failure-controller-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindCodeInterpreter: codeExecutor},
		},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(provider.URL, "provider-key")
	require.NoError(t, err)
	client := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(client.CloseIdleConnections)
	observer := &controllerObservationRecorder{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err = pipeline.NewFactory(client).Pipeline(
		responses.NewInboundTransformer(), conversion.NewOutbound(outbound),
		pipeline.WithToolLoopController(controller), pipeline.WithObserver(observer),
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","input":"run failing code","tools":[{"type":"code_interpreter","container":{"type":"auto"}}],"tool_choice":"required"}`),
	})
	require.Error(t, err)
	require.Equal(t, "gateway tool loop failed: hosted tool execution failed", err.Error())
	require.NotContains(t, err.Error(), "private executor diagnostic")
	require.EqualValues(t, 1, providerRounds.Load())
	require.EqualValues(t, 1, executorCalls.Load())
	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 1, summary.Failures)
	require.EqualValues(t, 1, summary.HostedCodeCalls)
	require.EqualValues(t, 1, totalHostedExecutions(summary))
}

func TestControllerDoesNotInterceptResponsesClientOwnedComputerOverRealHTTP(t *testing.T) {
	t.Parallel()
	observer := &controllerObservationRecorder{}
	var executorCalls atomic.Int64
	executorServer := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		executorCalls.Add(1)
		http.Error(writer, "client-owned computer must not reach hosted executor", http.StatusInternalServerError)
	}))
	t.Cleanup(executorServer.Close)
	allowedEndpoint, err := url.Parse(executorServer.URL)
	require.NoError(t, err)
	computerExecutor, err := hosted.NewHTTPExecutor(hosted.HTTPExecutorConfig{
		Kind: llm.ToolKindComputer, Endpoint: executorServer.URL, Client: executorServer.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedEndpoint.Scheme || candidate.Host != allowedEndpoint.Host {
				return fmt.Errorf("executor endpoint is not allowlisted")
			}
			return nil
		},
	})
	require.NoError(t, err)

	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		providerRounds.Add(1)
		var payload struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if json.NewDecoder(request.Body).Decode(&payload) != nil || len(payload.Tools) != 1 ||
			payload.Tools[0].Type != "function" || !strings.HasPrefix(payload.Tools[0].Function.Name, "axc_") ||
			strings.HasPrefix(payload.Tools[0].Function.Name, "axh_") {
			http.Error(writer, "client computer was not lowered reversibly", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(writer, `{"id":"chat-client-computer","object":"chat.completion","created":1785381002,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"computer_client_1","type":"function","function":{"name":%q,"arguments":"{\"type\":\"click\",\"x\":12,\"y\":34,\"button\":\"left\"}"}}]},"finish_reason":"tool_calls"}]}`, payload.Tools[0].Function.Name)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted: hosted.Config{
			SyntheticNameKey: []byte(strings.Repeat("hosted-client-computer-key-", 2)),
			Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindComputer: computerExecutor},
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
		Body: []byte(`{"model":"fixture-model","input":"click","tools":[{"type":"computer_use_preview","display_width":1024,"display_height":768,"environment":"browser"}],"tool_choice":"required"}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, providerRounds.Load())
	require.Zero(t, executorCalls.Load())
	require.Zero(t, totalHostedExecutions(observer.emulation()))
	var responseBody struct {
		Output []struct {
			Type   string          `json:"type"`
			CallID string          `json:"call_id"`
			Action json.RawMessage `json:"action"`
		} `json:"output"`
	}
	wire := string(result.Response.Body)
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), wire)
	require.Len(t, responseBody.Output, 1, wire)
	require.Equal(t, "computer_call", responseBody.Output[0].Type, wire)
	require.Equal(t, "computer_client_1", responseBody.Output[0].CallID, wire)
	require.JSONEq(t, `{"type":"click","x":12,"y":34,"button":"left"}`, string(responseBody.Output[0].Action))
}

func hostedExecutionCount(summary *llm.EmulationTraceSummary, kind llm.ToolKind) uint32 {
	if summary == nil {
		return 0
	}
	switch kind {
	case llm.ToolKindWebSearch:
		return summary.HostedWebSearchCalls
	case llm.ToolKindWebFetch:
		return summary.HostedWebFetchCalls
	case llm.ToolKindFileSearch:
		return summary.HostedFileSearchCalls
	case llm.ToolKindCodeInterpreter, llm.ToolKindCodeExecution:
		return summary.HostedCodeCalls
	case llm.ToolKindShell, llm.ToolKindLocalShell:
		return summary.HostedShellCalls
	case llm.ToolKindComputer:
		return summary.HostedComputerCalls
	case llm.ToolKindImageGeneration:
		return summary.HostedImageCalls
	case llm.ToolKindToolSearch:
		return summary.HostedToolSearchCalls
	default:
		return summary.HostedOtherCalls
	}
}

func totalHostedExecutions(summary *llm.EmulationTraceSummary) uint32 {
	if summary == nil {
		return 0
	}
	return summary.HostedWebSearchCalls + summary.HostedWebFetchCalls + summary.HostedFileSearchCalls +
		summary.HostedCodeCalls + summary.HostedShellCalls + summary.HostedComputerCalls +
		summary.HostedImageCalls + summary.HostedToolSearchCalls + summary.HostedOtherCalls
}

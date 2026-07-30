package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

const (
	originalCustomName  = "apply_patch"
	originalCustomInput = "*** Begin Patch\n*** End Patch\n"
	continuedCallID     = "call_patch_002"
)

func TestResponsesCustomToolRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		providerPath  string
		newOutbound   func(baseURL string) (transformer.Outbound, error)
		serveProvider func(t *testing.T, writer http.ResponseWriter, request *http.Request)
	}{
		{
			name:         "chat_completions",
			providerPath: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
			serveProvider: serveChatCustomFixture,
		},
		{
			name:         "anthropic_messages",
			providerPath: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
			serveProvider: serveAnthropicCustomFixture,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != tt.providerPath {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				tt.serveProvider(t, writer, request)
			}))
			t.Cleanup(provider.Close)

			target, err := tt.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target outbound: %v", err)
			}
			outbound := conversion.NewOutbound(target)
			observer := &observationRecorder{}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)

			ctx := llm.WithConversionDebugTrace(context.Background(), []byte(t.Name()), 2)
			result, err := pipeline.NewFactory(executor).
				Pipeline(
					responses.NewInboundTransformer(),
					outbound,
					pipeline.WithObserver(observer),
				).
				Process(ctx, responsesFixtureRequest(t))
			if err != nil {
				t.Fatalf("run real protocol pipeline: %v", err)
			}
			if result == nil || result.Stream || result.Response == nil {
				t.Fatalf("unexpected pipeline result: %#v", result)
			}

			assertResponsesCustomCall(t, result.Response.Body)
			assertConversionObservation(t, observer.snapshot(), target.APIFormat())
			assertConversionDebugObservation(t, observer.snapshot())
		})
	}
}

func TestResponsesCustomToolRepairsMalformedChatArgumentsOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		var body struct {
			Tools []struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tools"`
		}
		if request.URL.Path != "/v1/chat/completions" || json.NewDecoder(request.Body).Decode(&body) != nil ||
			len(body.Tools) != 1 || body.Tools[0].Function.Name == "" {
			http.Error(writer, "invalid lowered custom request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"id": "chatcmpl_repair", "object": "chat.completion", "model": "fixture-model",
			"choices": []any{map[string]any{
				"index": 0,
				"message": map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
					"id": continuedCallID, "type": "function",
					"function": map[string]any{"name": body.Tools[0].Function.Name, "arguments": `{"input":"repaired over tcp",}`},
				}}},
				"finish_reason": "tool_calls",
			}},
		})
	}))
	t.Cleanup(provider.Close)
	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	observer := &observationRecorder{}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	ctx := llm.WithConversionDebugTrace(llm.WithConversionTrace(context.Background()), []byte(t.Name()), 16)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target), pipeline.WithObserver(observer)).
		Process(ctx, responsesFixtureRequest(t))
	if err != nil {
		t.Fatalf("run malformed custom arguments over TCP: %v", err)
	}
	var response struct {
		Output []struct {
			Type  string `json:"type"`
			Input string `json:"input"`
		} `json:"output"`
	}
	if result == nil || result.Response == nil || json.Unmarshal(result.Response.Body, &response) != nil ||
		len(response.Output) != 1 || response.Output[0].Type != "custom_tool_call" || response.Output[0].Input != "repaired over tcp" {
		t.Fatalf("repaired Responses body = %#v result=%#v", response, result)
	}
	var restore *llm.ConversionTraceSummary
	var debug *llm.ConversionDebugTrace
	for _, event := range observer.snapshot() {
		if event.Stage == pipeline.StageConversionRestore {
			restore = event.Conversion
			debug = event.ConversionDebug
		}
	}
	if restore == nil || restore.CustomInputsRepaired != 1 || restore.RestoreMiss != 0 || !restore.Complete {
		t.Fatalf("custom repair observation = %#v", restore)
	}
	foundRepair := false
	if debug != nil {
		for _, action := range debug.Actions {
			if action.Action == "repair" && action.Strategy == "custom_as_function" {
				foundRepair = true
			}
		}
	}
	if !foundRepair {
		t.Fatalf("custom repair debug action missing: %#v", debug)
	}
}

func TestResponsesReasoningToolHistoryReachesChatAndAnthropicOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		assertBody  func(*testing.T, []byte)
		response    string
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			assertBody: assertChatReasoningToolHistory,
			response:   `{"id":"chat_reasoning","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			assertBody: assertAnthropicReasoningToolHistory,
			response:   `{"id":"msg_reasoning","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":8,"output_tokens":2}}`,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				test.assertBody(t, body)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.response)
			}))
			t.Cleanup(provider.Close)
			outbound, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/responses",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model","max_output_tokens":64,
						"input":[
							{"type":"message","role":"user","content":[{"type":"input_text","text":"inspect"}]},
							{"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"tool reasoning"}]},
							{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"sku\":\"A-1\"}"},
							{"type":"function_call_output","call_id":"call_1","output":"7"}
						],
						"tools":[{"type":"function","name":"lookup","parameters":{"type":"object","properties":{"sku":{"type":"string"}}}}]
					}`),
				})
			if err != nil {
				t.Fatalf("run Responses reasoning conversion: %v", err)
			}
			if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
				t.Fatalf("client response = %#v", result)
			}
		})
	}
}

func TestResponsesAdditionalNamespaceToolsRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, *http.Request)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveChatAdditionalNamespaceTools,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveAnthropicAdditionalNamespaceTools,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				test.serve(t, writer, request)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/responses",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model",
						"input":[
							{"type":"additional_tools","tools":[
								{"type":"namespace","name":"collaboration","tools":[
									{"type":"function","name":"send_message","description":"Send","parameters":{"type":"object","properties":{"message":{"type":"string"}},"required":["message"]}}
								]},
								{"type":"namespace","name":"terminal","tools":[
									{"type":"custom","name":"exec","description":"Run a command"}
								]}
							]},
							{"type":"message","role":"user","content":[{"type":"input_text","text":"send then run"}]}
						]
					}`),
				})
			if err != nil {
				t.Fatalf("run additional_tools conversion: %v", err)
			}
			assertResponsesAdditionalNamespaceCalls(t, result.Response.Body)
		})
	}
}

func TestResponsesAdditionalToolsRetainNativeInputPlacementOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		var payload struct {
			Tools []json.RawMessage `json:"tools"`
			Input []struct {
				Type  string `json:"type"`
				Role  string `json:"role"`
				Tools []struct {
					Type  string `json:"type"`
					Name  string `json:"name"`
					Tools []struct {
						Type string `json:"type"`
						Name string `json:"name"`
					} `json:"tools"`
				} `json:"tools"`
			} `json:"input"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
			http.Error(writer, "decode Responses request", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 0 || len(payload.Input) != 2 || payload.Input[0].Type != "additional_tools" ||
			payload.Input[0].Role != "developer" || len(payload.Input[0].Tools) != 2 ||
			payload.Input[0].Tools[0].Type != "namespace" || payload.Input[0].Tools[0].Name != "collaboration" ||
			len(payload.Input[0].Tools[0].Tools) != 1 || payload.Input[0].Tools[0].Tools[0].Type != "function" ||
			payload.Input[0].Tools[0].Tools[0].Name != "send_message" ||
			payload.Input[0].Tools[1].Type != "namespace" || payload.Input[0].Tools[1].Name != "terminal" ||
			len(payload.Input[0].Tools[1].Tools) != 1 || payload.Input[0].Tools[1].Tools[0].Type != "custom" ||
			payload.Input[0].Tools[1].Tools[0].Name != "exec" || payload.Input[1].Type != "message" {
			http.Error(writer, fmt.Sprintf("native additional_tools placement degraded: %#v", payload), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_native_additional","object":"response","status":"completed","model":"fixture-model","output":[],"usage":{"input_tokens":2,"output_tokens":0,"total_tokens":2}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses target: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), responsesAdditionalNamespaceRequest(false))
	if err != nil {
		t.Fatalf("run native additional_tools route: %v", err)
	}
	if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "resp_native_additional") {
		t.Fatalf("native Responses result = %#v", result)
	}
}

func TestAnthropicPrivateCitationProjectsPublicFieldsOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		assertBody  func(*testing.T, []byte)
		response    string
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			assertBody: assertChatPublicCitation,
			response:   `{"id":"chat_citation","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`,
		},
		{
			name: "responses", path: "/v1/responses",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			assertBody: assertResponsesPublicCitation,
			response:   `{"id":"resp_citation","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_done","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}`,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				test.assertBody(t, body)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.response)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/messages",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model","max_tokens":64,
						"messages":[
							{"role":"user","content":"cite it"},
							{"role":"assistant","content":[{"type":"text","text":"documented fact","citations":[{
								"type":"web_search_result_location","url":"https://example.invalid/source","title":"Source title",
								"encrypted_index":"private-index","cited_text":"private cited text"
							}]}]}
						]
					}`),
				})
			if err != nil {
				t.Fatalf("run Anthropic citation conversion: %v", err)
			}
			if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
				t.Fatalf("citation client result = %#v", result)
			}
		})
	}
}

func TestAnthropicHostedWebSearchHistoryProjectsToResponsesOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read provider request", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type    string `json:"type"`
				Filters struct {
					AllowedDomains []string `json:"allowed_domains"`
				} `json:"filters"`
			} `json:"tools"`
			Input []struct {
				Type   string `json:"type"`
				ID     string `json:"id"`
				CallID string `json:"call_id"`
				Action struct {
					Type    string `json:"type"`
					Query   string `json:"query"`
					Sources []struct {
						URL   string `json:"url"`
						Title string `json:"title"`
					} `json:"sources"`
				} `json:"action"`
			} `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(writer, "decode Responses web-search history", http.StatusBadRequest)
			return
		}
		var hosted *struct {
			Type   string `json:"type"`
			ID     string `json:"id"`
			CallID string `json:"call_id"`
			Action struct {
				Type    string `json:"type"`
				Query   string `json:"query"`
				Sources []struct {
					URL   string `json:"url"`
					Title string `json:"title"`
				} `json:"sources"`
			} `json:"action"`
		}
		for index := range payload.Input {
			if payload.Input[index].Type == "web_search_call" {
				hosted = &payload.Input[index]
				break
			}
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Type != "web_search" ||
			len(payload.Tools[0].Filters.AllowedDomains) != 1 || payload.Tools[0].Filters.AllowedDomains[0] != "example.invalid" ||
			hosted == nil || hosted.ID != "call_web" || hosted.Action.Query != "documentation" ||
			len(hosted.Action.Sources) != 1 || hosted.Action.Sources[0].URL != "https://example.invalid/doc" ||
			hosted.Action.Sources[0].Title != "Documentation" || bytes.Contains(body, []byte("private-result")) ||
			bytes.Contains(body, []byte(`"caller"`)) {
			http.Error(writer, fmt.Sprintf("Responses hosted history degraded: %s", body), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_hosted","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_done","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":8,"output_tokens":1,"total_tokens":9}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses target: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{
				"model":"fixture-model","max_tokens":64,
				"messages":[
					{"role":"user","content":"research"},
					{"role":"assistant","content":[
						{"type":"server_tool_use","id":"call_web","name":"web_search","input":{"query":"documentation"},"caller":{"type":"direct"}},
						{"type":"web_search_tool_result","tool_use_id":"call_web","content":[
							{"type":"web_search_result","url":"https://example.invalid/doc","title":"Documentation","encrypted_content":"private-result"}
						]}
					]}
				],
				"tools":[{"type":"web_search_20250305","name":"web_search","allowed_domains":["example.invalid"]}]
			}`),
		})
	if err != nil {
		t.Fatalf("run hosted web-search history conversion: %v", err)
	}
	if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
		t.Fatalf("hosted web-search result = %#v", result)
	}
}

func TestResponsesHostedWebSearchHistoryProjectsToAnthropicOverRealHTTP(t *testing.T) {
	t.Parallel()
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/messages" {
			http.Error(writer, "unexpected provider path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read provider request", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type           string   `json:"type"`
				AllowedDomains []string `json:"allowed_domains"`
			} `json:"tools"`
			Messages []struct {
				Role    string `json:"role"`
				Content []struct {
					Type      string          `json:"type"`
					ID        string          `json:"id"`
					Name      string          `json:"name"`
					Input     json.RawMessage `json:"input"`
					ToolUseID string          `json:"tool_use_id"`
					Content   []struct {
						Type  string `json:"type"`
						URL   string `json:"url"`
						Title string `json:"title"`
					} `json:"content"`
				} `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(writer, "decode Anthropic web-search history", http.StatusBadRequest)
			return
		}
		if len(payload.Tools) != 1 || payload.Tools[0].Type != anthropic.ToolTypeWebSearch20250305 ||
			len(payload.Tools[0].AllowedDomains) != 1 || payload.Tools[0].AllowedDomains[0] != "example.invalid" ||
			len(payload.Messages) != 2 || payload.Messages[1].Role != "assistant" || len(payload.Messages[1].Content) != 2 ||
			payload.Messages[1].Content[0].Type != "server_tool_use" || payload.Messages[1].Content[0].ID != "ws_1" ||
			payload.Messages[1].Content[0].Name != "web_search" || !bytes.Contains(payload.Messages[1].Content[0].Input, []byte(`"query":"documentation"`)) ||
			payload.Messages[1].Content[1].Type != "web_search_tool_result" || payload.Messages[1].Content[1].ToolUseID != "ws_1" ||
			len(payload.Messages[1].Content[1].Content) != 1 || payload.Messages[1].Content[1].Content[0].Type != "web_search_result" ||
			payload.Messages[1].Content[1].Content[0].URL != "https://example.invalid/doc" ||
			payload.Messages[1].Content[1].Content[0].Title != "Documentation" {
			http.Error(writer, fmt.Sprintf("Anthropic hosted history degraded: %s", body), http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"msg_hosted","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":8,"output_tokens":1}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Anthropic target: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{
				"model":"fixture-model","max_output_tokens":64,
				"input":[
					{"type":"message","role":"user","content":[{"type":"input_text","text":"research"}]},
					{"type":"web_search_call","id":"ws_1","status":"completed","action":{
						"type":"search","query":"documentation","sources":[
							{"type":"url","url":"https://example.invalid/doc","title":"Documentation"}
						]
					}}
				],
				"tools":[{"type":"web_search","filters":{"allowed_domains":["example.invalid"]}}]
			}`),
		})
	if err != nil {
		t.Fatalf("run Responses hosted web-search conversion: %v", err)
	}
	if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
		t.Fatalf("hosted web-search result = %#v", result)
	}
}

func TestResponsesLocalShellRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, []byte)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveChatLocalShell,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveAnthropicLocalShell,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				test.serve(t, writer, body)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/responses",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model","max_output_tokens":64,
						"input":[
							{"type":"message","role":"user","content":[{"type":"input_text","text":"inspect files"}]},
							{"type":"local_shell_call","id":"lsh_old","call_id":"shell_old","status":"completed","action":{
								"type":"exec","command":["cat","README.md"],"working_directory":"/workspace"
							}},
							{"type":"function_call_output","call_id":"shell_old","output":"old contents"}
						],
						"tools":[{"type":"local_shell"}]
					}`),
				})
			if err != nil {
				t.Fatalf("run local shell conversion: %v", err)
			}
			var payload struct {
				Output []struct {
					Type   string `json:"type"`
					CallID string `json:"call_id"`
					Action struct {
						Type    string   `json:"type"`
						Command []string `json:"command"`
					} `json:"action"`
				} `json:"output"`
			}
			if result == nil || result.Response == nil || json.Unmarshal(result.Response.Body, &payload) != nil ||
				len(payload.Output) != 1 || payload.Output[0].Type != "local_shell_call" || payload.Output[0].CallID != "shell_next" ||
				payload.Output[0].Action.Type != "exec" || len(payload.Output[0].Action.Command) != 1 || payload.Output[0].Action.Command[0] != "pwd" ||
				bytes.Contains(result.Response.Body, []byte("axc_")) {
				t.Fatalf("Responses local shell restoration degraded: result=%#v body=%s", result, responseBody(result))
			}
		})
	}
}

func serveChatLocalShell(t *testing.T, writer http.ResponseWriter, body []byte) {
	t.Helper()
	var payload struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name       string         `json:"name"`
				Parameters map[string]any `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(writer, "decode Chat local shell", http.StatusBadRequest)
		return
	}
	if len(payload.Tools) != 1 || payload.Tools[0].Type != "function" || !strings.HasPrefix(payload.Tools[0].Function.Name, "axc_") ||
		payload.Tools[0].Function.Parameters["type"] != "object" || len(payload.Messages) != 3 ||
		len(payload.Messages[1].ToolCalls) != 1 || payload.Messages[1].ToolCalls[0].ID != "shell_old" ||
		payload.Messages[1].ToolCalls[0].Function.Name != payload.Tools[0].Function.Name ||
		!strings.Contains(payload.Messages[1].ToolCalls[0].Function.Arguments, `"command":["cat","README.md"]`) ||
		payload.Messages[2].Role != "tool" || payload.Messages[2].ToolCallID != "shell_old" || payload.Messages[2].Content != "old contents" {
		http.Error(writer, fmt.Sprintf("Chat local shell lowering degraded: %s", body), http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"chat_shell","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"shell_next","type":"function","function":{"name":%q,"arguments":"{\"type\":\"exec\",\"command\":[\"pwd\"]}"}}]},"finish_reason":"tool_calls"}]}`, payload.Tools[0].Function.Name)
}

func serveAnthropicLocalShell(t *testing.T, writer http.ResponseWriter, body []byte) {
	t.Helper()
	var payload struct {
		Tools []struct {
			Name        string         `json:"name"`
			InputSchema map[string]any `json:"input_schema"`
		} `json:"tools"`
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string          `json:"type"`
				ID        string          `json:"id"`
				Name      string          `json:"name"`
				Input     json.RawMessage `json:"input"`
				ToolUseID string          `json:"tool_use_id"`
				Content   string          `json:"content"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(writer, "decode Anthropic local shell", http.StatusBadRequest)
		return
	}
	if len(payload.Tools) != 1 || !strings.HasPrefix(payload.Tools[0].Name, "axc_") || payload.Tools[0].InputSchema["type"] != "object" ||
		len(payload.Messages) != 3 || len(payload.Messages[1].Content) != 1 || payload.Messages[1].Content[0].Type != "tool_use" ||
		payload.Messages[1].Content[0].ID != "shell_old" || payload.Messages[1].Content[0].Name != payload.Tools[0].Name ||
		!bytes.Contains(payload.Messages[1].Content[0].Input, []byte(`"command":["cat","README.md"]`)) ||
		len(payload.Messages[2].Content) != 1 || payload.Messages[2].Content[0].Type != "tool_result" ||
		payload.Messages[2].Content[0].ToolUseID != "shell_old" || payload.Messages[2].Content[0].Content != "old contents" {
		http.Error(writer, fmt.Sprintf("Anthropic local shell lowering degraded: %s", body), http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"msg_shell","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"shell_next","name":%q,"input":{"type":"exec","command":["pwd"]}}],"stop_reason":"tool_use","usage":{"input_tokens":8,"output_tokens":4}}`, payload.Tools[0].Name)
}

func responseBody(result *pipeline.Result) []byte {
	if result == nil || result.Response == nil {
		return nil
	}
	return result.Response.Body
}

func TestResponsesClientToolSearchRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, []byte)
	}{
		{
			name: "chat", path: "/v1/chat/completions",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveChatToolSearch,
		},
		{
			name: "anthropic", path: "/v1/messages",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveAnthropicToolSearch,
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected provider path", http.StatusNotFound)
					return
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read provider request", http.StatusBadRequest)
					return
				}
				test.serve(t, writer, body)
			}))
			t.Cleanup(provider.Close)
			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create target: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: "/v1/responses",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body: []byte(`{
						"model":"fixture-model",
						"input":[
							{"type":"tool_search_call","id":"tsc_old","call_id":"search_old","status":"completed","execution":"client","arguments":{"query":"calendar"}},
							{"type":"tool_search_output","id":"tso_old","call_id":"search_old","status":"completed","execution":"client","tools":[
								{"type":"function","name":"create_event","description":"Create event","parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}
							]}
						],
						"tools":[{"type":"tool_search","execution":"client","description":"Search deferred tools","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]
					}`),
				})
			if err != nil {
				t.Fatalf("run client tool search conversion: %v", err)
			}
			var payload struct {
				Output []struct {
					Type      string          `json:"type"`
					CallID    string          `json:"call_id"`
					Execution string          `json:"execution"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"output"`
			}
			if result == nil || result.Response == nil || json.Unmarshal(result.Response.Body, &payload) != nil ||
				len(payload.Output) != 1 || payload.Output[0].Type != "tool_search_call" || payload.Output[0].CallID != "search_next" ||
				payload.Output[0].Execution != "client" || !bytes.Contains(payload.Output[0].Arguments, []byte(`"query":"mail"`)) ||
				bytes.Contains(result.Response.Body, []byte("axc_")) {
				t.Fatalf("Responses tool search restoration degraded: result=%#v body=%s", result, responseBody(result))
			}
		})
	}
}

func serveChatToolSearch(t *testing.T, writer http.ResponseWriter, body []byte) {
	t.Helper()
	var payload struct {
		Tools []struct {
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tools"`
		Messages []struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(writer, "decode Chat tool search", http.StatusBadRequest)
		return
	}
	synthetic := ""
	foundDiscovered := false
	for _, tool := range payload.Tools {
		if strings.HasPrefix(tool.Function.Name, "axc_") {
			synthetic = tool.Function.Name
		}
		if tool.Function.Name == "create_event" {
			foundDiscovered = true
		}
	}
	if synthetic == "" || !foundDiscovered || len(payload.Messages) != 2 || len(payload.Messages[0].ToolCalls) != 1 ||
		payload.Messages[0].ToolCalls[0].ID != "search_old" || payload.Messages[0].ToolCalls[0].Function.Name != synthetic ||
		payload.Messages[1].Role != "tool" || payload.Messages[1].ToolCallID != "search_old" {
		http.Error(writer, fmt.Sprintf("Chat tool search lowering degraded: %s", body), http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"chat_search","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"search_next","type":"function","function":{"name":%q,"arguments":"{\"query\":\"mail\"}"}}]},"finish_reason":"tool_calls"}]}`, synthetic)
}

func serveAnthropicToolSearch(t *testing.T, writer http.ResponseWriter, body []byte) {
	t.Helper()
	var payload struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				Name      string `json:"name"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		http.Error(writer, "decode Anthropic tool search", http.StatusBadRequest)
		return
	}
	synthetic := ""
	foundDiscovered := false
	for _, tool := range payload.Tools {
		if strings.HasPrefix(tool.Name, "axc_") {
			synthetic = tool.Name
		}
		if tool.Name == "create_event" {
			foundDiscovered = true
		}
	}
	if synthetic == "" || !foundDiscovered || len(payload.Messages) != 2 || len(payload.Messages[0].Content) != 1 ||
		payload.Messages[0].Content[0].Type != "tool_use" || payload.Messages[0].Content[0].ID != "search_old" ||
		payload.Messages[0].Content[0].Name != synthetic || len(payload.Messages[1].Content) != 1 ||
		payload.Messages[1].Content[0].Type != "tool_result" || payload.Messages[1].Content[0].ToolUseID != "search_old" {
		http.Error(writer, fmt.Sprintf("Anthropic tool search lowering degraded: %s", body), http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"msg_search","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"search_next","name":%q,"input":{"query":"mail"}}],"stop_reason":"tool_use","usage":{"input_tokens":8,"output_tokens":4}}`, synthetic)
}

func assertChatPublicCitation(t *testing.T, body []byte) {
	t.Helper()
	var payload struct {
		Messages []struct {
			Role        string `json:"role"`
			Annotations []struct {
				Type        string `json:"type"`
				URLCitation struct {
					URL   string `json:"url"`
					Title string `json:"title"`
				} `json:"url_citation"`
			} `json:"annotations"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Chat citation request: %v body=%s", err, body)
	}
	if len(payload.Messages) != 2 || payload.Messages[1].Role != "assistant" || len(payload.Messages[1].Annotations) != 1 ||
		payload.Messages[1].Annotations[0].Type != "url_citation" ||
		payload.Messages[1].Annotations[0].URLCitation.URL != "https://example.invalid/source" ||
		payload.Messages[1].Annotations[0].URLCitation.Title != "Source title" ||
		bytes.Contains(body, []byte("private-index")) || bytes.Contains(body, []byte("private cited text")) {
		t.Fatalf("Chat public citation projection degraded: %s", body)
	}
}

func assertResponsesPublicCitation(t *testing.T, body []byte) {
	t.Helper()
	var payload struct {
		Input []struct {
			Role    string `json:"role"`
			Content []struct {
				Text        string `json:"text"`
				Annotations []struct {
					Type        string `json:"type"`
					URLCitation struct {
						URL   string `json:"url"`
						Title string `json:"title"`
					} `json:"url_citation"`
				} `json:"annotations"`
			} `json:"content"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Responses citation request: %v body=%s", err, body)
	}
	if len(payload.Input) != 2 || payload.Input[1].Role != "assistant" || len(payload.Input[1].Content) != 1 ||
		len(payload.Input[1].Content[0].Annotations) != 1 ||
		payload.Input[1].Content[0].Annotations[0].Type != "url_citation" ||
		payload.Input[1].Content[0].Annotations[0].URLCitation.URL != "https://example.invalid/source" ||
		payload.Input[1].Content[0].Annotations[0].URLCitation.Title != "Source title" ||
		bytes.Contains(body, []byte("private-index")) || bytes.Contains(body, []byte("private cited text")) {
		t.Fatalf("Responses public citation projection degraded: %s", body)
	}
}

func responsesAdditionalNamespaceRequest(stream bool) *httpclient.Request {
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(fmt.Sprintf(`{
			"model":"fixture-model","stream":%t,
			"input":[
				{"type":"additional_tools","role":"developer","tools":[
					{"type":"namespace","name":"collaboration","tools":[{"type":"function","name":"send_message","parameters":{"type":"object","properties":{"message":{"type":"string"}}}}]},
					{"type":"namespace","name":"terminal","tools":[{"type":"custom","name":"exec"}]}
				]},
				{"type":"message","role":"user","content":[{"type":"input_text","text":"send then run"}]}
			]
		}`, stream)),
	}
}

type additionalNamespaceToolFixture struct {
	Name        string          `json:"name"`
	Schema      json.RawMessage `json:"-"`
	Parameters  json.RawMessage `json:"parameters"`
	InputSchema json.RawMessage `json:"input_schema"`
}

func serveChatAdditionalNamespaceTools(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	var body struct {
		Tools []struct {
			Function additionalNamespaceToolFixture `json:"function"`
		} `json:"tools"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		http.Error(writer, "decode Chat request", http.StatusBadRequest)
		return
	}
	tools := make([]additionalNamespaceToolFixture, 0, len(body.Tools))
	for _, tool := range body.Tools {
		tool.Function.Schema = tool.Function.Parameters
		tools = append(tools, tool.Function)
	}
	functionName, customName := classifyAdditionalNamespaceTools(t, tools)
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"chat_additional","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"call_send","type":"function","function":{"name":%q,"arguments":"{\"message\":\"ping\"}"}},{"id":"call_exec","type":"function","function":{"name":%q,"arguments":"{\"input\":\"pwd\"}"}}]},"finish_reason":"tool_calls"}]}`, functionName, customName)
}

func serveAnthropicAdditionalNamespaceTools(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	var body struct {
		Tools []additionalNamespaceToolFixture `json:"tools"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		http.Error(writer, "decode Anthropic request", http.StatusBadRequest)
		return
	}
	for index := range body.Tools {
		body.Tools[index].Schema = body.Tools[index].InputSchema
	}
	functionName, customName := classifyAdditionalNamespaceTools(t, body.Tools)
	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{"id":"msg_additional","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"call_send","name":%q,"input":{"message":"ping"}},{"type":"tool_use","id":"call_exec","name":%q,"input":{"input":"pwd"}}],"stop_reason":"tool_use","usage":{"input_tokens":8,"output_tokens":4}}`, functionName, customName)
}

func classifyAdditionalNamespaceTools(t *testing.T, tools []additionalNamespaceToolFixture) (string, string) {
	t.Helper()
	if len(tools) != 2 {
		t.Fatalf("provider tools = %#v, want two tools", tools)
	}
	var functionName, customName string
	for _, tool := range tools {
		var schema struct {
			Properties map[string]json.RawMessage `json:"properties"`
		}
		if err := json.Unmarshal(tool.Schema, &schema); err != nil {
			t.Fatalf("decode provider tool schema: %v schema=%s", err, tool.Schema)
		}
		if _, custom := schema.Properties["input"]; custom {
			customName = tool.Name
		} else {
			functionName = tool.Name
		}
	}
	if functionName != "collaboration__send_message" || customName == "" || customName == "terminal__exec" {
		t.Fatalf("provider tool identities function=%q custom=%q", functionName, customName)
	}
	return functionName, customName
}

func assertResponsesAdditionalNamespaceCalls(t *testing.T, body []byte) {
	t.Helper()
	var response struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
			Input     string `json:"input"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("decode Responses result: %v body=%s", err, body)
	}
	if len(response.Output) != 2 || response.Output[0].Type != "function_call" ||
		response.Output[0].Name != "send_message" || response.Output[0].Namespace != "collaboration" ||
		response.Output[1].Type != "custom_tool_call" || response.Output[1].Name != "terminal__exec" ||
		response.Output[1].Input != "pwd" {
		t.Fatalf("additional namespace calls degraded: %s", body)
	}
}

func assertChatReasoningToolHistory(t *testing.T, body []byte) {
	t.Helper()
	var request struct {
		Messages []struct {
			Role             string `json:"role"`
			ReasoningContent string `json:"reasoning_content"`
			ToolCallID       string `json:"tool_call_id"`
			ToolCalls        []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("decode Chat reasoning request: %v body=%s", err, body)
	}
	if len(request.Messages) != 3 || request.Messages[1].Role != "assistant" ||
		request.Messages[1].ReasoningContent != "tool reasoning" || len(request.Messages[1].ToolCalls) != 1 ||
		request.Messages[1].ToolCalls[0].ID != "call_1" || request.Messages[1].ToolCalls[0].Function.Name != "lookup" ||
		request.Messages[2].Role != "tool" || request.Messages[2].ToolCallID != "call_1" {
		t.Fatalf("Chat reasoning/tool history degraded: %s", body)
	}
}

func assertAnthropicReasoningToolHistory(t *testing.T, body []byte) {
	t.Helper()
	var request struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type      string `json:"type"`
				Thinking  string `json:"thinking"`
				ID        string `json:"id"`
				Name      string `json:"name"`
				ToolUseID string `json:"tool_use_id"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("decode Anthropic reasoning request: %v body=%s", err, body)
	}
	if len(request.Messages) != 3 || request.Messages[1].Role != "assistant" || len(request.Messages[1].Content) != 2 ||
		request.Messages[1].Content[0].Type != "thinking" || request.Messages[1].Content[0].Thinking != "tool reasoning" ||
		request.Messages[1].Content[1].Type != "tool_use" || request.Messages[1].Content[1].ID != "call_1" ||
		request.Messages[1].Content[1].Name != "lookup" || request.Messages[2].Role != "user" ||
		len(request.Messages[2].Content) != 1 || request.Messages[2].Content[0].Type != "tool_result" ||
		request.Messages[2].Content[0].ToolUseID != "call_1" {
		t.Fatalf("Anthropic reasoning/tool history degraded: %s", body)
	}
}

func responsesFixtureRequest(t *testing.T) *httpclient.Request {
	t.Helper()

	body, err := os.ReadFile("testdata/custom/responses-request.json")
	if err != nil {
		t.Fatalf("read Responses fixture: %v", err)
	}

	return &httpclient.Request{
		Method:  http.MethodPost,
		URL:     "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    body,
	}
}

func serveChatCustomFixture(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()

	var body struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name       string          `json:"name"`
				Parameters json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
		ToolChoice struct {
			Type     string `json:"type"`
			Function struct {
				Name string `json:"name"`
			} `json:"function"`
		} `json:"tool_choice"`
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.NewDecoder(request.Body).Decode(&body); err != nil {
		http.Error(writer, "invalid Chat fixture request", http.StatusBadRequest)
		return
	}
	if len(body.Tools) != 1 || body.Tools[0].Type != llm.ToolTypeFunction || body.Tools[0].Function.Name == "" ||
		body.Tools[0].Function.Name == originalCustomName {
		http.Error(writer, "custom definition was not reversibly lowered", http.StatusBadRequest)
		return
	}
	syntheticName := body.Tools[0].Function.Name
	if body.ToolChoice.Type != llm.ToolTypeFunction || body.ToolChoice.Function.Name != syntheticName {
		http.Error(writer, "custom tool choice was not lowered with its definition", http.StatusBadRequest)
		return
	}
	if !chatHistoryContainsCustomContinuation(body.Messages, syntheticName) {
		http.Error(writer, "custom call/result history was not lowered", http.StatusBadRequest)
		return
	}

	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{
  "id":"chatcmpl_fixture",
  "object":"chat.completion",
  "created":1765086001,
  "model":"conversion-fixture-model",
  "choices":[{
    "index":0,
    "message":{
      "role":"assistant",
      "tool_calls":[{
        "id":%q,
        "type":"function",
        "function":{"name":%q,"arguments":%q}
      }]
    },
    "finish_reason":"tool_calls"
  }],
  "usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
}`, continuedCallID, syntheticName, `{"input":"*** Begin Patch\n*** End Patch\n"}`)
}

func chatHistoryContainsCustomContinuation(messages []json.RawMessage, syntheticName string) bool {
	var hasCall, hasResult bool
	for _, raw := range messages {
		var message struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		}
		if json.Unmarshal(raw, &message) != nil {
			return false
		}
		if message.Role == "assistant" && len(message.ToolCalls) == 1 &&
			message.ToolCalls[0].ID == "call_patch_001" && message.ToolCalls[0].Function.Name == syntheticName {
			var arguments struct {
				Input string `json:"input"`
			}
			if json.Unmarshal([]byte(message.ToolCalls[0].Function.Arguments), &arguments) == nil && arguments.Input == originalCustomInput {
				hasCall = true
			}
		}
		if message.Role == "tool" && message.ToolCallID == "call_patch_001" {
			hasResult = true
		}
	}
	return hasCall && hasResult
}

func serveAnthropicCustomFixture(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	bodyBytes, err := io.ReadAll(request.Body)
	if err != nil {
		http.Error(writer, "failed to read Anthropic fixture request", http.StatusBadRequest)
		return
	}

	var body struct {
		Tools []struct {
			Name        string          `json:"name"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
		ToolChoice struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"tool_choice"`
		Messages []json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(bodyBytes, &body); err != nil {
		http.Error(writer, fmt.Sprintf("invalid Anthropic fixture request: %v: %s", err, bodyBytes), http.StatusBadRequest)
		return
	}
	if len(body.Tools) != 1 || body.Tools[0].Name == "" || body.Tools[0].Name == originalCustomName {
		http.Error(writer, "custom definition was not reversibly lowered", http.StatusBadRequest)
		return
	}
	syntheticName := body.Tools[0].Name
	if body.ToolChoice.Type != "tool" || body.ToolChoice.Name != syntheticName {
		http.Error(writer, "custom tool choice was not lowered with its definition", http.StatusBadRequest)
		return
	}
	if !anthropicHistoryContainsCustomContinuation(body.Messages, syntheticName) {
		http.Error(writer, "custom call/result history was not lowered", http.StatusBadRequest)
		return
	}

	writer.Header().Set("Content-Type", "application/json")
	_, _ = fmt.Fprintf(writer, `{
  "id":"msg_fixture",
  "type":"message",
  "role":"assistant",
  "model":"conversion-fixture-model",
  "content":[{
    "type":"tool_use",
    "id":%q,
    "name":%q,
    "input":{"input":%q}
  }],
  "stop_reason":"tool_use",
  "stop_sequence":null,
  "usage":{"input_tokens":10,"output_tokens":3}
}`, continuedCallID, syntheticName, originalCustomInput)
}

func anthropicHistoryContainsCustomContinuation(messages []json.RawMessage, syntheticName string) bool {
	var hasCall, hasResult bool
	for _, rawMessage := range messages {
		var message struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if json.Unmarshal(rawMessage, &message) != nil || len(message.Content) == 0 || message.Content[0] != '[' {
			continue
		}
		var content []json.RawMessage
		if json.Unmarshal(message.Content, &content) != nil {
			return false
		}
		for _, raw := range content {
			var block struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				Name      string `json:"name"`
				ToolUseID string `json:"tool_use_id"`
				Input     struct {
					Input string `json:"input"`
				} `json:"input"`
			}
			if json.Unmarshal(raw, &block) != nil {
				return false
			}
			if message.Role == "assistant" && block.Type == "tool_use" && block.ID == "call_patch_001" &&
				block.Name == syntheticName && block.Input.Input == originalCustomInput {
				hasCall = true
			}
			if message.Role == "user" && block.Type == "tool_result" && block.ToolUseID == "call_patch_001" {
				hasResult = true
			}
		}
	}
	return hasCall && hasResult
}

func assertResponsesCustomCall(t *testing.T, body []byte) {
	t.Helper()

	var response struct {
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
			Input  string `json:"input"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		t.Fatalf("decode Responses client body: %v\n%s", err, body)
	}
	if len(response.Output) != 1 {
		t.Fatalf("Responses output = %#v, want one custom call", response.Output)
	}
	call := response.Output[0]
	if call.Type != "custom_tool_call" || call.CallID != continuedCallID || call.Name != originalCustomName ||
		call.Input != originalCustomInput {
		t.Fatalf("restored custom call = %#v", call)
	}
}

type observationRecorder struct {
	mu     sync.Mutex
	events []pipeline.Observation
}

func (r *observationRecorder) Observe(_ context.Context, event pipeline.Observation) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *observationRecorder) snapshot() []pipeline.Observation {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]pipeline.Observation(nil), r.events...)
}

func assertConversionObservation(t *testing.T, events []pipeline.Observation, target llm.APIFormat) {
	t.Helper()

	for _, event := range events {
		if event.Stage != pipeline.StageConversionPlan {
			continue
		}
		if event.Conversion == nil {
			t.Fatal("conversion plan observation omitted its summary")
		}
		if event.Conversion.SourceFormat != llm.APIFormatOpenAIResponse || event.Conversion.TargetFormat != target ||
			event.Conversion.Lowered < 4 || !event.Conversion.Complete {
			t.Fatalf("conversion summary = %#v", event.Conversion)
		}
		encoded, err := json.Marshal(event)
		if err != nil {
			t.Fatalf("marshal conversion observation: %v", err)
		}
		for _, private := range []string{originalCustomName, originalCustomInput, "call_patch_001", "fixture-provider-key"} {
			if contains := string(encoded); len(private) > 0 && strings.Contains(contains, private) {
				t.Fatalf("conversion observation leaked private protocol data %q: %s", private, encoded)
			}
		}
		return
	}
	t.Fatal("conversion plan stage was not observed")
}

func assertConversionDebugObservation(t *testing.T, events []pipeline.Observation) {
	t.Helper()
	for _, event := range events {
		if event.Stage != pipeline.StageConversionPlan {
			continue
		}
		if event.ConversionDebug == nil || len(event.ConversionDebug.Actions) != 2 || event.ConversionDebug.Truncated == 0 {
			t.Fatalf("conversion debug trace was not bounded and propagated: %#v", event.ConversionDebug)
		}
		encoded, err := json.Marshal(event.ConversionDebug)
		if err != nil {
			t.Fatalf("marshal conversion debug trace: %v", err)
		}
		for _, private := range []string{originalCustomName, originalCustomInput, "call_patch_001", "fixture-provider-key"} {
			if strings.Contains(string(encoded), private) {
				t.Fatalf("conversion debug trace leaked private protocol data %q: %s", private, encoded)
			}
		}
		return
	}
	t.Fatal("conversion plan observation was not emitted")
}

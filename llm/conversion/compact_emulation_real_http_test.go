package conversion_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
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

func TestCompactConversionMatrixOverRealHTTP(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		path        string
		newOutbound func(string) (transformer.Outbound, error)
		serve       func(*testing.T, http.ResponseWriter, *http.Request)
		emulated    uint32
	}{
		{
			name: "chat", path: "/v1/chat/completions", emulated: 1,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveCompactChat,
		},
		{
			name: "anthropic", path: "/v1/messages", emulated: 1,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveCompactAnthropic,
		},
		{
			// Responses outbound also emulates compact as a normal /v1/responses
			// chat completion: live OpenAI-compatible gateways rarely implement
			// POST /v1/responses/compact, and native passthrough 404s.
			name: "responses_emulated", path: "/v1/responses", emulated: 1,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			serve: serveCompactResponsesEmulated,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != test.path {
					http.Error(writer, "unexpected path "+request.URL.Path, http.StatusNotFound)
					return
				}
				test.serve(t, writer, request)
			}))
			t.Cleanup(provider.Close)

			target, err := test.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create outbound: %v", err)
			}
			wrapped := conversion.NewOutbound(target)
			plan, err := wrapped.Preflight(compactLLMRequest("Preserve pending file edits."))
			if err != nil || plan.Summary.Emulated != test.emulated || !plan.Summary.Complete {
				t.Fatalf("compact preflight = %#v err=%v", plan, err)
			}

			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(responses.NewCompactInboundTransformer(), wrapped).
				Process(llm.WithConversionTrace(context.Background()), compactHTTPRequest("Preserve pending file edits."))
			if err != nil {
				t.Fatalf("run compact pipeline: %v", err)
			}
			assertCompactHTTPResponse(t, result, "Preserve pending file edits.", "COMPACTED_STATE")
		})
	}
}

func TestCompactEmulationSessionIsolationOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		var payload struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || len(payload.Messages) == 0 {
			http.Error(writer, "invalid compact emulation", http.StatusBadRequest)
			return
		}
		instruction := strings.TrimPrefix(payload.Messages[0].Content[strings.LastIndex(payload.Messages[0].Content, "instruction:")+len("instruction:"):], " ")
		writer.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(writer).Encode(map[string]any{
			"id": "chat_" + instruction, "object": "chat.completion", "created": 1770000000, "model": "fixture-model",
			"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": "state:" + instruction}, "finish_reason": "stop"}},
		})
	}))
	t.Cleanup(provider.Close)

	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create outbound: %v", err)
	}
	wrapped := conversion.NewOutbound(target)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)

	const workers = 24
	var wait sync.WaitGroup
	errors := make(chan error, workers)
	for index := 0; index < workers; index++ {
		index := index
		wait.Add(1)
		go func() {
			defer wait.Done()
			instruction := fmt.Sprintf("request-%02d", index)
			result, processErr := pipeline.NewFactory(executor).
				Pipeline(responses.NewCompactInboundTransformer(), wrapped).
				Process(context.Background(), compactHTTPRequest(instruction))
			if processErr != nil {
				errors <- processErr
				return
			}
			if verifyErr := checkCompactHTTPResponse(result, instruction, "state:"+instruction); verifyErr != nil {
				errors <- verifyErr
			}
		}()
	}
	wait.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
}

func TestCompactCustomToolHistoryLowersOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		var payload struct {
			Messages []struct {
				Role       string `json:"role"`
				ToolCallID string `json:"tool_call_id"`
				ToolCalls  []struct {
					ID       string `json:"id"`
					Function struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || len(payload.Messages) != 4 ||
			len(payload.Messages[2].ToolCalls) != 1 || payload.Messages[2].ToolCalls[0].ID != "call_patch" ||
			!strings.HasPrefix(payload.Messages[2].ToolCalls[0].Function.Name, "axc_") ||
			!strings.Contains(payload.Messages[2].ToolCalls[0].Function.Arguments, "*** Begin Patch") ||
			payload.Messages[3].Role != "tool" || payload.Messages[3].ToolCallID != "call_patch" {
			http.Error(writer, "custom compact history was not lowered", http.StatusBadRequest)
			return
		}
		writeCompactChatResponse(writer, "CUSTOM_STATE")
	}))
	t.Cleanup(provider.Close)

	target, err := openai.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewCompactInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), compactCustomHTTPRequest())
	if err != nil {
		t.Fatalf("run custom compact pipeline: %v", err)
	}
	assertCompactHTTPResponse(t, result, "Preserve tool state.", "CUSTOM_STATE")
}

func compactLLMRequest(instructions string) *llm.Request {
	return &llm.Request{
		Model:       "fixture-model",
		RequestType: llm.RequestTypeCompact,
		APIFormat:   llm.APIFormatOpenAIResponseCompact,
		Compact: &llm.CompactRequest{
			Instructions: instructions,
			Input:        []llm.Message{{Role: "user", Content: llm.MessageContent{Content: stringPointerForTest("Find ORBITAL_BEACON")}}},
		},
	}
}

func compactHTTPRequest(instructions string) *httpclient.Request {
	body, _ := json.Marshal(map[string]any{
		"model": "fixture-model", "instructions": instructions,
		"input": []any{
			map[string]any{"role": "user", "content": "Find ORBITAL_BEACON"},
			map[string]any{"type": "function_call", "call_id": "call_index", "name": "index_directory", "arguments": `{"path":"xxx"}`},
			map[string]any{"type": "function_call_output", "call_id": "call_index", "output": "indexed 2 files"},
			map[string]any{"role": "user", "content": "Continue from the indexed state"},
		},
	})
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses/compact",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	}
}

func compactCustomHTTPRequest() *httpclient.Request {
	body, _ := json.Marshal(map[string]any{
		"model": "fixture-model", "instructions": "Preserve tool state.",
		"input": []any{
			map[string]any{"role": "user", "content": "Apply the patch"},
			map[string]any{"type": "custom_tool_call", "call_id": "call_patch", "name": "apply_patch", "input": "*** Begin Patch\n*** End Patch"},
			map[string]any{"type": "custom_tool_call_output", "call_id": "call_patch", "output": "Done!"},
		},
	})
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses/compact",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	}
}

func serveCompactChat(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	var payload struct {
		Stream   bool `json:"stream"`
		Messages []struct {
			Role       string `json:"role"`
			Content    any    `json:"content"`
			ToolCallID string `json:"tool_call_id"`
			ToolCalls  []struct {
				ID       string                           `json:"id"`
				Function struct{ Name, Arguments string } `json:"function"`
			} `json:"tool_calls"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, "decode Chat compact request", http.StatusBadRequest)
		return
	}
	encoded, _ := json.Marshal(payload.Messages[0].Content)
	if payload.Stream || len(payload.Messages) != 5 || payload.Messages[0].Role != "system" ||
		!strings.Contains(string(encoded), "Preserve pending file edits.") ||
		payload.Messages[2].Role != "assistant" || len(payload.Messages[2].ToolCalls) != 1 ||
		payload.Messages[2].ToolCalls[0].ID != "call_index" || payload.Messages[2].ToolCalls[0].Function.Name != "index_directory" ||
		payload.Messages[3].Role != "tool" || payload.Messages[3].ToolCallID != "call_index" {
		http.Error(writer, "Chat compact history degraded", http.StatusBadRequest)
		return
	}
	writeCompactChatResponse(writer, "COMPACTED_STATE")
}

func serveCompactAnthropic(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	var payload struct {
		System   json.RawMessage `json:"system"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
		http.Error(writer, "decode Anthropic compact request", http.StatusBadRequest)
		return
	}
	encoded, _ := json.Marshal(payload.Messages)
	if !strings.Contains(string(payload.System), "Preserve pending file edits.") || len(payload.Messages) < 3 ||
		!strings.Contains(string(encoded), `"type":"tool_use"`) || !strings.Contains(string(encoded), `"id":"call_index"`) ||
		!strings.Contains(string(encoded), `"type":"tool_result"`) || !strings.Contains(string(encoded), `"tool_use_id":"call_index"`) {
		http.Error(writer, "Anthropic compact history degraded", http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = writer.Write([]byte(`{"id":"msg_compact","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"COMPACTED_STATE"}],"stop_reason":"end_turn","usage":{"input_tokens":20,"output_tokens":4}}`))
}

func serveCompactResponsesEmulated(t *testing.T, writer http.ResponseWriter, request *http.Request) {
	t.Helper()
	defer request.Body.Close()
	body, err := io.ReadAll(request.Body)
	if err != nil {
		http.Error(writer, "read Responses compact request", http.StatusBadRequest)
		return
	}
	// Emulation rewrites compact → RequestTypeChat on /v1/responses. The wire
	// is a normal response create carrying the compact system instruction and
	// lowered history — not object=response.compaction upstream.
	// Require BOTH the compact instruction and function-call history identity;
	// instruction alone must not pass (avoids false positive when history is dropped).
	payload := string(body)
	if !strings.Contains(payload, "Preserve pending file edits.") {
		http.Error(writer, "emulated Responses compact missing instruction", http.StatusBadRequest)
		return
	}
	hasFunctionHistory := strings.Contains(payload, `"type":"function_call"`) ||
		strings.Contains(payload, `"call_id":"call_index"`) ||
		strings.Contains(payload, "call_index")
	if !hasFunctionHistory {
		http.Error(writer, "emulated Responses compact lost function history", http.StatusBadRequest)
		return
	}
	writer.Header().Set("Content-Type", "application/json")
	_, _ = writer.Write([]byte(`{"id":"resp_compact","created_at":1770000000,"object":"response","model":"fixture-model","status":"completed","output":[{"id":"msg_compact","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"COMPACTED_STATE","annotations":[]}]}],"usage":{"input_tokens":20,"output_tokens":4,"total_tokens":24}}`))
}

func writeCompactChatResponse(writer http.ResponseWriter, content string) {
	writer.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(writer).Encode(map[string]any{
		"id": "chat_compact", "object": "chat.completion", "created": 1770000000, "model": "fixture-model",
		"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
	})
}

func assertCompactHTTPResponse(t *testing.T, result *pipeline.Result, instructions, output string) {
	t.Helper()
	if err := checkCompactHTTPResponse(result, instructions, output); err != nil {
		t.Fatal(err)
	}
}

func checkCompactHTTPResponse(result *pipeline.Result, instructions, output string) error {
	if result == nil || result.Stream || result.Response == nil {
		return fmt.Errorf("unexpected compact result: %#v", result)
	}
	var body struct {
		Object       string `json:"object"`
		Instructions string `json:"instructions"`
		Output       []struct {
			Type    string `json:"type"`
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(result.Response.Body, &body); err != nil {
		return fmt.Errorf("decode compact response: %w body=%s", err, result.Response.Body)
	}
	if body.Object != "response.compaction" || body.Instructions != instructions || len(body.Output) != 1 ||
		body.Output[0].Type != "message" || body.Output[0].Role != "assistant" || len(body.Output[0].Content) != 1 ||
		body.Output[0].Content[0].Type != "output_text" || body.Output[0].Content[0].Text != output {
		return fmt.Errorf("compact response degraded: %s", result.Response.Body)
	}
	return nil
}

func stringPointerForTest(value string) *string { return &value }

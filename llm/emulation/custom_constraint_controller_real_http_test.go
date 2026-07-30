package emulation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
	"github.com/stretchr/testify/require"
)

func TestControllerCorrectsConstrainedCustomToolThroughChatOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	var issueMu sync.Mutex
	var issues []string
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		var payload chatConstraintRequest
		if err := json.Unmarshal(body, &payload); err != nil || len(payload.Tools) != 1 ||
			payload.Tools[0].Type != "function" || payload.Tools[0].Function.Name == "" {
			http.Error(writer, "invalid lowered request", http.StatusBadRequest)
			return
		}
		syntheticName := payload.Tools[0].Function.Name
		writer.Header().Set("Content-Type", "application/json")
		if round == 1 {
			_, _ = io.WriteString(writer, chatCustomCallResponse("chat_bad", "call_bad", syntheticName, `{"input":"BAD"}`))
			return
		}
		if !hasConstraintFeedback(payload.Messages, "call_bad") {
			issueMu.Lock()
			issues = append(issues, "corrective round did not receive the grammar violation result")
			issueMu.Unlock()
			http.Error(writer, "missing correction feedback", http.StatusBadRequest)
			return
		}
		_, _ = io.WriteString(writer, chatCustomCallResponse("chat_fixed", "call_fixed", syntheticName, `{"input":"OK:fixed"}`))
	}))
	t.Cleanup(provider.Close)

	result, observer := runConstrainedResponsesRequest(t, provider.URL, 2)
	require.EqualValues(t, 2, rounds.Load(), "body=%s emulation=%+v", result.Response.Body, observer.emulation())
	issueMu.Lock()
	require.Empty(t, issues)
	issueMu.Unlock()

	var body struct {
		Status string `json:"status"`
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
			Input  string `json:"input"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &body), string(result.Response.Body))
	require.Equal(t, "completed", body.Status)
	require.Len(t, body.Output, 1)
	require.Equal(t, "custom_tool_call", body.Output[0].Type)
	require.Equal(t, "call_fixed", body.Output[0].CallID)
	require.Equal(t, "patch", body.Output[0].Name)
	require.Equal(t, "OK:fixed", body.Output[0].Input)
	require.NotContains(t, string(result.Response.Body), "BAD")

	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 1, summary.CustomConstraintCompiles)
	require.EqualValues(t, 2, summary.CustomConstraintValidations)
	require.EqualValues(t, 1, summary.CustomConstraintViolations)
	require.EqualValues(t, 1, summary.CustomConstraintRetries)
	require.Zero(t, summary.CustomConstraintFallbacks)
	require.Positive(t, summary.CustomConstraintCompileNanos)
	require.GreaterOrEqual(t, summary.CustomConstraintValidateNanos, int64(0))
}

func TestControllerPreservesInvalidCustomCallAfterCorrectionBudgetOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		var payload chatConstraintRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || len(payload.Tools) != 1 {
			http.Error(writer, "invalid request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		callID := fmt.Sprintf("call_bad_%d", round)
		_, _ = io.WriteString(writer, chatCustomCallResponse("chat_bad", callID, payload.Tools[0].Function.Name, `{"input":"BAD"}`))
	}))
	t.Cleanup(provider.Close)

	result, observer := runConstrainedResponsesRequest(t, provider.URL, 1)
	require.EqualValues(t, 2, rounds.Load(), "body=%s emulation=%+v", result.Response.Body, observer.emulation())
	var body struct {
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Input  string `json:"input"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &body), string(result.Response.Body))
	require.Len(t, body.Output, 1)
	require.Equal(t, "custom_tool_call", body.Output[0].Type)
	require.Equal(t, "call_bad_2", body.Output[0].CallID)
	require.Equal(t, "BAD", body.Output[0].Input)

	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 2, summary.CustomConstraintValidations)
	require.EqualValues(t, 2, summary.CustomConstraintViolations)
	require.EqualValues(t, 1, summary.CustomConstraintRetries)
	require.EqualValues(t, 1, summary.CustomConstraintFallbacks)
}

func TestControllerPreservesMixedClientCallsWithoutHiddenRetryOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		rounds.Add(1)
		var payload chatConstraintRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || len(payload.Tools) != 2 {
			http.Error(writer, "invalid request", http.StatusBadRequest)
			return
		}
		customName := ""
		for index := range payload.Tools {
			name := payload.Tools[index].Function.Name
			if name != "lookup" {
				customName = name
			}
		}
		if customName == "" {
			http.Error(writer, "custom tool was not lowered", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(writer, `{"id":"chat_mixed","object":"chat.completion","created":1785400000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_lookup","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"A-1\"}"}},{"id":"call_patch_bad","type":"function","function":{"name":%q,"arguments":"{\"input\":\"BAD\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, customName)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		EnableCustomConstraints: true, MaxCustomConstraintRetries: 2,
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
		Body: []byte(`{
			"model":"fixture-model","input":"look up an item and prepare a patch",
			"tools":[
				{"type":"function","name":"lookup","description":"Look up an item","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}},
				{"type":"custom","name":"patch","description":"Emit an OK patch token","format":{"type":"grammar","syntax":"regex","definition":"^OK:[a-z]+$"}}
			]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 1, rounds.Load(), "body=%s", result.Response.Body)
	var body struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Input     string `json:"input"`
			Arguments string `json:"arguments"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &body), string(result.Response.Body))
	require.Len(t, body.Output, 2)
	require.Equal(t, "function_call", body.Output[0].Type)
	require.Equal(t, "call_lookup", body.Output[0].CallID)
	require.Equal(t, "lookup", body.Output[0].Name)
	require.JSONEq(t, `{"key":"A-1"}`, body.Output[0].Arguments)
	require.Equal(t, "custom_tool_call", body.Output[1].Type)
	require.Equal(t, "call_patch_bad", body.Output[1].CallID)
	require.Equal(t, "patch", body.Output[1].Name)
	require.Equal(t, "BAD", body.Output[1].Input)

	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 1, summary.CustomConstraintValidations)
	require.EqualValues(t, 1, summary.CustomConstraintViolations)
	require.Zero(t, summary.CustomConstraintRetries)
	require.EqualValues(t, 1, summary.CustomConstraintFallbacks)
}

func TestControllerHoldsAndCorrectsConstrainedCustomToolStreamOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		var payload chatConstraintRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || !payload.Stream || len(payload.Tools) != 1 {
			http.Error(writer, "invalid streaming request", http.StatusBadRequest)
			return
		}
		if round == 2 && !hasConstraintFeedback(payload.Messages, "stream_bad") {
			http.Error(writer, "missing correction feedback", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			return
		}
		writeFrame := func(value any) {
			encoded, _ := json.Marshal(value)
			_, _ = writer.Write(append(append([]byte("data: "), encoded...), []byte("\n\n")...))
			flusher.Flush()
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": fmt.Sprintf("chat_constraint_stream_%d", round), "object": "chat.completion.chunk",
				"created": 1785400100 + round, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		callID := "stream_bad"
		firstArguments, secondArguments := `{"input":"B`, `AD"}`
		if round == 2 {
			callID = "stream_fixed"
			firstArguments, secondArguments = `{"input":"OK:st`, `ream"}`
		}
		writeFrame(chunk(map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
			"index": 0, "id": callID, "type": "function",
			"function": map[string]any{"name": payload.Tools[0].Function.Name, "arguments": firstArguments},
		}}}, nil))
		writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "function": map[string]any{"arguments": secondArguments},
		}}}, nil))
		writeFrame(chunk(map[string]any{}, "tool_calls"))
		writeFrame(map[string]any{
			"id": fmt.Sprintf("chat_constraint_stream_%d", round), "object": "chat.completion.chunk",
			"model": "fixture-model", "choices": []any{},
			"usage": map[string]any{"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
		})
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		EnableCustomConstraints: true, MaxCustomConstraintRetries: 2,
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
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"prepare a streamed patch",
			"tools":[{"type":"custom","name":"patch","description":"Emit an OK patch token",
				"format":{"type":"grammar","syntax":"regex","definition":"^OK:[a-z]+$"}}]
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	var wire strings.Builder
	var eventTypes []string
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		eventTypes = append(eventTypes, event.Type)
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.EqualValues(t, 2, rounds.Load(), wire.String())
	require.NotContains(t, wire.String(), "BAD")
	require.NotContains(t, wire.String(), "stream_bad")
	require.Contains(t, wire.String(), "OK:stream")
	require.Contains(t, eventTypes, "response.custom_tool_call_input.delta")
	require.Contains(t, eventTypes, "response.output_item.done")
	require.Contains(t, eventTypes, "response.completed")

	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 2, summary.CustomConstraintValidations)
	require.EqualValues(t, 1, summary.CustomConstraintViolations)
	require.EqualValues(t, 1, summary.CustomConstraintRetries)
	require.Zero(t, summary.CustomConstraintFallbacks)
}

func TestControllerReleasesLastInvalidCustomToolStreamAfterRetryBudgetOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		var payload chatConstraintRequest
		if err := json.NewDecoder(request.Body).Decode(&payload); err != nil || !payload.Stream || len(payload.Tools) != 1 {
			http.Error(writer, "invalid streaming request", http.StatusBadRequest)
			return
		}
		if round == 2 && !hasConstraintFeedback(payload.Messages, "stream_bad_1") {
			http.Error(writer, "missing correction feedback", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "text/event-stream")
		writer.Header().Set("Cache-Control", "no-cache")
		flusher, ok := writer.(http.Flusher)
		if !ok {
			return
		}
		writeFrame := func(value any) {
			encoded, _ := json.Marshal(value)
			_, _ = writer.Write(append(append([]byte("data: "), encoded...), []byte("\n\n")...))
			flusher.Flush()
		}
		chunk := func(delta map[string]any, finish any) map[string]any {
			return map[string]any{
				"id": fmt.Sprintf("chat_constraint_fallback_%d", round), "object": "chat.completion.chunk",
				"created": 1785400200 + round, "model": "fixture-model",
				"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
			}
		}
		callID := fmt.Sprintf("stream_bad_%d", round)
		writeFrame(chunk(map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{
			"index": 0, "id": callID, "type": "function",
			"function": map[string]any{"name": payload.Tools[0].Function.Name, "arguments": `{"input":"B`},
		}}}, nil))
		writeFrame(chunk(map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "function": map[string]any{"arguments": `AD"}`},
		}}}, nil))
		writeFrame(chunk(map[string]any{}, "tool_calls"))
		_, _ = io.WriteString(writer, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		EnableCustomConstraints: true, MaxCustomConstraintRetries: 1,
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
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"prepare a streamed patch",
			"tools":[{"type":"custom","name":"patch","description":"Emit an OK patch token",
				"format":{"type":"grammar","syntax":"regex","definition":"^OK:[a-z]+$"}}]
		}`),
	})
	require.NoError(t, err)
	require.True(t, result.Stream)
	var wire strings.Builder
	var eventTypes []string
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		eventTypes = append(eventTypes, event.Type)
		wire.Write(event.Data)
	}
	require.NoError(t, result.EventStream.Err())
	require.EqualValues(t, 2, rounds.Load(), wire.String())
	require.NotContains(t, wire.String(), "stream_bad_1")
	require.Contains(t, wire.String(), "stream_bad_2")
	require.Contains(t, wire.String(), "BAD")
	require.Contains(t, eventTypes, "response.custom_tool_call_input.delta")
	require.Contains(t, eventTypes, "response.output_item.done")
	require.Contains(t, eventTypes, "response.completed")

	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 2, summary.CustomConstraintValidations)
	require.EqualValues(t, 2, summary.CustomConstraintViolations)
	require.EqualValues(t, 1, summary.CustomConstraintRetries)
	require.EqualValues(t, 1, summary.CustomConstraintFallbacks)
}

func TestControllerCorrectsConstrainedCustomToolThroughAnthropicOverRealHTTP(t *testing.T) {
	var rounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := rounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Name string `json:"name"`
			} `json:"tools"`
			Messages []json.RawMessage `json:"messages"`
		}
		if json.Unmarshal(body, &payload) != nil || len(payload.Tools) != 1 || payload.Tools[0].Name == "" {
			http.Error(writer, "invalid Anthropic request", http.StatusBadRequest)
			return
		}
		if round == 2 {
			encoded, _ := json.Marshal(payload.Messages)
			if !strings.Contains(string(encoded), "anthropic_bad") ||
				!strings.Contains(string(encoded), "violated its required grammar") {
				http.Error(writer, "missing Anthropic correction feedback", http.StatusBadRequest)
				return
			}
		}
		callID, input := "anthropic_bad", "BAD"
		if round == 2 {
			callID, input = "anthropic_fixed", "OK:fixed"
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(writer, `{"id":"msg_%d","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":%q,"name":%q,"input":{"input":%q}}],"stop_reason":"tool_use","usage":{"input_tokens":4,"output_tokens":2}}`, round, callID, payload.Tools[0].Name, input)
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		EnableCustomConstraints: true, MaxCustomConstraintRetries: 2,
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := anthropic.NewOutboundTransformer(provider.URL, "provider-key")
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
		Body: []byte(`{
			"model":"fixture-model","input":"prepare a patch",
			"tools":[{"type":"custom","name":"patch","description":"Emit an OK patch token",
				"format":{"type":"grammar","syntax":"regex","definition":"^OK:[a-z]+$"}}]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, rounds.Load())
	var responseBody struct {
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Input  string `json:"input"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &responseBody), string(result.Response.Body))
	require.Len(t, responseBody.Output, 1)
	require.Equal(t, "custom_tool_call", responseBody.Output[0].Type)
	require.Equal(t, "anthropic_fixed", responseBody.Output[0].CallID)
	require.Equal(t, "OK:fixed", responseBody.Output[0].Input)
	require.NotContains(t, string(result.Response.Body), "BAD")
	summary := observer.emulation()
	require.NotNil(t, summary)
	require.EqualValues(t, 2, summary.CustomConstraintValidations)
	require.EqualValues(t, 1, summary.CustomConstraintRetries)
}

type chatConstraintRequest struct {
	Stream bool `json:"stream"`
	Tools  []struct {
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

func runConstrainedResponsesRequest(t *testing.T, providerURL string, retries int) (*pipeline.Result, *controllerObservationRecorder) {
	t.Helper()
	controller, err := emulation.NewController(emulation.ControllerConfig{
		EnableCustomConstraints: true, MaxCustomConstraintRetries: retries,
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := openai.NewOutboundTransformer(providerURL, "provider-key")
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
		Body: []byte(`{
			"model":"fixture-model","input":"prepare a patch",
			"tools":[{"type":"custom","name":"patch","description":"Emit an OK patch token",
				"format":{"type":"grammar","syntax":"regex","definition":"^OK:[a-z]+$"}}]
		}`),
	})
	require.NoError(t, err)
	require.NotNil(t, result.Response)
	return result, observer
}

func hasConstraintFeedback(messages []struct {
	Role       string `json:"role"`
	ToolCallID string `json:"tool_call_id"`
	Content    any    `json:"content"`
}, callID string) bool {
	for _, message := range messages {
		if message.Role != "tool" || message.ToolCallID != callID {
			continue
		}
		encoded, _ := json.Marshal(message.Content)
		return strings.Contains(string(encoded), "violated its required grammar")
	}
	return false
}

func chatCustomCallResponse(id, callID, name, arguments string) string {
	return fmt.Sprintf(`{"id":%q,"object":"chat.completion","created":1785400000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":%q,"type":"function","function":{"name":%q,"arguments":%q}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, id, callID, name, arguments)
}

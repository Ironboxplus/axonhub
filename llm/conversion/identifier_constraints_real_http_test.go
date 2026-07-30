package conversion_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

var targetPortableIdentifierPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

func TestOpenAIHistoriesReachAnthropicWithStablePortableIdentifiersOverRealHTTP(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name    string
		inbound func() transformer.Inbound
		url     string
		body    string
	}{
		{
			name: "chat", inbound: func() transformer.Inbound { return openai.NewInboundTransformer() }, url: "/v1/chat/completions",
			body: `{"model":"fixture-model","messages":[{"role":"assistant","tool_calls":[{"id":"call.with space:1","type":"function","function":{"name":"read.file","arguments":"{\"path\":\"README.md\"}"}}]},{"role":"tool","tool_call_id":"call.with space:1","content":"ok"}],"tools":[{"type":"function","function":{"name":"read.file","parameters":{"type":"object"}}}]}`,
		},
		{
			name: "responses", inbound: func() transformer.Inbound { return responses.NewInboundTransformer() }, url: "/v1/responses",
			body: `{"model":"fixture-model","input":[{"type":"function_call","call_id":"call.with space:1","name":"read.file","arguments":"{\"path\":\"README.md\"}"},{"type":"function_call_output","call_id":"call.with space:1","output":"ok"}],"tools":[{"type":"function","name":"read.file","parameters":{"type":"object"}}]}`,
		},
	}
	for _, test := range cases {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				if request.URL.Path != "/v1/messages" {
					http.Error(writer, "unexpected path", http.StatusNotFound)
					return
				}
				body, err := io.ReadAll(request.Body)
				if err != nil {
					http.Error(writer, "read request", http.StatusBadRequest)
					return
				}
				assertAnthropicPortableToolHistory(t, body)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, `{"id":"msg_ids","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":1}}`)
			}))
			t.Cleanup(provider.Close)
			target, err := anthropic.NewOutboundTransformer(provider.URL, "fixture-key")
			if err != nil {
				t.Fatalf("create Anthropic outbound: %v", err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			result, err := pipeline.NewFactory(executor).
				Pipeline(test.inbound(), conversion.NewOutbound(target)).
				Process(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: test.url,
					Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(test.body),
				})
			if err != nil {
				t.Fatalf("run %s to Anthropic: %v", test.name, err)
			}
			if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
				t.Fatalf("unexpected client response: %#v", result)
			}
		})
	}
}

func TestAnthropicLongToolHistoryReachesResponsesWithPairedCallIDOverRealHTTP(t *testing.T) {
	t.Parallel()
	longID := "toolu_" + strings.Repeat("a", 80)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/responses" {
			http.Error(writer, "unexpected path", http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		assertResponsesPairedCallID(t, body, longID)
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_ids","object":"response","status":"completed","model":"fixture-model","output":[{"id":"msg_done","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":1,"total_tokens":6}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"fixture-model","max_tokens":64,"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"` + longID + `","name":"lookup","input":{"q":"x"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + longID + `","content":"ok"}]}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`),
		})
	if err != nil {
		t.Fatalf("run Anthropic to Responses: %v", err)
	}
	if result == nil || result.Response == nil || !strings.Contains(string(result.Response.Body), "done") {
		t.Fatalf("unexpected client response: %#v", result)
	}
}

func TestResponsesProviderCallIDIsPortableForAnthropicClientOverRealHTTP(t *testing.T) {
	t.Parallel()
	original := "call.with spaces:" + strings.Repeat("z", 80)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_call","object":"response","status":"completed","model":"fixture-model","output":[{"id":"fc_1","type":"function_call","status":"completed","call_id":"`+original+`","name":"lookup","arguments":"{}"}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":"run"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`),
		})
	if err != nil {
		t.Fatalf("run Responses provider to Anthropic client: %v", err)
	}
	var payload struct {
		Content []struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		} `json:"content"`
	}
	if result == nil || result.Response == nil || json.Unmarshal(result.Response.Body, &payload) != nil || len(payload.Content) != 1 {
		t.Fatalf("decode Anthropic response: %#v", result)
	}
	if payload.Content[0].Type != "tool_use" || payload.Content[0].ID == original || !targetPortableIdentifierPattern.MatchString(payload.Content[0].ID) {
		t.Fatalf("Anthropic-facing tool ID is invalid: %#v", payload.Content[0])
	}
}

func TestResponsesProviderStreamCallIDIsPortableForAnthropicClientOverRealHTTP(t *testing.T) {
	t.Parallel()
	original := "call.with spaces:" + strings.Repeat("s", 80)
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream")
		writeNamedSSEJSON(t, writer, "response.created", map[string]any{
			"type": "response.created", "sequence_number": 0,
			"response": map[string]any{"id": "resp_stream_ids", "object": "response", "model": "fixture-model", "status": "in_progress", "output": []any{}},
		})
		writeResponsesFunctionAdded(t, writer, 1, 0, "fc_stream", original, "lookup")
		writeResponsesFunctionDelta(t, writer, 2, 0, "fc_stream", `{}`)
		writeResponsesFunctionDone(t, writer, 3, 0, "fc_stream", original, "lookup", `{}`)
		writeNamedSSEJSON(t, writer, "response.completed", map[string]any{
			"type": "response.completed", "sequence_number": 5,
			"response": map[string]any{
				"id": "resp_stream_ids", "object": "response", "model": "fixture-model", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
			},
		})
	}))
	t.Cleanup(provider.Close)
	target, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(anthropic.NewInboundTransformer(), conversion.NewOutbound(target)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body: []byte(`{"model":"fixture-model","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"run"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`),
		})
	if err != nil {
		t.Fatalf("run streaming Responses provider to Anthropic client: %v", err)
	}
	if result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("unexpected stream result: %#v", result)
	}
	defer result.EventStream.Close()
	toolID := ""
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil || len(event.Data) == 0 || bytes.Equal(event.Data, []byte("[DONE]")) {
			continue
		}
		var wire struct {
			Type         string `json:"type"`
			ContentBlock *struct {
				Type string `json:"type"`
				ID   string `json:"id"`
			} `json:"content_block"`
		}
		if json.Unmarshal(event.Data, &wire) == nil && wire.Type == "content_block_start" && wire.ContentBlock != nil && wire.ContentBlock.Type == "tool_use" {
			toolID = wire.ContentBlock.ID
		}
	}
	if err := result.EventStream.Err(); err != nil {
		t.Fatalf("consume Anthropic stream: %v", err)
	}
	if toolID == "" || toolID == original || !targetPortableIdentifierPattern.MatchString(toolID) {
		t.Fatalf("Anthropic streaming tool ID is invalid: %q", toolID)
	}
}

func assertAnthropicPortableToolHistory(t *testing.T, body []byte) {
	t.Helper()
	var payload struct {
		Tools []struct {
			Name string `json:"name"`
		} `json:"tools"`
		Messages []struct {
			Content []struct {
				Type      string `json:"type"`
				ID        string `json:"id"`
				ToolUseID string `json:"tool_use_id"`
				Name      string `json:"name"`
			} `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Anthropic wire: %v body=%s", err, body)
	}
	if len(payload.Tools) != 1 || len(payload.Messages) != 2 || len(payload.Messages[0].Content) != 1 || len(payload.Messages[1].Content) != 1 {
		t.Fatalf("unexpected Anthropic history shape: %s", body)
	}
	call := payload.Messages[0].Content[0]
	result := payload.Messages[1].Content[0]
	if call.Type != "tool_use" || result.Type != "tool_result" || call.ID != result.ToolUseID ||
		!targetPortableIdentifierPattern.MatchString(call.ID) || call.Name != payload.Tools[0].Name ||
		!targetPortableIdentifierPattern.MatchString(call.Name) {
		t.Fatalf("Anthropic identifiers diverged: call=%#v result=%#v tools=%#v body=%s", call, result, payload.Tools, body)
	}
}

func assertResponsesPairedCallID(t *testing.T, body []byte, original string) {
	t.Helper()
	var payload struct {
		Input []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Responses wire: %v body=%s", err, body)
	}
	var callID, resultID string
	for _, item := range payload.Input {
		switch item.Type {
		case "function_call":
			callID = item.CallID
		case "function_call_output":
			resultID = item.CallID
		}
	}
	if callID == "" || callID != resultID || callID == original || len(callID) > 64 {
		t.Fatalf("Responses call IDs diverged: call=%q result=%q body=%s", callID, resultID, body)
	}
}

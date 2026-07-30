package responses_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

// TestResponsesMCPLifecycleRoundTripOverFragmentedRealHTTP exercises the
// production HTTP transport, SSE parser, canonical state machine, and Responses
// stream encoder. Every byte is flushed separately so transport framing cannot
// accidentally substitute for semantic event framing.
func TestResponsesMCPLifecycleRoundTripOverFragmentedRealHTTP(t *testing.T) {
	t.Parallel()

	providerErrors := make(chan error, 1)
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			providerErrors <- fmt.Errorf("read request: %w", err)
			http.Error(w, "read request", http.StatusBadRequest)
			return
		}
		var request struct {
			Stream *bool `json:"stream"`
			Tools  []struct {
				Type        string `json:"type"`
				ServerLabel string `json:"server_label"`
			} `json:"tools"`
		}
		if err := json.Unmarshal(body, &request); err != nil || request.Stream == nil || !*request.Stream ||
			len(request.Tools) != 1 || request.Tools[0].Type != "mcp" || request.Tools[0].ServerLabel != "inventory" {
			providerErrors <- fmt.Errorf("unexpected provider request: %s", body)
			http.Error(w, "unexpected request", http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher, ok := w.(http.Flusher)
		if !ok {
			providerErrors <- fmt.Errorf("response writer does not support flush")
			return
		}
		for _, event := range mcpLifecycleSSEEvents() {
			frame := []byte("event: " + event.eventType + "\ndata: " + event.data + "\n\n")
			for _, value := range frame {
				if _, err := w.Write([]byte{value}); err != nil {
					providerErrors <- fmt.Errorf("write SSE byte: %w", err)
					return
				}
				flusher.Flush()
			}
		}
		providerErrors <- nil
	}))
	t.Cleanup(provider.Close)

	inbound := responses.NewInboundTransformer()
	outbound, err := responses.NewOutboundTransformer(provider.URL, "test-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).Pipeline(inbound, outbound).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model","stream":true,"input":"check inventory",
			"tools":[{"type":"mcp","server_label":"inventory","server_url":"https://mcp.example.invalid/rpc"}]
		}`),
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.NotNil(t, result.EventStream)
	defer result.EventStream.Close()

	seen := make(map[string]bool)
	var finalList, finalCall map[string]any
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		require.NotNil(t, event)
		seen[event.Type] = true
		if event.Type != "response.output_item.done" {
			continue
		}
		var envelope struct {
			Item map[string]any `json:"item"`
		}
		require.NoError(t, json.Unmarshal(event.Data, &envelope))
		switch envelope.Item["type"] {
		case "mcp_list_tools":
			finalList = envelope.Item
		case "mcp_call":
			finalCall = envelope.Item
		}
	}
	require.NoError(t, result.EventStream.Err())
	require.NoError(t, <-providerErrors)

	for _, eventType := range []string{
		"response.mcp_list_tools.in_progress", "response.mcp_list_tools.completed",
		"response.mcp_call_arguments.delta", "response.mcp_call_arguments.done",
		"response.mcp_call.in_progress", "response.mcp_call.completed", "response.completed",
	} {
		require.True(t, seen[eventType], "missing %s from %#v", eventType, seen)
	}
	require.Equal(t, "inventory", finalList["server_label"])
	require.Len(t, finalList["tools"], 1)
	require.Equal(t, "inventory", finalCall["server_label"])
	require.Equal(t, "lookup", finalCall["name"])
	require.Equal(t, `{"sku":"A-1"}`, finalCall["arguments"])
	require.Equal(t, "7", finalCall["output"])
}

type mcpSSEFixture struct {
	eventType string
	data      string
}

func mcpLifecycleSSEEvents() []mcpSSEFixture {
	return []mcpSSEFixture{
		{"response.created", `{"type":"response.created","sequence_number":0,"response":{"id":"resp_mcp","object":"response","created_at":1785380000,"model":"fixture-model","status":"in_progress","output":[]}}`},
		{"response.output_item.added", `{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"list_1","type":"mcp_list_tools","server_label":"inventory","tools":[]}}`},
		{"response.mcp_list_tools.in_progress", `{"type":"response.mcp_list_tools.in_progress","sequence_number":2,"output_index":0,"item_id":"list_1"}`},
		{"response.mcp_list_tools.completed", `{"type":"response.mcp_list_tools.completed","sequence_number":3,"output_index":0,"item_id":"list_1"}`},
		{"response.output_item.done", `{"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":{"id":"list_1","type":"mcp_list_tools","server_label":"inventory","tools":[{"name":"lookup","description":"find stock","input_schema":{"type":"object"}}]}}`},
		{"response.output_item.added", `{"type":"response.output_item.added","sequence_number":5,"output_index":1,"item":{"id":"call_1","type":"mcp_call","server_label":"inventory","name":"lookup","arguments":"","status":"in_progress"}}`},
		{"response.mcp_call_arguments.delta", `{"type":"response.mcp_call_arguments.delta","sequence_number":6,"output_index":1,"item_id":"call_1","delta":"{\"sku\":\""}`},
		{"response.mcp_call_arguments.delta", `{"type":"response.mcp_call_arguments.delta","sequence_number":7,"output_index":1,"item_id":"call_1","delta":"A-1\"}"}`},
		{"response.mcp_call_arguments.done", `{"type":"response.mcp_call_arguments.done","sequence_number":8,"output_index":1,"item_id":"call_1","arguments":"{\"sku\":\"A-1\"}"}`},
		{"response.mcp_call.in_progress", `{"type":"response.mcp_call.in_progress","sequence_number":9,"output_index":1,"item_id":"call_1"}`},
		{"response.mcp_call.completed", `{"type":"response.mcp_call.completed","sequence_number":10,"output_index":1,"item_id":"call_1"}`},
		{"response.output_item.done", `{"type":"response.output_item.done","sequence_number":11,"output_index":1,"item":{"id":"call_1","type":"mcp_call","server_label":"inventory","name":"lookup","arguments":"{\"sku\":\"A-1\"}","output":"7","status":"completed"}}`},
		{"response.completed", `{"type":"response.completed","sequence_number":12,"response":{"id":"resp_mcp","object":"response","created_at":1785380000,"model":"fixture-model","status":"completed","output":[],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`},
	}
}

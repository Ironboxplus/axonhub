package emulation_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestControllerEmulatesAnthropicServerToolSearchThroughResponsesRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRounds atomic.Int64
	var searchName string
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		round := providerRounds.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read", http.StatusBadRequest)
			return
		}
		var payload struct {
			Tools []struct {
				Type         string `json:"type"`
				Name         string `json:"name"`
				DeferLoading *bool  `json:"defer_loading"`
			} `json:"tools"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		names := make([]string, 0, len(payload.Tools))
		for _, tool := range payload.Tools {
			names = append(names, tool.Name)
			if strings.HasPrefix(tool.Name, "axh_") {
				searchName = tool.Name
			}
			if tool.DeferLoading != nil {
				http.Error(writer, "defer_loading leaked to emulated target", http.StatusBadRequest)
				return
			}
		}
		writer.Header().Set("Content-Type", "application/json")
		switch round {
		case 1:
			if searchName == "" || len(names) != 2 || !slices.Contains(names, "always_available") ||
				slices.Contains(names, "create_event") || slices.Contains(names, "delete_event") {
				http.Error(writer, fmt.Sprintf("initial catalog leaked: %v", names), http.StatusBadRequest)
				return
			}
			_, _ = fmt.Fprintf(writer, `{"id":"resp_server_search_1","object":"response","created_at":1785431000,"model":"fixture-model","status":"completed","output":[{"type":"function_call","id":"fc_search_1","call_id":"search_call_1","name":%q,"status":"completed","arguments":{"query":"create"}}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`, searchName)
		case 2:
			if len(names) != 3 || !slices.Contains(names, searchName) || !slices.Contains(names, "always_available") ||
				!slices.Contains(names, "create_event") || slices.Contains(names, "delete_event") ||
				!strings.Contains(string(payload.Input), "create_event") {
				http.Error(writer, fmt.Sprintf("discovered catalog missing: tools=%v input=%s", names, payload.Input), http.StatusBadRequest)
				return
			}
			_, _ = io.WriteString(writer, `{"id":"resp_server_search_2","object":"response","created_at":1785431001,"model":"fixture-model","status":"completed","output":[{"type":"function_call","id":"fc_create_1","call_id":"create_call_1","name":"create_event","status":"completed","arguments":{"title":"Architecture review"}}],"usage":{"input_tokens":6,"output_tokens":2,"total_tokens":8}}`)
		default:
			http.Error(writer, "too many rounds", http.StatusBadRequest)
		}
	}))
	t.Cleanup(provider.Close)

	controller, err := emulation.NewController(emulation.ControllerConfig{
		Hosted:    hosted.Config{SyntheticNameKey: []byte(strings.Repeat("responses-tool-search-key-", 2))},
		MaxRounds: 4, MaxToolCalls: 8, MaxParallelCalls: 2,
	})
	require.NoError(t, err)
	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-provider-key")
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
				{"name":"create_event","description":"Create a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}},
				{"name":"delete_event","description":"Delete a calendar event","defer_loading":true,"input_schema":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}}
			]
		}`),
	})
	require.NoError(t, err)
	require.EqualValues(t, 2, providerRounds.Load())
	wire := string(result.Response.Body)
	require.NotContains(t, wire, searchName)
	require.Contains(t, wire, `"type":"server_tool_use"`)
	require.Contains(t, wire, `"type":"tool_search_tool_result"`)
	require.Contains(t, wire, `"tool_name":"create_event"`)
	require.Contains(t, wire, `"type":"tool_use"`)
	require.Contains(t, wire, `"name":"create_event"`)
}

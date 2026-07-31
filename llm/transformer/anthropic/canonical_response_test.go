package anthropic

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestTransformResponseBuildsCanonicalHostedLifecycleFromAnthropic(t *testing.T) {
	transformer, err := NewOutboundTransformer("https://example.com", "test")
	require.NoError(t, err)
	response, err := transformer.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: http.StatusOK,
		Body:       []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"claude-test","content":[{"type":"server_tool_use","id":"srv_1","name":"web_search","input":{"query":"axon"}},{"type":"web_search_tool_result","tool_use_id":"srv_1","content":[{"type":"web_search_result","url":"https://example.com"}]},{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{"input_tokens":4,"output_tokens":2}}`),
	})
	require.NoError(t, err)
	require.Len(t, response.Output, 2)
	require.Equal(t, llm.ItemKindHostedCall, response.Output[0].Kind)
	require.Equal(t, "srv_1", response.Output[0].HostedCall.Invocation.CallID)
	require.Equal(t, llm.ExecutionOwnerProvider, response.Output[0].HostedCall.Invocation.Execution)
	require.NotNil(t, response.Output[0].HostedCall.Result)
	require.Equal(t, llm.ItemStatusCompleted, response.Output[0].Status)
	require.Equal(t, llm.ItemKindMessage, response.Output[1].Kind)
	require.Equal(t, "done", response.Output[1].Content[0].Text)
}

func TestTransformResponseBuildsCanonicalWebFetchObjectLifecycle(t *testing.T) {
	transformer, err := NewOutboundTransformer("https://example.com", "test")
	require.NoError(t, err)
	response, err := transformer.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: http.StatusOK,
		Body: []byte(`{
			"id":"msg_fetch","type":"message","role":"assistant","model":"claude-test",
			"content":[
				{"type":"server_tool_use","id":"srv_fetch","name":"web_fetch","input":{"url":"https://docs.example.com/page"}},
				{"type":"web_fetch_tool_result","tool_use_id":"srv_fetch","content":{"type":"web_fetch_result","url":"https://docs.example.com/page","retrieved_at":"2026-07-30T00:00:00Z","content":{"type":"document","title":"Axon","source":{"type":"text","media_type":"text/plain","data":"body"}}}},
				{"type":"text","text":"fetched"}
			],
			"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}
		}`),
	})
	require.NoError(t, err)
	require.Len(t, response.Output, 2)
	hosted := response.Output[0].HostedCall
	require.NotNil(t, hosted)
	require.Equal(t, llm.ToolKindWebFetch, hosted.Invocation.Kind)
	require.Equal(t, "srv_fetch", hosted.Invocation.CallID)
	require.NotNil(t, hosted.Result)
	require.Equal(t, llm.ToolKindWebFetch, hosted.Result.Kind)
	require.Len(t, hosted.Result.Content, 1)
	require.Equal(t, llm.ContentKindText, hosted.Result.Content[0].Kind)
	require.JSONEq(t, `{"type":"web_fetch_result","url":"https://docs.example.com/page","retrieved_at":"2026-07-30T00:00:00Z","content":{"type":"document","title":"Axon","source":{"type":"text","media_type":"text/plain","data":"body"}}}`, string(hosted.Result.StructuredContent))
	require.Equal(t, string(hosted.Result.StructuredContent), hosted.Result.Content[0].Text)
	require.Equal(t, "fetched", response.Output[1].Content[0].Text)
}

func TestCanonicalResponseProjectsMCPGatewayLifecycleToAnthropicBlocks(t *testing.T) {
	t.Parallel()
	message, encoded, err := canonicalResponseToAnthropic(&llm.Response{
		ID: "msg_mcp", Model: "fixture-model",
		Output: []llm.Item{
			{
				Kind: llm.ItemKindMCPListTools, ID: "list_inventory", Status: llm.ItemStatusCompleted,
				MCPListTools: &llm.MCPListTools{ServerLabel: "inventory", Tools: []llm.MCPDiscoveredTool{{Name: "lookup"}}},
			},
			{
				Kind: llm.ItemKindMCPCall, ID: "mcp_call_inventory_1", Status: llm.ItemStatusCompleted,
				MCPCall: &llm.MCPCall{
					ServerLabel: "inventory", LogicalName: "lookup", ArgumentsJSON: []byte(`{"sku":"A-1"}`),
					Output: `{"content":[{"type":"text","text":"7 units available"}],"structuredContent":{"available":7}}`, Status: llm.MCPCallStatusCompleted,
				},
			},
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "done"}}},
		},
	})
	require.NoError(t, err)
	require.True(t, encoded)
	require.Len(t, message.Content, 3, "MCP discovery is observable but has no fabricated Anthropic response block")
	require.Equal(t, "mcp_tool_use", message.Content[0].Type)
	require.Equal(t, "inventory", message.Content[0].ServerName)
	require.Equal(t, "lookup", *message.Content[0].Name)
	require.JSONEq(t, `{"sku":"A-1"}`, string(message.Content[0].Input))
	require.Equal(t, "mcp_tool_result", message.Content[1].Type)
	require.Equal(t, "mcp_call_inventory_1", *message.Content[1].ToolUseID)
	require.NotNil(t, message.Content[1].Content)
	require.Contains(t, string(message.Content[1].Content.Raw), "7 units available")
	require.NotNil(t, message.StopReason)
	require.Equal(t, "end_turn", *message.StopReason)
	require.Equal(t, "done", *message.Content[2].Text)
}

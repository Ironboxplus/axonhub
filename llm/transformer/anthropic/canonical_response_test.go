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

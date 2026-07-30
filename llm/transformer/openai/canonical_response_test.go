package openai

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestTransformResponseBuildsCanonicalOutputDirectlyFromChat(t *testing.T) {
	transformer, err := NewOutboundTransformer("https://example.com", "test")
	require.NoError(t, err)
	response, err := transformer.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: http.StatusOK,
		Body:       []byte(`{"id":"chat_1","object":"chat.completion","model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","content":"answer","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},"finish_reason":"tool_calls"}]}`),
	})
	require.NoError(t, err)
	require.Len(t, response.Output, 2)
	require.Equal(t, llm.ItemKindMessage, response.Output[0].Kind)
	require.Equal(t, "answer", response.Output[0].Content[0].Text)
	require.Equal(t, llm.ItemStatusCompleted, response.Output[0].Status)
	require.Equal(t, llm.ItemKindToolCall, response.Output[1].Kind)
	require.Equal(t, "call_1", response.Output[1].ToolCall.CallID)
	require.JSONEq(t, `{"q":"x"}`, string(response.Output[1].ToolCall.ArgumentsJSON))
	require.Equal(t, llm.ToolCallStatusCompleted, response.Output[1].ToolCall.Status)
}

func TestTransformResponseRejectsInvalidCanonicalChatTool(t *testing.T) {
	transformer, err := NewOutboundTransformer("https://example.com", "test")
	require.NoError(t, err)
	_, err = transformer.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: http.StatusOK,
		Body:       []byte(`{"id":"chat_1","object":"chat.completion","model":"gpt-test","choices":[{"index":0,"message":{"role":"assistant","tool_calls":[{"id":"","type":"function","function":{"name":"lookup","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`),
	})
	require.ErrorContains(t, err, "invalid canonical Chat response")
}

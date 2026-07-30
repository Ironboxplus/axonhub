package responses

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestTransformResponsePrefersCanonicalOutput(t *testing.T) {
	response, err := NewInboundTransformer().TransformResponse(context.Background(), &llm.Response{
		ID: "resp_public", Model: "gpt-test",
		Output: []llm.Item{
			{
				Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Status: llm.ItemStatusCompleted,
				Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "canonical"}},
			},
			{
				Kind: llm.ItemKindToolCall, ID: "call_1", Role: llm.RoleAssistant, Status: llm.ItemStatusCompleted,
				ToolCall: &llm.ToolInvocation{
					Kind: llm.ToolKindFunction, ID: "call_1", CallID: "call_1", LogicalName: "lookup",
					ArgumentsJSON: json.RawMessage(`{"q":"x"}`), Status: llm.ToolCallStatusCompleted,
					Execution: llm.ExecutionOwnerClient,
				},
			},
		},
		Choices: []llm.Choice{{Message: &llm.Message{Content: llm.MessageContent{Content: stringPointer("legacy")}}}},
	})
	require.NoError(t, err)
	var wire Response
	require.NoError(t, json.Unmarshal(response.Body, &wire))
	require.Len(t, wire.Output, 2)
	require.Equal(t, "canonical", wire.Output[0].GetContentItems()[0].Text)
	require.Equal(t, "function_call", wire.Output[1].Type)
	require.Equal(t, "call_1", wire.Output[1].CallID)
}

func TestTransformResponseRendersStoredInProgressLifecycleWithoutFakeMessage(t *testing.T) {
	response, err := NewInboundTransformer().TransformResponse(context.Background(), &llm.Response{
		ID: "resp_pending", Model: "gpt-test", Status: llm.ResponseStatusInProgress,
	})
	require.NoError(t, err)
	var wire Response
	require.NoError(t, json.Unmarshal(response.Body, &wire))
	require.NotNil(t, wire.Status)
	require.Equal(t, "in_progress", *wire.Status)
	require.Empty(t, wire.Output)
}

func TestTransformResponseRejectsHostedKindWithoutResponsesEncoding(t *testing.T) {
	_, err := NewInboundTransformer().TransformResponse(context.Background(), &llm.Response{
		ID: "resp_unsupported", Model: "gpt-test", Status: llm.ResponseStatusCompleted,
		Output: []llm.Item{{
			Kind: llm.ItemKindHostedCall, ID: "fetch_1", Status: llm.ItemStatusCompleted,
			HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{
				Kind: llm.ToolKindWebFetch, ID: "fetch_1", CallID: "fetch_call_1", LogicalName: "web_fetch",
				Execution: llm.ExecutionOwnerProvider,
			}},
		}},
	})
	require.ErrorContains(t, err, "has no Responses encoding")
}

package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestAnthropicNativeTerminalFieldsRoundTrip(t *testing.T) {
	provider := &Message{
		ID:           "msg_terminal",
		Type:         "message",
		Role:         "assistant",
		Model:        "claude-test",
		Content:      []MessageContentBlock{{Type: "text", Text: lo.ToPtr("continue later")}},
		StopReason:   lo.ToPtr("pause_turn"),
		StopSequence: lo.ToPtr("<END>"),
	}

	unified := convertToLlmResponse(provider, PlatformDirect)
	require.Equal(t, "pause_turn", unified.TerminalReason)
	restored := convertToAnthropicResponse(unified)

	require.NotNil(t, restored.StopReason)
	require.Equal(t, "pause_turn", *restored.StopReason)
	require.NotNil(t, restored.StopSequence)
	require.Equal(t, "<END>", *restored.StopSequence)
}

func TestAnthropicServerToolUsageJSONRoundTrip(t *testing.T) {
	var provider Usage
	require.NoError(t, json.Unmarshal([]byte(`{
		"input_tokens":10,"output_tokens":2,
		"server_tool_use":{"web_search_requests":3,"web_fetch_requests":2,"code_execution_requests":1,"tool_search_requests":4}
	}`), &provider))
	unified := convertToLlmUsage(&provider, PlatformDirect)
	require.Equal(t, &llm.ServerToolUsage{
		WebSearchRequests: 3, WebFetchRequests: 2, CodeExecutionRequests: 1, ToolSearchRequests: 4,
	}, unified.ServerToolUsage)

	restored, err := json.Marshal(convertToAnthropicUsage(unified))
	require.NoError(t, err)
	require.JSONEq(t, `{
		"input_tokens":10,"output_tokens":2,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,
		"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},
		"server_tool_use":{"web_search_requests":3,"web_fetch_requests":2,"code_execution_requests":1,"tool_search_requests":4}
	}`, string(restored))
}

func TestAnthropicMCPServerNameRoundTrip(t *testing.T) {
	provider := &Message{
		ID:    "msg_mcp",
		Type:  "message",
		Role:  "assistant",
		Model: "claude-test",
		Content: []MessageContentBlock{{
			Type:       "mcp_tool_use",
			ID:         "mcptoolu_1",
			Name:       lo.ToPtr("lookup"),
			ServerName: "retrieval",
			Input:      []byte(`{"query":"octopus"}`),
		}},
	}

	unified := convertToLlmResponse(provider, PlatformDirect)
	restored := convertToAnthropicResponse(unified)

	require.Len(t, restored.Content, 1)
	require.Equal(t, "mcp_tool_use", restored.Content[0].Type)
	require.Equal(t, "retrieval", restored.Content[0].ServerName)
}

func TestAnthropicMCPServerNameRequestRoundTrip(t *testing.T) {
	provider := &MessageRequest{
		Model:     "claude-test",
		MaxTokens: 1024,
		Messages: []MessageParam{{
			Role: "assistant",
			Content: MessageContent{MultipleContent: []MessageContentBlock{{
				Type:       "mcp_tool_use",
				ID:         "mcptoolu_request",
				Name:       lo.ToPtr("lookup"),
				ServerName: "retrieval",
				Input:      []byte(`{"query":"octopus"}`),
			}}},
		}},
	}

	unified, err := convertToLLMRequest(provider)
	require.NoError(t, err)
	restored := convertToAnthropicRequest(unified)

	require.Len(t, restored.Messages, 1)
	require.Len(t, restored.Messages[0].Content.MultipleContent, 1)
	require.Equal(t, "mcp_tool_use", restored.Messages[0].Content.MultipleContent[0].Type)
	require.Equal(t, "retrieval", restored.Messages[0].Content.MultipleContent[0].ServerName)
}

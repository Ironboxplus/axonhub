package openai

import (
	"context"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestCanonicalResponseBlocksAgentMessageOutputToChat(t *testing.T) {
	t.Parallel()
	item := testAgentMessageItem(llm.AgentMessageContentInputText, "first", "second")
	response, encoded, err := canonicalResponseToChat(&llm.Response{Output: []llm.Item{item}})
	if response != nil || !encoded || err == nil || strings.Contains(err.Error(), "first") || strings.Contains(err.Error(), "second") {
		t.Fatalf("Chat agent-message output response=%#v encoded=%v err=%v", response, encoded, err)
	}
}

func TestCanonicalResponseBlocksNonProjectableResponsesOutputToChat(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name string
		item llm.Item
	}{
		{name: "encrypted agent", item: testAgentMessageItem(llm.AgentMessageContentEncryptedContent, "PRIVATE_AGENT", "")},
		{name: "mixed agent", item: llm.Item{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "visible"}, {Kind: llm.AgentMessageContentEncryptedContent, EncryptedContent: "PRIVATE_AGENT"}}}}},
		{name: "context", item: llm.Item{Kind: llm.ItemKindContextCompaction, ContextCompaction: &llm.ContextCompactionItem{}}},
		{name: "future", item: llm.Item{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{Type: "future_behavior", Raw: []byte(`{"type":"future_behavior","secret":"PRIVATE_FUTURE"}`), Behavioral: true}, ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse}}},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, encoded, err := canonicalResponseToChat(&llm.Response{Output: []llm.Item{test.item}})
			if !encoded || err == nil || strings.Contains(err.Error(), "PRIVATE") {
				t.Fatalf("Chat non-projectable response encoded=%v err=%v", encoded, err)
			}
		})
	}
}

func TestCanonicalChatStreamBlocksAgentAndContext(t *testing.T) {
	t.Parallel()
	encoder := newCanonicalChatEncoder()
	outputIndex := 0
	plain := testAgentMessageItem(llm.AgentMessageContentInputText, "first", "second")
	_, err := encoder.encode(llm.Event{Kind: llm.EventKindResponseStarted, Sequence: 0})
	if err != nil {
		t.Fatalf("Chat stream start: %v", err)
	}
	chunks, err := encoder.encode(llm.Event{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: llm.ItemRef{ItemID: "am_1", OutputIndex: &outputIndex}, Snapshot: &plain})
	if err == nil || chunks != nil || strings.Contains(err.Error(), "first") {
		t.Fatalf("Chat stream agent-message chunks=%#v err=%v", chunks, err)
	}
	contextItem := llm.Item{Kind: llm.ItemKindContextCompaction, ContextCompaction: &llm.ContextCompactionItem{}}
	contextEncoder := newCanonicalChatEncoder()
	_, err = contextEncoder.encode(llm.Event{Kind: llm.EventKindResponseStarted, Sequence: 0})
	if err == nil {
		_, err = contextEncoder.encode(llm.Event{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: llm.ItemRef{ItemID: "ctx", OutputIndex: &outputIndex}, Snapshot: &contextItem})
	}
	if err == nil || strings.Contains(err.Error(), "encrypted") {
		t.Fatalf("Chat stream context error=%v", err)
	}
}

func testAgentMessageItem(kind llm.AgentMessageContentKind, first, second string) llm.Item {
	parts := []llm.AgentMessageContentPart{{Kind: kind}}
	if kind == llm.AgentMessageContentInputText {
		parts[0].Text = first
		if second != "" {
			parts = append(parts, llm.AgentMessageContentPart{Kind: kind, Text: second})
		}
	} else {
		parts[0].EncryptedContent = first
	}
	return llm.Item{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: parts}}
}

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

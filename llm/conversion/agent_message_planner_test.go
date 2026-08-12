package conversion

import (
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestActionForAgentMessageClassifiesOnlyTypedSafeShapes(t *testing.T) {
	t.Parallel()

	plaintext := &llm.Item{ProtocolHints: llm.ProtocolHints{SourceBytes: 73}, AgentMessage: &llm.AgentMessage{
		Author: "/root", Recipient: "/root/worker",
		Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "handoff"}},
	}}
	encrypted := &llm.Item{AgentMessage: &llm.AgentMessage{
		Author: "/root", Recipient: "/root/worker",
		Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentEncryptedContent, EncryptedContent: "opaque"}},
	}}
	mixed := &llm.Item{AgentMessage: &llm.AgentMessage{
		Author: "/root", Recipient: "/root/worker",
		Content: []llm.AgentMessageContentPart{
			{Kind: llm.AgentMessageContentInputText, Text: "visible"},
			{Kind: llm.AgentMessageContentEncryptedContent, EncryptedContent: "opaque"},
		},
	}}

	tests := []struct {
		name       string
		source     llm.APIFormat
		target     llm.APIFormat
		item       *llm.Item
		kind       ActionKind
		strategy   StrategyID
		reason     ReasonCode
		semantic   string
		reversible bool
	}{
		{
			name: "missing payload is unavailable", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIChatCompletion,
			kind: ActionUnknown, strategy: StrategyUnavailable, reason: ReasonNoStrategy, semantic: "agent_message_invalid",
		},
		{
			name: "Responses identity stays typed", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIResponse, item: plaintext,
			kind: ActionNative, strategy: StrategyNative, reason: ReasonTargetNative, semantic: "agent_message", reversible: true,
		},
		{
			name: "plaintext lowers deterministically", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatAnthropicMessage, item: plaintext,
			kind: ActionLower, strategy: StrategyAgentMessageLegacyInput, reason: ReasonSemanticProjection, semantic: "agent_message_plaintext",
		},
		{
			name: "encrypted is unavailable", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIChatCompletion, item: encrypted,
			kind: ActionUnknown, strategy: StrategyUnavailable, reason: ReasonProviderPrivate, semantic: "agent_message_encrypted",
		},
		{
			name: "mixed is unavailable", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIChatCompletion, item: mixed,
			kind: ActionUnknown, strategy: StrategyUnavailable, reason: ReasonProviderPrivate, semantic: "agent_message_mixed",
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			action := actionForAgentMessage(test.source, test.target, test.item, 9)
			if action.Ref.Kind != ObjectAgentMessage || action.Ref.ItemIndex != 9 || action.Kind != test.kind ||
				action.Strategy != test.strategy || action.Reason != test.reason || action.SemanticClass != test.semantic ||
				action.Reversible != test.reversible {
				t.Fatalf("agent-message action = %#v", action)
			}
			if test.item == plaintext && action.RawBytes != 73 {
				t.Fatalf("agent-message evidence raw bytes = %d, want 73", action.RawBytes)
			}
		})
	}
}

func TestAgentMessageSemanticClassDoesNotInferUnknownContent(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		message *llm.AgentMessage
		want    string
	}{
		{name: "nil", want: "agent_message_invalid"},
		{name: "empty", message: &llm.AgentMessage{}, want: "agent_message_invalid"},
		{name: "plaintext", message: &llm.AgentMessage{Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText}}}, want: "agent_message_plaintext"},
		{name: "encrypted", message: &llm.AgentMessage{Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentEncryptedContent}}}, want: "agent_message_encrypted"},
		{name: "mixed", message: &llm.AgentMessage{Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText}, {Kind: llm.AgentMessageContentEncryptedContent}}}, want: "agent_message_mixed"},
		{name: "future kind stays invalid", message: &llm.AgentMessage{Content: []llm.AgentMessageContentPart{{Kind: "future_agent_content"}}}, want: "agent_message_invalid"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if got := agentMessageSemanticClass(test.message); got != test.want {
				t.Fatalf("semantic class = %q, want %q", got, test.want)
			}
		})
	}
}

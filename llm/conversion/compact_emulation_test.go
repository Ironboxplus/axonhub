package conversion

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

func TestPlannerModelsCompactAsRequestLevelCapability(t *testing.T) {
	t.Parallel()

	request := &llm.Request{
		RequestType: llm.RequestTypeCompact,
		APIFormat:   llm.APIFormatOpenAIResponseCompact,
		Compact:     &llm.CompactRequest{},
	}

	tests := []struct {
		name       string
		target     llm.APIFormat
		kind       ActionKind
		strategy   StrategyID
		reversible bool
	}{
		// Responses targets also emulate: live OpenAI-compatible gateways often
		// lack /v1/responses/compact, so native passthrough is not reliable.
		{name: "responses_emulated", target: llm.APIFormatOpenAIResponse, kind: ActionEmulate, strategy: StrategyCompactAsChat},
		{name: "chat_emulated", target: llm.APIFormatOpenAIChatCompletion, kind: ActionEmulate, strategy: StrategyCompactAsChat},
		{name: "anthropic_emulated", target: llm.APIFormatAnthropicMessage, kind: ActionEmulate, strategy: StrategyCompactAsChat},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			plan, err := NewPlanner().Plan(request, test.target)
			if err != nil {
				t.Fatalf("plan compact conversion: %v", err)
			}
			if !plan.Complete() || len(plan.Actions) != 1 {
				t.Fatalf("compact plan = %#v", plan)
			}
			action := plan.Actions[0]
			if action.Ref.Kind != ObjectCompaction || action.Kind != test.kind || action.Strategy != test.strategy ||
				action.Reversible != test.reversible {
				t.Fatalf("compact action = %#v", action)
			}
			if plan.Summary.Emulated != map[bool]uint32{true: 1, false: 0}[test.kind == ActionEmulate] || !plan.Summary.Complete {
				t.Fatalf("compact summary = %#v", plan.Summary)
			}
		})
	}
}

func TestPlannerRejectsCompactWithoutPayload(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanner().Plan(&llm.Request{
		RequestType: llm.RequestTypeCompact,
		APIFormat:   llm.APIFormatOpenAIResponseCompact,
	}, llm.APIFormatOpenAIChatCompletion)
	if err == nil || plan == nil || plan.Complete() || plan.Summary.Unknown != 1 {
		t.Fatalf("invalid compact plan = %#v err=%v", plan, err)
	}
}

func TestCompactEmulationDoesNotMutateCallerRequest(t *testing.T) {
	t.Parallel()

	input := "*** Begin Patch\n*** End Patch"
	request := &llm.Request{
		Model:       "fixture-model",
		RequestType: llm.RequestTypeCompact,
		APIFormat:   llm.APIFormatOpenAIResponseCompact,
		Compact: &llm.CompactRequest{
			Instructions: "Keep the patch state.",
			Input: []llm.Message{{Role: "assistant", ToolCalls: []llm.ToolCall{{
				ID: "call_patch", Type: llm.ToolTypeResponsesCustomTool,
				ResponseCustomToolCall: &llm.ResponseCustomToolCall{CallID: "call_patch", Name: "apply_patch", Input: input},
			}}}},
		},
	}
	target, err := openai.NewOutboundTransformer("https://provider.invalid/v1", "fixture-key")
	if err != nil {
		t.Fatalf("create Chat outbound: %v", err)
	}
	outbound := NewOutbound(target)
	raw, err := outbound.TransformRequest(
		llm.WithConversionDebugTrace(llm.WithConversionTrace(context.Background()), []byte(t.Name()), 8),
		request,
	)
	if err != nil {
		t.Fatalf("transform compact request: %v", err)
	}
	if request.RequestType != llm.RequestTypeCompact || request.Compact == nil || len(request.Messages) != 0 ||
		request.Compact.Instructions != "Keep the patch state." || request.Compact.Input[0].ToolCalls[0].ResponseCustomToolCall.Input != input {
		t.Fatalf("caller request was mutated: %#v", request)
	}
	if raw == nil || raw.URL != "https://provider.invalid/v1/chat/completions" ||
		!strings.Contains(string(raw.Body), `"role":"system"`) || !strings.Contains(string(raw.Body), `"name":"axc_`) {
		t.Fatalf("emulated raw request = %#v body=%s", raw, raw.Body)
	}
	summary, ok := SummaryFromRequest(raw)
	if !ok || !summary.Complete || summary.Emulated != 1 || summary.Lowered != 1 {
		t.Fatalf("compact conversion summary = %#v ok=%v", summary, ok)
	}
	debug, ok := DebugTraceFromRequest(raw)
	if !ok || debug == nil || len(debug.Actions) < 2 || debug.Actions[0].ObjectKind != string(ObjectCompaction) ||
		debug.Actions[0].Strategy != string(StrategyCompactAsChat) {
		t.Fatalf("compact conversion debug = %#v ok=%v", debug, ok)
	}
}

func TestRestoreCompactEmulationFailsClosedWhenCanonicalOutputCannotBeProjected(t *testing.T) {
	t.Parallel()
	session := &Session{compactEmulation: &compactEmulationState{instructions: "preserve private state"}}
	tests := []struct {
		name     string
		response *llm.Response
	}{
		{
			name: "encrypted agent message",
			response: &llm.Response{Output: []llm.Item{{
				Kind: llm.ItemKindAgentMessage,
				AgentMessage: &llm.AgentMessage{
					Author: "/root", Recipient: "/root/worker",
					Content: []llm.AgentMessageContentPart{{
						Kind: llm.AgentMessageContentEncryptedContent, EncryptedContent: "PRIVATE_AGENT_FRAGMENT",
					}},
				},
			}}},
		},
		{
			name: "unknown provider item",
			response: &llm.Response{Output: []llm.Item{{
				Kind:    llm.ItemKindUnknown,
				Unknown: &llm.UnknownItem{Type: "future_compact_output", Raw: []byte(`{"secret":"PRIVATE_AGENT_FRAGMENT"}`), Behavioral: true},
			}}},
		},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			// A compact response cannot safely fall back to an ordinary response:
			// the compact inbound transformer will reject it later, after provider
			// completion. It must neither flatten encryption nor drop unknown output.
			restored, err := restoreCompactEmulation(test.response, session)
			if !errors.Is(err, ErrIncompletePlan) || restored != nil || strings.Contains(err.Error(), "PRIVATE_AGENT_FRAGMENT") {
				t.Fatalf("unsafe compact restoration response=%#v err=%v", restored, err)
			}
		})
	}
}

func TestRestoreCompactEmulationRehydratesOnlySafeContinuationState(t *testing.T) {
	t.Parallel()
	session := &Session{compactEmulation: &compactEmulationState{instructions: "keep decisions"}}
	response := &llm.Response{
		ID: "compact_1", Created: 42,
		Output: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "state"}},
		}},
	}
	restored, err := restoreCompactEmulation(response, session)
	if err != nil || restored != response || restored.RequestType != llm.RequestTypeCompact ||
		restored.APIFormat != llm.APIFormatOpenAIResponseCompact || restored.Object != "response.compaction" ||
		restored.Compact == nil || restored.Compact.ID != "compact_1" || restored.Compact.CreatedAt != 42 ||
		restored.Compact.Instructions != "keep decisions" || len(restored.Compact.Output) != 1 ||
		restored.Compact.Output[0].Content.Content == nil || *restored.Compact.Output[0].Content.Content != "state" {
		t.Fatalf("safe compact restoration = %#v err=%v", restored, err)
	}

	legacyContent := "choice fallback"
	fallback := &llm.Response{Choices: []llm.Choice{{Message: &llm.Message{
		Role: "assistant", Content: llm.MessageContent{Content: &legacyContent},
	}}}}
	restored, err = restoreCompactEmulation(fallback, session)
	if err != nil || restored.Compact == nil || len(restored.Compact.Output) != 1 ||
		restored.Compact.Output[0].Content.Content == nil || *restored.Compact.Output[0].Content.Content != legacyContent {
		t.Fatalf("legacy compact fallback = %#v err=%v", restored, err)
	}

	unchanged := &llm.Response{ID: "ordinary"}
	restored, err = restoreCompactEmulation(unchanged, nil)
	if err != nil || restored != unchanged || restored.RequestType == llm.RequestTypeCompact {
		t.Fatalf("non-emulated response changed = %#v err=%v", restored, err)
	}
}

func TestRestoreCompactEmulationRejectsPlaintextAgentOutput(t *testing.T) {
	t.Parallel()
	session := &Session{compactEmulation: &compactEmulationState{}}
	response := &llm.Response{Output: []llm.Item{{
		Kind: llm.ItemKindAgentMessage,
		AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{
			{Kind: llm.AgentMessageContentInputText, Text: "output first"},
			{Kind: llm.AgentMessageContentInputText, Text: "output second"},
		}},
	}}}
	restored, err := restoreCompactEmulation(response, session)
	if !errors.Is(err, ErrIncompletePlan) || restored != nil || strings.Contains(err.Error(), "output first") || strings.Contains(err.Error(), "output second") {
		t.Fatalf("compact plaintext agent output=%#v err=%v", restored, err)
	}
}

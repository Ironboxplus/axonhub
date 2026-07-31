package conversion

import (
	"context"
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

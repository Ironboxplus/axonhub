package conversion

import (
	"context"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestConversionDebugTraceIsOptInBoundedAndRequestScoped(t *testing.T) {
	actions := []Action{
		{
			Ref:  ObjectRef{Kind: ObjectToolDefinition, ToolIndex: 0, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionLower, Strategy: StrategyCustomAsFunction, Reason: ReasonTargetFunctionOnly, Reversible: true,
		},
		{
			Ref:  ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: 2, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionLower, Strategy: StrategyClientToolAsFunc, Reason: ReasonTargetFunctionOnly, Reversible: true,
		},
		{
			Ref:  ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: 3, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionLower, Strategy: StrategyClientToolAsFunc, Reason: ReasonTargetFunctionOnly, Reversible: true,
		},
	}

	if trace := buildConversionDebugTrace(context.Background(), actions); trace != nil {
		t.Fatalf("debug trace must be allocation-free when not enabled: %#v", trace)
	}
	if allocations := testing.AllocsPerRun(1000, func() {
		if buildConversionDebugTrace(context.Background(), actions) != nil {
			panic("unexpected debug trace")
		}
	}); allocations != 0 {
		t.Fatalf("disabled debug trace allocated %.2f objects per plan", allocations)
	}

	ctx := llm.WithConversionDebugTrace(context.Background(), []byte("request-a"), 2)
	trace := buildConversionDebugTrace(ctx, actions)
	if trace == nil {
		t.Fatal("expected sampled debug trace")
	}
	if len(trace.Actions) != 2 || trace.Truncated != 1 {
		t.Fatalf("unexpected cap result: %#v", trace)
	}
	first := trace.Actions[0]
	if first.Seq != 1 || first.Direction != llm.ConversionDirectionRequest ||
		first.ObjectKind != string(ObjectToolDefinition) || first.Action != string(ActionLower) ||
		first.Strategy != string(StrategyCustomAsFunction) || first.Reason != string(ReasonTargetFunctionOnly) ||
		!first.Reversible || len(first.LogicalIDHash) != 16 {
		t.Fatalf("unexpected first action: %#v", first)
	}
	if first.LogicalIDHash == trace.Actions[1].LogicalIDHash {
		t.Fatalf("distinct logical objects must not collide in one trace: %#v", trace.Actions)
	}

	repeat := buildConversionDebugTrace(ctx, actions)
	if repeat.Actions[0].LogicalIDHash != first.LogicalIDHash {
		t.Fatalf("same request must produce stable hashes: %q != %q", repeat.Actions[0].LogicalIDHash, first.LogicalIDHash)
	}
	other := buildConversionDebugTrace(llm.WithConversionDebugTrace(context.Background(), []byte("request-b"), 2), actions)
	if other.Actions[0].LogicalIDHash == first.LogicalIDHash {
		t.Fatalf("logical hashes must not correlate across requests: %q", first.LogicalIDHash)
	}
}

func TestConversionDebugTraceLimitIsHardClamped(t *testing.T) {
	actions := make([]Action, llm.MaxConversionDebugActions+7)
	for index := range actions {
		actions[index] = Action{
			Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: index, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true,
		}
	}
	trace := buildConversionDebugTrace(
		llm.WithConversionDebugTrace(context.Background(), []byte("request-cap"), llm.MaxConversionDebugActions+100),
		actions,
	)
	if len(trace.Actions) != llm.MaxConversionDebugActions {
		t.Fatalf("hard cap was not enforced: %d", len(trace.Actions))
	}
	if trace.Truncated != 7 {
		t.Fatalf("unexpected truncated count: %d", trace.Truncated)
	}
}

func TestConversionDebugTraceAddsResponseAndStreamRestorationEvidence(t *testing.T) {
	trace := buildConversionDebugTrace(
		llm.WithConversionDebugTrace(context.Background(), []byte("request-restore"), 8),
		[]Action{{
			Ref:  ObjectRef{Kind: ObjectToolCall, ItemIndex: 1, ToolIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionLower, Strategy: StrategyCustomAsFunction, Reason: ReasonTargetFunctionOnly, Reversible: true,
		}},
	)
	session := &Session{plan: &Plan{Debug: trace}}
	session.recordDebug(
		llm.ConversionDirectionResponse,
		ObjectRef{Kind: ObjectToolCall, ItemIndex: 2, ToolIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		"restore", StrategyCustomAsFunction, ReasonSemanticProjection, true,
	)
	session.recordDebug(
		llm.ConversionDirectionStream,
		ObjectRef{Kind: ObjectToolCall, ItemIndex: 3, ToolIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		"restore_miss", StrategyCustomAsFunction, ReasonNoStrategy, false,
	)

	snapshot := session.DebugTrace()
	if snapshot == nil || len(snapshot.Actions) != 3 {
		t.Fatalf("restoration actions missing: %#v", snapshot)
	}
	if got := snapshot.Actions[1]; got.Direction != llm.ConversionDirectionResponse || got.Action != "restore" || !got.Reversible {
		t.Fatalf("response restoration action = %#v", got)
	}
	if got := snapshot.Actions[2]; got.Direction != llm.ConversionDirectionStream || got.Action != "restore_miss" || got.Reversible {
		t.Fatalf("stream restoration miss = %#v", got)
	}
}

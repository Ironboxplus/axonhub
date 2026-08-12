package conversion

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestConversionDebugTraceSampleIsBoundedAndRequestScoped(t *testing.T) {
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
		first.ObjectID != "tools[0]" || first.FieldPath != "tools[0]" ||
		first.Stage != llm.ConversionStageRequestPlanning || first.Result != llm.ConversionResultLowered ||
		first.Severity != llm.ConversionSeverityWarning || !first.Reversible || len(first.LogicalIDHash) != 16 {
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

func TestConversionRequiredEvidenceCapturesAttentionActionsWithoutSampling(t *testing.T) {
	actions := []Action{
		{
			Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: 0, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true,
		},
		{
			Ref:  ObjectRef{Kind: ObjectContentBlock, ToolIndex: -1, ItemIndex: 3, ContentIndex: 2, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy, Reversible: false,
		},
		{
			Ref:  ObjectRef{Kind: ObjectToolDefinition, ToolIndex: 4, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar, Reason: ReasonSameProtocolOpaque, Reversible: true,
		},
	}

	trace := buildConversionDebugTrace(context.Background(), actions)
	if trace == nil {
		t.Fatal("attention actions must create required evidence without sampling")
	}
	if trace.Mode != llm.ConversionEvidenceRequired || len(trace.Actions) != 2 || trace.Truncated != 0 {
		t.Fatalf("unexpected required trace: %#v", trace)
	}
	unknown := trace.Actions[0]
	if unknown.ObjectID != "input[3].content[2]" || unknown.FieldPath != "input[3].content[2]" ||
		unknown.Stage != llm.ConversionStageRequestPlanning || unknown.Result != llm.ConversionResultUnknown ||
		unknown.Severity != llm.ConversionSeverityCritical || unknown.LogicalIDHash != "" {
		t.Fatalf("unknown evidence is not actionable: %#v", unknown)
	}
	opaque := trace.Actions[1]
	if opaque.ObjectID != "tools[4]" || opaque.Result != llm.ConversionResultOpaque ||
		opaque.Severity != llm.ConversionSeverityWarning {
		t.Fatalf("opaque evidence is not actionable: %#v", opaque)
	}
}

func TestConversionRequiredEvidenceKeepsNativeOnlyDefaultPathAllocationFree(t *testing.T) {
	actions := []Action{{
		Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: 0, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true,
	}}
	if trace := buildConversionDebugTrace(context.Background(), actions); trace != nil {
		t.Fatalf("native-only request must not allocate evidence: %#v", trace)
	}
	if allocations := testing.AllocsPerRun(1000, func() {
		if buildConversionDebugTrace(context.Background(), actions) != nil {
			panic("unexpected required evidence")
		}
	}); allocations != 0 {
		t.Fatalf("native-only default trace allocated %.2f objects", allocations)
	}
}

func BenchmarkConversionRequiredEvidenceSingleUnknown(b *testing.B) {
	actions := []Action{{
		Ref: ObjectRef{
			Kind: ObjectContentBlock, ToolIndex: -1, ItemIndex: 3, ContentIndex: 2,
			MessageIndex: -1, ToolCallIndex: -1,
		},
		Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy,
	}}
	b.ReportAllocs()
	for b.Loop() {
		trace := buildConversionDebugTrace(context.Background(), actions)
		if trace == nil || len(trace.Actions) != 1 {
			b.Fatal("required evidence missing")
		}
	}
}

func TestConversionRequiredEvidenceSingleUnknownAllocationBudget(t *testing.T) {
	actions := []Action{{
		Ref: ObjectRef{
			Kind: ObjectContentBlock, ToolIndex: -1, ItemIndex: 3, ContentIndex: 2,
			MessageIndex: -1, ToolCallIndex: -1,
		},
		Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy,
	}}
	allocations := testing.AllocsPerRun(1000, func() {
		trace := buildConversionDebugTrace(context.Background(), actions)
		if trace == nil || len(trace.Actions) != 1 {
			panic("required evidence missing")
		}
	})
	if allocations > 3 {
		t.Fatalf("single required evidence allocated %.2f objects, want <= 3", allocations)
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
	if got := snapshot.Actions[1]; got.Stage != llm.ConversionStageResponseRestore || got.Result != llm.ConversionResultRestored ||
		got.Severity != llm.ConversionSeverityInfo || got.ObjectID != "output[2].tool_call" {
		t.Fatalf("response restoration location = %#v", got)
	}
	if got := snapshot.Actions[2]; got.Direction != llm.ConversionDirectionStream || got.Action != "restore_miss" || got.Reversible ||
		got.Stage != llm.ConversionStageStreamRestore || got.Result != llm.ConversionResultRestoreMiss ||
		got.Severity != llm.ConversionSeverityCritical || got.ObjectID != "output[3].tool_call" {
		t.Fatalf("stream restoration miss = %#v", got)
	}
}

func TestConversionRestoreMissLazilyCreatesRequiredEvidence(t *testing.T) {
	session := &Session{plan: &Plan{}}
	ref := ObjectRef{Kind: ObjectToolCall, ItemIndex: 7, ToolIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
	session.recordDebug(
		llm.ConversionDirectionResponse,
		ref,
		"restore_miss",
		StrategyCustomAsFunction,
		ReasonNoStrategy,
		false,
	)

	trace := session.DebugTrace()
	if trace == nil || trace.Mode != llm.ConversionEvidenceRequired || len(trace.Actions) != 1 {
		t.Fatalf("restore miss evidence = %#v", trace)
	}
	action := trace.Actions[0]
	if action.ObjectID != "output[7].tool_call" || action.FieldPath != "output[7].tool_call" ||
		action.Stage != llm.ConversionStageResponseRestore || action.Result != llm.ConversionResultRestoreMiss ||
		action.Severity != llm.ConversionSeverityCritical {
		t.Fatalf("restore miss evidence is not actionable: %#v", action)
	}
}

func TestResponseOutputBlockersCarrySelectedEvidenceAndStreamsDeduplicateSafely(t *testing.T) {
	t.Parallel()
	raw := []byte(`{ "type":"future_behavior", "secret":"PRIVATE" }`)
	digest := sha256.Sum256(raw)
	blocked := llm.Item{
		Kind:          llm.ItemKindUnknown,
		Unknown:       &llm.UnknownItem{Type: "future_behavior", Raw: append(json.RawMessage(nil), raw...), Behavioral: true},
		ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "future_behavior", SourceBytes: uint32(len(raw)), SourceDigest: fmt.Sprintf("%x", digest)},
	}
	session := &Session{plan: &Plan{Source: llm.APIFormatOpenAIChatCompletion, Target: CapabilityProfile{APIFormat: llm.APIFormatOpenAIResponse}}, outputBlockers: make(map[string]struct{})}
	response := RestoreResponse(&llm.Response{Output: []llm.Item{blocked}}, session)
	if response == nil {
		t.Fatal("restore response returned nil")
	}
	trace := session.DebugTrace()
	if trace == nil || len(trace.Actions) != 1 {
		t.Fatalf("response blocker trace=%#v", trace)
	}
	assertOutputBlockerEvidence(t, trace.Actions[0], llm.ConversionDirectionResponse, "output[0]", "future_behavior", "future_unknown_behavioral", uint32(len(raw)), fmt.Sprintf("%x", digest))

	streamSession := &Session{plan: &Plan{Source: llm.APIFormatOpenAIChatCompletion, Target: CapabilityProfile{APIFormat: llm.APIFormatOpenAIResponse}}, outputBlockers: make(map[string]struct{})}
	index := 2
	ref := llm.ItemRef{ItemID: "same-item", OutputIndex: &index}
	stream := &llm.Response{Events: []llm.Event{
		{Kind: llm.EventKindItemAdded, Sequence: 1, ItemRef: ref, Snapshot: cloneOutputBlockerItem(&blocked)},
		{Kind: llm.EventKindItemDone, Sequence: 2, ItemRef: ref, Snapshot: cloneOutputBlockerItem(&blocked)},
	}}
	newStreamRestorer(streamSession).restore(stream)
	trace = streamSession.DebugTrace()
	if trace == nil || len(trace.Actions) != 1 {
		t.Fatalf("stream paired blocker trace=%#v", trace)
	}
	assertOutputBlockerEvidence(t, trace.Actions[0], llm.ConversionDirectionStream, "output[2]", "future_behavior", "future_unknown_behavioral", uint32(len(raw)), fmt.Sprintf("%x", digest))

	malformedSession := &Session{plan: &Plan{Source: llm.APIFormatOpenAIChatCompletion, Target: CapabilityProfile{APIFormat: llm.APIFormatOpenAIResponse}}, outputBlockers: make(map[string]struct{})}
	malformed := &llm.Response{Events: []llm.Event{
		{Kind: llm.EventKindItemAdded, Sequence: 11, ItemRef: llm.ItemRef{}, Snapshot: cloneOutputBlockerItem(&blocked)},
		{Kind: llm.EventKindItemAdded, Sequence: 12, ItemRef: llm.ItemRef{}, Snapshot: cloneOutputBlockerItem(&blocked)},
	}}
	newStreamRestorer(malformedSession).restore(malformed)
	trace = malformedSession.DebugTrace()
	if trace == nil || len(trace.Actions) != 2 {
		t.Fatalf("malformed stream blockers collided: %#v", trace)
	}
}

func TestResponsesProviderOutputBlockerRouteIsDirectional(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		source llm.APIFormat
		target llm.APIFormat
		want   bool
	}{
		{name: "chat client through Responses provider", source: llm.APIFormatOpenAIChatCompletion, target: llm.APIFormatOpenAIResponse, want: true},
		{name: "Anthropic client through Responses provider", source: llm.APIFormatAnthropicMessage, target: llm.APIFormatOpenAIResponse, want: true},
		{name: "Responses identity", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIResponse},
		{name: "Responses client through Chat provider", source: llm.APIFormatOpenAIResponse, target: llm.APIFormatOpenAIChatCompletion},
		{name: "Chat client through Anthropic provider", source: llm.APIFormatOpenAIChatCompletion, target: llm.APIFormatAnthropicMessage},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			session := &Session{plan: &Plan{Source: test.source, Target: CapabilityProfile{APIFormat: test.target}}}
			if got := shouldRecordResponsesProviderOutputBlocker(session); got != test.want {
				t.Fatalf("source=%s target=%s blocker=%v, want %v", test.source, test.target, got, test.want)
			}
		})
	}
}

func cloneOutputBlockerItem(item *llm.Item) *llm.Item {
	clone := *item
	if item.Unknown != nil {
		unknown := *item.Unknown
		unknown.Raw = append(json.RawMessage(nil), item.Unknown.Raw...)
		clone.Unknown = &unknown
	}
	return &clone
}

func assertOutputBlockerEvidence(
	t *testing.T,
	action llm.ConversionActionTrace,
	direction llm.ConversionDirection,
	objectID, sourceType, semantic string,
	rawBytes uint32,
	digest string,
) {
	t.Helper()
	if action.Direction != direction || action.ObjectID != objectID || action.FieldPath != objectID || action.Action != string(ActionUnknown) ||
		action.Result != llm.ConversionResultUnknown || action.Severity != llm.ConversionSeverityCritical || action.Reversible ||
		action.SourceType != sourceType || action.SemanticClass != semantic || action.RawBytes != rawBytes || action.SourceDigest != digest {
		t.Fatalf("output blocker evidence=%#v", action)
	}
}

func TestConversionEvidenceClassifiersAndLocationsCoverEveryBranch(t *testing.T) {
	planCases := []struct {
		kind     ActionKind
		result   llm.ConversionEvidenceResult
		severity llm.ConversionEvidenceSeverity
	}{
		{ActionNative, llm.ConversionResultNative, llm.ConversionSeverityInfo},
		{ActionLower, llm.ConversionResultLowered, llm.ConversionSeverityWarning},
		{ActionEmulate, llm.ConversionResultEmulated, llm.ConversionSeverityWarning},
		{ActionOpaque, llm.ConversionResultOpaque, llm.ConversionSeverityWarning},
		{ActionUnknown, llm.ConversionResultUnknown, llm.ConversionSeverityCritical},
	}
	for _, test := range planCases {
		result, severity := planEvidenceResult(test.kind)
		if result != test.result || severity != test.severity {
			t.Fatalf("plan evidence %q = (%q,%q), want (%q,%q)", test.kind, result, severity, test.result, test.severity)
		}
	}

	runtimeCases := []struct {
		action     string
		reversible bool
		result     llm.ConversionEvidenceResult
		severity   llm.ConversionEvidenceSeverity
	}{
		{"restore", true, llm.ConversionResultRestored, llm.ConversionSeverityInfo},
		{"restore_miss", false, llm.ConversionResultRestoreMiss, llm.ConversionSeverityCritical},
		{"normalize", true, llm.ConversionResultNormalized, llm.ConversionSeverityInfo},
		{"normalize", false, llm.ConversionResultNormalized, llm.ConversionSeverityWarning},
		{"repair", true, llm.ConversionResultRepaired, llm.ConversionSeverityWarning},
		{"future_action", true, llm.ConversionResultNative, llm.ConversionSeverityInfo},
		{"future_action", false, llm.ConversionResultUnknown, llm.ConversionSeverityCritical},
	}
	for _, test := range runtimeCases {
		result, severity := runtimeEvidenceResult(test.action, test.reversible)
		if result != test.result || severity != test.severity {
			t.Fatalf("runtime evidence %q/%v = (%q,%q), want (%q,%q)", test.action, test.reversible, result, severity, test.result, test.severity)
		}
	}

	stageCases := map[llm.ConversionDirection]llm.ConversionEvidenceStage{
		llm.ConversionDirectionRequest:  llm.ConversionStageRequestTransform,
		llm.ConversionDirectionResponse: llm.ConversionStageResponseRestore,
		llm.ConversionDirectionStream:   llm.ConversionStageStreamRestore,
	}
	for direction, want := range stageCases {
		if got := runtimeEvidenceStage(direction); got != want {
			t.Fatalf("stage %q = %q, want %q", direction, got, want)
		}
	}

	locationCases := []struct {
		direction llm.ConversionDirection
		ref       ObjectRef
		strategy  StrategyID
		objectID  string
		fieldPath string
	}{
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolDefinition, ToolIndex: -1, ItemIndex: -1}, StrategyNative, "tools", "tools"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolDefinition, ToolIndex: 2, ItemIndex: 4}, StrategySchemaNormalize, "input[4].tools[2]", "input[4].tools[2].parameters"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolChoice}, StrategyNative, "tool_choice", "tool_choice"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolCall, ItemIndex: -1, MessageIndex: 5, ToolCallIndex: 1}, StrategyNative, "messages[5].tool_calls[1]", "messages[5].tool_calls[1]"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolCall, ItemIndex: -1, MessageIndex: -1}, StrategyNative, "tool_call", "tool_call"},
		{llm.ConversionDirectionResponse, ObjectRef{Kind: ObjectToolResult, ItemIndex: 2}, StrategyNative, "output[2].tool_result", "output[2].tool_result"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolResult, ItemIndex: -1, MessageIndex: 3, ToolCallIndex: 1}, StrategyNative, "messages[3].tool_results[1]", "messages[3].tool_results[1]"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolResult, ItemIndex: -1, MessageIndex: 3, ToolCallIndex: -1}, StrategyNative, "messages[3].tool_result", "messages[3].tool_result"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectToolResult, ItemIndex: -1, MessageIndex: -1}, StrategyNative, "tool_result", "tool_result"},
		{llm.ConversionDirectionResponse, ObjectRef{Kind: ObjectContentBlock, ItemIndex: 6, ContentIndex: -1}, StrategyNative, "output[6].content", "output[6].content"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectProviderData, ItemIndex: -1}, StrategyNative, "provider_data", "provider_data"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectProviderData, ItemIndex: 2}, StrategyNative, "input[2].provider_data", "input[2].provider_data"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectRequestControl, ItemIndex: 8}, StrategyRequestControl, "input[8]", "input[8].position"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectCompaction, ItemIndex: 9}, StrategyNative, "input[9]", "input[9]"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectInputItem, ItemIndex: -1}, StrategyNative, "input", "input"},
		{llm.ConversionDirectionRequest, ObjectRef{Kind: ObjectKind("future_kind")}, StrategyNative, "future_kind", "future_kind"},
	}
	for _, test := range locationCases {
		objectID, fieldPath := objectEvidenceLocation(test.direction, test.ref, test.strategy)
		if objectID != test.objectID || fieldPath != test.fieldPath {
			t.Fatalf("location %+v = (%q,%q), want (%q,%q)", test.ref, objectID, fieldPath, test.objectID, test.fieldPath)
		}
	}

	if !requiredRuntimeEvidence(llm.ConversionActionTrace{Result: llm.ConversionResultRestoreMiss}) ||
		!requiredRuntimeEvidence(llm.ConversionActionTrace{Result: llm.ConversionResultNormalized}) ||
		!requiredRuntimeEvidence(llm.ConversionActionTrace{Result: llm.ConversionResultRepaired}) ||
		!requiredRuntimeEvidence(llm.ConversionActionTrace{Severity: llm.ConversionSeverityWarning}) ||
		requiredRuntimeEvidence(llm.ConversionActionTrace{Severity: llm.ConversionSeverityInfo}) {
		t.Fatal("required runtime evidence classifier is incomplete")
	}
}

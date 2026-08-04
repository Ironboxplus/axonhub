package llm

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestNormalizeRequestControlsMovesFinalControlWithoutMutatingSource(t *testing.T) {
	input := []Item{
		{Kind: ItemKindMessage, Role: RoleUser, Content: []ContentBlock{{Kind: ContentKindText, Text: "before"}}},
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
		{Kind: ItemKindReasoning, Reasoning: &ReasoningItem{Content: "restored"}},
	}

	normalized, adjustments, err := NormalizeRequestControls(input)
	if err != nil {
		t.Fatalf("normalize request controls: %v", err)
	}
	if len(adjustments) != 1 || adjustments[0].Kind != ItemKindCompactionTrigger ||
		adjustments[0].FromIndex != 1 || adjustments[0].ToIndex != 2 {
		t.Fatalf("unexpected control adjustments: %#v", adjustments)
	}
	if normalized[len(normalized)-1].Kind != ItemKindCompactionTrigger {
		t.Fatalf("final request control was not moved to the end: %#v", normalized)
	}
	if input[1].Kind != ItemKindCompactionTrigger || input[2].Kind != ItemKindReasoning {
		t.Fatalf("source input was mutated: %#v", input)
	}
	// A normalized result must be attempt-safe, not an alias of the source.
	normalized[0].Content[0].Text = "changed"
	if input[0].Content[0].Text != "before" {
		t.Fatalf("normalized input aliases source input")
	}
}

func TestNormalizeRequestControlsRejectsDuplicateUniqueControl(t *testing.T) {
	input := []Item{
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
	}

	_, _, err := NormalizeRequestControls(input)
	var controlErr *RequestControlError
	if !errors.As(err, &controlErr) || controlErr.Kind != ItemKindCompactionTrigger ||
		controlErr.Code != RequestControlDuplicate {
		t.Fatalf("expected typed duplicate control error, got %T %v", err, err)
	}
}

func TestConversationPersistentItemsExcludeRequestScopedControls(t *testing.T) {
	input := []Item{
		{Kind: ItemKindMessage, Role: RoleUser, Content: []ContentBlock{{Kind: ContentKindText, Text: "persist"}}},
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
	}

	persistent := ConversationPersistentItems(input)
	if len(persistent) != 1 || persistent[0].Kind != ItemKindMessage {
		t.Fatalf("request-scoped control leaked into conversation history: %#v", persistent)
	}
	persistent[0].Content[0].Text = "changed"
	if input[0].Content[0].Text != "persist" {
		t.Fatalf("persistent items alias request input")
	}
}

func TestConversationPersistentItemsFiltersOnlyPreciselyMatchedLegacyResponsesControl(t *testing.T) {
	unconfiguredKind := ItemKind("legacy_unconfigured_control")
	requestControlPolicies[unconfiguredKind] = RequestControlPolicy{Scope: RequestControlScopeCurrentRequest}
	t.Cleanup(func() { delete(requestControlPolicies, unconfiguredKind) })

	legacy := Item{
		Kind: ItemKindUnknown,
		Unknown: &UnknownItem{
			Type: "compaction_trigger", Raw: json.RawMessage(`{"type":"compaction_trigger"}`), Behavioral: true,
		},
		ProtocolHints: ProtocolHints{
			SourceFormat: APIFormatOpenAIResponse, SourceType: "compaction_trigger",
		},
	}
	persistent := ConversationPersistentItems([]Item{
		{Kind: ItemKindMessage, Role: RoleUser, Content: []ContentBlock{{Kind: ContentKindText, Text: "keep"}}},
		legacy,
	})
	if len(persistent) != 1 || persistent[0].Kind != ItemKindMessage {
		t.Fatalf("precise legacy request control was persisted: %#v", persistent)
	}

	mismatches := []Item{legacy, legacy, legacy, legacy, legacy}
	mismatches[0].ProtocolHints.SourceFormat = APIFormatOpenAIChatCompletion
	mismatches[1].ProtocolHints.SourceType = "future_control"
	mismatches[2].Unknown = &UnknownItem{Type: "future_control", Raw: legacy.Unknown.Raw, Behavioral: true}
	mismatches[3].Unknown = &UnknownItem{Type: "compaction_trigger", Raw: json.RawMessage(`{"type":"future_control"}`), Behavioral: true}
	mismatches[4].Unknown = &UnknownItem{Type: "compaction_trigger", Raw: json.RawMessage(`not-json`), Behavioral: true}
	for index := range mismatches {
		retained := ConversationPersistentItems([]Item{mismatches[index]})
		if len(retained) != 1 || retained[0].Kind != ItemKindUnknown {
			t.Fatalf("legacy mismatch %d was incorrectly removed: %#v", index, retained)
		}
	}
}

func TestNormalizeRequestControlsOrdinaryRequestAllocatesNothing(t *testing.T) {
	input := []Item{{Kind: ItemKindMessage}, {Kind: ItemKindReasoning}}
	allocations := testing.AllocsPerRun(100, func() {
		normalized, adjustments, err := NormalizeRequestControls(input)
		if err != nil || len(adjustments) != 0 || len(normalized) != len(input) {
			panic("unexpected normalization result")
		}
	})
	if allocations != 0 {
		t.Fatalf("ordinary request normalization allocated %.2f objects, want 0", allocations)
	}
}

func TestNormalizeRequestControlsHonorsRegisteredMaxCount(t *testing.T) {
	kind := ItemKind("test_control_max_two")
	requestControlPolicies[kind] = RequestControlPolicy{MaxCount: 2}
	t.Cleanup(func() { delete(requestControlPolicies, kind) })

	input := []Item{{Kind: kind}, {Kind: kind}}
	normalized, adjustments, err := NormalizeRequestControls(input)
	if err != nil || len(adjustments) != 0 || len(normalized) != len(input) {
		t.Fatalf("two registered controls should be accepted: normalized=%#v adjustments=%#v err=%v", normalized, adjustments, err)
	}

	input = append(input, Item{Kind: kind})
	_, _, err = NormalizeRequestControls(input)
	var controlErr *RequestControlError
	if !errors.As(err, &controlErr) || controlErr.FirstIndex != 0 || controlErr.Index != 2 {
		t.Fatalf("third registered control should be rejected with stable indexes, got %T %v", err, err)
	}
}

func TestRequestControlCoverageBranches(t *testing.T) {
	if got := (*RequestControlError)(nil).Error(); got != "request control error" {
		t.Fatalf("nil request control error text = %q", got)
	}
	errText := (&RequestControlError{Kind: ItemKindCompactionTrigger, FirstIndex: 1, Index: 3}).Error()
	if errText != `request control "compaction_trigger" is duplicated at input indexes 1 and 3` {
		t.Fatalf("request control error text = %q", errText)
	}

	normalized, adjustments, err := NormalizeRequestControls(nil)
	if err != nil || normalized != nil || adjustments != nil {
		t.Fatalf("empty normalization = %#v %#v %v", normalized, adjustments, err)
	}

	finalInput := []Item{
		{Kind: ItemKindMessage},
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
	}
	normalized, adjustments, err = NormalizeRequestControls(finalInput)
	if err != nil || len(adjustments) != 0 || &normalized[0] != &finalInput[0] {
		t.Fatalf("already-final input should be returned unchanged: %#v %#v %v", normalized, adjustments, err)
	}

	if persistent := ConversationPersistentItems(nil); persistent != nil {
		t.Fatalf("empty persistent input = %#v", persistent)
	}
}

func TestNormalizeRequestControlsMultipleFinalPoliciesKeepStableOrder(t *testing.T) {
	kind := ItemKind("test_final_control")
	requestControlPolicies[kind] = RequestControlPolicy{Placement: RequestControlPlacementFinal, MaxCount: 1}
	t.Cleanup(func() { delete(requestControlPolicies, kind) })

	input := []Item{
		{Kind: kind},
		{Kind: ItemKindMessage},
		{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}},
	}
	normalized, adjustments, err := NormalizeRequestControls(input)
	if err != nil {
		t.Fatalf("normalize multiple final policies: %v", err)
	}
	if len(normalized) != 3 || normalized[0].Kind != ItemKindMessage || normalized[1].Kind != kind || normalized[2].Kind != ItemKindCompactionTrigger {
		t.Fatalf("final policy order changed: %#v", normalized)
	}
	if len(adjustments) != 1 || adjustments[0].Kind != kind || adjustments[0].FromIndex != 0 || adjustments[0].ToIndex != 1 {
		t.Fatalf("unexpected adjustments: %#v", adjustments)
	}
}

func TestConversationPersistentItemsRetainsNonRequestScopedRegisteredControl(t *testing.T) {
	kind := ItemKind("test_persistent_control")
	requestControlPolicies[kind] = RequestControlPolicy{}
	t.Cleanup(func() { delete(requestControlPolicies, kind) })

	persistent := ConversationPersistentItems([]Item{{Kind: kind}})
	if len(persistent) != 1 || persistent[0].Kind != kind {
		t.Fatalf("non-request-scoped control was removed: %#v", persistent)
	}
}

func BenchmarkNormalizeRequestControlsOrdinary512Items(b *testing.B) {
	input := make([]Item, 512)
	for index := range input {
		input[index] = Item{Kind: ItemKindMessage}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := NormalizeRequestControls(input); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkNormalizeRequestControlsRelocatesFinalControl(b *testing.B) {
	input := make([]Item, 512)
	for index := range input {
		input[index] = Item{Kind: ItemKindMessage}
	}
	input[128] = Item{Kind: ItemKindCompactionTrigger, CompactionTrigger: &CompactionTriggerItem{}}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, _, err := NormalizeRequestControls(input); err != nil {
			b.Fatal(err)
		}
	}
}

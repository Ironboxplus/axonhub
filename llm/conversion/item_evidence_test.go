package conversion

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestPlannerItemEvidenceDistinguishesCompactionTriggerAndFutureUnknown(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		item       llm.Item
		ref        ObjectKind
		sourceType string
		semantic   string
		reason     ReasonCode
	}{
		{
			name: "compaction checkpoint", ref: ObjectCompaction,
			sourceType: "compaction_summary", semantic: "compaction_checkpoint", reason: ReasonProviderPrivate,
			item: llm.Item{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{}, ProtocolHints: llm.ProtocolHints{
				SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "compaction_summary", SourceBytes: 71, SourceDigest: testEvidenceDigest("compaction"),
			}},
		},
		{
			name: "compaction trigger request control", ref: ObjectRequestControl,
			sourceType: "compaction_trigger", semantic: "compaction_trigger_request_control", reason: ReasonProviderPrivate,
			item: llm.Item{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}, ProtocolHints: llm.ProtocolHints{
				SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "compaction_trigger", SourceBytes: 29, SourceDigest: testEvidenceDigest("trigger"),
			}},
		},
		{
			name: "context compaction checkpoint", ref: ObjectContextCompaction,
			sourceType: "context_compaction", semantic: "context_compaction_checkpoint", reason: ReasonProviderPrivate,
			item: llm.Item{Kind: llm.ItemKindContextCompaction, ContextCompaction: &llm.ContextCompactionItem{}, ProtocolHints: llm.ProtocolHints{
				SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "context_compaction", SourceBytes: 61, SourceDigest: testEvidenceDigest("context"),
			}},
		},
		{
			name: "future unknown behavioral", ref: ObjectInputItem,
			sourceType: "future_context_checkpoint", semantic: "future_unknown_behavioral", reason: ReasonNoStrategy,
			item: llm.Item{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{
				Type: "future_context_checkpoint", Raw: json.RawMessage(`{"type":"future_context_checkpoint","secret":"PRIVATE"}`), Behavioral: true,
			}, ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse, SourceBytes: 58, SourceDigest: testEvidenceDigest("future")}},
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			request := &llm.Request{APIFormat: llm.APIFormatOpenAIResponse, Input: []llm.Item{test.item}}
			identity, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
			if err != nil || len(identity.Actions) != 1 {
				t.Fatalf("identity plan = %#v err=%v", identity, err)
			}
			assertItemEvidenceAction(t, identity.Actions[0], test.ref, test.sourceType, test.semantic, test.item.ProtocolHints.SourceBytes, test.item.ProtocolHints.SourceDigest)
			if test.item.Kind == llm.ItemKindUnknown {
				if identity.Actions[0].Kind != ActionOpaque || identity.Actions[0].Reason != ReasonSameProtocolOpaque || !identity.Actions[0].Reversible {
					t.Fatalf("identity future unknown action = %#v", identity.Actions[0])
				}
			} else if identity.Actions[0].Kind != ActionNative || identity.Actions[0].Reason != ReasonTargetNative || !identity.Actions[0].Reversible {
				t.Fatalf("identity typed control action = %#v", identity.Actions[0])
			}

			cross, err := NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
			if !errors.Is(err, ErrIncompletePlan) || cross == nil || cross.Complete() || cross.Summary.Unknown != 1 || len(cross.Actions) != 1 {
				t.Fatalf("cross plan = %#v err=%v", cross, err)
			}
			action := cross.Actions[0]
			assertItemEvidenceAction(t, action, test.ref, test.sourceType, test.semantic, test.item.ProtocolHints.SourceBytes, test.item.ProtocolHints.SourceDigest)
			if action.Kind != ActionUnknown || action.Strategy != StrategyUnavailable || action.Reason != test.reason || action.Reversible {
				t.Fatalf("cross fail-closed action = %#v", action)
			}
			if cross.Debug == nil || len(cross.Debug.Actions) != 1 {
				t.Fatalf("cross debug = %#v", cross.Debug)
			}
			evidence := cross.Debug.Actions[0]
			if evidence.ObjectKind != string(test.ref) || evidence.ObjectID != "input[0]" || evidence.FieldPath != "input[0]" ||
				evidence.SourceType != test.sourceType || evidence.SemanticClass != test.semantic || evidence.RawBytes != test.item.ProtocolHints.SourceBytes ||
				evidence.SourceDigest != test.item.ProtocolHints.SourceDigest {
				t.Fatalf("cross payload-free evidence = %#v", evidence)
			}
			encoded, marshalErr := json.Marshal(cross.Debug)
			if marshalErr != nil || string(encoded) == "" || containsPrivateEvidence(encoded) {
				t.Fatalf("cross evidence leaked payload: err=%v debug=%s", marshalErr, encoded)
			}
		})
	}
}

func TestItemEvidenceDigestIsStrictAndOrdinaryItemsStayEmpty(t *testing.T) {
	t.Parallel()
	valid := testEvidenceDigest("selected raw union")
	item := &llm.Item{Kind: llm.ItemKindAgentMessage, ProtocolHints: llm.ProtocolHints{SourceBytes: 77, SourceDigest: valid}}
	action := itemEvidenceAction(item, 0, ObjectAgentMessage, "agent_message", "agent_message")
	if action.SourceDigest != valid || action.RawBytes != 77 {
		t.Fatalf("selected item evidence=%#v", action)
	}
	for _, forged := range []string{"", strings.Repeat("A", 64), strings.Repeat("a", 63), strings.Repeat("g", 64), "PRIVATE_DIGEST"} {
		item.ProtocolHints.SourceDigest = forged
		action = itemEvidenceAction(item, 0, ObjectAgentMessage, "agent_message", "agent_message")
		if action.SourceDigest != "" {
			t.Fatalf("forged digest entered evidence: %q -> %#v", forged, action)
		}
	}
	ordinary := &llm.Item{Kind: llm.ItemKindMessage}
	if action := itemEvidenceAction(ordinary, 0, ObjectInputItem, "message", "message"); action.RawBytes != 0 || action.SourceDigest != "" {
		t.Fatalf("ordinary item evidence=%#v", action)
	}
}

func TestContextCompactionCriticalEvidenceSurvivesLargeNativePlan(t *testing.T) {
	t.Parallel()
	items := make([]llm.Item, 0, 2001)
	for index := 0; index < 2000; index++ {
		items = append(items, llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "ordinary"}}})
	}
	items = append(items, llm.Item{
		Kind: llm.ItemKindContextCompaction, ContextCompaction: &llm.ContextCompactionItem{},
		ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "context_compaction", SourceBytes: 73, SourceDigest: testEvidenceDigest("context-large")},
	})
	plan, err := NewPlanner().Plan(&llm.Request{APIFormat: llm.APIFormatOpenAIResponse, Input: items}, llm.APIFormatOpenAIChatCompletion)
	if !errors.Is(err, ErrIncompletePlan) || plan == nil || plan.Summary.Unknown != 1 || plan.Summary.PlanVersion != PlanVersion || plan.Debug == nil ||
		plan.Debug.Mode != llm.ConversionEvidenceRequired || len(plan.Debug.Actions) != 1 {
		t.Fatalf("large context blocker plan=%#v err=%v", plan, err)
	}
	action := plan.Actions[len(plan.Actions)-1]
	evidence := plan.Debug.Actions[0]
	if action.Ref.Kind != ObjectContextCompaction || action.SourceDigest == "" || evidence.ObjectID != "input[2000]" || evidence.FieldPath != "input[2000]" ||
		evidence.SourceType != "context_compaction" || evidence.SemanticClass != "context_compaction_checkpoint" || evidence.SourceDigest != action.SourceDigest || evidence.RawBytes != 73 {
		t.Fatalf("large context critical evidence action=%#v evidence=%#v", action, evidence)
	}
}

func TestItemEvidenceSourceTypeIsBoundedWithoutMutatingUnknownIdentity(t *testing.T) {
	t.Parallel()
	invalid := []string{
		"", "future/context", "future.context", "future:context", "FUTURE", "future\ncontext", "未来", strings.Repeat("a", 97),
	}
	for _, sourceType := range invalid {
		sourceType := sourceType
		t.Run("invalid", func(t *testing.T) {
			t.Parallel()
			item := &llm.Item{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{Type: sourceType, Behavioral: true}, ProtocolHints: llm.ProtocolHints{SourceType: sourceType}}
			action := itemEvidenceAction(item, 4, ObjectInputItem, "unknown", unknownItemSemanticClass(item))
			if action.SourceType != "unknown" || item.Unknown.Type != sourceType || item.ProtocolHints.SourceType != sourceType {
				t.Fatalf("unsafe source evidence action=%#v item=%#v", action, item)
			}
		})
	}
	if got := itemEvidenceSourceType(&llm.Item{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{Type: "future_context", Behavioral: true}}, "unknown"); got != "future_context" {
		t.Fatalf("safe unknown fallback type = %q", got)
	}
	if got := itemEvidenceSourceType(nil, "future/context"); got != "unknown" {
		t.Fatalf("unsafe fallback escaped bounded evidence type = %q", got)
	}
	if got := unknownItemSemanticClass(nil); got != "future_unknown_nonbehavioral" {
		t.Fatalf("nil unknown semantic class = %q", got)
	}
	if got := unknownItemSemanticClass(&llm.Item{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{}}); got != "future_unknown_nonbehavioral" {
		t.Fatalf("non-behavioral unknown semantic class = %q", got)
	}
	for _, sourceType := range []string{"future_context", "future-context", "future_context_2"} {
		if !safeEvidenceSourceType(sourceType) {
			t.Fatalf("safe source type rejected: %q", sourceType)
		}
	}
}

func assertItemEvidenceAction(t *testing.T, action Action, ref ObjectKind, sourceType, semantic string, rawBytes uint32, sourceDigest string) {
	t.Helper()
	if action.Ref.Kind != ref || action.Ref.ItemIndex != 0 || action.SourceType != sourceType ||
		action.SemanticClass != semantic || action.RawBytes != rawBytes || action.SourceDigest != sourceDigest {
		t.Fatalf("item evidence action = %#v", action)
	}
}

func testEvidenceDigest(value string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(value)))
}

func containsPrivateEvidence(encoded []byte) bool {
	return string(encoded) != "" && (string(encoded) == "PRIVATE" || containsString(encoded, "PRIVATE"))
}

func containsString(value []byte, want string) bool {
	for index := 0; index+len(want) <= len(value); index++ {
		if string(value[index:index+len(want)]) == want {
			return true
		}
	}
	return false
}

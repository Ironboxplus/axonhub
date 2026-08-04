package llm

import (
	"encoding/json"
	"fmt"
)

type RequestControlScope string

const (
	RequestControlScopeCurrentRequest RequestControlScope = "current_request"
)

type RequestControlPlacement string

const (
	RequestControlPlacementFinal RequestControlPlacement = "final"
)

// RequestControlPolicy describes lifecycle rules for canonical objects whose
// behavior applies to the current provider request rather than conversation
// history. Adding a future control requires one registry entry instead of
// scattered protocol-specific conditionals.
type RequestControlPolicy struct {
	Scope              RequestControlScope
	Placement          RequestControlPlacement
	MaxCount           int
	LegacySourceFormat APIFormat
	LegacySourceType   string
}

var requestControlPolicies = map[ItemKind]RequestControlPolicy{
	ItemKindCompactionTrigger: {
		Scope:              RequestControlScopeCurrentRequest,
		Placement:          RequestControlPlacementFinal,
		MaxCount:           1,
		LegacySourceFormat: APIFormatOpenAIResponse,
		LegacySourceType:   "compaction_trigger",
	},
}

func RequestControlPolicyFor(kind ItemKind) (RequestControlPolicy, bool) {
	policy, ok := requestControlPolicies[kind]
	return policy, ok
}

// requestControlKindForPersistence recognizes both current canonical request
// controls and the exact unknown/raw representation produced by older Axon
// versions. Every legacy discriminator must agree so an unrelated future
// unknown object is never deleted from conversation history.
func requestControlKindForPersistence(item Item) (ItemKind, bool) {
	if _, controlled := RequestControlPolicyFor(item.Kind); controlled {
		return item.Kind, true
	}
	if item.Kind != ItemKindUnknown || item.Unknown == nil || !item.Unknown.Behavioral {
		return "", false
	}

	var raw struct {
		Type string `json:"type"`
	}
	if len(item.Unknown.Raw) == 0 || json.Unmarshal(item.Unknown.Raw, &raw) != nil {
		return "", false
	}
	for kind, policy := range requestControlPolicies {
		if policy.LegacySourceFormat == "" || policy.LegacySourceType == "" {
			continue
		}
		if item.ProtocolHints.SourceFormat == policy.LegacySourceFormat &&
			item.ProtocolHints.SourceType == policy.LegacySourceType &&
			item.Unknown.Type == policy.LegacySourceType &&
			raw.Type == policy.LegacySourceType {
			return kind, true
		}
	}
	return "", false
}

type RequestControlErrorCode string

const (
	RequestControlDuplicate RequestControlErrorCode = "duplicate"
)

// RequestControlError is intentionally structured so relays can expose a
// bounded object-level diagnostic without parsing an English error string.
type RequestControlError struct {
	Kind       ItemKind
	Code       RequestControlErrorCode
	FirstIndex int
	Index      int
}

func (err *RequestControlError) Error() string {
	if err == nil {
		return "request control error"
	}
	return fmt.Sprintf("request control %q is duplicated at input indexes %d and %d", err.Kind, err.FirstIndex, err.Index)
}

type RequestControlAdjustment struct {
	Kind      ItemKind
	FromIndex int
	ToIndex   int
}

// NormalizeRequestControls returns an attempt-safe canonical sequence whose
// request-local controls satisfy their registered placement and multiplicity
// rules. Ordinary requests allocate nothing; a relocated control causes one
// deep copy of the input graph.
func NormalizeRequestControls(input []Item) ([]Item, []RequestControlAdjustment, error) {
	if len(input) == 0 {
		return input, nil, nil
	}
	var firstIndex map[ItemKind]int
	var counts map[ItemKind]int
	var finalIndexes []int
	for index := range input {
		policy, controlled := RequestControlPolicyFor(input[index].Kind)
		if !controlled {
			continue
		}
		if counts == nil {
			counts = make(map[ItemKind]int)
			firstIndex = make(map[ItemKind]int)
		}
		kind := input[index].Kind
		count := counts[kind]
		if count == 0 {
			firstIndex[kind] = index
		}
		if policy.MaxCount > 0 && count >= policy.MaxCount {
			return nil, nil, &RequestControlError{
				Kind: kind, Code: RequestControlDuplicate,
				FirstIndex: firstIndex[kind], Index: index,
			}
		}
		counts[kind] = count + 1
		if policy.Placement == RequestControlPlacementFinal {
			finalIndexes = append(finalIndexes, index)
		}
	}
	if len(finalIndexes) == 0 {
		return input, nil, nil
	}
	firstFinal := len(input) - len(finalIndexes)
	alreadyFinal := true
	for offset, index := range finalIndexes {
		if index != firstFinal+offset {
			alreadyFinal = false
			break
		}
	}
	if alreadyFinal {
		return input, nil, nil
	}

	ordered := make([]Item, 0, len(input))
	adjustments := make([]RequestControlAdjustment, 0, len(finalIndexes))
	finalSet := make(map[int]struct{}, len(finalIndexes))
	for _, index := range finalIndexes {
		finalSet[index] = struct{}{}
	}
	for index := range input {
		if _, final := finalSet[index]; final {
			continue
		}
		ordered = append(ordered, input[index])
	}
	for _, index := range finalIndexes {
		toIndex := len(ordered)
		ordered = append(ordered, input[index])
		if index != toIndex {
			adjustments = append(adjustments, RequestControlAdjustment{
				Kind: input[index].Kind, FromIndex: index, ToIndex: toIndex,
			})
		}
	}
	return CloneCanonicalItems(ordered), adjustments, nil
}

// ConversationPersistentItems returns a deep copy of the canonical history
// with every current-request-only control removed. It is also used while
// hydrating legacy rows so controls written by an older binary cannot revive.
func ConversationPersistentItems(input []Item) []Item {
	if len(input) == 0 {
		return nil
	}
	persistent := make([]Item, 0, len(input))
	for index := range input {
		kind, controlled := requestControlKindForPersistence(input[index])
		if policy, registered := RequestControlPolicyFor(kind); controlled && registered &&
			policy.Scope == RequestControlScopeCurrentRequest {
			continue
		}
		persistent = append(persistent, CloneCanonicalItem(input[index]))
	}
	return persistent
}

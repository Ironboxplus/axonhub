package conversion

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
)

const portableIdentifierLimit = 64

type identifierRule struct {
	maxLength int
	portable  bool
	fallback  string
}

type identifierLedger struct {
	rule     identifierRule
	bySource map[string]string
	occupied map[string]string
}

func newIdentifierLedger(rule identifierRule) *identifierLedger {
	return &identifierLedger{
		rule: rule, bySource: make(map[string]string), occupied: make(map[string]string),
	}
}

func (ledger *identifierLedger) mapValue(source string) (string, bool) {
	if ledger == nil || source == "" {
		return source, false
	}
	if mapped, ok := ledger.bySource[source]; ok {
		return mapped, mapped != source
	}
	needsNormalization := identifierNeedsNormalization(source, ledger.rule)
	if !needsNormalization {
		if owner, occupied := ledger.occupied[source]; !occupied || owner == source {
			ledger.bySource[source] = source
			ledger.occupied[source] = source
			return source, false
		}
	}

	base := source
	if ledger.rule.portable {
		base = portableIdentifier(source)
	}
	base = strings.Trim(base, "_")
	if base == "" {
		base = ledger.rule.fallback
	}
	for attempt := 0; ; attempt++ {
		digest := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%d", source, attempt)))
		suffix := "_" + hex.EncodeToString(digest[:8])
		candidate := base
		if ledger.rule.maxLength > 0 && len(candidate)+len(suffix) > ledger.rule.maxLength {
			candidate = candidate[:max(0, ledger.rule.maxLength-len(suffix))]
		}
		candidate += suffix
		if owner, occupied := ledger.occupied[candidate]; !occupied || owner == source {
			ledger.bySource[source] = candidate
			ledger.occupied[candidate] = source
			return candidate, true
		}
	}
}

func portableIdentifier(value string) string {
	var builder strings.Builder
	builder.Grow(len(value))
	for index := 0; index < len(value); index++ {
		char := value[index]
		if char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '_' || char == '-' {
			builder.WriteByte(char)
		} else {
			builder.WriteByte('_')
		}
	}
	return builder.String()
}

func identifierNeedsNormalization(value string, rule identifierRule) bool {
	if value == "" || rule.maxLength > 0 && len(value) > rule.maxLength {
		return true
	}
	if !rule.portable {
		return false
	}
	for index := 0; index < len(value); index++ {
		char := value[index]
		if !(char >= 'a' && char <= 'z' || char >= 'A' && char <= 'Z' || char >= '0' && char <= '9' || char == '_' || char == '-') {
			return true
		}
	}
	return false
}

func toolNameRule(format llm.APIFormat) (identifierRule, bool) {
	switch format {
	case llm.APIFormatOpenAIChatCompletion, llm.APIFormatOpenAIResponse, llm.APIFormatAnthropicMessage:
		return identifierRule{maxLength: portableIdentifierLimit, portable: true, fallback: "tool"}, true
	default:
		return identifierRule{}, false
	}
}

func callIDRule(format llm.APIFormat) (identifierRule, bool) {
	switch format {
	case llm.APIFormatAnthropicMessage:
		return identifierRule{maxLength: portableIdentifierLimit, portable: true, fallback: "toolu"}, true
	case llm.APIFormatOpenAIResponse:
		return identifierRule{maxLength: portableIdentifierLimit, fallback: "call"}, true
	default:
		return identifierRule{}, false
	}
}

func appendIdentifierConstraintActions(plan *Plan, request *llm.Request, target llm.APIFormat) {
	if plan == nil || request == nil {
		return
	}
	nameRule, constrainNames := toolNameRule(target)
	idRule, constrainCallIDs := callIDRule(target)
	seen := make(map[ObjectRef]struct{})
	appendAction := func(ref ObjectRef) {
		if _, exists := seen[ref]; exists {
			return
		}
		seen[ref] = struct{}{}
		plan.Actions = append(plan.Actions, Action{
			Ref: ref, Kind: ActionLower, Strategy: StrategyIdentifierNormalize,
			Reason: ReasonProtocolConstraint, Reversible: true,
		})
	}
	needsName := func(name, namespace string, definition bool) bool {
		candidate, _ := toolIdentifierCandidate(name, namespace, definition, target)
		return constrainNames && candidate != "" && identifierNeedsNormalization(candidate, nameRule)
	}
	needsID := func(id string) bool {
		return constrainCallIDs && request.APIFormat != target && identifierNeedsNormalization(id, idRule)
	}

	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if needsName(definition.LogicalName, toolDefinitionNamespace(definition), true) {
			appendAction(ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
		}
	}
	for index := range request.Tools {
		name := request.Tools[index].Function.Name
		if request.Tools[index].ResponseCustomTool != nil {
			name = request.Tools[index].ResponseCustomTool.Name
		}
		if needsName(name, "", true) {
			appendAction(ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
		}
	}
	if request.ToolChoice != nil {
		if request.ToolChoice.NamedToolChoice != nil && needsName(request.ToolChoice.NamedToolChoice.Function.Name, "", false) {
			appendAction(ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
		}
		if request.ToolChoice.AllowedTools != nil {
			for _, ref := range request.ToolChoice.AllowedTools.Tools {
				if needsName(ref.Name, "", false) {
					appendAction(ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
					break
				}
			}
		}
	}
	for itemIndex := range request.Input {
		item := &request.Input[itemIndex]
		switch {
		case item.ToolCall != nil:
			if needsName(item.ToolCall.LogicalName, item.ToolCall.Namespace, false) || needsID(item.ToolCall.CallID) {
				appendAction(ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
			}
		case item.ToolResult != nil:
			if needsName(item.ToolResult.LogicalName, "", false) || needsID(item.ToolResult.CallID) {
				appendAction(ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
			}
		case item.HostedCall != nil:
			if needsName(item.HostedCall.Invocation.LogicalName, item.HostedCall.Invocation.Namespace, false) || needsID(item.HostedCall.Invocation.CallID) {
				appendAction(ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
			}
		}
	}
	for messageIndex := range request.Messages {
		message := &request.Messages[messageIndex]
		for callIndex := range message.ToolCalls {
			call := &message.ToolCalls[callIndex]
			name := call.Function.Name
			if call.ResponseCustomToolCall != nil {
				name = call.ResponseCustomToolCall.Name
			}
			if needsName(name, call.Function.Namespace, false) || needsID(call.ID) {
				appendAction(ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: callIndex})
			}
		}
		if message.ToolCallID != nil && needsID(*message.ToolCallID) {
			appendAction(ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: -1})
		}
	}
}

func planHasIdentifierNormalization(plan *Plan) bool {
	if plan == nil {
		return false
	}
	for _, action := range plan.Actions {
		if action.Strategy == StrategyIdentifierNormalize {
			return true
		}
	}
	return false
}

func normalizeRequestIdentifiers(request *llm.Request, session *Session, target llm.APIFormat) {
	if request == nil || session == nil {
		return
	}
	nameRule, namesSupported := toolNameRule(target)
	callRule, callIDsSupported := callIDRule(target)
	if namesSupported {
		session.targetToolNames = make(map[string]*identifierLedger)
	}
	if callIDsSupported && (session.plan == nil || session.plan.Source != target) {
		session.targetCallIDs = newIdentifierLedger(callRule)
	}
	normalizeName := func(value *string, namespace string, definition bool, ref ObjectRef) {
		if value == nil || *value == "" || session.targetToolNames == nil {
			return
		}
		original := *value
		candidate, scope := toolIdentifierCandidate(original, namespace, definition, target)
		ledger := session.targetToolNames[scope]
		if ledger == nil {
			ledger = newIdentifierLedger(nameRule)
			session.targetToolNames[scope] = ledger
		}
		mapped, changed := ledger.mapValue(candidate)
		*value = targetToolLogicalName(mapped, namespace, definition, target)
		identity, exists := session.identity(original, namespace)
		if !exists {
			identity = toolIdentity{
				SourceKind: llm.ToolKindFunction, SourceName: namespacedWireName(original, namespace, definition), SourceNamespace: namespace,
			}
		}
		session.registerToolIdentity(*value, namespace, identity)
		if changed {
			session.recordIdentifierNormalization(llm.ConversionDirectionRequest, ref)
		}
	}
	normalizeID := func(value *string, ref ObjectRef) {
		if value == nil || *value == "" || session.targetCallIDs == nil {
			return
		}
		mapped, changed := session.targetCallIDs.mapValue(*value)
		if changed {
			*value = mapped
			session.recordIdentifierNormalization(llm.ConversionDirectionRequest, ref)
		}
	}

	for index := range request.ToolDefinitions {
		ref := ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		definition := &request.ToolDefinitions[index]
		normalizeName(&definition.LogicalName, toolDefinitionNamespace(definition), true, ref)
	}
	for index := range request.Tools {
		ref := ObjectRef{Kind: ObjectToolDefinition, ToolIndex: index, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		tool := &request.Tools[index]
		normalizeName(&tool.Function.Name, "", true, ref)
		if tool.ResponseCustomTool != nil {
			normalizeName(&tool.ResponseCustomTool.Name, "", true, ref)
		}
	}
	if request.ToolChoice != nil {
		ref := ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		if request.ToolChoice.NamedToolChoice != nil {
			normalizeName(&request.ToolChoice.NamedToolChoice.Function.Name, "", false, ref)
		}
		if request.ToolChoice.AllowedTools != nil {
			for index := range request.ToolChoice.AllowedTools.Tools {
				normalizeName(&request.ToolChoice.AllowedTools.Tools[index].Name, "", false, ref)
			}
		}
	}
	for itemIndex := range request.Input {
		item := &request.Input[itemIndex]
		if item.ToolCall != nil {
			ref := ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
			normalizeName(&item.ToolCall.LogicalName, item.ToolCall.Namespace, false, ref)
			normalizeID(&item.ToolCall.CallID, ref)
		}
		if item.ToolResult != nil {
			ref := ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
			normalizeName(&item.ToolResult.LogicalName, "", false, ref)
			normalizeID(&item.ToolResult.CallID, ref)
			for discoveredIndex := range item.ToolResult.DiscoveredTools {
				definition := &item.ToolResult.DiscoveredTools[discoveredIndex]
				normalizeName(&definition.LogicalName, toolDefinitionNamespace(definition), true, ref)
			}
		}
		if item.HostedCall != nil {
			ref := ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
			normalizeName(&item.HostedCall.Invocation.LogicalName, item.HostedCall.Invocation.Namespace, false, ref)
			normalizeID(&item.HostedCall.Invocation.CallID, ref)
			if item.HostedCall.Result != nil {
				normalizeID(&item.HostedCall.Result.CallID, ref)
			}
		}
	}
	for messageIndex := range request.Messages {
		message := &request.Messages[messageIndex]
		for callIndex := range message.ToolCalls {
			ref := ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: callIndex}
			call := &message.ToolCalls[callIndex]
			normalizeName(&call.Function.Name, call.Function.Namespace, false, ref)
			normalizeID(&call.ID, ref)
			if call.ResponseCustomToolCall != nil {
				normalizeName(&call.ResponseCustomToolCall.Name, "", false, ref)
				normalizeID(&call.ResponseCustomToolCall.CallID, ref)
			}
		}
		if message.ToolCallID != nil {
			ref := ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: -1}
			normalizeID(message.ToolCallID, ref)
		}
		for resultIndex := range message.InlineToolResults {
			ref := ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: resultIndex}
			normalizeID(&message.InlineToolResults[resultIndex].ToolCallID, ref)
		}
	}
}

func namespacedWireName(name, namespace string, definition bool) string {
	if namespace == "" || !definition {
		return name
	}
	return strings.TrimPrefix(name, namespace+"__")
}

func toolIdentifierCandidate(name, namespace string, definition bool, target llm.APIFormat) (string, string) {
	child := namespacedWireName(name, namespace, definition)
	if namespace == "" || target == llm.APIFormatOpenAIResponse {
		return child, namespace
	}
	return namespace + "__" + child, ""
}

func targetToolLogicalName(mapped, namespace string, definition bool, target llm.APIFormat) string {
	if namespace != "" && definition && target == llm.APIFormatOpenAIResponse {
		return namespace + "__" + mapped
	}
	return mapped
}

func toolDefinitionNamespace(definition *llm.ToolDefinition) string {
	if definition == nil {
		return ""
	}
	switch {
	case definition.Function != nil:
		return definition.Function.Namespace
	case definition.Freeform != nil:
		return definition.Freeform.Namespace
	case definition.Hosted != nil:
		return definition.Hosted.Namespace
	default:
		return ""
	}
}

func (s *Session) normalizeSourceCallID(value string, direction llm.ConversionDirection, ref ObjectRef) string {
	if s == nil || s.plan == nil || value == "" {
		return value
	}
	if s.plan.Source == s.plan.Target.APIFormat {
		// Provider response call IDs represent newly created invocations. They
		// must not be guessed to be echoes of normalized request-history IDs
		// merely because the strings happen to match.
		return value
	}
	rule, ok := callIDRule(s.plan.Source)
	if !ok {
		return value
	}
	if s.sourceCallIDs == nil {
		s.sourceCallIDs = newIdentifierLedger(rule)
	}
	mapped, changed := s.sourceCallIDs.mapValue(value)
	if changed {
		s.recordIdentifierNormalization(direction, ref)
	}
	return mapped
}

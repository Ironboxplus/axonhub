package conversion

import (
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/looplj/axonhub/llm"
)

type Planner struct {
	profile *CapabilityProfile
}

func NewPlanner() *Planner {
	return &Planner{}
}

// NewPlannerWithProfile binds planning to one effective channel profile. The
// caller merges protocol defaults, channel overrides, and registered
// emulators before constructing the planner.
func NewPlannerWithProfile(profile CapabilityProfile) *Planner {
	copyProfile := profile
	return &Planner{profile: &copyProfile}
}

func (p *Planner) Plan(request *llm.Request, targetFormat llm.APIFormat) (*Plan, error) {
	_, plan, err := p.preparePlan(request, targetFormat, false)
	if plan != nil {
		plan.Debug = buildConversionDebugTrace(nil, plan.Actions)
	}
	return plan, err
}

func (p *Planner) preparePlan(request *llm.Request, targetFormat llm.APIFormat, trace bool) (*llm.Request, *Plan, error) {
	prepared, adjustments, err := normalizeRequestControlRequest(request)
	if err != nil {
		plan := p.requestControlFailurePlan(request, targetFormat, trace, err)
		return request, plan, &ConversionPlanError{Cause: err, Plan: plan}
	}
	plan, err := p.plan(prepared, targetFormat, trace)
	appendRequestControlActions(plan, adjustments)
	if err != nil && plan != nil {
		var planErr *ConversionPlanError
		if !errors.As(err, &planErr) {
			err = &ConversionPlanError{Cause: err, Plan: plan}
		}
	}
	return prepared, plan, err
}

func (p *Planner) profileFor(request *llm.Request, targetFormat llm.APIFormat) (CapabilityProfile, bool) {
	profile, ok := ProfileFor(targetFormat)
	if p != nil && p.profile != nil && p.profile.APIFormat == targetFormat {
		profile, ok = *p.profile, true
	}
	if !ok {
		return CapabilityProfile{ID: "unknown/v1", APIFormat: targetFormat}, false
	}
	return effectiveRequestCapabilityProfile(profile, request), true
}

const (
	responsesLiteProfileSuffix = "+responses-lite-namespace-function-only"
)

// effectiveRequestCapabilityProfile applies a known final-wire restriction to
// the concrete target profile. Responses Lite is still APIFormatOpenAIResponse
// at the protocol level, but its namespace declarations only accept function
// children. The restriction is intentionally scoped to namespace children:
// top-level Responses custom tools remain native. Keeping that fact in
// capability planning makes the existing reversible custom-to-function
// lowering run before encoding; a final wire validator can then fail closed
// if raw mutation somehow reintroduces a non-function child.
func effectiveRequestCapabilityProfile(profile CapabilityProfile, request *llm.Request) CapabilityProfile {
	if profile.APIFormat != llm.APIFormatOpenAIResponse || !llm.UsesResponsesLiteWireProfile(request) {
		return profile
	}
	profile.NamespaceChildNativeTools = CapabilityFunctionTool
	if !strings.Contains(profile.ID, responsesLiteProfileSuffix) {
		profile.ID += responsesLiteProfileSuffix
	}
	return profile
}

func namespaceChildCapabilityProfile(profile CapabilityProfile, namespace string) CapabilityProfile {
	if namespace == "" || profile.NamespaceChildNativeTools == 0 {
		return profile
	}
	profile.NativeTools = profile.NamespaceChildNativeTools
	return profile
}

func (p *Planner) requestControlFailurePlan(request *llm.Request, targetFormat llm.APIFormat, trace bool, cause error) *Plan {
	startedAt := time.Time{}
	if trace {
		startedAt = time.Now()
	}
	profile, _ := p.profileFor(request, targetFormat)
	itemIndex := -1
	var controlErr *llm.RequestControlError
	if errors.As(cause, &controlErr) {
		itemIndex = controlErr.Index
	}
	plan := &Plan{Target: profile}
	if request != nil {
		plan.Source = request.APIFormat
	}
	action := itemEvidenceAction(nil, itemIndex, ObjectRequestControl, "request_control", "request_control")
	if request != nil && itemIndex >= 0 && itemIndex < len(request.Input) {
		item := &request.Input[itemIndex]
		if item.Kind == llm.ItemKindCompactionTrigger {
			action = itemEvidenceAction(item, itemIndex, ObjectRequestControl, "compaction_trigger", "compaction_trigger_request_control")
		}
	}
	action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyRequestControlMultiplicity, ReasonDuplicateRequestControl
	plan.Actions = []Action{action}
	plan.Summary = summarizePlan(plan.Source, profile, plan.Actions, startedAt)
	return plan
}

func (p *Planner) plan(request *llm.Request, targetFormat llm.APIFormat, trace bool) (*Plan, error) {
	startedAt := time.Time{}
	if trace {
		startedAt = time.Now()
	}
	profile, ok := p.profileFor(request, targetFormat)
	plan := &Plan{
		Target: profile,
	}
	if request != nil {
		plan.Source = request.APIFormat
	}
	if request == nil || !ok {
		plan.Actions = append(plan.Actions, Action{
			Ref:      ObjectRef{Kind: ObjectProviderData, ToolIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			Kind:     ActionUnknown,
			Strategy: StrategyUnavailable,
			Reason:   ReasonNoStrategy,
		})
		plan.Summary = summarizePlan(plan.Source, profile, plan.Actions, startedAt)
		return plan, plan.Validate()
	}
	if request.RequestType == llm.RequestTypeCompact {
		plan.Actions = append(plan.Actions, actionForCompactRequest(request, profile))
	}
	if err := llm.ValidateItemStructure(request.Input); err != nil {
		plan.Actions = append(plan.Actions, unknownItemAction(-1, ReasonNoStrategy))
		plan.Summary = summarizePlan(plan.Source, profile, plan.Actions, startedAt)
		return plan, plan.Validate()
	}
	for toolIndex := range request.ToolDefinitions {
		if err := request.ToolDefinitions[toolIndex].Validate(); err != nil {
			plan.Actions = append(plan.Actions, Action{
				Ref:  ObjectRef{Kind: ObjectToolDefinition, ToolIndex: toolIndex, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
				Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy,
			})
			plan.Summary = summarizePlan(plan.Source, profile, plan.Actions, startedAt)
			return plan, plan.Validate()
		}
	}
	var allowedSelection map[int]struct{}
	if request.ToolChoice != nil && request.ToolChoice.AllowedTools != nil && profile.APIFormat != llm.APIFormatOpenAIResponse {
		allowed := request.ToolChoice.AllowedTools
		if (allowed.Mode == "auto" || allowed.Mode == "required") && len(allowed.Tools) > 0 {
			selected, _, _, selectErr := resolveAllowedDefinitions(request.ToolDefinitions, allowed.Tools)
			if selectErr == nil {
				allowedSelection = selected
			}
		}
	}

	if len(request.ToolDefinitions) > 0 {
		for toolIndex := range request.ToolDefinitions {
			if allowedSelection != nil {
				if _, selected := allowedSelection[toolIndex]; !selected {
					continue
				}
			}
			plan.Actions = append(plan.Actions, actionForToolDefinition(
				&request.ToolDefinitions[toolIndex], request.APIFormat, profile,
				ObjectRef{Kind: ObjectToolDefinition, ToolIndex: toolIndex, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			))
		}
	} else {
		for toolIndex := range request.Tools {
			plan.Actions = append(plan.Actions, actionForToolKind(
				request.Tools[toolIndex].Type,
				profile,
				ObjectRef{Kind: ObjectToolDefinition, ToolIndex: toolIndex, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			))
		}
	}

	if request.ToolChoice != nil && request.ToolChoice.AllowedTools != nil {
		ref := ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		if profile.APIFormat == llm.APIFormatOpenAIResponse {
			plan.Actions = append(plan.Actions, Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true})
		} else if allowed := request.ToolChoice.AllowedTools; (allowed.Mode != "auto" && allowed.Mode != "required") || len(allowed.Tools) == 0 {
			plan.Actions = append(plan.Actions, Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy})
		} else if _, _, _, err := resolveAllowedDefinitions(request.ToolDefinitions, allowed.Tools); err != nil {
			plan.Actions = append(plan.Actions, Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy})
		} else {
			plan.Actions = append(plan.Actions, Action{Ref: ref, Kind: ActionLower, Strategy: StrategyAllowedTools, Reason: ReasonSemanticProjection, Reversible: true})
		}
	} else if request.ToolChoice != nil && request.ToolChoice.NamedToolChoice != nil {
		choice := request.ToolChoice.NamedToolChoice
		ref := ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
		owner := llm.ExecutionOwner("")
		choiceProfile := profile
		for index := range request.ToolDefinitions {
			definition := &request.ToolDefinitions[index]
			if string(definition.Kind) == choice.Type && (choice.Function.Name == "" || definition.LogicalName == choice.Function.Name) {
				owner = definition.Execution
				choiceProfile = namespaceChildCapabilityProfile(profile, toolDefinitionNamespace(definition))
				break
			}
		}
		if owner == llm.ExecutionOwnerClient {
			plan.Actions = append(plan.Actions, actionForClientToolKind(llm.ToolKind(choice.Type), choiceProfile, ref))
		} else {
			plan.Actions = append(plan.Actions, actionForToolKind(choice.Type, choiceProfile, ref))
		}
	}

	if len(request.Input) > 0 {
		for itemIndex := range request.Input {
			item := &request.Input[itemIndex]
			switch item.Kind {
			case llm.ItemKindMessage:
				plan.Actions = append(plan.Actions, nativeItemAction(itemIndex))
				for contentIndex := range item.Content {
					plan.Actions = append(plan.Actions, actionForContentBlock(
						&item.Content[contentIndex], request.APIFormat, profile.APIFormat,
						ObjectRef{Kind: ObjectContentBlock, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: contentIndex, MessageIndex: -1, ToolCallIndex: -1},
					))
				}
			case llm.ItemKindToolCall:
				if item.ToolCall != nil {
					plan.Actions = append(plan.Actions, actionForToolCall(
						item.ToolCall, profile,
						ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: itemIndex, MessageIndex: -1, ToolCallIndex: -1},
					))
					if len(item.ToolCall.ProviderData) > 0 {
						plan.Actions = append(plan.Actions, actionForTypedProviderData(request.APIFormat, profile.APIFormat,
							ObjectRef{Kind: ObjectProviderData, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}))
					}
				}
			case llm.ItemKindToolResult:
				if item.ToolResult != nil {
					plan.Actions = append(plan.Actions, actionForToolResult(
						item.ToolResult, namespaceChildCapabilityProfile(profile, toolResultNamespace(request, item.ToolResult)),
						ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: itemIndex, MessageIndex: -1, ToolCallIndex: -1},
					))
					for contentIndex := range item.ToolResult.Content {
						plan.Actions = append(plan.Actions, actionForContentBlock(
							&item.ToolResult.Content[contentIndex], request.APIFormat, profile.APIFormat,
							ObjectRef{Kind: ObjectContentBlock, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: contentIndex, MessageIndex: -1, ToolCallIndex: -1},
						))
					}
					for discoveredIndex := range item.ToolResult.DiscoveredTools {
						plan.Actions = append(plan.Actions, actionForToolDefinition(
							&item.ToolResult.DiscoveredTools[discoveredIndex], request.APIFormat, profile,
							ObjectRef{Kind: ObjectToolDefinition, ToolIndex: discoveredIndex, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
						))
					}
					if len(item.ToolResult.ProviderData) > 0 {
						plan.Actions = append(plan.Actions, actionForTypedProviderData(request.APIFormat, profile.APIFormat,
							ObjectRef{Kind: ObjectProviderData, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}))
					}
				}
			case llm.ItemKindReasoning:
				plan.Actions = append(plan.Actions, actionForReasoningItem(request.APIFormat, profile.APIFormat, itemIndex))
			case llm.ItemKindAgentMessage:
				plan.Actions = append(plan.Actions, actionForAgentMessage(request.APIFormat, profile.APIFormat, item, itemIndex))
			case llm.ItemKindHostedCall:
				plan.Actions = append(plan.Actions, actionForHostedCall(request.APIFormat, profile, item, itemIndex))
			case llm.ItemKindCompaction:
				action := itemEvidenceAction(item, itemIndex, ObjectCompaction, "compaction", "compaction_checkpoint")
				if request.APIFormat == profile.APIFormat && profile.APIFormat == llm.APIFormatOpenAIResponse {
					action.Kind, action.Strategy, action.Reason, action.Reversible = ActionNative, StrategyNative, ReasonTargetNative, true
				} else {
					action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonProviderPrivate
				}
				plan.Actions = append(plan.Actions, action)
			case llm.ItemKindContextCompaction:
				action := itemEvidenceAction(item, itemIndex, ObjectContextCompaction, "context_compaction", "context_compaction_checkpoint")
				if request.APIFormat == profile.APIFormat && profile.APIFormat == llm.APIFormatOpenAIResponse {
					action.Kind, action.Strategy, action.Reason, action.Reversible = ActionNative, StrategyNative, ReasonTargetNative, true
				} else {
					action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonProviderPrivate
				}
				plan.Actions = append(plan.Actions, action)
			case llm.ItemKindCompactionTrigger:
				action := itemEvidenceAction(item, itemIndex, ObjectRequestControl, "compaction_trigger", "compaction_trigger_request_control")
				if request.APIFormat == profile.APIFormat && profile.APIFormat == llm.APIFormatOpenAIResponse {
					action.Kind, action.Strategy, action.Reason, action.Reversible = ActionNative, StrategyNative, ReasonTargetNative, true
				} else {
					action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonProviderPrivate
				}
				plan.Actions = append(plan.Actions, action)
			case llm.ItemKindToolDeclaration:
				if request.APIFormat == profile.APIFormat && profile.APIFormat == llm.APIFormatOpenAIResponse {
					plan.Actions = append(plan.Actions, nativeItemAction(itemIndex))
				} else {
					// The declaration is a Responses-specific placement container.
					// Its executable behavior already lives in ToolDefinitions, which
					// the target encoder projects into its native global tool list.
					plan.Actions = append(plan.Actions, Action{
						Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
						Kind: ActionLower, Strategy: StrategyToolDeclaration, Reason: ReasonSemanticProjection, Reversible: false,
					})
				}
			case llm.ItemKindMCPListTools, llm.ItemKindMCPApprovalRequest,
				llm.ItemKindMCPApprovalResponse, llm.ItemKindMCPCall:
				if profile.NativeTools.Supports(CapabilityMCPTool) {
					plan.Actions = append(plan.Actions, nativeItemAction(itemIndex))
				} else if profile.EmulatedTools.Supports(CapabilityMCPTool) && profile.NativeTools.Supports(CapabilityFunctionTool) {
					plan.Actions = append(plan.Actions, Action{
						Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
						Kind: ActionEmulate, Strategy: StrategyMCPGateway, Reason: ReasonGatewayExecution, Reversible: true,
					})
				} else {
					plan.Actions = append(plan.Actions, unknownItemAction(itemIndex, ReasonNoStrategy))
				}
			case llm.ItemKindUnknown:
				action := itemEvidenceAction(item, itemIndex, ObjectInputItem, "unknown", unknownItemSemanticClass(item))
				if request.APIFormat == profile.APIFormat {
					action.Kind, action.Strategy, action.Reason, action.Reversible = ActionOpaque, StrategyOpaqueSidecar, ReasonSameProtocolOpaque, true
				} else {
					action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonNoStrategy
				}
				plan.Actions = append(plan.Actions, action)
			}
		}
	} else {
		legacyMessages := request.Messages
		if request.RequestType == llm.RequestTypeCompact && request.Compact != nil {
			legacyMessages = request.Compact.Input
		}
		customCalls := make(map[string]struct{})
		for messageIndex := range legacyMessages {
			for callIndex := range legacyMessages[messageIndex].ToolCalls {
				call := legacyMessages[messageIndex].ToolCalls[callIndex]
				if isCustomToolCall(call) && call.ID != "" {
					customCalls[call.ID] = struct{}{}
				}
			}
		}
		for messageIndex := range legacyMessages {
			message := legacyMessages[messageIndex]
			for callIndex := range message.ToolCalls {
				call := message.ToolCalls[callIndex]
				toolKind := call.Type
				if isCustomToolCall(call) {
					toolKind = llm.ToolTypeResponsesCustomTool
				}
				plan.Actions = append(plan.Actions, actionForToolKind(
					toolKind, profile,
					ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: -1, MessageIndex: messageIndex, ToolCallIndex: callIndex},
				))
			}
			if message.Role == "tool" && message.ToolCallID != nil {
				if _, ok := customCalls[*message.ToolCallID]; ok {
					plan.Actions = append(plan.Actions, actionForToolKind(
						llm.ToolTypeResponsesCustomTool, profile,
						ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, MessageIndex: messageIndex, ToolCallIndex: -1},
					))
				}
			}
		}
	}
	appendResponsesExtensionActions(plan, request, profile.APIFormat)
	appendIdentifierConstraintActions(plan, request, profile.APIFormat)
	appendSchemaConstraintActions(plan, request, profile.APIFormat)

	plan.Summary = summarizePlan(plan.Source, profile, plan.Actions, startedAt)
	return plan, plan.Validate()
}

func actionForAgentMessage(source, target llm.APIFormat, item *llm.Item, itemIndex int) Action {
	action := itemEvidenceAction(item, itemIndex, ObjectAgentMessage, "agent_message", "agent_message")
	if item == nil || item.AgentMessage == nil {
		action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonNoStrategy
		action.SemanticClass = "agent_message_invalid"
		return action
	}
	if source == target && target == llm.APIFormatOpenAIResponse {
		action.Kind, action.Strategy, action.Reason, action.Reversible = ActionNative, StrategyNative, ReasonTargetNative, true
		return action
	}
	if _, err := item.AgentMessage.LegacyInterAgentMessageJSON(); err != nil {
		action.Kind, action.Strategy, action.Reason = ActionUnknown, StrategyUnavailable, ReasonProviderPrivate
		action.SemanticClass = agentMessageSemanticClass(item.AgentMessage)
		return action
	}
	// The legacy JSON adds trigger_turn because typed Responses agent_message
	// has no equivalent field. The lowering is intentionally lossy.
	action.Kind, action.Strategy, action.Reason, action.Reversible = ActionLower, StrategyAgentMessageLegacyInput, ReasonSemanticProjection, false
	action.SemanticClass = "agent_message_plaintext"
	return action
}

func agentMessageSemanticClass(message *llm.AgentMessage) string {
	if message == nil {
		return "agent_message_invalid"
	}
	plaintext, encrypted := false, false
	for index := range message.Content {
		switch message.Content[index].Kind {
		case llm.AgentMessageContentInputText:
			plaintext = true
		case llm.AgentMessageContentEncryptedContent:
			encrypted = true
		}
	}
	switch {
	case encrypted && plaintext:
		return "agent_message_mixed"
	case encrypted:
		return "agent_message_encrypted"
	case plaintext:
		return "agent_message_plaintext"
	default:
		return "agent_message_invalid"
	}
}

func actionForCompactRequest(request *llm.Request, profile CapabilityProfile) Action {
	ref := ObjectRef{
		Kind: ObjectCompaction, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1,
		MessageIndex: -1, ToolCallIndex: -1,
	}
	if request == nil || request.Compact == nil {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	// Always emulate compact as a chat-style completion. Live upstreams used by
	// Octopus (and many OpenAI-compatible gateways) do not implement
	// POST /v1/responses/compact; native passthrough 404s. Emulation rewrites to
	// an ordinary completion on the selected wire and restoreCompactEmulation
	// rehydrates object=response.compaction for the client.
	if profile.APIFormat == llm.APIFormatOpenAIResponse ||
		profile.APIFormat == llm.APIFormatOpenAIChatCompletion ||
		profile.APIFormat == llm.APIFormatAnthropicMessage {
		return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyCompactAsChat, Reason: ReasonTargetNoCompact}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
}

func nativeItemAction(itemIndex int) Action {
	return Action{
		Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true,
	}
}

func unknownItemAction(itemIndex int, reason ReasonCode) Action {
	return Action{
		Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: reason,
	}
}

// itemEvidenceAction constructs an item-level action with structural identity
// and payload-free source evidence. It never reads Unknown.Raw or any content
// payload: provider/UI evidence can identify the blocked union without turning
// source data into a persistence or log surface.
func itemEvidenceAction(item *llm.Item, itemIndex int, kind ObjectKind, fallbackSourceType, semanticClass string) Action {
	action := Action{
		Ref:        ObjectRef{Kind: kind, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
		SourceType: itemEvidenceSourceType(item, fallbackSourceType), SemanticClass: semanticClass,
	}
	if item != nil {
		action.RawBytes = item.ProtocolHints.SourceBytes
		action.SourceDigest = safeEvidenceSourceDigest(item.ProtocolHints.SourceDigest)
	}
	return action
}

// safeEvidenceSourceDigest accepts only the fixed SHA-256 representation that
// the Responses boundary emits. A canonical caller can construct Items, so
// trace evidence must not blindly persist a forged free-form digest value.
func safeEvidenceSourceDigest(value string) string {
	if len(value) != 64 {
		return ""
	}
	for index := 0; index < len(value); index++ {
		character := value[index]
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return ""
		}
	}
	return value
}

func itemEvidenceSourceType(item *llm.Item, fallback string) string {
	if item != nil {
		if safeEvidenceSourceType(item.ProtocolHints.SourceType) {
			return item.ProtocolHints.SourceType
		}
		if item.Unknown != nil && safeEvidenceSourceType(item.Unknown.Type) {
			return item.Unknown.Type
		}
	}
	if safeEvidenceSourceType(fallback) {
		return fallback
	}
	return "unknown"
}

// safeEvidenceSourceType matches the compact identifier grammar expected by
// Octopus evidence consumers. Source wire identity remains untouched; only the
// copied evidence label is bounded so a future provider discriminator cannot
// become an unbounded or control-character-bearing UI/persistence value.
func safeEvidenceSourceType(value string) bool {
	if len(value) == 0 || len(value) > 96 {
		return false
	}
	for index := 0; index < len(value); index++ {
		character := value[index]
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') &&
			character != '_' && character != '-' {
			return false
		}
	}
	return true
}

func unknownItemSemanticClass(item *llm.Item) string {
	if item != nil && item.Unknown != nil && item.Unknown.Behavioral {
		return "future_unknown_behavioral"
	}
	return "future_unknown_nonbehavioral"
}

func actionForContentBlock(block *llm.ContentBlock, source, target llm.APIFormat, ref ObjectRef) Action {
	if block == nil {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	switch block.Kind {
	case llm.ContentKindText, llm.ContentKindRefusal, llm.ContentKindImage:
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	case llm.ContentKindDocument:
		if block.Document == nil {
			break
		}
		if source == target || block.Document.SourceType == llm.DocumentSourceBase64 || block.Document.SourceType == llm.DocumentSourceFile {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
		if block.Document.SourceType == llm.DocumentSourceURL && target != llm.APIFormatOpenAIChatCompletion {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
	case llm.ContentKindCitation:
		if source == target || block.Citation == nil || block.Citation.EncryptedIndex == nil && block.Citation.CitedText == nil {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyCitationProject, Reason: ReasonSemanticProjection}
	case llm.ContentKindAudio:
		if target == llm.APIFormatOpenAIChatCompletion {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
}

func actionForReasoningItem(source, target llm.APIFormat, itemIndex int) Action {
	if source == target {
		return nativeItemAction(itemIndex)
	}
	return Action{
		Ref: ObjectRef{
			Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex,
			ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
		},
		Kind: ActionLower, Strategy: StrategyReasoningProject, Reason: ReasonSemanticProjection,
	}
}

func actionForHostedCall(source llm.APIFormat, target CapabilityProfile, item *llm.Item, itemIndex int) Action {
	ref := ObjectRef{
		Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex,
		ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
	}
	if item == nil || item.HostedCall == nil || item.HostedCall.Invocation.Execution != llm.ExecutionOwnerProvider {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	capability := capabilityForToolKind(string(item.HostedCall.Invocation.Kind))
	if source == target.APIFormat && profileAdmitsToolCapability(source, target, capability) {
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	}
	if target.NativeTools.Supports(capability) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyHostedProject, Reason: ReasonSemanticProjection}
	}
	if target.EmulatedTools.Supports(capability) && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyHostedGateway, Reason: ReasonGatewayExecution, Reversible: true}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
}

func profileAdmitsToolCapability(source llm.APIFormat, target CapabilityProfile, capability ToolCapabilitySet) bool {
	if capability == 0 {
		return source == target.APIFormat
	}
	return target.NativeTools.Supports(capability)
}

func actionForProviderData(source, target llm.APIFormat, ref ObjectRef) Action {
	if source == target {
		return Action{Ref: ref, Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar, Reason: ReasonSameProtocolOpaque, Reversible: true}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonProviderPrivate}
}

func actionForTypedProviderData(source, target llm.APIFormat, ref ObjectRef) Action {
	if source == target {
		return Action{Ref: ref, Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar, Reason: ReasonSameProtocolOpaque, Reversible: true}
	}
	// The behavioral subset has already been decoded into the surrounding
	// canonical tool call/result. Cross-protocol wire formats cannot carry the
	// source-private envelope, so record the projection explicitly instead of
	// treating a fully typed lifecycle as unavailable.
	return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyOpaqueSidecar, Reason: ReasonSemanticProjection}
}

func appendResponsesExtensionActions(plan *Plan, request *llm.Request, target llm.APIFormat) {
	if plan == nil || request == nil || request.ProviderExtensions == nil ||
		request.ProviderExtensions.OpenAIResponses == nil || request.ProviderExtensions.OpenAIResponses.Request == nil {
		return
	}
	extension := request.ProviderExtensions.OpenAIResponses.Request
	appendReasoningContextAction := func(ref ObjectRef) {
		if request.APIFormat == target {
			plan.Actions = append(plan.Actions, Action{
				Ref: ref, Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar,
				Reason: ReasonSameProtocolOpaque, Reversible: true,
			})
			return
		}
		// Responses reasoning.context controls how much request history the
		// provider may use for reasoning (for example, "all_turns"). Chat and
		// Anthropic have no equivalent wire field, but the canonical request
		// already carries the conversation history. Project that history and
		// record the dropped protocol control as an explicit, non-reversible
		// degradation instead of rejecting an otherwise valid request.
		plan.Actions = append(plan.Actions, Action{
			Ref: ref, Kind: ActionLower, Strategy: StrategyReasoningProject,
			Reason: ReasonProtocolConstraint, Reversible: false,
		})
	}
	if extension.ReasoningContext != "" {
		appendReasoningContextAction(ObjectRef{Kind: ObjectProviderData, ToolIndex: -1, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
	}
}

func actionForToolKind(toolKind string, target CapabilityProfile, ref ObjectRef) Action {
	capability := capabilityForToolKind(toolKind)
	if target.NativeTools.Supports(capability) {
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	}
	if capability == CapabilityCustomTool && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyCustomAsFunction, Reason: ReasonTargetFunctionOnly, Reversible: true}
	}
	if capability == CapabilityMCPTool && target.EmulatedTools.Supports(CapabilityMCPTool) && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyMCPGateway, Reason: ReasonGatewayExecution, Reversible: true}
	}
	// ItemKindToolCall/ToolResult are client-executed lifecycles. Any typed
	// client capability can travel through a function-only protocol because the
	// session ledger restores its original kind and logical name on the way back.
	if capability != 0 && capability != CapabilityFunctionTool && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyClientToolAsFunc, Reason: ReasonTargetFunctionOnly, Reversible: true}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
}

func capabilityForToolKind(toolKind string) ToolCapabilitySet {
	switch toolKind {
	case "", llm.ToolTypeFunction:
		return CapabilityFunctionTool
	case llm.ToolTypeResponsesCustomTool, "custom":
		return CapabilityCustomTool
	case llm.ToolTypeWebSearch:
		return CapabilityWebSearchTool
	case string(llm.ToolKindWebFetch):
		return CapabilityWebFetchTool
	case string(llm.ToolKindFileSearch):
		return CapabilityFileSearchTool
	case llm.ToolTypeImageGeneration:
		return CapabilityImageGenerationTool
	case string(llm.ToolKindMCP):
		return CapabilityMCPTool
	case string(llm.ToolKindLocalShell):
		return CapabilityLocalShellTool
	case string(llm.ToolKindToolSearch):
		return CapabilityToolSearch
	case string(llm.ToolKindCodeInterpreter), string(llm.ToolKindCodeExecution):
		return CapabilityCodeExecutionTool
	case string(llm.ToolKindComputer):
		return CapabilityComputerTool
	case string(llm.ToolKindShell):
		return CapabilityShellTool
	case string(llm.ToolKindApplyPatch):
		return CapabilityApplyPatchTool
	case string(llm.ToolKindAnthropicServer):
		return CapabilityAnthropicServerTool
	default:
		return 0
	}
}

func actionForToolCall(call *llm.ToolInvocation, target CapabilityProfile, ref ObjectRef) Action {
	if call == nil {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	target = namespaceChildCapabilityProfile(target, call.Namespace)
	// Anthropic tool_use.input is JSON, unlike Chat/Responses function
	// arguments which are strings. Never let the encoder silently replace an
	// invalid argument string with an empty object.
	if call.Kind == llm.ToolKindFunction && target.APIFormat == llm.APIFormatAnthropicMessage &&
		len(call.ArgumentsJSON) == 0 && call.ArgumentsText != "" && !json.Valid([]byte(call.ArgumentsText)) {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if call.Execution == llm.ExecutionOwnerClient {
		return actionForClientToolKind(call.Kind, target, ref)
	}
	return actionForToolKind(string(call.Kind), target, ref)
}

func actionForToolResult(result *llm.ToolResult, target CapabilityProfile, ref ObjectRef) Action {
	if result == nil {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if result.Execution == llm.ExecutionOwnerClient {
		return actionForClientToolKind(result.Kind, target, ref)
	}
	return actionForToolKind(string(result.Kind), target, ref)
}

func actionForToolDefinition(definition *llm.ToolDefinition, source llm.APIFormat, target CapabilityProfile, ref ObjectRef) Action {
	if definition == nil {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	target = namespaceChildCapabilityProfile(target, toolDefinitionNamespace(definition))
	capability := capabilityForToolKind(string(definition.Kind))
	if definition.Execution == llm.ExecutionOwnerProvider && definition.Kind == llm.ToolKindMCP {
		// A standard remote-MCP definition is native on Responses. A read_only
		// filter is not a wire property there: its semantics are preserved only
		// by discovering readOnlyHint annotations through the MCP gateway.
		portable := definition.MCP != nil && (definition.MCP.AllowedTools == nil || definition.MCP.AllowedTools.ReadOnly == nil)
		if portable && target.NativeTools.Supports(CapabilityMCPTool) {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
		if target.EmulatedTools.Supports(capability) && target.NativeTools.Supports(CapabilityFunctionTool) {
			return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyMCPGateway, Reason: ReasonGatewayExecution, Reversible: true}
		}
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if source == target.APIFormat && profileAdmitsToolCapability(source, target, capability) {
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	}
	if definition.Kind == llm.ToolKindToolSearch && definition.Execution != llm.ExecutionOwnerClient &&
		!target.NativeTools.Supports(CapabilityToolSearch) {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if definition.Kind == llm.ToolKindWebSearch && source != target.APIFormat &&
		(definition.Hosted == nil || definition.Hosted.WebSearch == nil) {
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if definition.Execution == llm.ExecutionOwnerProvider && definition.Hosted != nil {
		if profileAdmitsToolCapability(source, target, capability) && HostedToolNativeEquivalent(*definition, target.APIFormat) {
			return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
		}
		if target.EmulatedTools.Supports(capability) && target.NativeTools.Supports(CapabilityFunctionTool) {
			return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyHostedGateway, Reason: ReasonGatewayExecution, Reversible: true}
		}
		return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
	}
	if definition.Execution == llm.ExecutionOwnerClient && capability == 0 && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyClientToolAsFunc, Reason: ReasonTargetFunctionOnly, Reversible: true}
	}
	if definition.Execution == llm.ExecutionOwnerClient {
		return actionForClientToolKind(definition.Kind, target, ref)
	}
	return actionForToolKind(string(definition.Kind), target, ref)
}

func actionForClientToolKind(kind llm.ToolKind, target CapabilityProfile, ref ObjectRef) Action {
	capability := capabilityForToolKind(string(kind))
	if clientToolNativeEquivalent(kind, target.APIFormat) && target.NativeTools.Supports(capability) {
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	}
	if kind == llm.ToolKindCustom && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyCustomAsFunction, Reason: ReasonTargetFunctionOnly, Reversible: true}
	}
	if kind != llm.ToolKindFunction && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyClientToolAsFunc, Reason: ReasonTargetFunctionOnly, Reversible: true}
	}
	return actionForToolKind(string(kind), target, ref)
}

func clientToolNativeEquivalent(kind llm.ToolKind, target llm.APIFormat) bool {
	switch target {
	case llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage:
		return kind == llm.ToolKindFunction
	case llm.APIFormatOpenAIResponse:
		switch kind {
		case llm.ToolKindFunction, llm.ToolKindCustom, llm.ToolKindLocalShell,
			llm.ToolKindToolSearch, llm.ToolKindComputer, llm.ToolKindApplyPatch:
			return true
		}
	}
	return false
}

func isCustomToolCall(call llm.ToolCall) bool {
	return call.ResponseCustomToolCall != nil || call.Type == llm.ToolTypeResponsesCustomTool || call.Type == "custom"
}

func toolResultNamespace(request *llm.Request, result *llm.ToolResult) string {
	if request == nil || result == nil || result.CallID == "" {
		return ""
	}
	for index := range request.Input {
		call := request.Input[index].ToolCall
		if call != nil && call.CallID == result.CallID {
			return call.Namespace
		}
	}
	return ""
}

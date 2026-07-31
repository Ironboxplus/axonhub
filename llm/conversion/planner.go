package conversion

import (
	"encoding/json"
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
	return p.plan(request, targetFormat, false)
}

func (p *Planner) plan(request *llm.Request, targetFormat llm.APIFormat, trace bool) (*Plan, error) {
	startedAt := time.Time{}
	if trace {
		startedAt = time.Now()
	}
	profile, ok := ProfileFor(targetFormat)
	if p != nil && p.profile != nil && p.profile.APIFormat == targetFormat {
		profile = *p.profile
		ok = true
	}
	if !ok {
		profile = CapabilityProfile{ID: "unknown/v1", APIFormat: targetFormat}
	}
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
		for index := range request.ToolDefinitions {
			definition := &request.ToolDefinitions[index]
			if string(definition.Kind) == choice.Type && (choice.Function.Name == "" || definition.LogicalName == choice.Function.Name) {
				owner = definition.Execution
				break
			}
		}
		if owner == llm.ExecutionOwnerClient {
			plan.Actions = append(plan.Actions, actionForClientToolKind(llm.ToolKind(choice.Type), profile, ref))
		} else {
			plan.Actions = append(plan.Actions, actionForToolKind(choice.Type, profile, ref))
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
						item.ToolResult, profile,
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
			case llm.ItemKindHostedCall:
				plan.Actions = append(plan.Actions, actionForHostedCall(request.APIFormat, profile, item, itemIndex))
			case llm.ItemKindCompaction:
				if request.APIFormat == profile.APIFormat && profile.APIFormat == llm.APIFormatOpenAIResponse {
					plan.Actions = append(plan.Actions, nativeItemAction(itemIndex))
				} else {
					plan.Actions = append(plan.Actions, unknownItemAction(itemIndex, ReasonProviderPrivate))
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
				if request.APIFormat == profile.APIFormat {
					plan.Actions = append(plan.Actions, Action{
						Ref:  ObjectRef{Kind: ObjectInputItem, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
						Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar, Reason: ReasonSameProtocolOpaque, Reversible: true,
					})
				} else {
					plan.Actions = append(plan.Actions, unknownItemAction(itemIndex, ReasonNoStrategy))
				}
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
	if source == target.APIFormat {
		return Action{Ref: ref, Kind: ActionNative, Strategy: StrategyNative, Reason: ReasonTargetNative, Reversible: true}
	}
	capability := capabilityForToolKind(string(item.HostedCall.Invocation.Kind))
	if target.NativeTools.Supports(capability) {
		return Action{Ref: ref, Kind: ActionLower, Strategy: StrategyHostedProject, Reason: ReasonSemanticProjection}
	}
	if target.EmulatedTools.Supports(capability) && target.NativeTools.Supports(CapabilityFunctionTool) {
		return Action{Ref: ref, Kind: ActionEmulate, Strategy: StrategyHostedGateway, Reason: ReasonGatewayExecution, Reversible: true}
	}
	return Action{Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable, Reason: ReasonNoStrategy}
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
	appendAction := func(ref ObjectRef) {
		if request.APIFormat == target {
			plan.Actions = append(plan.Actions, Action{
				Ref: ref, Kind: ActionOpaque, Strategy: StrategyOpaqueSidecar,
				Reason: ReasonSameProtocolOpaque, Reversible: true,
			})
			return
		}
		plan.Actions = append(plan.Actions, Action{
			Ref: ref, Kind: ActionUnknown, Strategy: StrategyUnavailable,
			Reason: ReasonProviderPrivate,
		})
	}
	for index := range extension.RawTools {
		appendAction(ObjectRef{
			Kind: ObjectToolDefinition, ToolIndex: extension.RawTools[index].OriginalIndex,
			ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
		})
	}
	for index := range extension.RawInputItems {
		appendAction(ObjectRef{
			Kind: ObjectProviderData, ToolIndex: -1, ItemIndex: extension.RawInputItems[index].OriginalIndex,
			MessageIndex: -1, ToolCallIndex: -1,
		})
	}
	if len(extension.RawToolChoice) > 0 {
		appendAction(ObjectRef{Kind: ObjectToolChoice, ToolIndex: -1, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
	}
	if extension.ReasoningContext != "" {
		appendAction(ObjectRef{Kind: ObjectProviderData, ToolIndex: -1, ItemIndex: -1, MessageIndex: -1, ToolCallIndex: -1})
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
	if source == target.APIFormat {
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
		if HostedToolNativeEquivalent(*definition, target.APIFormat) {
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

package conversion

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/samber/lo"
)

var customFunctionParameters = json.RawMessage(`{"type":"object","properties":{"input":{"type":"string"}},"required":["input"],"additionalProperties":false}`)
var clientToolFunctionParameters = json.RawMessage(`{"type":"object","additionalProperties":true}`)

func Lower(request *llm.Request, plan *Plan) (*llm.Request, *Session, error) {
	return lower(request, plan, false)
}

func lower(request *llm.Request, plan *Plan, trace bool) (*llm.Request, *Session, error) {
	if err := plan.Validate(); err != nil {
		return nil, nil, err
	}
	startedAt := time.Time{}
	if trace {
		startedAt = time.Now()
	}
	session := newSession(plan, request, trace)
	if !planNeedsLowering(plan) {
		if trace {
			session.setLowerNanos(time.Since(startedAt).Nanoseconds())
		}
		return request, session, nil
	}

	lowered := request.Clone()
	var allowedSelection map[int]struct{}
	var allowedMCPNames map[int]map[string]struct{}
	var allowedMCPUnrestricted map[int]bool
	allowedMode := ""
	if lowered.ToolChoice != nil && lowered.ToolChoice.AllowedTools != nil && planLowersToolChoice(plan) {
		allowed := lowered.ToolChoice.AllowedTools
		var err error
		allowedSelection, allowedMCPNames, allowedMCPUnrestricted, err = resolveAllowedDefinitions(lowered.ToolDefinitions, allowed.Tools)
		if err != nil {
			return nil, nil, err
		}
		if allowed.Mode != "auto" && allowed.Mode != "required" {
			return nil, nil, fmt.Errorf("%w: allowed_tools mode %q is not portable", ErrIncompletePlan, allowed.Mode)
		}
		allowedMode = allowed.Mode
		applyAllowedMCPRestrictions(lowered.ToolDefinitions, allowedMCPNames, allowedMCPUnrestricted)
	}
	definitionIndexes := make(map[string]int, len(lowered.ToolDefinitions))
	customChoiceNames := make(map[string][]string, len(lowered.ToolDefinitions))
	for index := range lowered.ToolDefinitions {
		definitionIndexes[lowered.ToolDefinitions[index].LogicalName] = index
	}
	lowerDeferredCatalog := false
	for index := range lowered.ToolDefinitions {
		definition := &lowered.ToolDefinitions[index]
		if definition.Kind == llm.ToolKindToolSearch && definition.Execution == llm.ExecutionOwnerClient && planLowersDefinition(plan, index) {
			lowerDeferredCatalog = true
			break
		}
	}
	activatedDefinitions := make(map[string]struct{})
	for itemIndex := range lowered.Input {
		result := lowered.Input[itemIndex].ToolResult
		if result == nil || result.Kind != llm.ToolKindToolSearch || !planLowersItem(plan, ObjectToolResult, itemIndex) {
			continue
		}
		for discoveredIndex := range result.DiscoveredTools {
			discovered := result.DiscoveredTools[discoveredIndex]
			activatedDefinitions[discovered.LogicalName] = struct{}{}
			if existingIndex, exists := definitionIndexes[discovered.LogicalName]; exists {
				if !equivalentDiscoveredDefinition(lowered.ToolDefinitions[existingIndex], discovered) {
					return nil, nil, fmt.Errorf("%w: discovered tool %q conflicts with an existing definition", ErrIncompletePlan, discovered.LogicalName)
				}
				lowered.ToolDefinitions[existingIndex].DeferLoading = nil
				continue
			}
			discovered.DeferLoading = nil
			definitionIndexes[discovered.LogicalName] = len(lowered.ToolDefinitions)
			lowered.ToolDefinitions = append(lowered.ToolDefinitions, discovered)
		}
		catalog, err := json.Marshal(struct {
			Tools []llm.ToolDefinition `json:"tools"`
		}{Tools: result.DiscoveredTools})
		if err != nil {
			return nil, nil, fmt.Errorf("encode tool search result %q: %w", result.CallID, err)
		}
		result.Content = []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(catalog)}}
		result.DiscoveredTools = nil
		result.Execution = ""
	}
	for toolIndex := range lowered.ToolDefinitions {
		definition := &lowered.ToolDefinitions[toolIndex]
		if !planLowersDefinition(plan, toolIndex) {
			continue
		}
		switch definition.Kind {
		case llm.ToolKindCustom:
			if definition.Freeform == nil {
				continue
			}
			namespace := definition.Freeform.Namespace
			sourceName := strings.TrimPrefix(definition.LogicalName, namespace+"__")
			custom := &llm.ResponseCustomTool{
				Name: definition.LogicalName, Description: definition.Description,
				Format: &llm.ResponseCustomToolFormat{
					Type: definition.Freeform.Format, Syntax: definition.Freeform.Syntax, Definition: definition.Freeform.Definition,
				},
			}
			definition.Kind = llm.ToolKindFunction
			definition.LogicalName = session.syntheticNameInNamespace(sourceName, namespace)
			customChoiceNames[custom.Name] = append(customChoiceNames[custom.Name], definition.LogicalName)
			if sourceName != custom.Name {
				customChoiceNames[sourceName] = append(customChoiceNames[sourceName], definition.LogicalName)
			}
			definition.Description = customFunctionDescription(custom)
			definition.Function = &llm.FunctionDefinition{
				Parameters: append(json.RawMessage(nil), customFunctionParameters...), Strict: lo.ToPtr(true), Namespace: namespace,
			}
			definition.Freeform = nil
		default:
			lowerClientToolDefinition(definition, session)
		}
	}
	var hiddenDeferredNames map[string]struct{}
	if lowerDeferredCatalog {
		hiddenDeferredNames = make(map[string]struct{})
		for toolIndex := range lowered.ToolDefinitions {
			definition := &lowered.ToolDefinitions[toolIndex]
			if definition.DeferLoading == nil || !*definition.DeferLoading {
				continue
			}
			if _, activated := activatedDefinitions[definition.LogicalName]; activated {
				definition.DeferLoading = nil
				continue
			}
			hiddenDeferredNames[definition.LogicalName] = struct{}{}
		}
	}
	if allowedSelection != nil {
		lowered.ToolDefinitions = filterAllowedDefinitions(lowered.ToolDefinitions, allowedSelection)
		lowered.ToolChoice = &llm.ToolChoice{ToolChoice: &allowedMode}
	}
	lowered.ToolDefinitions = filterDefinitionsByName(lowered.ToolDefinitions, hiddenDeferredNames)
	for toolIndex := range lowered.Tools {
		tool := &lowered.Tools[toolIndex]
		if !planLowersDefinition(plan, toolIndex) || tool.Type != llm.ToolTypeResponsesCustomTool && tool.Type != "custom" {
			continue
		}
		if tool.ResponseCustomTool == nil {
			return nil, nil, fmt.Errorf("%w: custom tool %d has no definition", ErrIncompletePlan, toolIndex)
		}
		source := tool.ResponseCustomTool
		*tool = llm.Tool{
			Type: llm.ToolTypeFunction,
			Function: llm.Function{
				Name:        session.syntheticName(source.Name),
				Description: customFunctionDescription(source),
				Parameters:  append(json.RawMessage(nil), customFunctionParameters...),
				Strict:      lo.ToPtr(true),
			},
		}
	}

	if lowered.ToolChoice != nil && lowered.ToolChoice.NamedToolChoice != nil && planLowersToolChoice(plan) {
		choice := lowered.ToolChoice.NamedToolChoice
		if choice.Type == llm.ToolTypeResponsesCustomTool || choice.Type == "custom" {
			choice.Type = llm.ToolTypeFunction
			if candidates := customChoiceNames[choice.Function.Name]; len(candidates) == 1 {
				choice.Function.Name = candidates[0]
			} else {
				choice.Function.Name = session.syntheticName(choice.Function.Name)
			}
		} else if choice.Type == string(llm.ToolKindToolSearch) {
			choice.Type = llm.ToolTypeFunction
			choice.Function.Name = session.syntheticToolName(llm.ToolKindToolSearch, "tool_search")
		} else if sourceKind := llm.ToolKind(choice.Type); sourceKind != llm.ToolKindFunction {
			sourceName := choice.Function.Name
			if sourceName == "" {
				sourceName = string(sourceKind)
			}
			choice.Type = llm.ToolTypeFunction
			choice.Function.Name = session.syntheticToolName(sourceKind, sourceName)
		}
	}

	for messageIndex := range lowered.Messages {
		message := &lowered.Messages[messageIndex]
		for callIndex := range message.ToolCalls {
			call := &message.ToolCalls[callIndex]
			if !planLowersLegacyCall(plan, messageIndex, callIndex) || !isCustomToolCall(*call) {
				continue
			}
			if call.ResponseCustomToolCall == nil {
				return nil, nil, fmt.Errorf("%w: custom call at message %d index %d has no payload", ErrIncompletePlan, messageIndex, callIndex)
			}
			customCall := call.ResponseCustomToolCall
			arguments, err := json.Marshal(struct {
				Input string `json:"input"`
			}{Input: customCall.Input})
			if err != nil {
				return nil, nil, fmt.Errorf("encode custom input: %w", err)
			}
			call.ID = customCall.CallID
			call.Type = llm.ToolTypeFunction
			call.Function = llm.FunctionCall{
				Name:      session.syntheticNameInNamespace(customCall.Name, customCall.Namespace),
				Namespace: customCall.Namespace,
				Arguments: string(arguments),
			}
			call.ResponseCustomToolCall = nil
		}
	}
	callIdentities := make(map[string]toolIdentity)
	for itemIndex := range lowered.Input {
		item := &lowered.Input[itemIndex]
		if item.ToolCall != nil && planLowersItem(plan, ObjectToolCall, itemIndex) && item.ToolCall.Kind == llm.ToolKindCustom {
			call := item.ToolCall
			callIdentities[call.CallID] = toolIdentity{SourceKind: call.Kind, SourceName: call.LogicalName, SourceNamespace: call.Namespace}
			arguments, err := json.Marshal(struct {
				Input string `json:"input"`
			}{Input: call.InputText})
			if err != nil {
				return nil, nil, fmt.Errorf("encode canonical custom input: %w", err)
			}
			call.Kind = llm.ToolKindFunction
			call.LogicalName = session.syntheticNameInNamespace(call.LogicalName, call.Namespace)
			call.ArgumentsJSON = arguments
			call.ArgumentsText = ""
			call.InputText = ""
			continue
		}
		if item.ToolCall != nil && planLowersItem(plan, ObjectToolCall, itemIndex) && item.ToolCall.Kind != llm.ToolKindCustom {
			call := item.ToolCall
			callIdentities[call.CallID] = toolIdentity{SourceKind: call.Kind, SourceName: call.LogicalName, SourceNamespace: call.Namespace}
			sourceKind, sourceName := call.Kind, call.LogicalName
			call.Kind = llm.ToolKindFunction
			call.LogicalName = session.syntheticToolNameInNamespace(sourceKind, sourceName, call.Namespace)
		}
		if item.ToolResult != nil && planLowersItem(plan, ObjectToolResult, itemIndex) {
			result := item.ToolResult
			sourceKind, sourceName := result.Kind, result.LogicalName
			if identity, ok := callIdentities[result.CallID]; ok {
				if sourceKind == "" {
					sourceKind = identity.SourceKind
				}
				if sourceName == "" {
					sourceName = identity.SourceName
				}
				if result.LogicalName == "" {
					result.LogicalName = identity.SourceName
				}
				if sourceKind == llm.ToolKindCustom {
					result.LogicalName = session.syntheticNameInNamespace(sourceName, identity.SourceNamespace)
				}
			}
			if sourceName == "" {
				sourceName = string(sourceKind)
			}
			item.ToolResult.Kind = llm.ToolKindFunction
			if sourceKind == llm.ToolKindCustom {
				if sourceName != "" {
					namespace := ""
					if identity, ok := callIdentities[result.CallID]; ok {
						namespace = identity.SourceNamespace
					}
					item.ToolResult.LogicalName = session.syntheticNameInNamespace(sourceName, namespace)
				}
			} else {
				item.ToolResult.LogicalName = session.syntheticToolName(sourceKind, sourceName)
			}
		}
	}
	if planHasIdentifierNormalization(plan) {
		normalizeRequestIdentifiers(lowered, session, plan.Target.APIFormat)
	}
	normalizeRequestSchemas(lowered, session, plan.Target.APIFormat)

	if trace {
		session.setLowerNanos(time.Since(startedAt).Nanoseconds())
	}
	return lowered, session, nil
}

func equivalentDiscoveredDefinition(existing, discovered llm.ToolDefinition) bool {
	existing.DeferLoading = nil
	discovered.DeferLoading = nil
	existing.ProtocolHints = llm.ProtocolHints{}
	discovered.ProtocolHints = llm.ProtocolHints{}
	existingJSON, existingErr := json.Marshal(existing)
	discoveredJSON, discoveredErr := json.Marshal(discovered)
	return existingErr == nil && discoveredErr == nil && bytes.Equal(existingJSON, discoveredJSON)
}

func filterDefinitionsByName(definitions []llm.ToolDefinition, excluded map[string]struct{}) []llm.ToolDefinition {
	if len(excluded) == 0 {
		return definitions
	}
	filtered := definitions[:0]
	for index := range definitions {
		if _, hidden := excluded[definitions[index].LogicalName]; hidden {
			continue
		}
		filtered = append(filtered, definitions[index])
	}
	return filtered
}

func lowerClientToolDefinition(definition *llm.ToolDefinition, session *Session) {
	if definition == nil || session == nil || definition.Kind == llm.ToolKindFunction {
		return
	}
	sourceKind, sourceName := definition.Kind, definition.LogicalName
	if sourceName == "" {
		sourceName = string(sourceKind)
	}
	parameters := append(json.RawMessage(nil), clientToolFunctionParameters...)
	strict := (*bool)(nil)
	if definition.Function != nil {
		if len(definition.Function.Parameters) > 0 {
			parameters = append(json.RawMessage(nil), definition.Function.Parameters...)
		}
		strict = definition.Function.Strict
	} else if definition.Hosted != nil && len(definition.Hosted.Configuration) > 0 {
		var wire struct {
			InputSchema json.RawMessage `json:"input_schema"`
		}
		if json.Unmarshal(definition.Hosted.Configuration, &wire) == nil && len(wire.InputSchema) > 0 && json.Valid(wire.InputSchema) {
			parameters = append(json.RawMessage(nil), wire.InputSchema...)
		}
	}
	definition.Kind = llm.ToolKindFunction
	definition.LogicalName = session.syntheticToolName(sourceKind, sourceName)
	definition.Function = &llm.FunctionDefinition{Parameters: parameters, Strict: strict}
	definition.Freeform = nil
	definition.Hosted = nil
	definition.MCP = nil
}

func planLowersDefinition(plan *Plan, toolIndex int) bool {
	for _, action := range plan.Actions {
		if action.Kind == ActionLower && !isStructuralNormalization(action.Strategy) && action.Ref.Kind == ObjectToolDefinition && action.Ref.ItemIndex == -1 && action.Ref.ToolIndex == toolIndex {
			return true
		}
	}
	return false
}

func planLowersItem(plan *Plan, kind ObjectKind, itemIndex int) bool {
	for _, action := range plan.Actions {
		if action.Kind == ActionLower && !isStructuralNormalization(action.Strategy) && action.Ref.Kind == kind && action.Ref.ItemIndex == itemIndex {
			return true
		}
	}
	return false
}

func planLowersToolChoice(plan *Plan) bool {
	for _, action := range plan.Actions {
		if action.Kind == ActionLower && !isStructuralNormalization(action.Strategy) && action.Ref.Kind == ObjectToolChoice {
			return true
		}
	}
	return false
}

func planLowersLegacyCall(plan *Plan, messageIndex, callIndex int) bool {
	for _, action := range plan.Actions {
		if action.Kind == ActionLower && !isStructuralNormalization(action.Strategy) && action.Ref.Kind == ObjectToolCall && action.Ref.ItemIndex == -1 &&
			action.Ref.MessageIndex == messageIndex && action.Ref.ToolCallIndex == callIndex {
			return true
		}
	}
	return false
}

func isStructuralNormalization(strategy StrategyID) bool {
	return strategy == StrategyIdentifierNormalize || strategy == StrategySchemaNormalize
}

func planNeedsLowering(plan *Plan) bool {
	if plan == nil {
		return false
	}
	for _, action := range plan.Actions {
		if action.Kind == ActionLower {
			return true
		}
	}
	return false
}

func customFunctionDescription(custom *llm.ResponseCustomTool) string {
	if custom == nil || strings.TrimSpace(custom.Description) == "" {
		// A lowered function must not manufacture a semantic description for a
		// free-form custom tool. Responses Lite requires that field and its final
		// wire contract will reject it before transport, preserving the original
		// fail-closed rule rather than treating the generic envelope instruction
		// as a description of client behavior.
		return ""
	}
	parts := make([]string, 0, 3)
	parts = append(parts, strings.TrimSpace(custom.Description))
	parts = append(parts, "Pass the complete free-form tool input verbatim in the input field.")
	if custom.Format != nil && custom.Format.Type != "" {
		constraint := "Input format: " + custom.Format.Type
		if custom.Format.Syntax != "" {
			constraint += " (" + custom.Format.Syntax + ")"
		}
		if custom.Format.Definition != "" {
			constraint += ": " + custom.Format.Definition
		}
		parts = append(parts, constraint)
	}
	return strings.Join(parts, "\n")
}

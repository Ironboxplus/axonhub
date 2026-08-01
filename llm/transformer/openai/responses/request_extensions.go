package responses

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
)

func attachOpenAIResponsesRequestExtensions(chatReq *llm.Request, req *Request, rawBody []byte) {
	if chatReq == nil || req == nil {
		return
	}

	raw := parseRawRequestFragments(rawBody)
	reasoningContext := ""
	if req.Reasoning != nil {
		reasoningContext = req.Reasoning.Context
	}
	requestExt := &llm.OpenAIResponsesRequestExtensions{
		ReasoningContext: reasoningContext,
		RawTools:         buildRawOnlyToolFragments(req.Tools, raw.Tools),
		ToolSignatures:   buildRepresentedToolSignatures(chatReq),
		RawToolChoice:    rawUnsupportedToolChoice(req.ToolChoice, raw.ToolChoice),
		RawInputItems:    buildRawOnlyInputFragments(req.Input, raw.InputItems),
	}

	if requestExt.ReasoningContext == "" && len(requestExt.RawTools) == 0 && len(requestExt.RawToolChoice) == 0 && len(requestExt.RawInputItems) == 0 {
		return
	}

	ext := llm.EnsureOpenAIResponsesProviderExtensions(chatReq)
	if ext == nil {
		return
	}
	ext.Request = requestExt
}

type rawRequestFragments struct {
	Tools      []json.RawMessage
	ToolChoice json.RawMessage
	InputItems []json.RawMessage
}

func parseRawRequestFragments(rawBody []byte) rawRequestFragments {
	if len(rawBody) == 0 {
		return rawRequestFragments{}
	}

	var raw struct {
		Tools      []json.RawMessage `json:"tools"`
		ToolChoice json.RawMessage   `json:"tool_choice"`
		Input      json.RawMessage   `json:"input"`
	}
	if err := json.Unmarshal(rawBody, &raw); err != nil {
		return rawRequestFragments{}
	}

	var inputItems []json.RawMessage
	if len(raw.Input) > 0 && json.Unmarshal(raw.Input, &inputItems) != nil {
		inputItems = nil
	}

	return rawRequestFragments{
		Tools:      raw.Tools,
		ToolChoice: raw.ToolChoice,
		InputItems: inputItems,
	}
}

func buildRepresentedToolSignatures(request *llm.Request) []string {
	tools, represented, err := canonicalRequestTools(request)
	if err != nil || !represented || len(tools) == 0 {
		return nil
	}

	signatures := make([]string, 0, len(tools))
	for _, tool := range tools {
		signature, ok := responseToolSignature(tool)
		if !ok {
			return nil
		}
		signatures = append(signatures, signature)
	}

	return signatures
}

func buildRawOnlyToolFragments(tools []Tool, rawTools []json.RawMessage) []llm.OpenAIResponsesRawFragment {
	if len(tools) == 0 {
		return nil
	}

	fragments := make([]llm.OpenAIResponsesRawFragment, 0, len(tools))
	for i := range tools {
		if i >= len(rawTools) || len(rawTools[i]) == 0 || isStructurallyRepresentedTool(tools[i]) {
			continue
		}

		fragments = append(fragments, llm.OpenAIResponsesRawFragment{
			Type:                 tools[i].Type,
			Name:                 tools[i].Name,
			OriginalIndex:        i,
			RepresentedToolCount: representedNamespaceToolCount(tools[i]),
			Raw:                  cloneRaw(rawTools[i]),
		})
	}

	return fragments
}

func representedNamespaceToolCount(tool Tool) int {
	if tool.Type != "namespace" {
		return 0
	}

	for _, subTool := range tool.Tools {
		if isStructurallyRepresentedNamespaceChild(subTool) {
			// Canonical Responses outbound groups every typed child back into
			// one top-level namespace object. A raw namespace fragment therefore
			// replaces one structured tool, not one tool per child.
			return 1
		}
	}
	return 0
}

func isStructurallyRepresentedToolType(toolType string) bool {
	switch toolType {
	case "function", "image_generation", "web_search", "custom", "mcp", "namespace", "local_shell", "tool_search",
		"file_search", "code_interpreter", "computer", "computer_use_preview", "shell", "apply_patch":
		return true
	default:
		return false
	}
}

func isStructurallyRepresentedTool(tool Tool) bool {
	if tool.Type == "tool_search" {
		return tool.Execution == string(llm.ExecutionOwnerClient)
	}
	if tool.Type == "namespace" {
		if len(tool.Tools) == 0 {
			return false
		}
		for _, subTool := range tool.Tools {
			if !isStructurallyRepresentedNamespaceChild(subTool) {
				return false
			}
		}
		return true
	}
	return isStructurallyRepresentedToolType(tool.Type)
}

func isStructurallyRepresentedNamespaceChild(tool Tool) bool {
	return tool.Type == "" || tool.Type == "function" || tool.Type == "custom"
}

func responseToolSignature(tool Tool) (string, bool) {
	encoded, err := json.Marshal(tool)
	if err != nil {
		return "", false
	}
	digest := sha256.Sum256(encoded)
	return fmt.Sprintf("%x", digest), true
}

func rawUnsupportedToolChoice(choice *ToolChoice, rawChoice json.RawMessage) json.RawMessage {
	if choice == nil || len(rawChoice) == 0 {
		return nil
	}

	if len(choice.Tools) > 0 && (choice.Type == nil || *choice.Type != "allowed_tools") {
		return cloneRaw(rawChoice)
	}

	return nil
}

func buildRawOnlyInputFragments(input Input, rawItems []json.RawMessage) []llm.OpenAIResponsesRawFragment {
	if len(input.Items) == 0 {
		return nil
	}

	fragments := make([]llm.OpenAIResponsesRawFragment, 0)
	for i := range input.Items {
		item := input.Items[i]
		if i >= len(rawItems) || len(rawItems[i]) == 0 || isStructurallyRepresentedInputItemValue(item) {
			continue
		}

		fragments = append(fragments, llm.OpenAIResponsesRawFragment{
			Type:          item.Type,
			Name:          item.Name,
			CallID:        item.CallID,
			OriginalIndex: i,
			Raw:           cloneRaw(rawItems[i]),
		})
	}

	return fragments
}

func isStructurallyRepresentedInputItem(itemType string) bool {
	switch itemType {
	case "", "message", "input_text", "input_image", "function_call", "function_call_output",
		"custom_tool_call", "custom_tool_call_output", "mcp_list_tools", "mcp_approval_request",
		"mcp_approval_response", "mcp_call", "web_search_call", "image_generation_call", "local_shell_call",
		"local_shell_call_output", "computer_call", "computer_call_output", "file_search_call", "code_interpreter_call",
		"shell_call", "shell_call_output", "apply_patch_call", "apply_patch_call_output",
		"tool_search_call", "tool_search_output", "reasoning",
		"compaction", "compaction_summary", "additional_tools":
		return true
	default:
		return false
	}
}

func isStructurallyRepresentedInputItemValue(item Item) bool {
	if item.Type == "tool_search_call" || item.Type == "tool_search_output" {
		return item.Execution == string(llm.ExecutionOwnerClient)
	}
	return isStructurallyRepresentedInputItem(item.Type)
}

func openAIResponsesRequestExtensions(llmReq *llm.Request) *llm.OpenAIResponsesRequestExtensions {
	if llmReq == nil || llmReq.ProviderExtensions == nil || llmReq.ProviderExtensions.OpenAIResponses == nil {
		return nil
	}
	requestExt := llmReq.ProviderExtensions.OpenAIResponses.Request

	return requestExt
}

func marshalRequestPayload(payload Request, llmReq *llm.Request) ([]byte, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	requestExt := openAIResponsesRequestExtensions(llmReq)
	if requestExt == nil {
		return body, nil
	}

	var obj map[string]json.RawMessage
	if err := json.Unmarshal(body, &obj); err != nil {
		return nil, err
	}

	if tools, ok, err := mergeRawOnlyTools(obj["tools"], requestExt); err != nil {
		return nil, err
	} else if ok {
		toolsRaw, err := json.Marshal(tools)
		if err != nil {
			return nil, err
		}
		obj["tools"] = toolsRaw
	}

	if len(requestExt.RawToolChoice) > 0 && rawToolChoiceMatchesCurrentTools(requestExt.RawToolChoice, payload.ToolChoice) {
		obj["tool_choice"] = cloneRaw(requestExt.RawToolChoice)
	}

	if input, ok := mergeRawOnlyInputItems(obj["input"], requestExt); ok {
		inputRaw, err := json.Marshal(input)
		if err != nil {
			return nil, err
		}
		obj["input"] = inputRaw
	}

	return json.Marshal(obj)
}

func mergeRawOnlyInputItems(structuredRaw json.RawMessage, requestExt *llm.OpenAIResponsesRequestExtensions) ([]json.RawMessage, bool) {
	if requestExt == nil || len(requestExt.RawInputItems) == 0 {
		return nil, false
	}

	var structuredItems []json.RawMessage
	if len(structuredRaw) > 0 {
		if err := json.Unmarshal(structuredRaw, &structuredItems); err != nil {
			return nil, false
		}
	}

	total := len(structuredItems) + len(requestExt.RawInputItems)
	items := make([]json.RawMessage, 0, total)
	structuredIndex := 0
	rawByIndex := make(map[int]json.RawMessage, len(requestExt.RawInputItems))
	for _, fragment := range requestExt.RawInputItems {
		if len(fragment.Raw) == 0 || fragment.OriginalIndex < 0 {
			return nil, false
		}
		rawByIndex[fragment.OriginalIndex] = cloneRaw(fragment.Raw)
	}

	for i := 0; i < total; i++ {
		if raw, ok := rawByIndex[i]; ok {
			items = append(items, raw)
			continue
		}
		if structuredIndex >= len(structuredItems) {
			return nil, false
		}
		items = append(items, cloneRaw(structuredItems[structuredIndex]))
		structuredIndex++
	}

	if structuredIndex != len(structuredItems) {
		return nil, false
	}

	return items, true
}

func mergeRawOnlyTools(structuredRaw json.RawMessage, requestExt *llm.OpenAIResponsesRequestExtensions) ([]json.RawMessage, bool, error) {
	if requestExt == nil || len(requestExt.RawTools) == 0 {
		return nil, false, nil
	}

	var structuredTools []json.RawMessage
	if len(structuredRaw) > 0 {
		if err := json.Unmarshal(structuredRaw, &structuredTools); err != nil {
			return nil, false, fmt.Errorf("decode canonical Responses tools for raw merge: %w", err)
		}
	}
	if len(structuredTools) != len(requestExt.ToolSignatures) {
		return nil, false, fmt.Errorf(
			"cannot safely merge opaque Responses tools after canonical tool count changed: got %d, want %d",
			len(structuredTools), len(requestExt.ToolSignatures),
		)
	}

	representedCount := 0
	for _, fragment := range requestExt.RawTools {
		if fragment.RepresentedToolCount < 0 {
			return nil, false, fmt.Errorf("invalid represented tool count %d for raw Responses tool", fragment.RepresentedToolCount)
		}
		representedCount += fragment.RepresentedToolCount
	}
	if representedCount > len(structuredTools) {
		return nil, false, fmt.Errorf("opaque Responses tools replace %d canonical tools, only %d available", representedCount, len(structuredTools))
	}

	total := len(structuredTools) - representedCount + len(requestExt.RawTools)
	tools := make([]json.RawMessage, 0, total)
	structuredIndex := 0
	rawByIndex := make(map[int]llm.OpenAIResponsesRawFragment, len(requestExt.RawTools))
	for _, fragment := range requestExt.RawTools {
		if len(fragment.Raw) == 0 || fragment.OriginalIndex < 0 {
			return nil, false, fmt.Errorf("invalid opaque Responses tool fragment at index %d", fragment.OriginalIndex)
		}
		if _, duplicate := rawByIndex[fragment.OriginalIndex]; duplicate {
			return nil, false, fmt.Errorf("duplicate opaque Responses tool fragment at index %d", fragment.OriginalIndex)
		}
		rawByIndex[fragment.OriginalIndex] = fragment
	}

	for i := 0; i < total; i++ {
		if fragment, ok := rawByIndex[i]; ok {
			switch fragment.RepresentedToolCount {
			case 0:
				tools = append(tools, cloneRaw(fragment.Raw))
			case 1:
				if structuredIndex >= len(structuredTools) {
					return nil, false, fmt.Errorf("opaque Responses tool at index %d has no canonical counterpart", i)
				}
				if structuredToolSignatureMatches(structuredTools[structuredIndex], requestExt.ToolSignatures[structuredIndex]) {
					tools = append(tools, cloneRaw(fragment.Raw))
				} else {
					merged, err := mergeChangedNamespaceTool(structuredTools[structuredIndex], fragment.Raw)
					if err != nil {
						return nil, false, fmt.Errorf("merge opaque Responses namespace %q at index %d: %w", fragment.Name, i, err)
					}
					tools = append(tools, merged)
				}
				structuredIndex++
			default:
				return nil, false, fmt.Errorf(
					"cannot safely merge opaque Responses tool at index %d replacing %d canonical tools",
					i, fragment.RepresentedToolCount,
				)
			}
			continue
		}
		if structuredIndex >= len(structuredTools) {
			return nil, false, fmt.Errorf("canonical Responses tool layout ended before output index %d", i)
		}
		tools = append(tools, cloneRaw(structuredTools[structuredIndex]))
		structuredIndex++
	}

	if structuredIndex != len(structuredTools) {
		return nil, false, fmt.Errorf("canonical Responses tool layout left %d unmerged tools", len(structuredTools)-structuredIndex)
	}

	return tools, true, nil
}

func mergeChangedNamespaceTool(currentRaw, originalRaw json.RawMessage) (json.RawMessage, error) {
	var currentObject map[string]json.RawMessage
	if err := json.Unmarshal(currentRaw, &currentObject); err != nil {
		return nil, fmt.Errorf("decode current namespace: %w", err)
	}
	var originalObject map[string]json.RawMessage
	if err := json.Unmarshal(originalRaw, &originalObject); err != nil {
		return nil, fmt.Errorf("decode original namespace: %w", err)
	}

	currentType, currentName, err := rawToolIdentity(currentObject)
	if err != nil {
		return nil, fmt.Errorf("current namespace identity: %w", err)
	}
	originalType, originalName, err := rawToolIdentity(originalObject)
	if err != nil {
		return nil, fmt.Errorf("original namespace identity: %w", err)
	}
	if currentType != "namespace" || originalType != "namespace" || currentName == "" || currentName != originalName {
		return nil, fmt.Errorf(
			"canonical counterpart changed identity from %q/%q to %q/%q",
			originalType, originalName, currentType, currentName,
		)
	}

	var currentChildren []json.RawMessage
	if err := json.Unmarshal(currentObject["tools"], &currentChildren); err != nil {
		return nil, fmt.Errorf("decode current namespace children: %w", err)
	}
	for index, child := range currentChildren {
		childType, err := rawToolType(child)
		if err != nil {
			return nil, fmt.Errorf("current namespace child %d: %w", index, err)
		}
		if !isStructurallyRepresentedNamespaceChild(Tool{Type: childType}) {
			return nil, fmt.Errorf("current namespace child %d has non-canonical type %q", index, childType)
		}
	}

	var originalChildren []json.RawMessage
	if err := json.Unmarshal(originalObject["tools"], &originalChildren); err != nil {
		return nil, fmt.Errorf("decode original namespace children: %w", err)
	}
	mergedChildren := make([]json.RawMessage, 0, len(currentChildren)+len(originalChildren))
	currentIndex := 0
	for index, child := range originalChildren {
		childType, err := rawToolType(child)
		if err != nil {
			return nil, fmt.Errorf("original namespace child %d: %w", index, err)
		}
		if isStructurallyRepresentedNamespaceChild(Tool{Type: childType}) {
			if currentIndex < len(currentChildren) {
				mergedChildren = append(mergedChildren, cloneRaw(currentChildren[currentIndex]))
				currentIndex++
			}
			continue
		}
		mergedChildren = append(mergedChildren, cloneRaw(child))
	}
	for currentIndex < len(currentChildren) {
		mergedChildren = append(mergedChildren, cloneRaw(currentChildren[currentIndex]))
		currentIndex++
	}

	mergedChildrenRaw, err := json.Marshal(mergedChildren)
	if err != nil {
		return nil, fmt.Errorf("encode merged namespace children: %w", err)
	}
	for key, value := range currentObject {
		originalObject[key] = cloneRaw(value)
	}
	originalObject["tools"] = mergedChildrenRaw
	merged, err := json.Marshal(originalObject)
	if err != nil {
		return nil, fmt.Errorf("encode merged namespace: %w", err)
	}
	return merged, nil
}

func rawToolIdentity(object map[string]json.RawMessage) (string, string, error) {
	var identity struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	raw, err := json.Marshal(object)
	if err != nil {
		return "", "", err
	}
	if err := json.Unmarshal(raw, &identity); err != nil {
		return "", "", err
	}
	return identity.Type, identity.Name, nil
}

func rawToolType(raw json.RawMessage) (string, error) {
	var identity struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(raw, &identity); err != nil {
		return "", err
	}
	return identity.Type, nil
}

func structuredToolSignatureMatches(rawTool json.RawMessage, expected string) bool {
	if expected == "" {
		return false
	}
	var tool Tool
	if err := json.Unmarshal(rawTool, &tool); err != nil {
		return false
	}
	signature, ok := responseToolSignature(tool)
	return ok && signature == expected
}

func rawToolChoiceMatchesCurrentTools(raw json.RawMessage, current *ToolChoice) bool {
	if current == nil {
		return true
	}

	var rawChoice ToolChoice
	if err := json.Unmarshal(raw, &rawChoice); err != nil {
		return false
	}

	currentSignature := toolChoiceSignature(current)
	if currentSignature == "" {
		return true
	}

	return toolChoiceSignature(&rawChoice) == currentSignature
}

func toolChoiceSignature(choice *ToolChoice) string {
	if choice == nil {
		return ""
	}

	if choice.Mode != nil {
		return "mode:" + *choice.Mode
	}

	if choice.Type != nil && choice.Name != nil {
		return "named:" + *choice.Type + ":" + *choice.Name
	}

	if len(choice.Tools) > 0 {
		return "tools"
	}

	return ""
}

func cloneRaw(src json.RawMessage) json.RawMessage {
	if len(src) == 0 {
		return nil
	}

	return append(json.RawMessage(nil), src...)
}

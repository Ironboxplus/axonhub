package responses

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

var localShellParametersJSON = json.RawMessage(`{"type":"object","properties":{"type":{"type":"string","enum":["exec"]},"command":{"type":"array","items":{"type":"string"}},"timeout_ms":{"type":"integer","minimum":0},"working_directory":{"type":"string"},"env":{"type":"object","additionalProperties":{"type":"string"}},"user":{"type":"string"}},"required":["type","command"],"additionalProperties":false}`)

// convertRequestToCanonical decodes the Responses wire model directly into
// the ordered canonical model. The legacy Message projection remains in the
// request for adapters that have not migrated yet, but it is not the source of
// truth for item identity, ordering, or tool-call lifecycle.
func convertRequestToCanonical(req *Request, rawBody []byte) ([]llm.Item, []llm.ToolDefinition, error) {
	if req == nil {
		return nil, nil, nil
	}

	items := make([]llm.Item, 0, len(req.Input.Items)+1)
	if req.Instructions != "" {
		items = append(items, llm.Item{
			Kind:    llm.ItemKindMessage,
			Role:    llm.RoleSystem,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: req.Instructions}},
			ProtocolHints: llm.ProtocolHints{
				SourceFormat: llm.APIFormatOpenAIResponse,
				SourceType:   "instructions",
				Ordinal:      -1,
			},
		})
	}

	raw := parseRawRequestFragments(rawBody)
	additionalDefinitions := make([]llm.ToolDefinition, 0)
	if req.Input.Text != nil {
		items = append(items, llm.Item{
			Kind:    llm.ItemKindMessage,
			Role:    llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *req.Input.Text}},
			ProtocolHints: llm.ProtocolHints{
				SourceFormat: llm.APIFormatOpenAIResponse,
				SourceType:   "input_text",
				Ordinal:      0,
			},
		})
	} else {
		callKinds := make(map[string]llm.ToolKind)
		for index := range req.Input.Items {
			item := &req.Input.Items[index]
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			switch item.Type {
			case "function_call":
				callKinds[callID] = llm.ToolKindFunction
			case "custom_tool_call":
				callKinds[callID] = llm.ToolKindCustom
			case "local_shell_call":
				callKinds[callID] = llm.ToolKindLocalShell
			case "computer_call":
				callKinds[callID] = llm.ToolKindComputer
			case "apply_patch_call":
				callKinds[callID] = llm.ToolKindApplyPatch
			case "tool_search_call":
				callKinds[callID] = llm.ToolKindToolSearch
			}
		}
		for index := range req.Input.Items {
			if req.Input.Items[index].Type == "additional_tools" {
				rawItem := rawInputAt(raw.InputItems, index)
				additionalTools, role, err := responseAdditionalTools(&req.Input.Items[index], rawItem, index)
				if err != nil {
					return nil, nil, err
				}
				groupID := fmt.Sprintf("responses-additional-tools:%d", index)
				definitions, err := responseToolsToCanonical(additionalTools)
				if err != nil {
					return nil, nil, fmt.Errorf("decode Responses additional_tools item %d: %w", index, err)
				}
				for definitionIndex := range definitions {
					hints := definitions[definitionIndex].ProtocolHints
					hints.SourceFormat = llm.APIFormatOpenAIResponse
					hints.SourceType = "additional_tools"
					hints.SourceRole = responseRole(role)
					hints.SourceGroup = groupID
					hints.Ordinal = index
					definitions[definitionIndex].ProtocolHints = hints
				}
				additionalDefinitions = append(additionalDefinitions, definitions...)
				items = append(items, llm.Item{
					Kind: llm.ItemKindToolDeclaration, Role: responseRole(role),
					ToolDeclaration: &llm.ToolDeclarationItem{
						SourceGroup: groupID,
						Namespaces:  responseToolNamespaces(additionalTools),
					},
					ProtocolHints: llm.ProtocolHints{
						SourceFormat:      llm.APIFormatOpenAIResponse,
						SourceType:        "additional_tools",
						SourceRole:        responseRole(role),
						SourceGroup:       groupID,
						Ordinal:           index,
						ResidualOwnerType: req.Input.Items[index].Type,
						SourceResidual:    cloneRaw(req.Input.Items[index].Residual),
					},
				})
				continue
			}
			item, err := responseItemToCanonical(&req.Input.Items[index], rawInputAt(raw.InputItems, index), index)
			if err != nil {
				return nil, nil, err
			}
			if item != nil {
				if item.ToolResult != nil {
					if kind, ok := callKinds[item.ToolResult.CallID]; ok {
						item.ToolResult.Kind = kind
						if kind == llm.ToolKindLocalShell && item.ToolResult.LogicalName == "" {
							item.ToolResult.LogicalName = "local_shell"
						}
						if item.ToolResult.Execution == "" {
							item.ToolResult.Execution = llm.ExecutionOwnerClient
						}
					}
				}
				if item.ProtocolHints.SourceType == "shell_call_output" && item.HostedCall != nil && item.HostedCall.Result != nil && len(items) > 0 {
					previous := &items[len(items)-1]
					if previous.HostedCall != nil && previous.HostedCall.Result == nil &&
						previous.HostedCall.Invocation.Kind == item.HostedCall.Invocation.Kind &&
						previous.HostedCall.Invocation.CallID == item.HostedCall.Invocation.CallID {
						previous.HostedCall.Result = item.HostedCall.Result
						continue
					}
				}
				items = append(items, *item)
			}
		}
	}

	definitions, err := responseToolsToCanonical(req.Tools)
	if err != nil {
		return nil, nil, err
	}
	definitions = append(definitions, additionalDefinitions...)
	return items, definitions, nil
}

func responseAdditionalTools(item *Item, raw json.RawMessage, ordinal int) ([]Tool, string, error) {
	if item != nil && item.Type == "additional_tools" && item.AdditionalTools != nil {
		return append([]Tool(nil), item.AdditionalTools...), item.Role, nil
	}
	if len(raw) == 0 {
		return nil, "", fmt.Errorf("Responses additional_tools item %d has no declaration", ordinal)
	}
	var wire struct {
		Type  string `json:"type"`
		Role  string `json:"role"`
		Tools []Tool `json:"tools"`
	}
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, "", fmt.Errorf("decode Responses additional_tools item %d: %w", ordinal, err)
	}
	if wire.Type != "additional_tools" {
		return nil, "", fmt.Errorf("Responses input item %d is %q, want additional_tools", ordinal, wire.Type)
	}
	return wire.Tools, wire.Role, nil
}

func responseToolNamespaces(tools []Tool) []llm.ToolNamespaceDeclaration {
	namespaces := make([]llm.ToolNamespaceDeclaration, 0)
	for index := range tools {
		tool := &tools[index]
		if tool.Type != "namespace" {
			continue
		}
		namespaces = append(namespaces, llm.ToolNamespaceDeclaration{
			Name: tool.Name, Description: tool.Description,
			SourceResidual: cloneRaw(tool.Residual),
		})
	}
	return namespaces
}

func responseItemToCanonical(item *Item, raw json.RawMessage, ordinal int) (*llm.Item, error) {
	if item == nil {
		return nil, nil
	}
	residual := cloneRaw(item.Residual)
	hints := llm.ProtocolHints{
		SourceFormat:      llm.APIFormatOpenAIResponse,
		SourceType:        item.Type,
		Ordinal:           ordinal,
		ResidualOwnerType: item.Type,
		SourceResidual:    residual,
	}
	status := canonicalItemStatus(item.Status)

	switch item.Type {
	case "", "message", "input_text", "output_text", "input_image":
		content, err := responseItemContent(item)
		if err != nil {
			return nil, err
		}
		if len(content) == 0 {
			return nil, nil
		}
		role := responseRole(item.Role)
		if item.Role == "" && (item.Type == "output_text") {
			role = llm.RoleAssistant
		}
		return &llm.Item{
			Kind: llm.ItemKindMessage, ID: item.ID, Role: role, Status: status,
			Content: content, ProtocolHints: hints,
		}, nil

	case "function_call":
		invocation := &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, ID: item.ID, CallID: item.CallID,
			LogicalName: item.Name, Namespace: item.Namespace,
			Status: canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
		}
		if json.Valid([]byte(item.Arguments)) {
			invocation.ArgumentsJSON = append(json.RawMessage(nil), item.Arguments...)
		} else {
			invocation.ArgumentsText = item.Arguments
		}
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: invocation, ProtocolHints: hints,
		}, nil

	case "custom_tool_call":
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindCustom, ID: item.ID, CallID: item.CallID,
				LogicalName: item.Name, Namespace: item.Namespace, InputText: stringValue(item.Input),
				Status: canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
			},
			ProtocolHints: hints,
		}, nil

	case "local_shell_call":
		if item.Action == nil || item.Action.LocalShell == nil {
			return nil, fmt.Errorf("Responses local_shell_call item %q has no exec action", item.ID)
		}
		callID := item.CallID
		if callID == "" {
			callID = item.ID
		}
		arguments, err := json.Marshal(item.Action.LocalShell)
		if err != nil {
			return nil, fmt.Errorf("marshal Responses local shell action %q: %w", item.ID, err)
		}
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindLocalShell, ID: item.ID, CallID: callID, LogicalName: "local_shell",
				ArgumentsJSON: arguments, Status: canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
			},
			ProtocolHints: hints,
		}, nil

	case "computer_call":
		if len(item.ComputerAction) == 0 || !json.Valid(item.ComputerAction) {
			return nil, fmt.Errorf("Responses computer_call item %q has no valid action", item.ID)
		}
		callID := item.CallID
		if callID == "" {
			callID = item.ID
		}
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindComputer, ID: item.ID, CallID: callID, LogicalName: "computer",
				ArgumentsJSON: append(json.RawMessage(nil), item.ComputerAction...),
				Status:        canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
				ProviderData:        append(json.RawMessage(nil), raw...),
				PendingSafetyChecks: responseSafetyChecksToCanonical(item.PendingSafetyChecks),
			},
			ProtocolHints: hints,
		}, nil

	case "apply_patch_call":
		if len(item.Operation) == 0 || !json.Valid(item.Operation) {
			return nil, fmt.Errorf("Responses apply_patch_call item %q has no valid operation", item.ID)
		}
		callID := item.CallID
		if callID == "" {
			callID = item.ID
		}
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindApplyPatch, ID: item.ID, CallID: callID, LogicalName: "apply_patch",
				ArgumentsJSON: append(json.RawMessage(nil), item.Operation...),
				Status:        canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
				ProviderData: append(json.RawMessage(nil), raw...),
			},
			ProtocolHints: hints,
		}, nil

	case "tool_search_call":
		if item.Execution != string(llm.ExecutionOwnerClient) {
			break
		}
		if item.CallID == "" || !json.Valid([]byte(item.Arguments)) {
			return nil, fmt.Errorf("Responses client tool_search_call item %q requires call_id and JSON arguments", item.ID)
		}
		return &llm.Item{
			Kind: llm.ItemKindToolCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindToolSearch, ID: item.ID, CallID: item.CallID, LogicalName: "tool_search",
				ArgumentsJSON: append(json.RawMessage(nil), item.Arguments...),
				Status:        canonicalToolCallStatus(item.Status), Execution: llm.ExecutionOwnerClient,
			},
			ProtocolHints: hints,
		}, nil

	case "tool_search_output":
		if item.Execution != string(llm.ExecutionOwnerClient) {
			break
		}
		if item.CallID == "" {
			return nil, fmt.Errorf("Responses client tool_search_output item %q requires call_id", item.ID)
		}
		discovered, err := responseToolsToCanonical(item.AdditionalTools)
		if err != nil {
			return nil, fmt.Errorf("decode Responses tool_search_output %q: %w", item.ID, err)
		}
		return &llm.Item{
			Kind: llm.ItemKindToolResult, ID: item.ID, Status: status,
			ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindToolSearch, CallID: item.CallID, LogicalName: "tool_search",
				DiscoveredTools: discovered, Execution: llm.ExecutionOwnerClient,
				Status: canonicalToolResultStatus(item.Status),
			},
			ProtocolHints: hints,
		}, nil

	case "function_call_output", "custom_tool_call_output", "local_shell_call_output", "apply_patch_call_output":
		if item.Output == nil {
			return nil, fmt.Errorf("Responses %s item %q has no output", item.Type, item.ID)
		}
		content, err := responseInputContent(item.Output)
		if err != nil {
			return nil, err
		}
		kind := llm.ToolKindFunction
		if item.Type == "custom_tool_call_output" {
			kind = llm.ToolKindCustom
		} else if item.Type == "local_shell_call_output" {
			kind = llm.ToolKindLocalShell
		} else if item.Type == "apply_patch_call_output" {
			kind = llm.ToolKindApplyPatch
		}
		return &llm.Item{
			Kind: llm.ItemKindToolResult, ID: item.ID, Status: status,
			ToolResult: &llm.ToolResult{
				Kind: kind, CallID: item.CallID, LogicalName: item.Name,
				Content: content, Status: canonicalToolResultStatus(item.Status), Execution: llm.ExecutionOwnerClient,
			},
			ProtocolHints: hints,
		}, nil

	case "computer_call_output":
		if item.ComputerOutput == nil {
			return nil, fmt.Errorf("Responses computer_call_output item %q has no output", item.ID)
		}
		encoded, err := json.Marshal(item.ComputerOutput)
		if err != nil {
			return nil, fmt.Errorf("marshal Responses computer output %q: %w", item.ID, err)
		}
		content := make([]llm.ContentBlock, 0, 1)
		if item.ComputerOutput.ImageURL != "" {
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: item.ComputerOutput.ImageURL}})
		} else {
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindText, Text: string(encoded)})
		}
		return &llm.Item{
			Kind: llm.ItemKindToolResult, ID: item.ID, Status: status,
			ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindComputer, CallID: item.CallID, LogicalName: "computer",
				Content: content, StructuredContent: encoded,
				Status: canonicalToolResultStatus(item.Status), Execution: llm.ExecutionOwnerClient,
				ProviderData:             append(json.RawMessage(nil), raw...),
				AcknowledgedSafetyChecks: responseSafetyChecksToCanonical(item.AcknowledgedSafetyChecks),
			},
			ProtocolHints: hints,
		}, nil

	case "mcp_list_tools":
		tools := make([]llm.MCPDiscoveredTool, 0, len(item.Tools))
		for index := range item.Tools {
			tool := &item.Tools[index]
			tools = append(tools, llm.MCPDiscoveredTool{
				Name: tool.Name, Description: tool.Description,
				InputSchema:    append(json.RawMessage(nil), tool.InputSchema...),
				Annotations:    append(json.RawMessage(nil), tool.Annotations...),
				SourceResidual: cloneRaw(tool.Residual),
			})
		}
		return &llm.Item{
			Kind: llm.ItemKindMCPListTools, ID: item.ID, Status: status,
			MCPListTools: &llm.MCPListTools{
				ServerLabel: item.ServerLabel, Tools: tools, Error: stringValue(item.Error),
			},
			ProtocolHints: hints,
		}, nil

	case "mcp_approval_request":
		argumentsJSON, argumentsText := canonicalMCPArguments(item.Arguments)
		return &llm.Item{
			Kind: llm.ItemKindMCPApprovalRequest, ID: item.ID, Status: status,
			MCPApprovalRequest: &llm.MCPApprovalRequest{
				ServerLabel: item.ServerLabel, LogicalName: item.Name,
				ArgumentsJSON: argumentsJSON, ArgumentsText: argumentsText,
			},
			ProtocolHints: hints,
		}, nil

	case "mcp_approval_response":
		if item.Approve == nil {
			return nil, fmt.Errorf("Responses mcp_approval_response item %q has no approve decision", item.ID)
		}
		return &llm.Item{
			Kind: llm.ItemKindMCPApprovalResponse, ID: item.ID, Status: status,
			MCPApprovalResponse: &llm.MCPApprovalResponse{
				ApprovalRequestID: item.ApprovalRequestID, Approve: *item.Approve, Reason: item.Reason,
			},
			ProtocolHints: hints,
		}, nil

	case "mcp_call":
		argumentsJSON, argumentsText := canonicalMCPArguments(item.Arguments)
		return &llm.Item{
			Kind: llm.ItemKindMCPCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			MCPCall: &llm.MCPCall{
				ServerLabel: item.ServerLabel, LogicalName: item.Name,
				ApprovalRequestID: item.ApprovalRequestID,
				ArgumentsJSON:     argumentsJSON, ArgumentsText: argumentsText,
				Output: responseMCPOutput(item.Output), Error: stringValue(item.Error),
				Status: llm.MCPCallStatus(stringValue(item.Status)),
			},
			ProtocolHints: hints,
		}, nil

	case "web_search_call", "image_generation_call", "file_search_call", "code_interpreter_call", "shell_call", "shell_call_output":
		return responseHostedCallToCanonical(item, raw, hints, status)

	case "reasoning", "reasoning_text":
		var content strings.Builder
		summaryParts := make([]llm.ReasoningPart, 0, len(item.Summary))
		for index := range item.Summary {
			content.WriteString(item.Summary[index].Text)
			summaryParts = append(summaryParts, llm.ReasoningPart{
				Type: item.Summary[index].Type, Text: item.Summary[index].Text,
				ResidualOwnerType: item.Summary[index].Type, SourceResidual: cloneRaw(item.Summary[index].Residual),
			})
		}
		contentParts := make([]llm.ReasoningPart, 0, len(item.ReasoningContent))
		contentField := ""
		for index := range item.ReasoningContent {
			content.WriteString(item.ReasoningContent[index].Text)
			contentParts = append(contentParts, llm.ReasoningPart{
				Type: item.ReasoningContent[index].Type, Text: item.ReasoningContent[index].Text,
				ResidualOwnerType: item.ReasoningContent[index].Type, SourceResidual: cloneRaw(item.ReasoningContent[index].Residual),
			})
			contentField = "reasoning_content"
		}
		if item.Content != nil && len(item.Content.Items) > 0 {
			contentParts = contentParts[:0]
			contentField = "content"
			for index := range item.Content.Items {
				part := &item.Content.Items[index]
				text := stringValue(part.Text)
				content.WriteString(text)
				contentParts = append(contentParts, llm.ReasoningPart{
					Type: part.Type, Text: text, ResidualOwnerType: part.Type, SourceResidual: cloneRaw(part.Residual),
				})
			}
		}
		return &llm.Item{
			Kind: llm.ItemKindReasoning, ID: item.ID, Role: llm.RoleAssistant, Status: status,
			Reasoning: &llm.ReasoningItem{
				ID: item.ID, Content: content.String(), Signature: stringValue(item.EncryptedContent),
				SummaryParts: summaryParts, ContentParts: contentParts, ContentField: contentField,
			},
			ProtocolHints: hints,
		}, nil

	case "compaction", "compaction_summary":
		return &llm.Item{
			Kind: llm.ItemKindCompaction, ID: item.ID, Status: status,
			Compaction: &llm.CompactionItem{
				EncryptedContent: stringValue(item.EncryptedContent), CreatedBy: stringValue(item.CreatedBy),
			},
			ProtocolHints: hints,
		}, nil

	case "compaction_trigger":
		return &llm.Item{
			Kind:              llm.ItemKindCompactionTrigger,
			CompactionTrigger: &llm.CompactionTriggerItem{},
			ProtocolHints:     hints,
		}, nil
	}

	if len(raw) == 0 {
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("marshal unknown Responses item %q: %w", item.Type, err)
		}
		raw = encoded
	}
	return &llm.Item{
		Kind: llm.ItemKindUnknown, ID: item.ID, Status: status,
		Unknown: &llm.UnknownItem{
			Type: item.Type, Raw: append(json.RawMessage(nil), raw...), Behavioral: true,
		},
		ProtocolHints: hints,
	}, nil
}

func canonicalMCPArguments(arguments string) (json.RawMessage, string) {
	if json.Valid([]byte(arguments)) {
		return append(json.RawMessage(nil), arguments...), ""
	}
	return nil, arguments
}

func responseSafetyChecksToCanonical(checks []ComputerSafetyCheck) []llm.ToolSafetyCheck {
	if len(checks) == 0 {
		return nil
	}
	result := make([]llm.ToolSafetyCheck, 0, len(checks))
	for index := range checks {
		check := &checks[index]
		result = append(result, llm.ToolSafetyCheck{
			ID: check.ID, Code: check.Code, Message: check.Message,
			SourceResidual: cloneRaw(check.Residual),
		})
	}
	return result
}

func responseMCPOutput(output *Input) string {
	if output == nil || output.Text == nil {
		return ""
	}
	return *output.Text
}

func responseHostedCallToCanonical(item *Item, raw json.RawMessage, hints llm.ProtocolHints, status llm.ItemStatus) (*llm.Item, error) {
	if item == nil {
		return nil, nil
	}
	kind, logicalName := responseHostedKind(item.Type)
	callID := item.CallID
	if callID == "" {
		callID = item.ID
	}
	providerData := append(json.RawMessage(nil), raw...)
	if len(providerData) == 0 {
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("marshal Responses hosted item %q: %w", item.Type, err)
		}
		providerData = encoded
	}
	arguments, err := responseHostedArguments(item)
	if err != nil {
		return nil, err
	}
	hosted := &llm.HostedToolCall{Invocation: llm.ToolInvocation{
		Kind: kind, ID: item.ID, CallID: callID, LogicalName: logicalName,
		ArgumentsJSON: arguments, Status: canonicalToolCallStatus(item.Status),
		Execution: llm.ExecutionOwnerProvider, ProviderData: providerData,
	}}
	if result := responseHostedResult(item, kind, callID, providerData); result != nil {
		hosted.Result = result
	}
	return &llm.Item{
		Kind: llm.ItemKindHostedCall, ID: item.ID, Role: llm.RoleAssistant, Status: status,
		HostedCall: hosted, ProtocolHints: hints,
	}, nil
}

func responseHostedKind(itemType string) (llm.ToolKind, string) {
	switch itemType {
	case "image_generation_call":
		return llm.ToolKindImageGeneration, "image_generation"
	case "file_search_call":
		return llm.ToolKindFileSearch, "file_search"
	case "code_interpreter_call":
		return llm.ToolKindCodeInterpreter, "code_interpreter"
	case "shell_call", "shell_call_output":
		return llm.ToolKindShell, "shell"
	default:
		return llm.ToolKindWebSearch, "web_search"
	}
}

func responseHostedArguments(item *Item) (json.RawMessage, error) {
	if item == nil {
		return nil, nil
	}
	var value any
	switch item.Type {
	case "file_search_call":
		value = struct {
			Queries []string `json:"queries,omitempty"`
		}{Queries: item.Queries}
	case "code_interpreter_call":
		value = struct {
			Code        *string `json:"code,omitempty"`
			ContainerID *string `json:"container_id,omitempty"`
		}{Code: item.Code, ContainerID: item.ContainerID}
	case "shell_call":
		if len(item.ComputerAction) == 0 {
			return nil, nil
		}
		return append(json.RawMessage(nil), item.ComputerAction...), nil
	case "shell_call_output":
		return nil, nil
	default:
		value = item.Action
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal Responses hosted action %q: %w", item.Type, err)
	}
	return encoded, nil
}

func responseHostedResult(item *Item, kind llm.ToolKind, callID string, providerData json.RawMessage) *llm.ToolResult {
	if item == nil {
		return nil
	}
	_, logicalName := responseHostedKind(item.Type)
	result := &llm.ToolResult{
		Kind: kind, CallID: callID, LogicalName: logicalName,
		Status: canonicalToolResultStatus(item.Status), ProviderData: append(json.RawMessage(nil), providerData...),
	}
	if item.Type == "web_search_call" && item.Action != nil && item.Action.WebSearch != nil {
		for index := range item.Action.WebSearch.Sources {
			source := &item.Action.WebSearch.Sources[index]
			result.Content = append(result.Content, llm.ContentBlock{
				Kind:     llm.ContentKindCitation,
				Citation: &llm.URLCitation{URL: source.URL, Title: source.Title, SourceResidual: cloneRaw(source.Residual)},
			})
		}
		if len(result.Content) == 0 {
			return nil
		}
		return result
	}
	if item.Type == "image_generation_call" && item.Result != nil && *item.Result != "" {
		format := "png"
		if item.OutputFormat != nil && *item.OutputFormat != "" {
			format = *item.OutputFormat
		}
		result.Content = []llm.ContentBlock{{
			Kind:  llm.ContentKindImage,
			Image: &llm.ImageURL{URL: "data:image/" + format + ";base64," + *item.Result},
		}}
		return result
	}
	if item.Type == "file_search_call" && len(item.Results) > 0 && json.Valid(item.Results) {
		result.StructuredContent = append(json.RawMessage(nil), item.Results...)
		result.Content = []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(item.Results)}}
		return result
	}
	if item.Type == "code_interpreter_call" && len(item.Outputs) > 0 && json.Valid(item.Outputs) && string(item.Outputs) != "null" {
		result.StructuredContent = append(json.RawMessage(nil), item.Outputs...)
		result.Content = []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(item.Outputs)}}
		return result
	}
	if item.Type == "shell_call_output" {
		if len(item.RawOutput) > 0 && json.Valid(item.RawOutput) {
			result.StructuredContent = append(json.RawMessage(nil), item.RawOutput...)
			result.Content = []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(item.RawOutput)}}
		}
		return result
	}
	return nil
}

func responseItemContent(item *Item) ([]llm.ContentBlock, error) {
	if item.Content != nil {
		return responseInputContent(item.Content)
	}
	if item.Text != nil {
		return []llm.ContentBlock{{
			Kind: llm.ContentKindText, ID: item.ID, Text: *item.Text,
			ResidualOwnerType: item.Type, SourceResidual: cloneRaw(item.Residual),
		}}, nil
	}
	if item.ImageURL != nil {
		return []llm.ContentBlock{{
			Kind: llm.ContentKindImage, ID: item.ID,
			Image:             &llm.ImageURL{URL: *item.ImageURL, Detail: item.Detail},
			ResidualOwnerType: item.Type, SourceResidual: cloneRaw(item.Residual),
		}}, nil
	}
	if item.Type == "input_file" {
		if document := responseDocument(item); document != nil {
			return []llm.ContentBlock{{
				Kind: llm.ContentKindDocument, ID: item.ID, Document: document,
				ResidualOwnerType: item.Type, SourceResidual: cloneRaw(item.Residual),
			}}, nil
		}
	}
	return nil, nil
}

func responseInputContent(input *Input) ([]llm.ContentBlock, error) {
	if input == nil {
		return nil, nil
	}
	if input.Text != nil {
		return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: *input.Text}}, nil
	}
	blocks := make([]llm.ContentBlock, 0, len(input.Items))
	for index := range input.Items {
		item := &input.Items[index]
		residual := cloneRaw(item.Residual)
		switch item.Type {
		case "", "text", "input_text", "output_text":
			if item.Text != nil {
				blocks = append(blocks, llm.ContentBlock{
					Kind: llm.ContentKindText, ID: item.ID, Text: *item.Text,
					ResidualOwnerType: item.Type, SourceResidual: residual,
				})
			}
			for annotationIndex := range item.Annotations {
				annotation := &item.Annotations[annotationIndex]
				if annotation.URLCitation == nil {
					continue
				}
				blocks = append(blocks, llm.ContentBlock{
					Kind: llm.ContentKindCitation, StartIndex: annotation.StartIndex, EndIndex: annotation.EndIndex,
					Citation: &llm.URLCitation{
						Type: annotation.Type, URL: annotation.URLCitation.URL, Title: annotation.URLCitation.Title,
						SourceResidual: cloneRaw(annotation.URLCitation.Residual),
					},
					ResidualOwnerType: annotation.Type, SourceResidual: cloneRaw(annotation.Residual),
				})
			}
		case "input_image":
			if item.ImageURL != nil {
				blocks = append(blocks, llm.ContentBlock{
					Kind: llm.ContentKindImage, ID: item.ID,
					Image:             &llm.ImageURL{URL: *item.ImageURL, Detail: item.Detail},
					ResidualOwnerType: item.Type, SourceResidual: residual,
				})
			}
		case "input_file":
			if document := responseDocument(item); document != nil {
				blocks = append(blocks, llm.ContentBlock{
					Kind: llm.ContentKindDocument, ID: item.ID, Document: document,
					ResidualOwnerType: item.Type, SourceResidual: residual,
				})
			}
		default:
			encoded, err := json.Marshal(item)
			if err != nil {
				return nil, fmt.Errorf("marshal unknown Responses content item %q: %w", item.Type, err)
			}
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindUnknown, ID: item.ID, UnknownRaw: encoded})
		}
	}
	return blocks, nil
}

func responseDocument(item *Item) *llm.DocumentURL {
	if item == nil {
		return nil
	}
	document := &llm.DocumentURL{Filename: item.Filename}
	switch {
	case item.FileData != "":
		document.SourceType = llm.DocumentSourceBase64
		document.Data = item.FileData
		if parsed := xurl.ParseDataURL(item.FileData); parsed != nil {
			document.Data = parsed.Data
			document.MIMEType = parsed.MediaType
		}
	case item.FileID != "":
		document.SourceType = llm.DocumentSourceFile
		document.FileID = item.FileID
	case item.FileURL != "":
		document.SourceType = llm.DocumentSourceURL
		document.URL = item.FileURL
	default:
		return nil
	}
	return document
}

func responseToolsToCanonical(tools []Tool) ([]llm.ToolDefinition, error) {
	definitions := make([]llm.ToolDefinition, 0, len(tools))
	mcpLabels := make(map[string]struct{})
	for index := range tools {
		tool := &tools[index]
		definitionStart := len(definitions)
		switch tool.Type {
		case "function":
			parameters, err := json.Marshal(tool.Parameters)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses function %q parameters: %w", tool.Name, err)
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindFunction, LogicalName: tool.Name, Description: tool.Description,
				Function:  &llm.FunctionDefinition{Parameters: parameters, Strict: tool.Strict},
				Execution: llm.ExecutionOwnerClient, DeferLoading: cloneResponsesBool(tool.DeferLoading),
			})
		case "custom":
			definition := llm.ToolDefinition{
				Kind: llm.ToolKindCustom, LogicalName: tool.Name, Description: tool.Description,
				Freeform: &llm.FreeformDefinition{}, Execution: llm.ExecutionOwnerClient,
				DeferLoading: cloneResponsesBool(tool.DeferLoading),
			}
			if tool.Format != nil {
				definition.Freeform.Format = tool.Format.Type
				definition.Freeform.Syntax = tool.Format.Syntax
				definition.Freeform.Definition = tool.Format.Definition
			}
			definitions = append(definitions, definition)
		case "web_search":
			webSearch := &llm.WebSearch{}
			if tool.Filters != nil {
				webSearch.AllowedDomains = append([]string(nil), tool.Filters.AllowedDomains...)
			}
			if tool.UserLocation != nil {
				webSearch.UserLocation = llm.WebSearchToolUserLocation{
					Type: tool.UserLocation.Type, City: tool.UserLocation.City, Country: tool.UserLocation.Country,
					Region: tool.UserLocation.Region, Timezone: tool.UserLocation.Timezone,
				}
			}
			raw, err := json.Marshal(tool)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses web search configuration: %w", err)
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindWebSearch, LogicalName: "web_search",
				Hosted: &llm.HostedToolDefinition{Type: tool.Type, WebSearch: webSearch, Configuration: raw}, Execution: llm.ExecutionOwnerProvider,
			})
		case "image_generation":
			raw, err := json.Marshal(tool)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses image generation configuration: %w", err)
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindImageGeneration, LogicalName: "image_generation",
				Hosted: &llm.HostedToolDefinition{Type: tool.Type, Configuration: raw}, Execution: llm.ExecutionOwnerProvider,
			})
		case "file_search", "code_interpreter", "shell", "computer", "computer_use_preview", "apply_patch":
			raw, err := json.Marshal(tool)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses %s configuration: %w", tool.Type, err)
			}
			kind, name, owner := responseDefinitionKind(tool.Type)
			definitions = append(definitions, llm.ToolDefinition{
				Kind: kind, LogicalName: name, Description: tool.Description,
				Hosted: &llm.HostedToolDefinition{Type: tool.Type, Configuration: raw}, Execution: owner,
			})
		case "local_shell":
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindLocalShell, LogicalName: "local_shell", Description: tool.Description,
				Function:  &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), localShellParametersJSON...)},
				Execution: llm.ExecutionOwnerClient,
			})
		case "tool_search":
			if tool.Execution != string(llm.ExecutionOwnerClient) {
				raw, err := json.Marshal(tool)
				if err != nil {
					return nil, fmt.Errorf("marshal Responses provider tool_search: %w", err)
				}
				name := tool.Name
				if name == "" {
					name = "tool_search"
				}
				definitions = append(definitions, llm.ToolDefinition{
					Kind: llm.ToolKindUnknownBehavioral, LogicalName: name, Description: tool.Description,
					Hosted:    &llm.HostedToolDefinition{Type: tool.Type, Configuration: raw},
					Execution: llm.ExecutionOwnerProvider,
				})
				break
			}
			parameters, err := json.Marshal(tool.Parameters)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses tool_search parameters: %w", err)
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindToolSearch, LogicalName: "tool_search", Description: tool.Description,
				Function: &llm.FunctionDefinition{Parameters: parameters}, Execution: llm.ExecutionOwnerClient,
			})
		case "mcp":
			if _, duplicate := mcpLabels[tool.ServerLabel]; duplicate {
				return nil, fmt.Errorf("duplicate Responses MCP server_label %q", tool.ServerLabel)
			}
			mcpLabels[tool.ServerLabel] = struct{}{}
			allowedTools, err := canonicalMCPToolFilter(tool.AllowedTools)
			if err != nil {
				return nil, fmt.Errorf("decode Responses MCP %q allowed_tools: %w", tool.ServerLabel, err)
			}
			approval, err := canonicalMCPApprovalPolicy(tool.RequireApproval)
			if err != nil {
				return nil, fmt.Errorf("decode Responses MCP %q require_approval: %w", tool.ServerLabel, err)
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindMCP, LogicalName: tool.ServerLabel, Description: tool.ServerDescription,
				MCP: &llm.MCPDefinition{
					ServerLabel: tool.ServerLabel, ServerDescription: tool.ServerDescription, ServerURL: tool.ServerURL,
					ConnectorID: tool.ConnectorID, TunnelID: tool.TunnelID, DeferLoading: tool.DeferLoading,
					AllowedCallers: append([]string(nil), tool.AllowedCallers...), AllowedTools: allowedTools,
					RequireApproval: approval,
				},
				Execution: llm.ExecutionOwnerProvider,
			})
		case "namespace":
			for subIndex := range tool.Tools {
				subTool := &tool.Tools[subIndex]
				switch subTool.Type {
				case "function", "":
					parameters, err := json.Marshal(subTool.Parameters)
					if err != nil {
						return nil, fmt.Errorf("marshal Responses namespace function %q parameters: %w", subTool.Name, err)
					}
					definitions = append(definitions, llm.ToolDefinition{
						Kind: llm.ToolKindFunction, LogicalName: namespaceFunctionName(tool.Name, subTool.Name), Description: subTool.Description,
						Function:  &llm.FunctionDefinition{Parameters: parameters, Strict: subTool.Strict, Namespace: tool.Name},
						Execution: llm.ExecutionOwnerClient, DeferLoading: cloneResponsesBool(subTool.DeferLoading),
					})
					definitions[len(definitions)-1].ProtocolHints = llm.ProtocolHints{
						SourceType: "function", ResidualOwnerType: "function", SourceResidual: cloneRaw(subTool.Residual),
					}
				case "custom":
					definition := llm.ToolDefinition{
						Kind: llm.ToolKindCustom, LogicalName: namespaceFunctionName(tool.Name, subTool.Name), Description: subTool.Description,
						Freeform: &llm.FreeformDefinition{Namespace: tool.Name}, Execution: llm.ExecutionOwnerClient,
						DeferLoading: cloneResponsesBool(subTool.DeferLoading),
					}
					if subTool.Format != nil {
						definition.Freeform.Format = subTool.Format.Type
						definition.Freeform.Syntax = subTool.Format.Syntax
						definition.Freeform.Definition = subTool.Format.Definition
					}
					definitions = append(definitions, definition)
					definitions[len(definitions)-1].ProtocolHints = llm.ProtocolHints{
						SourceType: "custom", ResidualOwnerType: "custom", SourceResidual: cloneRaw(subTool.Residual),
					}
				default:
					raw, err := json.Marshal(subTool)
					if err != nil {
						return nil, fmt.Errorf("marshal Responses namespace tool %q: %w", subTool.Name, err)
					}
					definitions = append(definitions, llm.ToolDefinition{
						Kind: llm.ToolKindUnknownBehavioral, LogicalName: namespaceFunctionName(tool.Name, subTool.Name),
						Description: subTool.Description,
						Hosted:      &llm.HostedToolDefinition{Type: subTool.Type, Configuration: raw, Namespace: tool.Name},
						Execution:   llm.ExecutionOwnerProvider,
					})
					definitions[len(definitions)-1].ProtocolHints = llm.ProtocolHints{
						SourceType: subTool.Type, ResidualOwnerType: subTool.Type, SourceResidual: cloneRaw(subTool.Residual),
					}
				}
			}
		default:
			raw, err := json.Marshal(tool)
			if err != nil {
				return nil, fmt.Errorf("marshal Responses tool %q: %w", tool.Type, err)
			}
			name := tool.Name
			if name == "" {
				name = tool.Type
			}
			definitions = append(definitions, llm.ToolDefinition{
				Kind: llm.ToolKindUnknownBehavioral, LogicalName: name, Description: tool.Description,
				Hosted:    &llm.HostedToolDefinition{Type: tool.Type, Configuration: raw},
				Execution: llm.ExecutionOwnerProvider,
			})
		}
		for definitionIndex := definitionStart; definitionIndex < len(definitions); definitionIndex++ {
			hints := definitions[definitionIndex].ProtocolHints
			hints.SourceFormat = llm.APIFormatOpenAIResponse
			hints.Ordinal = index
			if tool.Type != "namespace" {
				hints.SourceType = tool.Type
				hints.ResidualOwnerType = tool.Type
				hints.SourceResidual = cloneRaw(tool.Residual)
			} else if hints.ResidualOwnerType == "" {
				hints.ResidualOwnerType = hints.SourceType
			}
			definitions[definitionIndex].ProtocolHints = hints
		}
	}
	return definitions, nil
}

func cloneResponsesBool(value *bool) *bool {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func responseDefinitionKind(toolType string) (llm.ToolKind, string, llm.ExecutionOwner) {
	switch toolType {
	case "file_search":
		return llm.ToolKindFileSearch, "file_search", llm.ExecutionOwnerProvider
	case "code_interpreter":
		return llm.ToolKindCodeInterpreter, "code_interpreter", llm.ExecutionOwnerProvider
	case "shell":
		return llm.ToolKindShell, "shell", llm.ExecutionOwnerProvider
	case "computer", "computer_use_preview":
		return llm.ToolKindComputer, "computer", llm.ExecutionOwnerClient
	case "apply_patch":
		return llm.ToolKindApplyPatch, "apply_patch", llm.ExecutionOwnerClient
	default:
		return llm.ToolKindUnknownBehavioral, toolType, llm.ExecutionOwnerProvider
	}
}

func canonicalMCPToolFilter(raw json.RawMessage) (*llm.MCPToolFilter, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var names []string
	if err := json.Unmarshal(raw, &names); err == nil {
		return &llm.MCPToolFilter{ToolNames: append([]string(nil), names...)}, nil
	}
	var filter struct {
		ToolNames []string `json:"tool_names"`
		ReadOnly  *bool    `json:"read_only"`
	}
	if err := json.Unmarshal(raw, &filter); err != nil {
		return nil, err
	}
	return &llm.MCPToolFilter{ToolNames: append([]string(nil), filter.ToolNames...), ReadOnly: filter.ReadOnly}, nil
}

func canonicalMCPApprovalPolicy(raw json.RawMessage) (*llm.MCPApprovalPolicy, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var mode string
	if err := json.Unmarshal(raw, &mode); err == nil {
		return &llm.MCPApprovalPolicy{Mode: llm.MCPApprovalMode(mode)}, nil
	}
	var filters struct {
		Always json.RawMessage `json:"always"`
		Never  json.RawMessage `json:"never"`
	}
	if err := json.Unmarshal(raw, &filters); err != nil {
		return nil, err
	}
	always, err := canonicalMCPToolFilter(filters.Always)
	if err != nil {
		return nil, fmt.Errorf("always: %w", err)
	}
	never, err := canonicalMCPToolFilter(filters.Never)
	if err != nil {
		return nil, fmt.Errorf("never: %w", err)
	}
	return &llm.MCPApprovalPolicy{Always: always, Never: never}, nil
}

func responseRole(role string) llm.Role {
	switch role {
	case "system":
		return llm.RoleSystem
	case "developer":
		return llm.RoleDeveloper
	case "assistant":
		return llm.RoleAssistant
	default:
		return llm.RoleUser
	}
}

func canonicalItemStatus(status *string) llm.ItemStatus {
	if status == nil {
		return ""
	}
	switch *status {
	case "in_progress":
		return llm.ItemStatusInProgress
	case "completed":
		return llm.ItemStatusCompleted
	case "incomplete":
		return llm.ItemStatusIncomplete
	case "failed":
		return llm.ItemStatusFailed
	default:
		return ""
	}
}

func canonicalToolCallStatus(status *string) llm.ToolCallStatus {
	switch canonicalItemStatus(status) {
	case llm.ItemStatusInProgress:
		return llm.ToolCallStatusInProgress
	case llm.ItemStatusCompleted:
		return llm.ToolCallStatusCompleted
	case llm.ItemStatusIncomplete:
		return llm.ToolCallStatusIncomplete
	case llm.ItemStatusFailed:
		return llm.ToolCallStatusFailed
	default:
		return ""
	}
}

func canonicalToolResultStatus(status *string) llm.ToolResultStatus {
	if canonicalItemStatus(status) == llm.ItemStatusFailed {
		return llm.ToolResultStatusFailed
	}
	if status != nil {
		return llm.ToolResultStatusCompleted
	}
	return ""
}

func stringValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func rawInputAt(items []json.RawMessage, index int) json.RawMessage {
	if index < 0 || index >= len(items) {
		return nil
	}
	return items[index]
}

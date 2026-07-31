package responses

import (
	"encoding/json"
	"fmt"
	"maps"
	"sort"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
)

func canonicalRequestTools(request *llm.Request) ([]Tool, bool, error) {
	if request == nil || len(request.ToolDefinitions) == 0 {
		return nil, false, nil
	}

	tools := make([]Tool, 0, len(request.ToolDefinitions))
	namespaceIndexes := make(map[string]int)
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse &&
			definition.ProtocolHints.SourceType == "additional_tools" {
			continue
		}
		if err := definition.Validate(); err != nil {
			return nil, true, fmt.Errorf("canonical tool %d: %w", index, err)
		}
		switch definition.Kind {
		case llm.ToolKindFunction:
			parameters := make(map[string]any)
			if len(definition.Function.Parameters) > 0 {
				if err := json.Unmarshal(definition.Function.Parameters, &parameters); err != nil {
					return nil, true, fmt.Errorf("canonical function %q parameters: %w", definition.LogicalName, err)
				}
			}
			function := Tool{
				Type: "function", Name: definition.LogicalName, Description: definition.Description,
				Parameters: parameters, Strict: definition.Function.Strict,
				DeferLoading: definition.DeferLoading,
			}
			if namespace := definition.Function.Namespace; namespace != "" {
				function.Name = strings.TrimPrefix(definition.LogicalName, namespace+"__")
				if namespaceIndex, ok := namespaceIndexes[namespace]; ok {
					tools[namespaceIndex].Tools = append(tools[namespaceIndex].Tools, function)
				} else {
					namespaceIndexes[namespace] = len(tools)
					tools = append(tools, Tool{Type: "namespace", Name: namespace, Tools: []Tool{function}})
				}
				continue
			}
			tools = append(tools, function)
		case llm.ToolKindCustom:
			format := (*CustomToolFormat)(nil)
			if freeform := definition.Freeform; freeform != nil && (freeform.Format != "" || freeform.Syntax != "" || freeform.Definition != "") {
				format = &CustomToolFormat{Type: freeform.Format, Syntax: freeform.Syntax, Definition: freeform.Definition}
			}
			custom := Tool{Type: "custom", Name: definition.LogicalName, Description: definition.Description, Format: format, DeferLoading: definition.DeferLoading}
			if namespace := definition.Freeform.Namespace; namespace != "" {
				custom.Name = strings.TrimPrefix(definition.LogicalName, namespace+"__")
				if namespaceIndex, ok := namespaceIndexes[namespace]; ok {
					tools[namespaceIndex].Tools = append(tools[namespaceIndex].Tools, custom)
				} else {
					namespaceIndexes[namespace] = len(tools)
					tools = append(tools, Tool{Type: "namespace", Name: namespace, Tools: []Tool{custom}})
				}
				continue
			}
			tools = append(tools, custom)
		case llm.ToolKindWebSearch:
			tool := Tool{Type: "web_search"}
			if definition.Hosted != nil && len(definition.Hosted.Configuration) > 0 {
				if err := json.Unmarshal(definition.Hosted.Configuration, &tool); err != nil {
					return nil, true, fmt.Errorf("canonical web search configuration: %w", err)
				}
				tool.Type = "web_search"
			}
			if definition.Hosted != nil && definition.Hosted.WebSearch != nil {
				web := definition.Hosted.WebSearch
				if len(web.AllowedDomains) > 0 {
					tool.Filters = &WebSearchFilters{AllowedDomains: append([]string(nil), web.AllowedDomains...)}
				}
				location := web.UserLocation
				if location.Type != "" || location.City != "" || location.Country != "" || location.Region != "" || location.Timezone != "" {
					tool.UserLocation = &WebSearchUserLocation{
						Type: location.Type, City: location.City, Country: location.Country,
						Region: location.Region, Timezone: location.Timezone,
					}
				}
			}
			tools = append(tools, tool)
		case llm.ToolKindImageGeneration:
			tool := Tool{Type: "image_generation"}
			if definition.Hosted != nil && len(definition.Hosted.Configuration) > 0 {
				if err := json.Unmarshal(definition.Hosted.Configuration, &tool); err != nil {
					return nil, true, fmt.Errorf("canonical image generation configuration: %w", err)
				}
				tool.Type = "image_generation"
			}
			tools = append(tools, tool)
		case llm.ToolKindFileSearch:
			tool, err := canonicalResponsesNativeTool(definition, "file_search")
			if err != nil {
				return nil, true, fmt.Errorf("canonical file search configuration: %w", err)
			}
			tools = append(tools, tool)
		case llm.ToolKindCodeInterpreter:
			tool, err := canonicalResponsesNativeTool(definition, "code_interpreter")
			if err != nil {
				return nil, true, fmt.Errorf("canonical code interpreter configuration: %w", err)
			}
			tools = append(tools, tool)
		case llm.ToolKindShell:
			tool, err := canonicalResponsesNativeTool(definition, "shell")
			if err != nil {
				return nil, true, fmt.Errorf("canonical shell configuration: %w", err)
			}
			tools = append(tools, tool)
		case llm.ToolKindComputer:
			tool, err := canonicalResponsesNativeTool(definition, "computer")
			if err != nil {
				return nil, true, fmt.Errorf("canonical computer configuration: %w", err)
			}
			tools = append(tools, tool)
		case llm.ToolKindApplyPatch:
			tool, err := canonicalResponsesNativeTool(definition, "apply_patch")
			if err != nil {
				return nil, true, fmt.Errorf("canonical apply patch configuration: %w", err)
			}
			tools = append(tools, tool)
		case llm.ToolKindLocalShell:
			tools = append(tools, Tool{Type: "local_shell"})
		case llm.ToolKindToolSearch:
			parameters := make(map[string]any)
			if definition.Function != nil && len(definition.Function.Parameters) > 0 {
				if err := json.Unmarshal(definition.Function.Parameters, &parameters); err != nil {
					return nil, true, fmt.Errorf("canonical tool_search parameters: %w", err)
				}
			}
			tools = append(tools, Tool{
				Type: "tool_search", Description: definition.Description,
				Execution: string(definition.Execution), Parameters: parameters,
			})
		case llm.ToolKindMCP:
			tool, err := canonicalMCPToolToResponses(request, definition)
			if err != nil {
				return nil, true, fmt.Errorf("encode canonical MCP tool %d: %w", index, err)
			}
			tools = append(tools, tool)
		default:
			return nil, true, fmt.Errorf("canonical tool %d kind %q has no Responses encoding", index, definition.Kind)
		}
	}
	return tools, true, nil
}

func canonicalResponsesNativeTool(definition *llm.ToolDefinition, fallbackType string) (Tool, error) {
	tool := Tool{Type: fallbackType}
	if definition == nil || definition.Hosted == nil || len(definition.Hosted.Configuration) == 0 {
		return tool, nil
	}
	if err := json.Unmarshal(definition.Hosted.Configuration, &tool); err != nil {
		return Tool{}, err
	}
	// Raw is only safe when it already is a Responses definition of the
	// requested family. Cross-protocol definitions use the portable fields.
	validType := tool.Type == fallbackType || (fallbackType == "computer" && tool.Type == "computer_use_preview")
	if validType {
		return tool, nil
	}
	tool.Raw = nil
	tool.Type = fallbackType
	return tool, nil
}

func canonicalMCPToolToResponses(request *llm.Request, definition *llm.ToolDefinition) (Tool, error) {
	if definition == nil || definition.MCP == nil {
		return Tool{}, fmt.Errorf("MCP definition is missing")
	}
	if err := definition.Validate(); err != nil {
		return Tool{}, err
	}
	mcp := definition.MCP
	allowedTools, err := responsesMCPToolFilter(mcp.AllowedTools)
	if err != nil {
		return Tool{}, err
	}
	requireApproval, err := responsesMCPApprovalPolicy(mcp.RequireApproval)
	if err != nil {
		return Tool{}, err
	}
	tool := Tool{
		Type: "mcp", ServerLabel: mcp.ServerLabel, ServerDescription: mcp.ServerDescription,
		ServerURL: mcp.ServerURL, ConnectorID: mcp.ConnectorID, TunnelID: mcp.TunnelID,
		DeferLoading: mcp.DeferLoading, AllowedCallers: append([]string(nil), mcp.AllowedCallers...),
		AllowedTools: allowedTools, RequireApproval: requireApproval,
	}
	if request.ToolExecutionSecrets != nil {
		if secret, ok := request.ToolExecutionSecrets.MCP[mcp.ServerLabel]; ok {
			tool.Authorization = secret.Authorization
			tool.Headers = maps.Clone(secret.Headers)
		}
	}
	return tool, nil
}

func responsesMCPToolFilter(filter *llm.MCPToolFilter) (json.RawMessage, error) {
	if filter == nil {
		return nil, nil
	}
	if filter.ReadOnly != nil {
		// read_only filters are semantic constraints over tools/list metadata.
		// Current Responses accepts only a tool-name sequence at this position;
		// serializing the historical object form makes compatible providers
		// reject the entire request with a 4xx. The relay must select the MCP
		// gateway, discover the annotated tools, and lower the definition first.
		return nil, fmt.Errorf("Responses MCP read_only requires MCP gateway projection")
	}
	return json.Marshal(filter.ToolNames)
}

func responsesMCPApprovalPolicy(policy *llm.MCPApprovalPolicy) (json.RawMessage, error) {
	if policy == nil {
		return nil, nil
	}
	if policy.Mode != "" {
		return json.Marshal(policy.Mode)
	}
	always, err := responsesMCPToolFilterObject(policy.Always)
	if err != nil {
		return nil, fmt.Errorf("encode always approval filter: %w", err)
	}
	never, err := responsesMCPToolFilterObject(policy.Never)
	if err != nil {
		return nil, fmt.Errorf("encode never approval filter: %w", err)
	}
	object := make(map[string]json.RawMessage, 2)
	if len(always) > 0 {
		object["always"] = always
	}
	if len(never) > 0 {
		object["never"] = never
	}
	return json.Marshal(object)
}

func responsesMCPToolFilterObject(filter *llm.MCPToolFilter) (json.RawMessage, error) {
	if filter == nil {
		return nil, nil
	}
	return json.Marshal(struct {
		ToolNames []string `json:"tool_names,omitempty"`
		ReadOnly  *bool    `json:"read_only,omitempty"`
	}{ToolNames: filter.ToolNames, ReadOnly: filter.ReadOnly})
}

// canonicalRequestInput converts ordered canonical items directly to Responses
// wire items. Responses needs this path on both identity and cross-protocol
// routes because item_id and call_id are distinct identities; unknown native
// items are still replayed by the raw sidecar merge in marshalRequestPayload.
func canonicalRequestInput(request *llm.Request) (Input, string, bool, error) {
	if request == nil || len(request.Input) == 0 {
		return Input{}, "", false, nil
	}
	type placedItem struct {
		ordinal  int
		sequence int
		wire     Item
	}
	placed := make([]placedItem, 0, len(request.Input))
	var instructions []string
	for index := range request.Input {
		item := &request.Input[index]
		if item.Kind == llm.ItemKindMessage {
			// Only reconstruct the top-level instructions field when the item
			// actually originated there. System/developer message items are valid
			// Responses input and must retain their distinct roles.
			if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && item.ProtocolHints.SourceType == "instructions" {
				if text := canonicalText(item.Content); text != "" {
					instructions = append(instructions, text)
				}
				continue
			}
		}
		// Unknown Responses items are replayed by the raw sidecar. Hosted items
		// are structurally represented and must pass through the typed encoder;
		// on an identity route it overlays canonical state onto ProviderData so
		// Responses-private fields survive without duplicating the item.
		if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse &&
			item.Kind == llm.ItemKindUnknown {
			continue
		}
		wire, ok := canonicalItemToResponses(item)
		if !ok {
			return Input{}, "", true, fmt.Errorf("canonical input item %d kind %q has no Responses encoding", index, item.Kind)
		}
		ordinal := len(request.Input) + index
		if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && item.ProtocolHints.Ordinal >= 0 {
			ordinal = item.ProtocolHints.Ordinal
		}
		placed = append(placed, placedItem{ordinal: ordinal, sequence: len(placed), wire: wire})
		if item.Kind == llm.ItemKindHostedCall && item.HostedCall != nil && item.HostedCall.Result != nil {
			if output, outputOK := canonicalHostedResultToResponses(item.HostedCall.Invocation.Kind, item.HostedCall.Result); outputOK {
				placed = append(placed, placedItem{ordinal: ordinal, sequence: len(placed), wire: output})
			}
		}
	}

	type declarationGroup struct {
		role        llm.Role
		definitions []llm.ToolDefinition
	}
	groups := make(map[int]*declarationGroup)
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		hints := definition.ProtocolHints
		if hints.SourceFormat != llm.APIFormatOpenAIResponse || hints.SourceType != "additional_tools" {
			continue
		}
		group := groups[hints.Ordinal]
		if group == nil {
			group = &declarationGroup{role: hints.SourceRole}
			groups[hints.Ordinal] = group
		}
		definition.ProtocolHints = llm.ProtocolHints{}
		group.definitions = append(group.definitions, definition)
	}
	for ordinal, group := range groups {
		tools, _, err := canonicalRequestTools(&llm.Request{ToolDefinitions: group.definitions})
		if err != nil {
			return Input{}, "", true, fmt.Errorf("encode Responses additional_tools item %d: %w", ordinal, err)
		}
		placed = append(placed, placedItem{
			ordinal: ordinal, sequence: len(placed),
			wire: Item{Type: "additional_tools", Role: string(group.role), AdditionalTools: tools},
		})
	}
	sort.SliceStable(placed, func(i, j int) bool {
		if placed[i].ordinal == placed[j].ordinal {
			return placed[i].sequence < placed[j].sequence
		}
		return placed[i].ordinal < placed[j].ordinal
	})
	items := make([]Item, 0, len(placed))
	for index := range placed {
		items = append(items, placed[index].wire)
	}
	return Input{Items: items}, strings.Join(instructions, "\n"), true, nil
}

func canonicalHostedResultToResponses(kind llm.ToolKind, result *llm.ToolResult) (Item, bool) {
	if result == nil {
		return Item{}, false
	}
	var wire Item
	if len(result.ProviderData) > 0 && json.Valid(result.ProviderData) {
		_ = json.Unmarshal(result.ProviderData, &wire)
	}
	wire.CallID = result.CallID
	status := "completed"
	if result.Status == llm.ToolResultStatusFailed || result.IsError {
		status = "failed"
	}
	wire.Status = &status
	switch kind {
	case llm.ToolKindShell:
		wire.Type = "shell_call_output"
		if raw := canonicalUnknownResult(result); len(raw) > 0 {
			wire.RawOutput = raw
		}
		if len(wire.RawOutput) == 0 {
			wire.RawOutput = json.RawMessage(`[]`)
		}
	case llm.ToolKindComputer:
		wire.Type = "computer_call_output"
		wire.ComputerOutput = canonicalComputerOutput(result)
		wire.AcknowledgedSafetyChecks = canonicalSafetyChecksToResponses(result.AcknowledgedSafetyChecks)
		if wire.ComputerOutput == nil {
			return Item{}, false
		}
	default:
		return Item{}, false
	}
	return wire, true
}

func canonicalItemToResponses(item *llm.Item) (Item, bool) {
	if item == nil {
		return Item{}, false
	}
	switch item.Kind {
	case llm.ItemKindMessage:
		content := canonicalContentToResponses(item.Content, item.Role == llm.RoleAssistant)
		if len(content) == 0 {
			return Item{}, false
		}
		return Item{ID: item.ID, Type: "message", Role: string(item.Role), Content: &Input{Items: content}, Status: responsesStatus(item.Status)}, true
	case llm.ItemKindReasoning:
		if item.Reasoning == nil {
			return Item{}, false
		}
		summary := make([]ReasoningSummary, 0, 1)
		if item.Reasoning.Content != "" {
			summary = append(summary, ReasoningSummary{Type: "summary_text", Text: item.Reasoning.Content})
		}
		var encrypted *string
		if item.Reasoning.Signature != "" {
			value := item.Reasoning.Signature
			encrypted = &value
		}
		return Item{ID: item.ID, Type: "reasoning", Summary: summary, EncryptedContent: encrypted, Status: responsesStatus(item.Status)}, true
	case llm.ItemKindToolCall:
		if item.ToolCall == nil {
			return Item{}, false
		}
		call := item.ToolCall
		if call.Kind == llm.ToolKindCustom {
			input := call.InputText
			return Item{ID: item.ID, Type: "custom_tool_call", CallID: call.CallID, Name: call.LogicalName, Input: &input, Status: responsesStatus(item.Status)}, true
		}
		if call.Kind == llm.ToolKindLocalShell {
			var wire Item
			if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && len(call.ProviderData) > 0 && json.Valid(call.ProviderData) {
				_ = json.Unmarshal(call.ProviderData, &wire)
			}
			var action LocalShellAction
			if len(call.ArgumentsJSON) == 0 || json.Unmarshal(call.ArgumentsJSON, &action) != nil {
				return Item{}, false
			}
			wire.ID, wire.Type, wire.CallID = item.ID, "local_shell_call", call.CallID
			wire.Name, wire.Action, wire.Status = "", NewLocalShellAction(&action), responsesStatus(item.Status)
			return wire, true
		}
		if call.Kind == llm.ToolKindComputer {
			var wire Item
			if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && len(call.ProviderData) > 0 && json.Valid(call.ProviderData) {
				_ = json.Unmarshal(call.ProviderData, &wire)
			}
			wire.ID, wire.Type, wire.CallID = item.ID, "computer_call", call.CallID
			wire.Name, wire.Status = "", responsesStatus(item.Status)
			wire.ComputerAction = append(json.RawMessage(nil), call.ArgumentsJSON...)
			wire.PendingSafetyChecks = canonicalSafetyChecksToResponses(call.PendingSafetyChecks)
			return wire, true
		}
		if call.Kind == llm.ToolKindApplyPatch {
			var wire Item
			if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && len(call.ProviderData) > 0 && json.Valid(call.ProviderData) {
				_ = json.Unmarshal(call.ProviderData, &wire)
			}
			wire.ID, wire.Type, wire.CallID = item.ID, "apply_patch_call", call.CallID
			wire.Name, wire.Status = "", responsesStatus(item.Status)
			wire.Operation = append(json.RawMessage(nil), call.ArgumentsJSON...)
			return wire, true
		}
		if call.Kind == llm.ToolKindToolSearch {
			return Item{
				ID: item.ID, Type: "tool_search_call", CallID: call.CallID,
				Arguments: canonicalInvocationArguments(call), Execution: string(call.Execution),
				Status: responsesStatus(item.Status),
			}, true
		}
		return Item{ID: item.ID, Type: "function_call", CallID: call.CallID, Name: call.LogicalName, Namespace: call.Namespace, Arguments: canonicalInvocationArguments(call), Status: responsesStatus(item.Status)}, true
	case llm.ItemKindToolResult:
		if item.ToolResult == nil {
			return Item{}, false
		}
		result := item.ToolResult
		itemType := "function_call_output"
		if result.Kind == llm.ToolKindCustom {
			itemType = "custom_tool_call_output"
		}
		if result.Kind == llm.ToolKindToolSearch {
			tools, _, err := canonicalRequestTools(&llm.Request{ToolDefinitions: result.DiscoveredTools})
			if err != nil {
				return Item{}, false
			}
			return Item{
				ID: item.ID, Type: "tool_search_output", CallID: result.CallID,
				Execution: string(result.Execution), AdditionalTools: tools, Status: responsesStatus(item.Status),
			}, true
		}
		if result.Kind == llm.ToolKindComputer {
			var wire Item
			if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && len(result.ProviderData) > 0 && json.Valid(result.ProviderData) {
				_ = json.Unmarshal(result.ProviderData, &wire)
			}
			wire.ID, wire.Type, wire.CallID = item.ID, "computer_call_output", result.CallID
			wire.Name, wire.Status = "", responsesStatus(item.Status)
			wire.ComputerOutput = canonicalComputerOutput(result)
			wire.AcknowledgedSafetyChecks = canonicalSafetyChecksToResponses(result.AcknowledgedSafetyChecks)
			return wire, wire.ComputerOutput != nil
		}
		if result.Kind == llm.ToolKindApplyPatch {
			output := canonicalResultOutput(result.Content)
			return Item{ID: item.ID, Type: "apply_patch_call_output", CallID: result.CallID, Output: &output, Status: responsesStatus(item.Status)}, true
		}
		if result.Kind == llm.ToolKindLocalShell {
			output := canonicalResultOutput(result.Content)
			return Item{ID: item.ID, Type: "local_shell_call_output", CallID: result.CallID, Output: &output, Status: responsesStatus(item.Status)}, true
		}
		output := canonicalResultOutput(result.Content)
		return Item{ID: item.ID, Type: itemType, CallID: result.CallID, Name: result.LogicalName, Output: &output, Status: responsesStatus(item.Status)}, true
	case llm.ItemKindMCPListTools:
		if item.MCPListTools == nil {
			return Item{}, false
		}
		list := item.MCPListTools
		tools := make([]MCPListedTool, 0, len(list.Tools))
		for index := range list.Tools {
			tool := &list.Tools[index]
			tools = append(tools, MCPListedTool{
				Name: tool.Name, Description: tool.Description,
				InputSchema: append(json.RawMessage(nil), tool.InputSchema...),
				Annotations: append(json.RawMessage(nil), tool.Annotations...),
			})
		}
		var itemError *string
		if list.Error != "" {
			itemError = &list.Error
		}
		return Item{
			ID: item.ID, Type: "mcp_list_tools", ServerLabel: list.ServerLabel,
			Tools: tools, Error: itemError, Status: responsesStatus(item.Status),
		}, true
	case llm.ItemKindMCPApprovalRequest:
		if item.MCPApprovalRequest == nil {
			return Item{}, false
		}
		request := item.MCPApprovalRequest
		return Item{
			ID: item.ID, Type: "mcp_approval_request", ServerLabel: request.ServerLabel,
			Name: request.LogicalName, Arguments: canonicalMCPItemArguments(request.ArgumentsJSON, request.ArgumentsText),
			Status: responsesStatus(item.Status),
		}, true
	case llm.ItemKindMCPApprovalResponse:
		if item.MCPApprovalResponse == nil {
			return Item{}, false
		}
		response := item.MCPApprovalResponse
		return Item{
			ID: item.ID, Type: "mcp_approval_response", ApprovalRequestID: response.ApprovalRequestID,
			Approve: &response.Approve, Reason: response.Reason, Status: responsesStatus(item.Status),
		}, true
	case llm.ItemKindMCPCall:
		if item.MCPCall == nil {
			return Item{}, false
		}
		call := item.MCPCall
		status := responsesStatus(item.Status)
		if call.Status != "" {
			value := string(call.Status)
			status = &value
		}
		var output *Input
		if call.Output != "" {
			value := call.Output
			output = &Input{Text: &value}
		}
		var itemError *string
		if call.Error != "" {
			itemError = &call.Error
		}
		return Item{
			ID: item.ID, Type: "mcp_call", ServerLabel: call.ServerLabel, Name: call.LogicalName,
			ApprovalRequestID: call.ApprovalRequestID,
			Arguments:         canonicalMCPItemArguments(call.ArgumentsJSON, call.ArgumentsText),
			Output:            output, Error: itemError, Status: status,
		}, true
	case llm.ItemKindHostedCall:
		if item.HostedCall == nil {
			return Item{}, false
		}
		invocation := &item.HostedCall.Invocation
		var wire Item
		if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && len(invocation.ProviderData) > 0 && json.Valid(invocation.ProviderData) {
			_ = json.Unmarshal(invocation.ProviderData, &wire)
		}
		wire.ID = item.ID
		if wire.ID == "" {
			wire.ID = invocation.ID
		}
		wire.CallID = invocation.CallID
		wire.Name = invocation.LogicalName
		wire.Status = responsesStatus(item.Status)
		switch invocation.Kind {
		case llm.ToolKindImageGeneration:
			wire.Type = "image_generation_call"
			var arguments struct {
				Action string `json:"action"`
			}
			_ = json.Unmarshal(invocation.ArgumentsJSON, &arguments)
			if arguments.Action == "" {
				arguments.Action = "generate"
			}
			wire.Action = &ItemAction{ImageGenerationAction: arguments.Action}
			if data, format := canonicalImageGenerationResult(item.HostedCall.Result); data != "" {
				wire.Result = &data
				wire.OutputFormat = &format
			}
		case llm.ToolKindFileSearch:
			wire.Type = "file_search_call"
			var arguments struct {
				Queries []string `json:"queries"`
			}
			_ = json.Unmarshal(invocation.ArgumentsJSON, &arguments)
			wire.Queries = append([]string(nil), arguments.Queries...)
			wire.Results = canonicalUnknownResult(item.HostedCall.Result)
		case llm.ToolKindCodeInterpreter:
			wire.Type = "code_interpreter_call"
			var arguments struct {
				Code        *string `json:"code"`
				ContainerID *string `json:"container_id"`
			}
			_ = json.Unmarshal(invocation.ArgumentsJSON, &arguments)
			wire.Code, wire.ContainerID = arguments.Code, arguments.ContainerID
			wire.Outputs = canonicalUnknownResult(item.HostedCall.Result)
		case llm.ToolKindShell:
			wire.Type = "shell_call"
			wire.ComputerAction = append(json.RawMessage(nil), invocation.ArgumentsJSON...)
		case llm.ToolKindComputer:
			wire.Type = "computer_call"
			wire.Name = ""
			wire.ComputerAction = append(json.RawMessage(nil), invocation.ArgumentsJSON...)
			wire.PendingSafetyChecks = canonicalSafetyChecksToResponses(invocation.PendingSafetyChecks)
		case llm.ToolKindWebSearch:
			wire.Type = "web_search_call"
			wire.Action = canonicalResponsesWebSearchAction(invocation, item.HostedCall.Result)
		default:
			return Item{}, false
		}
		return wire, true
	case llm.ItemKindCompaction:
		if item.Compaction == nil {
			return Item{}, false
		}
		encrypted := item.Compaction.EncryptedContent
		var createdBy *string
		if item.Compaction.CreatedBy != "" {
			createdBy = &item.Compaction.CreatedBy
		}
		itemType := "compaction"
		if item.ProtocolHints.SourceFormat == llm.APIFormatOpenAIResponse && item.ProtocolHints.SourceType == "compaction_summary" {
			itemType = "compaction_summary"
		}
		return Item{ID: item.ID, Type: itemType, EncryptedContent: &encrypted, CreatedBy: createdBy, Status: responsesStatus(item.Status)}, true
	case llm.ItemKindUnknown:
		if item.Unknown == nil || item.ProtocolHints.SourceFormat != llm.APIFormatOpenAIResponse {
			return Item{}, false
		}
		var wire Item
		if json.Unmarshal(item.Unknown.Raw, &wire) != nil {
			return Item{}, false
		}
		return wire, true
	default:
		return Item{}, false
	}
}

func canonicalImageGenerationResult(result *llm.ToolResult) (string, string) {
	if result == nil {
		return "", ""
	}
	for index := range result.Content {
		block := &result.Content[index]
		if block.Kind != llm.ContentKindImage || block.Image == nil {
			continue
		}
		parsed := xurl.ParseDataURL(block.Image.URL)
		if parsed == nil || !parsed.IsBase64 || !strings.HasPrefix(parsed.MediaType, "image/") {
			continue
		}
		return parsed.Data, strings.TrimPrefix(parsed.MediaType, "image/")
	}
	return "", ""
}

func canonicalUnknownResult(result *llm.ToolResult) json.RawMessage {
	if result == nil {
		return nil
	}
	if len(result.StructuredContent) > 0 && json.Valid(result.StructuredContent) {
		return append(json.RawMessage(nil), result.StructuredContent...)
	}
	for index := range result.Content {
		if result.Content[index].Kind == llm.ContentKindUnknown && json.Valid(result.Content[index].UnknownRaw) {
			return append(json.RawMessage(nil), result.Content[index].UnknownRaw...)
		}
	}
	return nil
}

func canonicalResponsesWebSearchAction(invocation *llm.ToolInvocation, result *llm.ToolResult) *ItemAction {
	action := &WebSearchAction{Type: "search"}
	if invocation != nil && len(invocation.ArgumentsJSON) > 0 {
		var wrapped ItemAction
		if json.Unmarshal(invocation.ArgumentsJSON, &wrapped) == nil && wrapped.WebSearch != nil {
			*action = *wrapped.WebSearch
			action.Queries = append([]string(nil), wrapped.WebSearch.Queries...)
			action.Sources = append([]WebSearchSource(nil), wrapped.WebSearch.Sources...)
		} else {
			var arguments struct {
				Type    string   `json:"type"`
				Query   string   `json:"query"`
				Queries []string `json:"queries"`
			}
			if json.Unmarshal(invocation.ArgumentsJSON, &arguments) == nil {
				if arguments.Type != "" {
					action.Type = arguments.Type
				}
				action.Query = arguments.Query
				action.Queries = append([]string(nil), arguments.Queries...)
			}
		}
	}
	if action.Type == "" {
		action.Type = "search"
	}
	if result != nil {
		action.Sources = action.Sources[:0]
		for index := range result.Content {
			citation := result.Content[index].Citation
			if result.Content[index].Kind != llm.ContentKindCitation || citation == nil {
				continue
			}
			action.Sources = append(action.Sources, WebSearchSource{Type: "url", URL: citation.URL, Title: citation.Title})
		}
	}
	return NewWebSearchAction(action)
}

func canonicalSafetyChecksToResponses(checks []llm.ToolSafetyCheck) []ComputerSafetyCheck {
	if len(checks) == 0 {
		return nil
	}
	result := make([]ComputerSafetyCheck, 0, len(checks))
	for index := range checks {
		check := &checks[index]
		result = append(result, ComputerSafetyCheck{ID: check.ID, Code: check.Code, Message: check.Message})
	}
	return result
}

func canonicalComputerOutput(result *llm.ToolResult) *ComputerScreenshot {
	if result == nil {
		return nil
	}
	if len(result.StructuredContent) > 0 && json.Valid(result.StructuredContent) {
		var screenshot ComputerScreenshot
		if json.Unmarshal(result.StructuredContent, &screenshot) == nil && screenshot.Type == "computer_screenshot" {
			return &screenshot
		}
	}
	for index := range result.Content {
		block := &result.Content[index]
		if block.Kind == llm.ContentKindImage && block.Image != nil && block.Image.URL != "" {
			return &ComputerScreenshot{Type: "computer_screenshot", ImageURL: block.Image.URL}
		}
		if block.Kind == llm.ContentKindUnknown && json.Valid(block.UnknownRaw) {
			var screenshot ComputerScreenshot
			if json.Unmarshal(block.UnknownRaw, &screenshot) == nil && screenshot.Type == "computer_screenshot" {
				return &screenshot
			}
		}
	}
	return nil
}

func canonicalMCPItemArguments(argumentsJSON json.RawMessage, argumentsText string) string {
	if len(argumentsJSON) > 0 {
		return string(argumentsJSON)
	}
	return argumentsText
}

func canonicalContentToResponses(blocks []llm.ContentBlock, assistant bool) []Item {
	items := make([]Item, 0, len(blocks))
	for index := range blocks {
		block := &blocks[index]
		switch block.Kind {
		case llm.ContentKindText, llm.ContentKindRefusal:
			value := block.Text
			itemType := "input_text"
			if assistant {
				itemType = "output_text"
			}
			items = append(items, Item{ID: block.ID, Type: itemType, Text: &value})
		case llm.ContentKindImage:
			if block.Image != nil {
				url := block.Image.URL
				items = append(items, Item{ID: block.ID, Type: "input_image", ImageURL: &url, Detail: block.Image.Detail})
			}
		case llm.ContentKindDocument:
			if assistant || block.Document == nil {
				continue
			}
			item := Item{ID: block.ID, Type: "input_file", Filename: block.Document.Filename}
			switch block.Document.SourceType {
			case llm.DocumentSourceBase64:
				item.FileData = xurl.BuildDataURL(block.Document.MIMEType, block.Document.Data, true)
			case llm.DocumentSourceURL:
				item.FileURL = block.Document.URL
			case llm.DocumentSourceFile:
				item.FileID = block.Document.FileID
			default:
				continue
			}
			items = append(items, item)
		case llm.ContentKindCitation:
			if len(items) == 0 || block.Citation == nil {
				continue
			}
			items[len(items)-1].Annotations = append(items[len(items)-1].Annotations, Annotation{
				Type: "url_citation", StartIndex: block.StartIndex, EndIndex: block.EndIndex,
				URLCitation: &URLCitation{URL: block.Citation.URL, Title: block.Citation.Title},
			})
		}
	}
	return items
}

// ensureResponsesOutputAnnotations applies the Responses response contract
// without changing assistant messages that appear in request history.
func ensureResponsesOutputAnnotations(item *Item) {
	if item == nil || item.Type != "message" || item.Content == nil {
		return
	}
	for index := range item.Content.Items {
		part := &item.Content.Items[index]
		if part.Type == "output_text" && part.Annotations == nil {
			part.Annotations = []Annotation{}
		}
	}
}

func canonicalResultOutput(blocks []llm.ContentBlock) Input {
	if len(blocks) == 1 && blocks[0].Kind == llm.ContentKindText {
		value := blocks[0].Text
		return Input{Text: &value}
	}
	return Input{Items: canonicalContentToResponses(blocks, false)}
}

func canonicalInvocationArguments(call *llm.ToolInvocation) string {
	if len(call.ArgumentsJSON) > 0 {
		return string(call.ArgumentsJSON)
	}
	return call.ArgumentsText
}

func canonicalText(blocks []llm.ContentBlock) string {
	var builder strings.Builder
	for index := range blocks {
		if blocks[index].Kind != llm.ContentKindText {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(blocks[index].Text)
	}
	return builder.String()
}

func responsesStatus(status llm.ItemStatus) *string {
	if status == "" {
		return nil
	}
	value := string(status)
	return &value
}

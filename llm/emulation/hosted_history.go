package emulation

import (
	"errors"
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/hosted"
)

func lowerHostedHistory(request *llm.Request, registry *hosted.Registry) (*llm.Request, error) {
	if request == nil || registry == nil || registry.Empty() {
		return request, nil
	}
	prepared := request.Clone()
	definitions := prepared.ToolDefinitions[:0]
	for index := range prepared.ToolDefinitions {
		definition := prepared.ToolDefinitions[index]
		if registry.OwnsDefinition(definition) {
			continue
		}
		definitions = append(definitions, definition)
	}
	prepared.ToolDefinitions = append(definitions, registry.FunctionDefinitions()...)
	prepared.Input = prepared.Input[:0]
	for index := range request.Input {
		item := request.Input[index]
		if item.Kind != llm.ItemKindHostedCall || item.HostedCall == nil {
			prepared.Input = append(prepared.Input, llm.CloneCanonicalItem(item))
			continue
		}
		invocation := item.HostedCall.Invocation
		binding, ok := registry.BindingFor(invocation.Kind, invocation.LogicalName)
		if !ok {
			prepared.Input = append(prepared.Input, llm.CloneCanonicalItem(item))
			continue
		}
		if invocation.CallID == "" {
			return nil, errors.New("hosted history invocation has no call_id")
		}
		call := llm.Item{
			Kind: llm.ItemKindToolCall, ID: invocation.ID, Role: llm.RoleAssistant, Status: item.Status,
			ToolCall: &llm.ToolInvocation{
				Kind: llm.ToolKindFunction, ID: invocation.ID, CallID: invocation.CallID,
				LogicalName:   binding.SyntheticName,
				ArgumentsJSON: append([]byte(nil), invocation.ArgumentsJSON...), ArgumentsText: invocation.ArgumentsText,
				Status: invocation.Status, Execution: llm.ExecutionOwnerGateway,
			},
		}
		prepared.Input = append(prepared.Input, call)
		if item.HostedCall.Result != nil {
			result := item.HostedCall.Result
			resultClone := llm.CloneCanonicalItem(llm.Item{Kind: llm.ItemKindToolResult, ToolResult: result})
			prepared.Input = append(prepared.Input, llm.Item{
				Kind: llm.ItemKindToolResult, Status: item.Status,
				ToolResult: &llm.ToolResult{
					Kind: llm.ToolKindFunction, CallID: invocation.CallID, LogicalName: binding.SyntheticName,
					Content:           resultClone.ToolResult.Content,
					StructuredContent: resultClone.ToolResult.StructuredContent,
					IsError:           result.IsError, Status: result.Status,
				},
			})
		}
	}
	if err := llm.ValidateItemStructure(prepared.Input); err != nil {
		return nil, fmt.Errorf("lower hosted history: %w", err)
	}
	return prepared, nil
}

func refreshHostedDefinitions(request *llm.Request, registry *hosted.Registry) {
	if request == nil || registry == nil || registry.Empty() {
		return
	}
	definitions := request.ToolDefinitions[:0]
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if _, owned := registry.Binding(definition.LogicalName); owned || registry.OwnsDefinition(definition) {
			continue
		}
		definitions = append(definitions, definition)
	}
	request.ToolDefinitions = append(definitions, registry.FunctionDefinitions()...)
}

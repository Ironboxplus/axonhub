package emulation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/mcp"
)

func (controller *Controller) lowerHistory(ctx context.Context, request *llm.Request, registry *mcp.Registry) (*llm.Request, []llm.Item, error) {
	prepared := request.Clone()
	prepared.CanonicalEncodingRequired = true
	clonedDefinitions := prepared.ToolDefinitions
	prepared.ToolDefinitions = nil
	for index := range clonedDefinitions {
		if clonedDefinitions[index].Kind != llm.ToolKindMCP {
			prepared.ToolDefinitions = append(prepared.ToolDefinitions, clonedDefinitions[index])
		}
	}
	prepared.ToolDefinitions = append(prepared.ToolDefinitions, registry.FunctionDefinitions()...)

	requests := make(map[string]llm.Item)
	decisions := make(map[string]llm.Item)
	for index := range request.Input {
		item := request.Input[index]
		switch item.Kind {
		case llm.ItemKindMCPApprovalRequest:
			if item.ID == "" || item.MCPApprovalRequest == nil {
				return nil, nil, fmt.Errorf("%w: malformed approval request", ErrApprovalState)
			}
			if _, duplicate := requests[item.ID]; duplicate {
				return nil, nil, fmt.Errorf("%w: duplicate approval request", ErrApprovalState)
			}
			requests[item.ID] = item
		case llm.ItemKindMCPApprovalResponse:
			if item.MCPApprovalResponse == nil || item.MCPApprovalResponse.ApprovalRequestID == "" {
				return nil, nil, fmt.Errorf("%w: malformed approval response", ErrApprovalState)
			}
			approvalID := item.MCPApprovalResponse.ApprovalRequestID
			if _, duplicate := decisions[approvalID]; duplicate {
				return nil, nil, fmt.Errorf("%w: duplicate approval response", ErrApprovalState)
			}
			decisions[approvalID] = item
		}
	}

	prepared.Input = make([]llm.Item, 0, len(request.Input)+len(decisions))
	resumed := make([]llm.Item, 0, len(decisions))
	usedDecisions := make(map[string]struct{}, len(decisions))
	for index := 0; index < len(request.Input); {
		item := request.Input[index]
		if isHistoryCallItem(item) {
			var loweredCalls, loweredResults, resumedItems []llm.Item
			var err error
			index, loweredCalls, loweredResults, resumedItems, err = lowerHistoryCallBatch(
				ctx, request.Input, index, registry, decisions, usedDecisions,
			)
			if err != nil {
				return nil, nil, err
			}
			prepared.Input = append(prepared.Input, loweredCalls...)
			prepared.Input = append(prepared.Input, loweredResults...)
			resumed = append(resumed, resumedItems...)
			continue
		}
		switch item.Kind {
		case llm.ItemKindMCPListTools:
			// Discovery is represented to the target by generated function
			// definitions, not by a protocol-foreign history item.
		case llm.ItemKindMCPApprovalResponse:
			// A response is consumed by the call batch containing its request.
		default:
			prepared.Input = append(prepared.Input, llm.CloneCanonicalItem(item))
		}
		index++
	}
	for approvalID := range decisions {
		if _, found := requests[approvalID]; !found {
			return nil, nil, fmt.Errorf("%w: response references an unknown approval request", ErrApprovalState)
		}
		if _, used := usedDecisions[approvalID]; !used {
			return nil, nil, fmt.Errorf("%w: approval response was not consumed", ErrApprovalState)
		}
	}
	if err := llm.ValidateItemStructure(prepared.Input); err != nil {
		return nil, nil, fmt.Errorf("lower MCP history: %w", err)
	}
	return prepared, resumed, nil
}

func isHistoryCallItem(item llm.Item) bool {
	return item.Kind == llm.ItemKindToolCall || item.Kind == llm.ItemKindMCPCall || item.Kind == llm.ItemKindMCPApprovalRequest
}

// lowerHistoryCallBatch expands protocol-specific combined MCP items without
// serializing calls that were emitted by the same model turn. Target encoders
// merge the returned consecutive calls into one Chat assistant message or one
// Anthropic assistant block list; results follow in the same call order.
func lowerHistoryCallBatch(
	ctx context.Context,
	input []llm.Item,
	start int,
	registry *mcp.Registry,
	decisions map[string]llm.Item,
	usedDecisions map[string]struct{},
) (next int, calls, results, resumed []llm.Item, err error) {
	type callSlot struct {
		callID string
		result *llm.Item
	}

	slots := make([]callSlot, 0)
	clientSlots := make(map[string]int)
	next = start
	for next < len(input) && isHistoryCallItem(input[next]) {
		item := input[next]
		switch item.Kind {
		case llm.ItemKindToolCall:
			if item.ToolCall == nil {
				return start, nil, nil, nil, errors.New("tool call payload is missing")
			}
			call := llm.CloneCanonicalItem(item)
			clientSlots[call.ToolCall.CallID] = len(slots)
			slots = append(slots, callSlot{callID: call.ToolCall.CallID})
			calls = append(calls, call)
		case llm.ItemKindMCPCall:
			call, result, lowerErr := lowerCompletedMCPCall(item, registry)
			if lowerErr != nil {
				return start, nil, nil, nil, lowerErr
			}
			resultClone := result
			slots = append(slots, callSlot{callID: call.ToolCall.CallID, result: &resultClone})
			calls = append(calls, call)
		case llm.ItemKindMCPApprovalRequest:
			decisionItem, answered := decisions[item.ID]
			if !answered {
				return start, nil, nil, nil, fmt.Errorf("%w: approval request has no response", ErrApprovalState)
			}
			usedDecisions[item.ID] = struct{}{}
			call, callErr := approvalGatewayCall(item, registry)
			if callErr != nil {
				return start, nil, nil, nil, callErr
			}
			decision := decisionItem.MCPApprovalResponse
			var execution callExecution
			if decision.Approve {
				execution = executeCall(ctx, registry, call)
			} else {
				execution = rejectedExecution(call, registry, item.ID)
			}
			execution.public.MCPCall.ApprovalRequestID = item.ID
			resultClone := execution.result
			slots = append(slots, callSlot{callID: call.item.ToolCall.CallID, result: &resultClone})
			calls = append(calls, call.item)
			resumed = append(resumed, execution.public)
		}
		next++
	}

	// Responses places client results and approval decisions after the output
	// call run. Consume only items correlated to this batch; unrelated items
	// remain for the outer history traversal.
	for next < len(input) {
		item := input[next]
		switch item.Kind {
		case llm.ItemKindToolResult:
			if item.ToolResult == nil {
				return start, nil, nil, nil, errors.New("tool result payload is missing")
			}
			position, belongs := clientSlots[item.ToolResult.CallID]
			if !belongs {
				goto assembled
			}
			resultClone := llm.CloneCanonicalItem(item)
			slots[position].result = &resultClone
			delete(clientSlots, item.ToolResult.CallID)
			next++
		case llm.ItemKindMCPApprovalResponse:
			if item.MCPApprovalResponse == nil {
				return start, nil, nil, nil, fmt.Errorf("%w: malformed approval response", ErrApprovalState)
			}
			if _, consumed := usedDecisions[item.MCPApprovalResponse.ApprovalRequestID]; !consumed {
				goto assembled
			}
			next++
		default:
			goto assembled
		}
	}

assembled:
	for index := range slots {
		if slots[index].result != nil {
			results = append(results, *slots[index].result)
		}
	}
	return next, calls, results, resumed, nil
}

func lowerCompletedMCPCall(item llm.Item, registry *mcp.Registry) (llm.Item, llm.Item, error) {
	if item.MCPCall == nil {
		return llm.Item{}, llm.Item{}, errors.New("MCP call payload is missing")
	}
	binding, ok := registry.BindingFor(item.MCPCall.ServerLabel, item.MCPCall.LogicalName)
	if !ok {
		return llm.Item{}, llm.Item{}, fmt.Errorf("MCP call references an unavailable discovered tool")
	}
	callID := registry.SyntheticCallID(item.ID)
	call := syntheticToolCall(callID, binding, item.MCPCall.ArgumentsJSON, item.MCPCall.ArgumentsText)
	failed := item.Status == llm.ItemStatusFailed || item.MCPCall.Status == llm.MCPCallStatusFailed || item.MCPCall.Error != ""
	output := item.MCPCall.Output
	if output == "" {
		output = item.MCPCall.Error
	}
	return call, toolResultItem(gatewayCall{item: call, binding: binding}, output, failed), nil
}

func approvalGatewayCall(item llm.Item, registry *mcp.Registry) (gatewayCall, error) {
	request := item.MCPApprovalRequest
	if request == nil {
		return gatewayCall{}, fmt.Errorf("%w: approval request payload is missing", ErrApprovalState)
	}
	binding, ok := registry.BindingFor(request.ServerLabel, request.LogicalName)
	if !ok {
		return gatewayCall{}, fmt.Errorf("%w: approval tool is not available", ErrApprovalState)
	}
	callID := registry.SyntheticCallID(item.ID)
	call := syntheticToolCall(callID, binding, request.ArgumentsJSON, request.ArgumentsText)
	return gatewayCall{item: call, binding: binding}, nil
}

func syntheticToolCall(callID string, binding mcp.Binding, argumentsJSON json.RawMessage, argumentsText string) llm.Item {
	return llm.Item{
		Kind: llm.ItemKindToolCall, ID: callID, Role: llm.RoleAssistant, Status: llm.ItemStatusCompleted,
		ToolCall: &llm.ToolInvocation{
			Kind: llm.ToolKindFunction, ID: callID, CallID: callID,
			LogicalName:   binding.SyntheticName,
			ArgumentsJSON: append(json.RawMessage(nil), argumentsJSON...), ArgumentsText: argumentsText,
			Status: llm.ToolCallStatusCompleted, Execution: llm.ExecutionOwnerGateway,
		},
	}
}

func rejectedExecution(call gatewayCall, registry *mcp.Registry, approvalID string) callExecution {
	const message = "MCP tool call rejected by client"
	execution := callExecution{
		call:   call,
		result: toolResultItem(call, message, true),
		public: llm.Item{
			Kind: llm.ItemKindMCPCall, ID: registry.MCPCallItemID(call.item.ToolCall.CallID), Status: llm.ItemStatusFailed,
			MCPCall: &llm.MCPCall{
				ServerLabel: call.binding.ServerLabel, LogicalName: call.binding.Tool.Name,
				ApprovalRequestID: approvalID,
				Error:             message, Status: llm.MCPCallStatusFailed,
			},
		},
	}
	copyInvocationArguments(execution.public.MCPCall, call.item.ToolCall)
	return execution
}

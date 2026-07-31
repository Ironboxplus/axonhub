package conversion

import (
	"context"
	"time"

	"github.com/looplj/axonhub/llm"
)

func RestoreResponse(response *llm.Response, session *Session) *llm.Response {
	return RestoreResponseContext(context.Background(), response, session)
}

// RestoreResponseContext is the production restore entry point. ctx carries the
// trusted SessionScope used when publishing raw provider argument bytes into
// the host ContinuationBinding for multi-turn Responses replay.
func RestoreResponseContext(ctx context.Context, response *llm.Response, session *Session) *llm.Response {
	if response == nil || session == nil {
		return response
	}
	if ctx == nil {
		ctx = context.Background()
	}
	startedAt := time.Time{}
	if session.traceEnabled {
		startedAt = time.Now()
	}
	for choiceIndex := range response.Choices {
		choice := &response.Choices[choiceIndex]
		restoreMessage(choice.Message, session, llm.ConversionDirectionResponse, choice.Index)
		restoreMessage(choice.Delta, session, llm.ConversionDirectionResponse, choice.Index)
	}
	restoreCanonicalOutput(response.Output, session, llm.ConversionDirectionResponse)
	normalizeResponseIdentifiers(ctx, response, session, llm.ConversionDirectionResponse)
	if session.traceEnabled {
		session.addRestoreNanos(time.Since(startedAt).Nanoseconds())
		setResponseSummary(response, session.Summary())
	}
	setResponseDebug(response, session.DebugTrace())
	return response
}

func normalizeResponseIdentifiers(ctx context.Context, response *llm.Response, session *Session, direction llm.ConversionDirection) {
	if response == nil || session == nil {
		return
	}
	for itemIndex := range response.Output {
		item := &response.Output[itemIndex]
		ref := canonicalToolRef(itemIndex)
		if item.ToolCall != nil {
			providerCallID := item.ToolCall.CallID
			providerName := item.ToolCall.LogicalName
			item.ToolCall.CallID = session.normalizeSourceCallID(providerCallID, direction, ref)
			// Publish under the provider-facing name used when recording raw
			// bytes, before identity restore rewrites LogicalName for clients.
			session.publishProviderArgumentBytes(ctx, providerCallID, item.ToolCall.CallID, providerName)
			if item.ToolCall.LogicalName != providerName && item.ToolCall.LogicalName != "" {
				session.publishProviderArgumentBytes(ctx, providerCallID, item.ToolCall.CallID, item.ToolCall.LogicalName)
			}
		}
		if item.ToolResult != nil {
			item.ToolResult.CallID = session.normalizeSourceCallID(item.ToolResult.CallID, direction, ObjectRef{
				Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1,
			})
		}
		if item.HostedCall != nil {
			item.HostedCall.Invocation.CallID = session.normalizeSourceCallID(item.HostedCall.Invocation.CallID, direction, ref)
			if item.HostedCall.Result != nil {
				item.HostedCall.Result.CallID = session.normalizeSourceCallID(item.HostedCall.Result.CallID, direction, ref)
			}
		}
	}
	for choiceIndex := range response.Choices {
		choice := &response.Choices[choiceIndex]
		for _, message := range []*llm.Message{choice.Message, choice.Delta} {
			if message == nil {
				continue
			}
			for callIndex := range message.ToolCalls {
				call := &message.ToolCalls[callIndex]
				ref := messageToolRef(choice.Index, call.Index)
				providerCallID := call.ID
				providerName := call.Function.Name
				call.ID = session.normalizeSourceCallID(providerCallID, direction, ref)
				session.publishProviderArgumentBytes(ctx, providerCallID, call.ID, providerName)
				if call.Function.Name != providerName && call.Function.Name != "" {
					session.publishProviderArgumentBytes(ctx, providerCallID, call.ID, call.Function.Name)
				}
				if call.ResponseCustomToolCall != nil {
					call.ResponseCustomToolCall.CallID = session.normalizeSourceCallID(call.ResponseCustomToolCall.CallID, direction, ref)
				}
			}
			if message.ToolCallID != nil {
				ref := ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: choice.Index, ToolCallIndex: -1}
				*message.ToolCallID = session.normalizeSourceCallID(*message.ToolCallID, direction, ref)
			}
			for resultIndex := range message.InlineToolResults {
				ref := ObjectRef{Kind: ObjectToolResult, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: choice.Index, ToolCallIndex: resultIndex}
				result := &message.InlineToolResults[resultIndex]
				result.ToolCallID = session.normalizeSourceCallID(result.ToolCallID, direction, ref)
			}
		}
	}
}

func restoreCanonicalOutput(output []llm.Item, session *Session, direction llm.ConversionDirection) {
	for index := range output {
		item := &output[index]
		if item.ToolCall == nil || item.ToolCall.Kind != llm.ToolKindFunction {
			continue
		}
		restoreInvocationSchemaArguments(item.ToolCall, session, direction, canonicalToolRef(index))
		identity, ok := session.identity(item.ToolCall.LogicalName)
		if !ok {
			continue
		}
		if identity.SourceKind == llm.ToolKindFunction {
			item.ToolCall.LogicalName = identity.SourceName
			item.ToolCall.Namespace = identity.SourceNamespace
			recordIdentityRestore(session, direction, canonicalToolRef(index), identity, true)
			continue
		}
		if identity.SourceKind != llm.ToolKindCustom {
			item.ProtocolHints.SourceFormat = llm.APIFormatOpenAIResponse
			item.ProtocolHints.SourceType = sourceItemType(identity.SourceKind)
			item.ToolCall.Kind = identity.SourceKind
			item.ToolCall.LogicalName = identity.SourceName
			recordIdentityRestore(session, direction, canonicalToolRef(index), identity, true)
			continue
		}
		arguments := item.ToolCall.ArgumentsText
		if len(item.ToolCall.ArgumentsJSON) > 0 {
			arguments = string(item.ToolCall.ArgumentsJSON)
		}
		input, valid := restoreCustomInput(arguments, item.ToolCall.CallID, session, direction, canonicalToolRef(index))
		if !valid {
			input = arguments
		}
		item.ProtocolHints.SourceFormat = llm.APIFormatOpenAIResponse
		item.ProtocolHints.SourceType = "custom_tool_call"
		item.ToolCall.Kind = llm.ToolKindCustom
		item.ToolCall.LogicalName = identity.SourceName
		item.ToolCall.ArgumentsJSON = nil
		item.ToolCall.ArgumentsText = ""
		item.ToolCall.InputText = input
		recordIdentityRestore(session, direction, canonicalToolRef(index), identity, valid)
	}
}

func restoreMessage(message *llm.Message, session *Session, direction llm.ConversionDirection, choiceIndex int) {
	if message == nil {
		return
	}
	for callIndex := range message.ToolCalls {
		call := &message.ToolCalls[callIndex]
		restoreLegacySchemaArguments(call, session, direction, messageToolRef(choiceIndex, callIndex))
		identity, ok := session.identity(call.Function.Name)
		if !ok {
			continue
		}
		if identity.SourceKind == llm.ToolKindFunction {
			call.Function.Name = identity.SourceName
			call.Function.Namespace = identity.SourceNamespace
			recordIdentityRestore(session, direction, messageToolRef(choiceIndex, callIndex), identity, true)
			continue
		}
		if identity.SourceKind != llm.ToolKindCustom {
			call.Function.Name = identity.SourceName
			recordIdentityRestore(session, direction, messageToolRef(choiceIndex, callIndex), identity, true)
			continue
		}
		input, valid := restoreCustomInput(call.Function.Arguments, call.ID, session, direction, messageToolRef(choiceIndex, callIndex))
		if !valid {
			input = call.Function.Arguments
		}
		call.Type = llm.ToolTypeResponsesCustomTool
		call.ResponseCustomToolCall = &llm.ResponseCustomToolCall{
			CallID: call.ID,
			Name:   identity.SourceName,
			Input:  input,
		}
		call.Function = llm.FunctionCall{}
		recordIdentityRestore(session, direction, messageToolRef(choiceIndex, callIndex), identity, valid)
	}
}

func canonicalToolRef(itemIndex int) ObjectRef {
	return ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: itemIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1}
}

func messageToolRef(messageIndex, toolCallIndex int) ObjectRef {
	return ObjectRef{Kind: ObjectToolCall, ToolIndex: -1, ItemIndex: -1, ContentIndex: -1, MessageIndex: messageIndex, ToolCallIndex: toolCallIndex}
}

func recordIdentityRestore(session *Session, direction llm.ConversionDirection, ref ObjectRef, identity toolIdentity, reversible bool) {
	strategy := StrategyClientToolAsFunc
	if identity.SourceKind == llm.ToolKindFunction {
		strategy = StrategyNative
	} else if identity.SourceKind == llm.ToolKindCustom {
		strategy = StrategyCustomAsFunction
	}
	action, reason := "restore", ReasonSemanticProjection
	if !reversible {
		action, reason = "restore_miss", ReasonNoStrategy
	}
	session.recordDebug(direction, ref, action, strategy, reason, reversible)
}

func sourceItemType(kind llm.ToolKind) string {
	switch kind {
	case llm.ToolKindLocalShell:
		return "local_shell_call"
	default:
		return string(kind) + "_call"
	}
}

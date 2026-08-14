package conversion

import (
	"context"
	"errors"

	"github.com/looplj/axonhub/llm"
)

type inlineCompactionProjectionKind uint8

const (
	inlineCompactionProjectionBlock inlineCompactionProjectionKind = iota
	inlineCompactionProjectionSummarize
	inlineCompactionProjectionDrop
	inlineCompactionProjectionStrip
)

// inlineCompactionProjectionDecision is the shared pure classification used
// by both planning and execution. Its closed canonical union prevents a new
// provider item/content arm from becoming summary input by accident.
type inlineCompactionProjectionDecision struct {
	kind     inlineCompactionProjectionKind
	reason   ReasonCode
	semantic string
}

func classifyInlineCompactionItemForTarget(item llm.Item, target llm.APIFormat) inlineCompactionProjectionDecision {
	reject := func(reason ReasonCode, semantic string) inlineCompactionProjectionDecision {
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionBlock, reason: reason, semantic: semantic}
	}
	summarize := func(semantic string) inlineCompactionProjectionDecision {
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionSummarize, reason: ReasonGatewayCompaction, semantic: semantic}
	}
	switch item.Kind {
	case llm.ItemKindMessage:
		if item.Role != llm.RoleSystem && item.Role != llm.RoleDeveloper && item.Role != llm.RoleUser && item.Role != llm.RoleAssistant {
			return reject(ReasonProtocolConstraint, "inline_history_unsupported_message_role")
		}
		if hasInlinePrivateItemData(item) {
			return reject(ReasonProviderPrivate, "inline_history_private_message")
		}
		for index := range item.Content {
			content := item.Content[index]
			switch content.Kind {
			case llm.ContentKindText, llm.ContentKindRefusal:
			case llm.ContentKindImage:
				if content.Image == nil || content.Image.URL == "" || len(content.Image.URL) > maxInlineCompactionRetainedItemBytes || len(content.Image.URL) > maxInlineCompactionSummaryVisibleBytes {
					return reject(ReasonProtocolConstraint, "inline_history_oversize_image")
				}
			case llm.ContentKindDocument:
				if _, err := inlineCompactionDocumentProjection(content.Document, target); err != nil {
					var inlineErr *InlineCompactionError
					if errors.As(err, &inlineErr) && inlineErr.Code == InlineCompactionOversize {
						return reject(ReasonProtocolConstraint, "inline_history_oversize_document")
					}
					return reject(ReasonNoStrategy, "inline_history_unsupported_document")
				}
			default:
				return reject(ReasonNoStrategy, "inline_history_unsupported_content")
			}
		}
		return summarize("inline_history_message")
	case llm.ItemKindToolCall:
		if item.ToolCall == nil {
			return reject(ReasonNoStrategy, "inline_history_invalid_tool_call")
		}
		return summarize("inline_history_tool_call")
	case llm.ItemKindToolResult:
		if item.ToolResult == nil {
			return reject(ReasonNoStrategy, "inline_history_invalid_tool_result")
		}
		if len(item.ToolResult.Content) == 0 && (len(item.ToolResult.ProviderData) > 0 || len(item.ToolResult.StructuredContent) > 0) {
			return reject(ReasonProviderPrivate, "inline_history_private_only_tool_result")
		}
		provenance, err := inlineCompactionToolResultContentProvenance(item.ToolResult.Content)
		if err != nil {
			return reject(ReasonProviderPrivate, "inline_history_private_tool_result_content")
		}
		if _, err = inlineCompactionToolResultMessage(item.ToolResult, target, nil); err != nil {
			var inlineErr *InlineCompactionError
			if errors.As(err, &inlineErr) && inlineErr.Code == InlineCompactionOversize {
				return reject(ReasonProtocolConstraint, "inline_history_oversize_tool_result")
			}
			return reject(ReasonNoStrategy, "inline_history_unsupported_tool_result")
		}
		if provenance > 0 {
			return summarize("inline_history_tool_result_provenance_stripped")
		}
		return summarize("inline_history_tool_result")
	case llm.ItemKindMCPListTools:
		if item.MCPListTools == nil {
			return reject(ReasonNoStrategy, "inline_history_invalid_mcp_discovery")
		}
		return summarize("inline_history_mcp_discovery")
	case llm.ItemKindMCPApprovalRequest, llm.ItemKindMCPApprovalResponse:
		return summarize("inline_history_mcp_approval")
	case llm.ItemKindMCPCall:
		if item.MCPCall == nil {
			return reject(ReasonNoStrategy, "inline_history_invalid_mcp_call")
		}
		return summarize("inline_history_mcp_call")
	case llm.ItemKindAgentMessage:
		if item.AgentMessage == nil {
			return reject(ReasonNoStrategy, "inline_history_invalid_agent_message")
		}
		if _, err := item.AgentMessage.LegacyInterAgentMessageJSON(); err != nil {
			return reject(ReasonProviderPrivate, "inline_history_private_agent_message")
		}
		return summarize("inline_history_agent_message")
	case llm.ItemKindReasoning:
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionDrop, reason: ReasonProviderPrivate, semantic: "inline_history_reasoning_dropped"}
	case llm.ItemKindHostedCall:
		if item.HostedCall == nil || item.HostedCall.Invocation.Kind != llm.ToolKindWebSearch && item.HostedCall.Invocation.Kind != llm.ToolKindWebFetch {
			return reject(ReasonNoStrategy, "inline_history_unsupported_hosted_call")
		}
		if item.HostedCall.Result != nil {
			if _, err := inlineHostedCompactionText(item.HostedCall.Result.Content, nil); err != nil {
				return reject(ReasonNoStrategy, "inline_history_unsupported_hosted_result")
			}
		}
		return summarize("inline_history_hosted_web")
	case llm.ItemKindToolDeclaration:
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionStrip, reason: ReasonGatewayCompaction, semantic: "inline_history_tool_declaration_stripped"}
	case llm.ItemKindCompaction:
		if item.Compaction == nil || item.Compaction.EncryptedContent == "" {
			return reject(ReasonCheckpointInvalid, "inline_checkpoint_missing_opaque_token")
		}
		if len(item.Compaction.EncryptedContent) > maxInlineCompactionTokenBytes {
			return reject(ReasonProtocolConstraint, "inline_checkpoint_token_oversize")
		}
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionSummarize, reason: ReasonGatewayCheckpoint, semantic: "compaction_checkpoint"}
	case llm.ItemKindCompactionTrigger:
		return inlineCompactionProjectionDecision{kind: inlineCompactionProjectionStrip, reason: ReasonGatewayCompaction, semantic: "compaction_trigger_request_control"}
	case llm.ItemKindContextCompaction:
		return reject(ReasonProviderPrivate, "inline_history_context_compaction")
	case llm.ItemKindUnknown:
		return reject(ReasonNoStrategy, "inline_history_unknown")
	default:
		return reject(ReasonNoStrategy, "inline_history_unsupported")
	}
}

func inlineCompactionCheckpointStructuralError(request *llm.Request) *InlineCompactionError {
	if request == nil {
		return nil
	}
	count := 0
	for index := range request.Input {
		item := request.Input[index]
		if item.Kind != llm.ItemKindCompaction {
			continue
		}
		count++
		if item.Compaction == nil || item.Compaction.EncryptedContent == "" {
			return &InlineCompactionError{Code: InlineCompactionInvalidToken, Err: errors.New("checkpoint has no opaque content")}
		}
		if len(item.Compaction.EncryptedContent) > maxInlineCompactionTokenBytes {
			return &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("checkpoint token exceeds the gateway limit")}
		}
	}
	if count > 1 {
		return &InlineCompactionError{Code: InlineCompactionInvalidToken, Err: errors.New("multiple gateway checkpoints are not allowed")}
	}
	return nil
}

const inlineCompactionGenerationMetadataKey = "axon.inline_compaction_generation.v1"

func planRequiresGatewayInlineCompactionCodec(plan *Plan) bool {
	if plan == nil {
		return false
	}
	for index := range plan.Actions {
		action := &plan.Actions[index]
		if ((action.Ref.Kind == ObjectRequestControl && action.Strategy == StrategyInlineCompactionGateway) || (action.Ref.Kind == ObjectCompaction && action.Strategy == StrategyInlineCompactionHydrate)) && action.Kind == ActionEmulate {
			return true
		}
	}
	return false
}

func planExecutesGatewayInlineCompactionSummary(plan *Plan) bool {
	if plan == nil {
		return false
	}
	for index := range plan.Actions {
		action := &plan.Actions[index]
		if action.Ref.Kind == ObjectRequestControl && action.Strategy == StrategyInlineCompactionGateway && action.Kind == ActionEmulate {
			return true
		}
	}
	return false
}

func inlineCompactionTrigger(request *llm.Request) (int, *llm.Item, bool) {
	if request == nil {
		return -1, nil, false
	}
	for index := range request.Input {
		if request.Input[index].Kind == llm.ItemKindCompactionTrigger {
			return index, &request.Input[index], true
		}
	}
	return -1, nil, false
}

func hydrateGatewayInlineCompaction(ctx context.Context, request *llm.Request, plan *Plan, codec CompactionStateCodec) (*llm.Request, bool, error) {
	if request == nil || plan == nil {
		return request, false, nil
	}
	checkpoint := -1
	for index := range request.Input {
		if request.Input[index].Kind != llm.ItemKindCompaction {
			continue
		}
		if checkpoint >= 0 {
			return nil, false, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("multiple compaction checkpoints require an explicit provider owner")}
		}
		checkpoint = index
	}
	if checkpoint < 0 {
		return request, false, nil
	}
	if codec == nil {
		if inlineCompactionAllowsResponsesOpaqueIdentity(request, plan) && !inlineCompactionHasTrigger(request) {
			return request, false, nil
		}
		return nil, false, &InlineCompactionError{Code: InlineCompactionCodecMissing}
	}
	payload := request.Input[checkpoint].Compaction
	if payload == nil || payload.EncryptedContent == "" {
		return nil, false, &InlineCompactionError{Code: InlineCompactionInvalidToken, Err: errors.New("checkpoint has no opaque content")}
	}
	if len(payload.EncryptedContent) > maxInlineCompactionTokenBytes {
		return nil, false, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("checkpoint token exceeds the gateway limit")}
	}
	state, owned, err := codec.Open(ctx, payload.EncryptedContent)
	if err != nil {
		return nil, false, &InlineCompactionError{Code: inlineCompactionCodecErrorCode(err), Err: err}
	}
	if !owned {
		// A codec determines ownership but never grants a foreign token a
		// conversion route. The exact source/target Responses path is the sole
		// safe opaque identity route; every cross-protocol route stops here,
		// before projectCanonical or provider dispatch.
		if inlineCompactionAllowsResponsesOpaqueIdentity(request, plan) && !inlineCompactionHasTrigger(request) {
			return request, false, nil
		}
		return nil, false, &InlineCompactionError{Code: InlineCompactionForeignToken}
	}
	safeState, err := sanitizeInlineCompactionState(state)
	if err != nil {
		return nil, false, err
	}
	hydrated := request.Clone()
	input := make([]llm.Item, 0, len(safeState.Retained)+1+len(request.Input)-checkpoint-1)
	input = append(input, llm.CloneCanonicalItems(safeState.Retained)...)
	input = append(input, llm.CloneCanonicalItem(safeState.Continuation))
	for index := checkpoint + 1; index < len(request.Input); index++ {
		input = append(input, llm.CloneCanonicalItem(request.Input[index]))
	}
	hydrated.Input = input
	metadata := make(map[string]any, len(hydrated.TransformerMetadata)+1)
	for key, value := range hydrated.TransformerMetadata {
		metadata[key] = value
	}
	metadata[inlineCompactionGenerationMetadataKey] = safeState.Generation
	hydrated.TransformerMetadata = metadata
	return hydrated, true, nil
}

func inlineCompactionAllowsResponsesOpaqueIdentity(request *llm.Request, plan *Plan) bool {
	return request != nil && plan != nil && request.APIFormat == llm.APIFormatOpenAIResponse && plan.Target.APIFormat == llm.APIFormatOpenAIResponse
}

func inlineCompactionHasTrigger(request *llm.Request) bool {
	_, _, found := inlineCompactionTrigger(request)
	return found
}

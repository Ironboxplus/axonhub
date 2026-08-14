package conversion

import (
	"context"
	"errors"

	"github.com/looplj/axonhub/llm"
)

type inlineCompactionSummaryCanonicalizer func([]llm.Message) ([]llm.Item, error)
type inlineCompactionStateByteCounter func(CompactionState) (uint32, error)

// checkedInlineCompactionSummaryItems is the sole checked boundary between the
// safe legacy-message selector and canonical summary input. Keeping the
// canonicalizer injectable makes the error contract executable without ever
// substituting a provider transport in tests.
func checkedInlineCompactionSummaryItems(messages []llm.Message, canonicalize inlineCompactionSummaryCanonicalizer) ([]llm.Item, error) {
	items, err := canonicalize(messages)
	if err != nil {
		return nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: err}
	}
	return items, nil
}

// checkedInlineCompactionStateBytes is the only state-size boundary used by
// execution/restore. The byte counter returns the typed limit/encoding error;
// this helper deliberately preserves it rather than treating a prior state
// sanitizer pass as permission to ignore serialization.
func checkedInlineCompactionStateBytes(state CompactionState, count inlineCompactionStateByteCounter) (uint32, error) {
	bytes, err := count(state)
	if err != nil {
		return 0, err
	}
	return bytes, nil
}

// projectInlineCompactionExecution creates the one explicit, tool-free,
// non-streaming summary request. It is an allowlist: input extensions,
// provider sidecars, tools, metadata, headers, and client controls never leak
// into this internal exchange.
func projectInlineCompactionExecution(request *llm.Request, plan *Plan, codec CompactionStateCodec) (*llm.Request, *inlineCompactionState, error) {
	return projectInlineCompactionExecutionWithDependencies(request, plan, codec, inlineCompactionSummaryCanonicalItems, inlineCompactionStateBytes)
}

func projectInlineCompactionExecutionWithDependencies(request *llm.Request, plan *Plan, codec CompactionStateCodec, canonicalize inlineCompactionSummaryCanonicalizer, count inlineCompactionStateByteCounter) (*llm.Request, *inlineCompactionState, error) {
	if !planExecutesGatewayInlineCompactionSummary(plan) {
		return request, nil, nil
	}
	generation, err := inlineCompactionGeneration(request)
	if err != nil {
		return nil, nil, err
	}
	triggerIndex, trigger, found := inlineCompactionTrigger(request)
	if !found {
		return nil, nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("gateway inline compaction plan has no final trigger")}
	}
	if codec == nil {
		return nil, nil, &InlineCompactionError{Code: InlineCompactionCodecMissing}
	}
	history := make([]llm.Item, 0, len(request.Input)-1)
	for index := range request.Input {
		if index != triggerIndex {
			history = append(history, llm.CloneCanonicalItem(request.Input[index]))
		}
	}
	if len(history) == 0 {
		return nil, nil, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("inline compaction has no history")}
	}
	retained, retention, err := retainedInlineCompactionItemsWithBudget(history)
	if err != nil {
		return nil, nil, err
	}
	if len(retained) == 0 {
		return nil, nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("history has no retainable user, developer, or system context")}
	}
	pendingStateBytes, err := checkedInlineCompactionStateBytes(CompactionState{Version: inlineCompactionStateVersion, Generation: generation, Retained: retained, Continuation: pendingInlineContinuation()}, count)
	if err != nil {
		return nil, nil, err
	}
	messages, drops, summary, err := inlineCompactionSummaryMessages(history, plan.Target.APIFormat)
	if err != nil {
		return nil, nil, err
	}
	summaryItems, err := checkedInlineCompactionSummaryItems(messages, canonicalize)
	if err != nil {
		return nil, nil, err
	}
	drops.RetainedTruncated = retention.Truncated
	drops.RetainedTruncatedBytes = retention.TruncatedBytes
	drops.SummaryTruncated = summary.Truncated
	drops.SummaryTruncatedBytes = summary.TruncatedBytes
	stream := false
	maxTokens := inlineCompactionMaxOutputTokens
	execution := &llm.Request{Model: request.Model, RequestType: llm.RequestTypeChat, APIFormat: request.APIFormat, Stream: &stream, MaxCompletionTokens: &maxTokens, CanonicalEncodingRequired: true, Input: append([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionInstruction}}}}, summaryItems...)}
	return execution, &inlineCompactionState{codec: codec, retained: retained, generation: generation, drops: drops, stateBytes: pendingStateBytes, sourceBytes: trigger.ProtocolHints.SourceBytes, sourceDigest: safeEvidenceSourceDigest(trigger.ProtocolHints.SourceDigest)}, nil
}

func inlineCompactionGeneration(request *llm.Request) (uint32, error) {
	if request != nil && request.TransformerMetadata != nil {
		if generation, ok := request.TransformerMetadata[inlineCompactionGenerationMetadataKey].(uint32); ok && generation > 0 {
			if generation == ^uint32(0) {
				return 0, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("gateway checkpoint generation cannot advance")}
			}
			return generation + 1, nil
		}
		if generation, ok := request.TransformerMetadata[inlineCompactionGenerationMetadataKey].(int); ok && generation > 0 {
			if uint64(generation) >= uint64(^uint32(0)) {
				return 0, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("gateway checkpoint generation cannot advance")}
			}
			return uint32(generation + 1), nil
		}
	}
	return 1, nil
}

func armInlineCompaction(session *Session, state *inlineCompactionState) {
	if session == nil || state == nil {
		return
	}
	session.inlineCompaction = state
	session.recordInlineCompactionEvidence(llm.ConversionDirectionRequest, "summary", "emulate", StrategyInlineCompactionGateway, ReasonGatewayCompaction, state.sourceBytes, state.sourceDigest, state.generation, state.stateBytes, uint32(len(state.retained)), state.drops, true)
}

func restoreInlineCompaction(ctx context.Context, response *llm.Response, session *Session) (*llm.Response, error) {
	return restoreInlineCompactionWithStateBytes(ctx, response, session, inlineCompactionStateBytes)
}

func restoreInlineCompactionWithStateBytes(ctx context.Context, response *llm.Response, session *Session, count inlineCompactionStateByteCounter) (*llm.Response, error) {
	if response == nil || session == nil || session.inlineCompaction == nil {
		return response, nil
	}
	continuation, err := inlineCompactionContinuation(response)
	if err != nil {
		recordInlineCompactionFailure(session.plan, "summary_output", err)
		return nil, err
	}
	state := CompactionState{Version: inlineCompactionStateVersion, Generation: session.inlineCompaction.generation, Retained: llm.CloneCanonicalItems(session.inlineCompaction.retained), Continuation: continuation}
	safeState, err := sanitizeInlineCompactionState(state)
	if err != nil {
		recordInlineCompactionFailure(session.plan, "seal", err)
		return nil, err
	}
	stateBytes, err := checkedInlineCompactionStateBytes(safeState, count)
	if err != nil {
		recordInlineCompactionFailure(session.plan, "seal", err)
		return nil, err
	}
	token, err := session.inlineCompaction.codec.Seal(ctx, safeState)
	if err != nil {
		inlineErr := &InlineCompactionError{Code: inlineCompactionCodecErrorCode(err), Err: err}
		recordInlineCompactionFailure(session.plan, "seal", inlineErr)
		return nil, inlineErr
	}
	if token == "" {
		inlineErr := &InlineCompactionError{Code: InlineCompactionInvalidToken, Err: errors.New("codec returned an empty checkpoint token")}
		recordInlineCompactionFailure(session.plan, "seal", inlineErr)
		return nil, inlineErr
	}
	if len(token) > maxInlineCompactionTokenBytes {
		inlineErr := &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("codec returned an oversized checkpoint token")}
		recordInlineCompactionFailure(session.plan, "seal", inlineErr)
		return nil, inlineErr
	}
	var usage *llm.Usage
	if response.Usage != nil {
		copyUsage := *response.Usage
		usage = &copyUsage
	}
	responseID, itemID := inlineCompactionPublicIDs(token)
	response = &llm.Response{ID: responseID, Object: "response", Model: response.Model, Created: response.Created, RequestType: llm.RequestTypeChat, APIFormat: llm.APIFormatOpenAIResponse, Status: llm.ResponseStatusCompleted, Usage: usage, Output: []llm.Item{{Kind: llm.ItemKindCompaction, ID: itemID, Compaction: &llm.CompactionItem{EncryptedContent: token}}}}
	inline := session.inlineCompaction
	session.recordInlineCompactionEvidence(llm.ConversionDirectionResponse, "restore", "restore", StrategyInlineCompactionGateway, ReasonGatewayCheckpoint, 0, digestInlineCompactionToken(token), inline.generation, stateBytes, uint32(len(inline.retained)), inline.drops, true)
	return response, nil
}

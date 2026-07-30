package conversion

import (
	"fmt"

	"github.com/looplj/axonhub/llm"
)

const compactEmulationInstruction = "Compact the conversation into a concise, self-contained continuation state. Preserve facts, decisions, constraints, pending work, tool-call results, identifiers, and file state. Do not answer the conversation or start new work."

func projectCompactEmulationInput(request *llm.Request, plan *Plan) *llm.Request {
	if request == nil || request.Compact == nil || !planEmulatesCompact(plan) {
		return request
	}
	projected := request.Clone()
	projected.Messages = projected.Compact.Input
	projected.Input = nil
	return projected
}

func prepareCompactEmulation(request *llm.Request, session *Session) (*llm.Request, error) {
	if !sessionEmulatesCompact(session) {
		return request, nil
	}
	if request == nil || request.Compact == nil {
		return nil, fmt.Errorf("%w: compact emulation requires compact request data", ErrIncompletePlan)
	}

	// The attempt clone is the only value mutated here. Compact input remains an
	// ordered message sequence; it is never flattened into a lossy text prompt.
	lowered := request.Clone()
	state := &compactEmulationState{instructions: lowered.Compact.Instructions}
	prompt := compactEmulationInstruction
	if state.instructions != "" {
		prompt += "\n\nAdditional compact instruction: " + state.instructions
	}
	messages := make([]llm.Message, 0, len(lowered.Messages)+1)
	messages = append(messages, llm.Message{
		Role:    "system",
		Content: llm.MessageContent{Content: stringPointer(prompt)},
	})
	messages = append(messages, lowered.Messages...)
	if len(lowered.Messages) == 0 {
		messages = append(messages, llm.Message{
			Role:    "user",
			Content: llm.MessageContent{Content: stringPointer("Produce the compact continuation state.")},
		})
	}
	if lowered.Compact.PromptCacheKey != "" {
		cacheKey := lowered.Compact.PromptCacheKey
		lowered.PromptCacheKey = &cacheKey
	}

	stream := false
	lowered.RequestType = llm.RequestTypeChat
	lowered.Stream = &stream
	lowered.StreamOptions = nil
	lowered.Messages = messages
	lowered.Input = nil
	lowered.Tools = nil
	lowered.ToolDefinitions = nil
	lowered.ToolChoice = nil
	lowered.Compact = nil
	session.compactEmulation = state
	return lowered, nil
}

func sessionEmulatesCompact(session *Session) bool {
	if session == nil || session.plan == nil {
		return false
	}
	return planEmulatesCompact(session.plan)
}

func planEmulatesCompact(plan *Plan) bool {
	if plan == nil {
		return false
	}
	for index := range plan.Actions {
		action := &plan.Actions[index]
		if action.Ref.Kind == ObjectCompaction && action.Kind == ActionEmulate && action.Strategy == StrategyCompactAsChat {
			return true
		}
	}
	return false
}

func restoreCompactEmulation(response *llm.Response, session *Session) *llm.Response {
	if response == nil || session == nil || session.compactEmulation == nil {
		return response
	}

	output := canonicalItemsToMessages(response.Output)
	if len(output) == 0 {
		legacy := make([]llm.Message, 0, len(response.Choices))
		for index := range response.Choices {
			if response.Choices[index].Message != nil {
				legacy = append(legacy, *response.Choices[index].Message)
			}
		}
		if len(legacy) > 0 {
			output = (&llm.Request{Messages: legacy}).Clone().Messages
		}
	}

	response.RequestType = llm.RequestTypeCompact
	response.APIFormat = llm.APIFormatOpenAIResponseCompact
	response.Object = "response.compaction"
	response.Compact = &llm.CompactResponse{
		ID:           response.ID,
		CreatedAt:    response.Created,
		Object:       "response.compaction",
		Instructions: session.compactEmulation.instructions,
		Output:       output,
	}
	return response
}

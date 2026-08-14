package conversion

import (
	"fmt"

	"github.com/looplj/axonhub/llm"
)

// codexMultiAgentV2UserInputBlocks implements precisely the portable input
// transform authorized by CPA's OptimizeMultiAgentV2 is-compat mode: each
// known encrypted_content part is retyped as an input_text part and the outer
// typed item becomes a normal user message. In this explicitly admitted path,
// Axon sends that encrypted_content string to the provider as ordinary user
// text, exactly as CPA declares for an is-compat API-key model. It is not a
// generic encrypted-content projection: the only caller is the plan-selected
// CodexMultiAgentV2Compat strategy after both target admission and the
// Responses Lite source-profile gate have passed.
//
// Axon does not decrypt, parse, or infer a meaning from the value. It trusts
// the declared CPA contract and forwards the source-order strings as the
// target's input_text payloads; every other source or target path is blocked.
func codexMultiAgentV2UserInputBlocks(message *llm.AgentMessage) ([]llm.ContentBlock, error) {
	if err := message.Validate(); err != nil {
		return nil, fmt.Errorf("invalid typed agent_message for Codex MultiAgentV2 compatibility: %w", err)
	}
	blocks := make([]llm.ContentBlock, 0, len(message.Content))
	for index := range message.Content {
		part := &message.Content[index]
		switch part.Kind {
		case llm.AgentMessageContentInputText:
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindText, Text: part.Text})
		case llm.AgentMessageContentEncryptedContent:
			blocks = append(blocks, llm.ContentBlock{Kind: llm.ContentKindText, Text: part.EncryptedContent})
		default:
			// Keep a second closed-union boundary even though Validate currently
			// enforces it, so a future grammar extension cannot silently cross
			// this explicitly admitted portability path.
			return nil, fmt.Errorf("unsupported agent_message content type %q", part.Kind)
		}
	}
	if len(blocks) == 0 {
		return nil, fmt.Errorf("Codex MultiAgentV2 user input is empty")
	}
	return blocks, nil
}

// projectAgentMessagesForPlan performs the request-local representation
// change that legacy Chat and Anthropic encoders consume. It is intentionally
// keyed by the checked Plan action rather than by agent-message contents, a
// request header, a user agent, or an arbitrary provider string.
func projectAgentMessagesForPlan(items []llm.Item, plan *Plan) ([]llm.Item, error) {
	if len(items) == 0 || plan == nil {
		return items, nil
	}
	projected := append([]llm.Item(nil), items...)
	for itemIndex := range projected {
		item := &projected[itemIndex]
		if item.Kind != llm.ItemKindAgentMessage {
			continue
		}
		action, ok := agentMessageActionForInput(plan, itemIndex)
		if !ok {
			return nil, fmt.Errorf("canonical agent_message item %d has no checked plan action", itemIndex)
		}
		if item.AgentMessage == nil {
			return nil, fmt.Errorf("canonical agent_message item %d has no payload", itemIndex)
		}

		switch action.Strategy {
		case StrategyAgentMessageLegacyInput:
			envelope, err := item.AgentMessage.LegacyInterAgentMessageJSON()
			if err != nil {
				return nil, fmt.Errorf("canonical agent_message item %d has no safe checked legacy projection: %w", itemIndex, err)
			}
			// Do not leave the typed agent message in Input after a legacy
			// projection. Downstream generic encoders would otherwise re-evaluate
			// it without this checked conversion decision.
			projected[itemIndex] = llm.Item{
				Kind: llm.ItemKindMessage,
				ID:   item.ID,
				Role: llm.RoleAssistant,
				Content: []llm.ContentBlock{{
					Kind: llm.ContentKindText,
					Text: envelope,
				}},
			}
		case StrategyAgentMessageMultiAgentV2UserInput:
			blocks, err := codexMultiAgentV2UserInputBlocks(item.AgentMessage)
			if err != nil {
				return nil, fmt.Errorf("canonical agent_message item %d has no safe checked Codex MultiAgentV2 user projection: %w", itemIndex, err)
			}
			// CPA's explicit is-compat policy emits type=message, role=user and
			// preserves the ordered input_text + formerly encrypted payload as
			// input_text parts. Apply that exact semantic projection only after
			// the selected target profile proved the same capability.
			projected[itemIndex] = llm.Item{Kind: llm.ItemKindMessage, ID: item.ID, Role: llm.RoleUser, Content: blocks}
		default:
			return nil, fmt.Errorf("canonical agent_message item %d has no safe checked legacy projection", itemIndex)
		}
	}
	return projected, nil
}

func agentMessageActionForInput(plan *Plan, itemIndex int) (Action, bool) {
	if plan == nil {
		return Action{}, false
	}
	for actionIndex := range plan.Actions {
		action := plan.Actions[actionIndex]
		if action.Ref.Kind == ObjectAgentMessage && action.Ref.ItemIndex == itemIndex {
			return action, true
		}
	}
	return Action{}, false
}

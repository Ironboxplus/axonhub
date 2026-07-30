package llm

import (
	"encoding/json"
	"errors"
	"fmt"
)

// EventKind is the provider-neutral lifecycle vocabulary used between stream
// decoders and encoders. Transport sentinels such as SSE [DONE] are deliberately
// excluded: they belong to a protocol encoder, not to the semantic stream.
type EventKind string

const (
	EventKindResponseStarted    EventKind = "response_started"
	EventKindResponseInProgress EventKind = "response_in_progress"
	EventKindResponseCompleted  EventKind = "response_completed"
	EventKindResponseFailed     EventKind = "response_failed"
	EventKindResponseIncomplete EventKind = "response_incomplete"
	EventKindResponseCancelled  EventKind = "response_cancelled"

	EventKindItemAdded EventKind = "item_added"
	EventKindItemDone  EventKind = "item_done"

	EventKindTextDelta      EventKind = "text_delta"
	EventKindReasoningDelta EventKind = "reasoning_delta"
	EventKindRefusalDelta   EventKind = "refusal_delta"
	EventKindToolInputDelta EventKind = "tool_input_delta"
	EventKindToolInputDone  EventKind = "tool_input_done"
	EventKindHostedStatus   EventKind = "hosted_status"
	EventKindMCPStatus      EventKind = "mcp_status"
	EventKindUsage          EventKind = "usage"
	EventKindError          EventKind = "error"
)

// ItemRef keeps protocol object identity and positional identity separate.
// In particular, Responses item_id is never substituted for call_id, and a
// Chat choice/tool index is not confused with a Responses output_index.
type ItemRef struct {
	ChoiceIndex *int   `json:"choice_index,omitempty"`
	OutputIndex *int   `json:"output_index,omitempty"`
	ToolIndex   *int   `json:"tool_index,omitempty"`
	ItemID      string `json:"item_id,omitempty"`
	CallID      string `json:"call_id,omitempty"`
}

// Delta contains only incremental data. Final typed values belong in the
// Snapshot on item_done; this prevents an encoder from treating accumulated
// data as another delta.
type Delta struct {
	Text          string          `json:"text,omitempty"`
	ArgumentsJSON string          `json:"arguments_json,omitempty"`
	InputText     string          `json:"input_text,omitempty"`
	HostedStatus  string          `json:"hosted_status,omitempty"`
	MCPStatus     string          `json:"mcp_status,omitempty"`
	ProviderData  json.RawMessage `json:"provider_data,omitempty"`
}

// Event is the ordered canonical stream unit shared by Chat Completions,
// Responses and Anthropic Messages.
type Event struct {
	Kind           EventKind      `json:"kind"`
	Sequence       uint64         `json:"sequence"`
	TerminalReason string         `json:"terminal_reason,omitempty"`
	SourceSequence *uint64        `json:"source_sequence,omitempty"`
	ItemRef        ItemRef        `json:"item_ref,omitempty"`
	ContentIndex   *int           `json:"content_index,omitempty"`
	Delta          Delta          `json:"delta,omitempty"`
	Snapshot       *Item          `json:"snapshot,omitempty"`
	Usage          *Usage         `json:"usage,omitempty"`
	Error          *ResponseError `json:"error,omitempty"`
}

func (event *Event) Validate() error {
	if event == nil {
		return errors.New("nil event")
	}

	switch event.Kind {
	case EventKindResponseStarted, EventKindResponseInProgress,
		EventKindResponseCompleted, EventKindResponseIncomplete,
		EventKindResponseCancelled:
		if event.Snapshot != nil || event.Usage != nil || event.Error != nil {
			return fmt.Errorf("%s cannot carry item, usage, or error payload", event.Kind)
		}
	case EventKindResponseFailed, EventKindError:
		if event.Error == nil {
			return fmt.Errorf("%s requires an error payload", event.Kind)
		}
	case EventKindItemAdded, EventKindItemDone:
		if event.Snapshot == nil {
			return fmt.Errorf("%s requires an item snapshot", event.Kind)
		}
		if err := event.Snapshot.Validate(); err != nil {
			return fmt.Errorf("%s snapshot: %w", event.Kind, err)
		}
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
	case EventKindTextDelta, EventKindReasoningDelta, EventKindRefusalDelta:
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
	case EventKindToolInputDelta:
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
		if event.Delta.ArgumentsJSON == "" && event.Delta.InputText == "" {
			return errors.New("tool_input_delta requires JSON arguments or freeform input")
		}
		if event.Delta.ArgumentsJSON != "" && event.Delta.InputText != "" {
			return errors.New("tool_input_delta cannot mix JSON arguments and freeform input")
		}
	case EventKindToolInputDone:
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
	case EventKindHostedStatus:
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
		if event.Delta.HostedStatus == "" {
			return errors.New("hosted_status requires a status")
		}
	case EventKindMCPStatus:
		if _, err := event.ItemRef.key(); err != nil {
			return err
		}
		if event.Delta.MCPStatus == "" {
			return errors.New("mcp_status requires a status")
		}
	case EventKindUsage:
		if event.Usage == nil {
			return errors.New("usage requires a usage payload")
		}
	default:
		return fmt.Errorf("unknown event kind %q", event.Kind)
	}

	if len(event.Delta.ProviderData) > 0 && !json.Valid(event.Delta.ProviderData) {
		return errors.New("event provider_data is invalid JSON")
	}
	if event.TerminalReason != "" {
		switch event.Kind {
		case EventKindResponseCompleted, EventKindResponseIncomplete, EventKindResponseCancelled, EventKindResponseFailed:
		default:
			return fmt.Errorf("%s cannot carry a terminal reason", event.Kind)
		}
	}
	return nil
}

// StableKey returns the request-local correlation key used by state machines,
// lowering ledgers and protocol encoders. The key is metadata only and is not
// exposed as a provider object ID.
func (ref ItemRef) StableKey() (string, error) {
	if ref.ItemID != "" {
		return "item:" + ref.ItemID, nil
	}
	if ref.CallID != "" {
		return "call:" + ref.CallID, nil
	}
	if ref.ToolIndex != nil && ref.ChoiceIndex != nil {
		return fmt.Sprintf("choice:%d/tool:%d", *ref.ChoiceIndex, *ref.ToolIndex), nil
	}
	if ref.OutputIndex != nil {
		return fmt.Sprintf("output:%d", *ref.OutputIndex), nil
	}
	if ref.ChoiceIndex != nil {
		return fmt.Sprintf("choice:%d", *ref.ChoiceIndex), nil
	}
	return "", errors.New("item event requires item_id, call_id, output_index, or choice_index")
}

func (ref ItemRef) key() (string, error) { return ref.StableKey() }

package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
)

type anthropicCanonicalEncoder struct {
	machine       *llm.StreamStateMachine
	blockIndexes  map[string]int64
	items         map[string]*llm.Item
	arguments     map[string]*strings.Builder
	nextBlock     int64
	pendingUsage  *Usage
	sawClientTool bool
}

func newAnthropicCanonicalEncoder() *anthropicCanonicalEncoder {
	return &anthropicCanonicalEncoder{
		machine: llm.NewStreamStateMachine(), blockIndexes: make(map[string]int64),
		items: make(map[string]*llm.Item), arguments: make(map[string]*strings.Builder),
	}
}

func (encoder *anthropicCanonicalEncoder) encode(source *anthropicInboundStream, event llm.Event) error {
	if err := encoder.machine.Apply(event); err != nil {
		return fmt.Errorf("encode canonical %s as Anthropic: %w", event.Kind, err)
	}
	emit := func(wire *StreamEvent) error {
		if err := source.enqueEvent(wire); err != nil {
			return fmt.Errorf("emit Anthropic %s: %w", wire.Type, err)
		}
		return nil
	}

	switch event.Kind {
	case llm.EventKindResponseStarted:
		source.hasStarted = true
		usage := &Usage{}
		if source.pendingUsage != nil {
			usage = source.pendingUsage
		}
		return emit(&StreamEvent{Type: "message_start", Message: &StreamMessage{
			ID: source.messageID, Type: "message", Role: "assistant", Model: source.model,
			Content: []MessageContentBlock{}, Usage: usage,
		}})
	case llm.EventKindResponseInProgress:
		return nil
	case llm.EventKindItemAdded:
		return encoder.addItem(event, emit)
	case llm.EventKindTextDelta, llm.EventKindRefusalDelta:
		return encoder.textDelta(event, emit)
	case llm.EventKindReasoningDelta:
		return encoder.reasoningDelta(event, emit)
	case llm.EventKindToolInputDelta:
		return encoder.toolDelta(event, emit)
	case llm.EventKindToolInputDone, llm.EventKindHostedStatus:
		return nil
	case llm.EventKindItemDone:
		return encoder.doneItem(event, emit)
	case llm.EventKindUsage:
		encoder.pendingUsage = convertToAnthropicUsage(event.Usage)
		source.pendingUsage = encoder.pendingUsage
		return nil
	case llm.EventKindResponseCompleted, llm.EventKindResponseIncomplete, llm.EventKindResponseCancelled:
		stopReason := "end_turn"
		if event.TerminalReason != "" {
			stopReason = event.TerminalReason
		} else if event.Kind == llm.EventKindResponseIncomplete {
			stopReason = "max_tokens"
		} else if encoder.sawClientTool {
			stopReason = "tool_use"
		}
		usage := encoder.pendingUsage
		if usage == nil {
			usage = &Usage{}
		}
		if err := emit(&StreamEvent{Type: "message_delta", Delta: &StreamDelta{StopReason: &stopReason}, Usage: usage}); err != nil {
			return err
		}
		if err := emit(&StreamEvent{Type: "message_stop"}); err != nil {
			return err
		}
		source.hasFinished = true
		source.messageStoped = true
		return nil
	case llm.EventKindResponseFailed, llm.EventKindError:
		return event.Error
	default:
		return fmt.Errorf("canonical event %q has no Anthropic stream encoding", event.Kind)
	}
}

func (encoder *anthropicCanonicalEncoder) addItem(event llm.Event, emit func(*StreamEvent) error) error {
	key, _ := event.ItemRef.StableKey()
	index := encoder.nextBlock
	encoder.nextBlock++
	encoder.blockIndexes[key] = index
	encoder.items[key] = event.Snapshot
	var block MessageContentBlock

	switch event.Snapshot.Kind {
	case llm.ItemKindMessage:
		block = MessageContentBlock{Type: "text", Text: lo.ToPtr("")}
	case llm.ItemKindReasoning:
		blockType := "thinking"
		if event.Snapshot.ProtocolHints.SourceFormat == llm.APIFormatAnthropicMessage && event.Snapshot.ProtocolHints.SourceType == "redacted_thinking" {
			blockType = "redacted_thinking"
		}
		block = MessageContentBlock{Type: blockType, Thinking: lo.ToPtr("")}
		if blockType == "redacted_thinking" && event.Snapshot.Reasoning != nil {
			block.Data = event.Snapshot.Reasoning.Signature
		}
	case llm.ItemKindToolCall:
		call := event.Snapshot.ToolCall
		if call == nil || call.Kind != llm.ToolKindFunction {
			return fmt.Errorf("Anthropic client tool item is not a lowered function")
		}
		encoder.sawClientTool = true
		encoder.arguments[key] = &strings.Builder{}
		block = MessageContentBlock{Type: "tool_use", ID: call.CallID, Name: &call.LogicalName, Input: json.RawMessage(`{}`)}
	case llm.ItemKindHostedCall:
		if event.Snapshot.HostedCall == nil {
			return fmt.Errorf("Anthropic hosted item has no hosted payload")
		}
		call := &event.Snapshot.HostedCall.Invocation
		encoder.arguments[key] = &strings.Builder{}
		block = canonicalAnthropicHostedUseBlock(call)
		block.Input = json.RawMessage(`{}`)
		if event.Snapshot.ProtocolHints.SourceFormat == llm.APIFormatAnthropicMessage && isAnthropicSpecialToolUseBlock(event.Snapshot.ProtocolHints.SourceType) {
			block.Type = event.Snapshot.ProtocolHints.SourceType
		}
	case llm.ItemKindAgentMessage, llm.ItemKindCompaction, llm.ItemKindContextCompaction, llm.ItemKindUnknown:
		return fmt.Errorf("Anthropic cannot encode canonical stream item kind %q", event.Snapshot.Kind)
	default:
		return fmt.Errorf("canonical item kind %q has no Anthropic stream block", event.Snapshot.Kind)
	}
	return emit(&StreamEvent{Type: "content_block_start", Index: &index, ContentBlock: &block})
}

func (encoder *anthropicCanonicalEncoder) textDelta(event llm.Event, emit func(*StreamEvent) error) error {
	_, index, err := encoder.coordinates(event)
	if err != nil {
		return err
	}
	typeName := "text_delta"
	text := event.Delta.Text
	return emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, Text: &text}})
}

func (encoder *anthropicCanonicalEncoder) reasoningDelta(event llm.Event, emit func(*StreamEvent) error) error {
	_, index, err := encoder.coordinates(event)
	if err != nil {
		return err
	}
	typeName := "thinking_delta"
	text := event.Delta.Text
	return emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, Thinking: &text}})
}

func (encoder *anthropicCanonicalEncoder) toolDelta(event llm.Event, emit func(*StreamEvent) error) error {
	key, index, err := encoder.coordinates(event)
	if err != nil {
		return err
	}
	if event.Delta.ArgumentsJSON == "" {
		return fmt.Errorf("Anthropic tool delta requires lowered JSON")
	}
	arguments := encoder.arguments[key]
	if arguments == nil {
		return fmt.Errorf("Anthropic tool delta references item without argument state %q", key)
	}
	arguments.WriteString(event.Delta.ArgumentsJSON)
	typeName := "input_json_delta"
	fragment := event.Delta.ArgumentsJSON
	return emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, PartialJSON: &fragment}})
}

func (encoder *anthropicCanonicalEncoder) doneItem(event llm.Event, emit func(*StreamEvent) error) error {
	key, index, err := encoder.coordinates(event)
	if err != nil {
		return err
	}
	if event.Snapshot.Kind == llm.ItemKindMessage {
		for _, content := range event.Snapshot.Content {
			if content.Kind != llm.ContentKindCitation || content.Citation == nil {
				continue
			}
			typeName := "citations_delta"
			citation := &TextCitation{
				Type: "web_search_result_location", URL: content.Citation.URL, Title: content.Citation.Title,
				EncryptedIndex: content.Citation.EncryptedIndex, CitedText: content.Citation.CitedText,
			}
			if err := emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, Citation: citation}}); err != nil {
				return err
			}
		}
	}
	if event.Snapshot.Kind == llm.ItemKindReasoning && event.Snapshot.Reasoning != nil &&
		event.Snapshot.ProtocolHints.SourceType != "redacted_thinking" {
		typeName := "signature_delta"
		signature := event.Snapshot.Reasoning.Signature
		// A reasoning signature is opaque canonical data. Replacing a non-empty
		// source value with a random token destroys round-trip identity and gives
		// the client a value that no provider actually issued.
		if signature == "" {
			signature = generateSignature()
		}
		if err := emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, Signature: &signature}}); err != nil {
			return err
		}
	}
	if arguments := encoder.arguments[key]; arguments != nil && arguments.Len() == 0 {
		var invocation *llm.ToolInvocation
		if event.Snapshot.ToolCall != nil {
			invocation = event.Snapshot.ToolCall
		} else if event.Snapshot.HostedCall != nil {
			invocation = &event.Snapshot.HostedCall.Invocation
		}
		if invocation != nil {
			full := canonicalAnthropicArguments(invocation)
			if string(full) != "{}" {
				typeName := "input_json_delta"
				fragment := string(full)
				if err := emit(&StreamEvent{Type: "content_block_delta", Index: &index, Delta: &StreamDelta{Type: &typeName, PartialJSON: &fragment}}); err != nil {
					return err
				}
				arguments.Write(full)
			}
		}
	}
	if err := emit(&StreamEvent{Type: "content_block_stop", Index: &index}); err != nil {
		return err
	}
	if event.Snapshot.Kind == llm.ItemKindHostedCall && event.Snapshot.HostedCall != nil && event.Snapshot.HostedCall.Result != nil {
		resultIndex := encoder.nextBlock
		encoder.nextBlock++
		resultBlock := canonicalAnthropicHostedResultBlock(&event.Snapshot.HostedCall.Invocation, event.Snapshot.HostedCall.Result)
		if err := emit(&StreamEvent{Type: "content_block_start", Index: &resultIndex, ContentBlock: &resultBlock}); err != nil {
			return err
		}
		if err := emit(&StreamEvent{Type: "content_block_stop", Index: &resultIndex}); err != nil {
			return err
		}
	}
	delete(encoder.arguments, key)
	return nil
}

func (encoder *anthropicCanonicalEncoder) coordinates(event llm.Event) (string, int64, error) {
	key, err := event.ItemRef.StableKey()
	if err != nil {
		return "", 0, err
	}
	index, ok := encoder.blockIndexes[key]
	if !ok {
		return "", 0, fmt.Errorf("canonical %s references unknown item %q", event.Kind, key)
	}
	return key, index, nil
}

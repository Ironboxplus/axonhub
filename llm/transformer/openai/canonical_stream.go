package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

type chatCanonicalStream struct {
	source            streams.Stream[*llm.Response]
	decoder           *chatCanonicalDecoder
	current           *llm.Response
	err               error
	pending           []*llm.Response
	ended             bool
	incompleteHandled bool
}

func newChatCanonicalStream(source streams.Stream[*llm.Response]) streams.Stream[*llm.Response] {
	return &chatCanonicalStream{source: source, decoder: newChatCanonicalDecoder()}
}

func (stream *chatCanonicalStream) Next() bool {
	if len(stream.pending) > 0 {
		stream.current = stream.pending[0]
		stream.pending = stream.pending[1:]
		return true
	}
	if stream.ended {
		return false
	}
	if !stream.source.Next() {
		stream.ended = true
		if errors.Is(stream.source.Err(), shared.ErrStreamIncomplete) {
			events, err := stream.decoder.decode(&llm.Response{Object: "[DONE]"})
			if err != nil {
				stream.err = err
				return false
			}
			stream.incompleteHandled = true
			stream.pending = append(stream.pending, stream.incompleteResponse(events), llm.DoneResponse)
			return stream.Next()
		}
		stream.err = stream.source.Err()
		return false
	}
	response := stream.source.Current()
	if response == nil {
		return stream.Next()
	}
	events, err := stream.decoder.decode(response)
	if err != nil {
		stream.err = err
		return false
	}
	if len(events) > 0 {
		if response == llm.DoneResponse {
			clone := *response
			response = &clone
		}
		response.Events = append(response.Events, events...)
	}
	stream.current = response
	return true
}

func (stream *chatCanonicalStream) Current() *llm.Response { return stream.current }

func (stream *chatCanonicalStream) Err() error {
	if stream.err != nil {
		return stream.err
	}
	if stream.incompleteHandled {
		return nil
	}
	return stream.source.Err()
}

func (stream *chatCanonicalStream) Close() error { return stream.source.Close() }

func (stream *chatCanonicalStream) incompleteResponse(events []llm.Event) *llm.Response {
	response := &llm.Response{Object: "chat.completion.chunk", APIFormat: llm.APIFormatOpenAIChatCompletion, Events: events}
	if stream.current != nil {
		response.ID = stream.current.ID
		response.Model = stream.current.Model
		response.Created = stream.current.Created
	}
	indexes := make([]int, 0, len(stream.decoder.choices))
	for index := range stream.decoder.choices {
		indexes = append(indexes, index)
	}
	if len(indexes) == 0 {
		indexes = append(indexes, 0)
	}
	sort.Ints(indexes)
	reason := "length"
	response.Choices = make([]llm.Choice, 0, len(indexes))
	for _, index := range indexes {
		response.Choices = append(response.Choices, llm.Choice{Index: index, Delta: &llm.Message{}, FinishReason: &reason})
	}
	return response
}

type chatCanonicalDecoder struct {
	machine         *llm.StreamStateMachine
	nextSequence    uint64
	nextOutput      int
	started         bool
	terminalPending bool
	terminalKind    llm.EventKind
	terminalEmitted bool
	choices         map[int]*chatChoiceLifecycle
	openOrder       []chatOpenRef
}

type chatChoiceLifecycle struct {
	message   *chatItemLifecycle
	reasoning *chatItemLifecycle
	tools     map[int]*chatToolLifecycle
}

type chatItemLifecycle struct {
	ref      llm.ItemRef
	snapshot *llm.Item
	done     bool
}

type chatToolLifecycle struct {
	item       *chatItemLifecycle
	index      int
	callID     string
	name       string
	namespace  string
	arguments  string
	declared   bool
	pendingArg string
}

type chatOpenRef struct {
	choice int
	kind   llm.ItemKind
	tool   int
}

func newChatCanonicalDecoder() *chatCanonicalDecoder {
	return &chatCanonicalDecoder{machine: llm.NewStreamStateMachine(), choices: make(map[int]*chatChoiceLifecycle)}
}

func (decoder *chatCanonicalDecoder) decode(response *llm.Response) ([]llm.Event, error) {
	var events []llm.Event
	emit := func(event llm.Event) error {
		event.Sequence = decoder.nextSequence
		decoder.nextSequence++
		if err := decoder.machine.Apply(event); err != nil {
			return fmt.Errorf("Chat stream: %w", err)
		}
		events = append(events, event)
		return nil
	}

	if response.Object == "[DONE]" {
		if !decoder.started {
			if err := emit(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
				return nil, err
			}
			decoder.started = true
		}
		if !decoder.terminalPending {
			decoder.terminalPending = true
			decoder.terminalKind = llm.EventKindResponseIncomplete
		}
		if err := decoder.emitTerminal(emit); err != nil {
			return nil, err
		}
		return events, nil
	}

	if !decoder.started {
		if err := emit(llm.Event{Kind: llm.EventKindResponseStarted}); err != nil {
			return nil, err
		}
		if err := emit(llm.Event{Kind: llm.EventKindResponseInProgress}); err != nil {
			return nil, err
		}
		decoder.started = true
	}

	for choiceIndex := range response.Choices {
		choice := &response.Choices[choiceIndex]
		state := decoder.choice(choice.Index)
		if choice.Delta != nil {
			if err := decoder.decodeMessageDelta(choice.Index, state, choice.Delta, emit); err != nil {
				return nil, err
			}
		}
		if choice.FinishReason != nil {
			if err := decoder.closeChoice(choice.Index, state, terminalItemStatus(*choice.FinishReason), emit); err != nil {
				return nil, err
			}
			decoder.terminalPending = true
			decoder.terminalKind = terminalEventKind(*choice.FinishReason)
		}
	}

	if response.Usage != nil {
		if err := emit(llm.Event{Kind: llm.EventKindUsage, Usage: response.Usage}); err != nil {
			return nil, err
		}
	}
	if decoder.terminalPending && response.Usage != nil {
		if err := decoder.emitTerminal(emit); err != nil {
			return nil, err
		}
	}
	return events, nil
}

func (decoder *chatCanonicalDecoder) decodeMessageDelta(choiceIndex int, state *chatChoiceLifecycle, delta *llm.Message, emit func(llm.Event) error) error {
	reasoning := delta.ReasoningContent
	if reasoning == nil {
		reasoning = delta.Reasoning
	}
	if reasoning != nil || delta.ReasoningSignature != nil {
		if state.reasoning == nil {
			outputIndex := decoder.allocateOutput()
			state.reasoning = &chatItemLifecycle{
				ref: llm.ItemRef{ChoiceIndex: lo.ToPtr(choiceIndex), OutputIndex: lo.ToPtr(outputIndex)},
				snapshot: &llm.Item{
					Kind: llm.ItemKindReasoning, Role: llm.RoleAssistant, Status: llm.ItemStatusInProgress,
					Reasoning:     &llm.ReasoningItem{},
					ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIChatCompletion, SourceType: "reasoning", Ordinal: outputIndex},
				},
			}
			decoder.openOrder = append(decoder.openOrder, chatOpenRef{choice: choiceIndex, kind: llm.ItemKindReasoning})
			if err := emit(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: state.reasoning.ref, Snapshot: cloneChatStreamItem(state.reasoning.snapshot)}); err != nil {
				return err
			}
		}
		if reasoning != nil && *reasoning != "" {
			state.reasoning.snapshot.Reasoning.Content += *reasoning
			if err := emit(llm.Event{Kind: llm.EventKindReasoningDelta, ItemRef: state.reasoning.ref, Delta: llm.Delta{Text: *reasoning}}); err != nil {
				return err
			}
		}
		if delta.ReasoningSignature != nil {
			state.reasoning.snapshot.Reasoning.Signature = *delta.ReasoningSignature
		}
	}

	if (delta.Content.Content != nil && *delta.Content.Content != "") || delta.Refusal != "" || len(delta.Content.MultipleContent) > 0 || len(delta.Annotations) > 0 {
		if state.message == nil {
			outputIndex := decoder.allocateOutput()
			state.message = &chatItemLifecycle{
				ref: llm.ItemRef{ChoiceIndex: lo.ToPtr(choiceIndex), OutputIndex: lo.ToPtr(outputIndex)},
				snapshot: &llm.Item{
					Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Status: llm.ItemStatusInProgress,
					Content:       []llm.ContentBlock{{Kind: llm.ContentKindText}},
					ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIChatCompletion, SourceType: "message", Ordinal: outputIndex},
				},
			}
			decoder.openOrder = append(decoder.openOrder, chatOpenRef{choice: choiceIndex, kind: llm.ItemKindMessage})
			if err := emit(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: state.message.ref, Snapshot: cloneChatStreamItem(state.message.snapshot)}); err != nil {
				return err
			}
		}
		if delta.Content.Content != nil && *delta.Content.Content != "" {
			state.message.snapshot.Content[0].Text += *delta.Content.Content
			if err := emit(llm.Event{Kind: llm.EventKindTextDelta, ItemRef: state.message.ref, ContentIndex: lo.ToPtr(0), Delta: llm.Delta{Text: *delta.Content.Content}}); err != nil {
				return err
			}
		}
		if delta.Refusal != "" {
			refusalIndex := len(state.message.snapshot.Content)
			state.message.snapshot.Content = append(state.message.snapshot.Content, llm.ContentBlock{Kind: llm.ContentKindRefusal, Text: delta.Refusal})
			if err := emit(llm.Event{Kind: llm.EventKindRefusalDelta, ItemRef: state.message.ref, ContentIndex: lo.ToPtr(refusalIndex), Delta: llm.Delta{Text: delta.Refusal}}); err != nil {
				return err
			}
		}
		appendChatStreamContent(state.message.snapshot, delta)
	}

	for index := range delta.ToolCalls {
		if err := decoder.decodeToolDelta(choiceIndex, state, &delta.ToolCalls[index], emit); err != nil {
			return err
		}
	}
	return nil
}

func (decoder *chatCanonicalDecoder) decodeToolDelta(choiceIndex int, state *chatChoiceLifecycle, delta *llm.ToolCall, emit func(llm.Event) error) error {
	tool := state.tools[delta.Index]
	if tool == nil {
		tool = &chatToolLifecycle{index: delta.Index}
		state.tools[delta.Index] = tool
	}
	if delta.ID != "" {
		tool.callID = delta.ID
	}
	if delta.Function.Name != "" {
		tool.name += delta.Function.Name
	}
	if delta.Function.Namespace != "" {
		tool.namespace += delta.Function.Namespace
	}
	tool.pendingArg += delta.Function.Arguments

	if !tool.declared && tool.callID != "" && tool.name != "" {
		outputIndex := decoder.allocateOutput()
		tool.item = &chatItemLifecycle{
			ref: llm.ItemRef{ChoiceIndex: lo.ToPtr(choiceIndex), OutputIndex: lo.ToPtr(outputIndex), ToolIndex: lo.ToPtr(delta.Index), CallID: tool.callID},
			snapshot: &llm.Item{
				Kind: llm.ItemKindToolCall, ID: tool.callID, Role: llm.RoleAssistant, Status: llm.ItemStatusInProgress,
				ToolCall: &llm.ToolInvocation{
					Kind: llm.ToolKindFunction, ID: tool.callID, CallID: tool.callID, LogicalName: tool.name,
					Namespace: tool.namespace, Status: llm.ToolCallStatusInProgress, Execution: llm.ExecutionOwnerClient,
				},
				ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIChatCompletion, SourceType: "function_call", Ordinal: outputIndex},
			},
		}
		tool.declared = true
		decoder.openOrder = append(decoder.openOrder, chatOpenRef{choice: choiceIndex, kind: llm.ItemKindToolCall, tool: delta.Index})
		if err := emit(llm.Event{Kind: llm.EventKindItemAdded, ItemRef: tool.item.ref, Snapshot: cloneChatStreamItem(tool.item.snapshot)}); err != nil {
			return err
		}
	}
	if tool.declared && tool.pendingArg != "" {
		fragment := tool.pendingArg
		tool.pendingArg = ""
		tool.arguments += fragment
		if err := emit(llm.Event{Kind: llm.EventKindToolInputDelta, ItemRef: tool.item.ref, Delta: llm.Delta{ArgumentsJSON: fragment}}); err != nil {
			return err
		}
	}
	return nil
}

func (decoder *chatCanonicalDecoder) closeChoice(choiceIndex int, state *chatChoiceLifecycle, status llm.ItemStatus, emit func(llm.Event) error) error {
	for _, open := range decoder.openOrder {
		if open.choice != choiceIndex {
			continue
		}
		var item *chatItemLifecycle
		switch open.kind {
		case llm.ItemKindMessage:
			item = state.message
		case llm.ItemKindReasoning:
			item = state.reasoning
		case llm.ItemKindToolCall:
			tool := state.tools[open.tool]
			if tool == nil || !tool.declared {
				continue
			}
			item = tool.item
			if item.done {
				continue
			}
			if err := emit(llm.Event{Kind: llm.EventKindToolInputDone, ItemRef: item.ref}); err != nil {
				return err
			}
			if json.Valid([]byte(tool.arguments)) {
				item.snapshot.ToolCall.ArgumentsJSON = append(json.RawMessage(nil), tool.arguments...)
			} else {
				item.snapshot.ToolCall.ArgumentsText = tool.arguments
			}
		}
		if item == nil || item.done {
			continue
		}
		item.done = true
		item.snapshot.Status = status
		if item.snapshot.ToolCall != nil {
			if status == llm.ItemStatusCompleted {
				item.snapshot.ToolCall.Status = llm.ToolCallStatusCompleted
			} else {
				item.snapshot.ToolCall.Status = llm.ToolCallStatusIncomplete
			}
		}
		if err := emit(llm.Event{Kind: llm.EventKindItemDone, ItemRef: item.ref, Snapshot: cloneChatStreamItem(item.snapshot)}); err != nil {
			return err
		}
	}
	return nil
}

func (decoder *chatCanonicalDecoder) emitTerminal(emit func(llm.Event) error) error {
	if decoder.terminalEmitted {
		return nil
	}
	for choiceIndex, state := range decoder.choices {
		if err := decoder.closeChoice(choiceIndex, state, llm.ItemStatusIncomplete, emit); err != nil {
			return err
		}
	}
	if err := emit(llm.Event{Kind: decoder.terminalKind}); err != nil {
		return err
	}
	decoder.terminalEmitted = true
	return nil
}

func (decoder *chatCanonicalDecoder) choice(index int) *chatChoiceLifecycle {
	state := decoder.choices[index]
	if state == nil {
		state = &chatChoiceLifecycle{tools: make(map[int]*chatToolLifecycle)}
		decoder.choices[index] = state
	}
	return state
}

func (decoder *chatCanonicalDecoder) allocateOutput() int {
	index := decoder.nextOutput
	decoder.nextOutput++
	return index
}

func terminalEventKind(reason string) llm.EventKind {
	switch reason {
	case "length", "content_filter":
		return llm.EventKindResponseIncomplete
	case "cancelled", "canceled":
		return llm.EventKindResponseCancelled
	default:
		return llm.EventKindResponseCompleted
	}
}

func terminalItemStatus(reason string) llm.ItemStatus {
	if terminalEventKind(reason) == llm.EventKindResponseCompleted {
		return llm.ItemStatusCompleted
	}
	return llm.ItemStatusIncomplete
}

func cloneChatStreamItem(item *llm.Item) *llm.Item {
	if item == nil {
		return nil
	}
	clone := *item
	clone.Content = append([]llm.ContentBlock(nil), item.Content...)
	if item.ToolCall != nil {
		call := *item.ToolCall
		call.ArgumentsJSON = append(json.RawMessage(nil), item.ToolCall.ArgumentsJSON...)
		clone.ToolCall = &call
	}
	if item.Reasoning != nil {
		reasoning := *item.Reasoning
		clone.Reasoning = &reasoning
	}
	return &clone
}

func appendChatStreamContent(item *llm.Item, message *llm.Message) {
	if item == nil || message == nil {
		return
	}
	for _, part := range message.Content.MultipleContent {
		switch part.Type {
		case "image_url":
			if part.ImageURL != nil {
				image := *part.ImageURL
				item.Content = append(item.Content, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &image})
			}
		case "input_audio":
			if part.InputAudio != nil {
				audio := *part.InputAudio
				item.Content = append(item.Content, llm.ContentBlock{Kind: llm.ContentKindAudio, Audio: &audio})
			}
		case "document":
			if part.Document != nil {
				document := *part.Document
				item.Content = append(item.Content, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: &document})
			}
		}
	}
	for _, annotation := range message.Annotations {
		if annotation.URLCitation != nil {
			item.Content = append(item.Content, llm.ContentBlock{
				Kind:     llm.ContentKindCitation,
				Citation: &llm.URLCitation{URL: annotation.URLCitation.URL, Title: annotation.URLCitation.Title},
			})
		}
	}
}

package openai

import (
	"context"
	"fmt"
	"sort"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

type canonicalChatInboundStream struct {
	ctx         context.Context
	transformer *InboundTransformer
	source      streams.Stream[*llm.Response]
	encoder     *canonicalChatEncoder
	queue       []*httpclient.StreamEvent
	queueIndex  int
	current     *httpclient.StreamEvent
	err         error
	canonical   bool
	identity    bool
}

func newCanonicalChatInboundStream(ctx context.Context, transformer *InboundTransformer, source streams.Stream[*llm.Response]) streams.Stream[*httpclient.StreamEvent] {
	return &canonicalChatInboundStream{
		ctx: ctx, transformer: transformer, source: source, encoder: newCanonicalChatEncoder(),
	}
}

func (stream *canonicalChatInboundStream) Next() bool {
	for {
		if stream.queueIndex < len(stream.queue) {
			stream.current = stream.queue[stream.queueIndex]
			stream.queueIndex++
			return true
		}
		stream.queue = nil
		stream.queueIndex = 0

		if !stream.source.Next() {
			stream.err = stream.source.Err()
			return false
		}
		chunk := stream.source.Current()
		if chunk == nil {
			continue
		}
		if chunk.APIFormat == llm.APIFormatOpenAIChatCompletion {
			stream.identity = true
		}
		if len(chunk.Events) > 0 && !stream.identity {
			stream.canonical = true
			stream.encoder.updateMetadata(chunk)
			for _, event := range chunk.Events {
				responses, err := stream.encoder.encode(event)
				if err != nil {
					stream.err = err
					return false
				}
				for _, response := range responses {
					wire, err := stream.transformer.TransformStreamChunk(stream.ctx, response)
					if err != nil {
						stream.err = err
						return false
					}
					if wire != nil {
						stream.queue = append(stream.queue, wire)
					}
				}
			}
			continue
		}
		if stream.canonical {
			// Canonical terminal events synthesize the single Chat [DONE]. Ignore
			// trailing compatibility usage/sentinel chunks to avoid duplicates.
			continue
		}
		wire, err := stream.transformer.TransformStreamChunk(stream.ctx, chunk)
		if err != nil {
			stream.err = err
			return false
		}
		if wire == nil {
			continue
		}
		stream.current = wire
		return true
	}
}

func (stream *canonicalChatInboundStream) Current() *httpclient.StreamEvent { return stream.current }

func (stream *canonicalChatInboundStream) Err() error {
	if stream.err != nil {
		return stream.err
	}
	return stream.source.Err()
}

func (stream *canonicalChatInboundStream) Close() error { return stream.source.Close() }

type canonicalChatEncoder struct {
	machine       *llm.StreamStateMachine
	id            string
	model         string
	created       int64
	itemKinds     map[string]llm.ItemKind
	toolIndexes   map[string]int
	nextToolIndex map[int]int
	roleEmitted   map[int]bool
	choices       map[int]bool
	sawTool       map[int]bool
	pendingUsage  *llm.Usage
}

func newCanonicalChatEncoder() *canonicalChatEncoder {
	return &canonicalChatEncoder{
		machine: llm.NewStreamStateMachine(), itemKinds: make(map[string]llm.ItemKind),
		toolIndexes: make(map[string]int), nextToolIndex: make(map[int]int),
		roleEmitted: make(map[int]bool), choices: make(map[int]bool), sawTool: make(map[int]bool),
	}
}

func (encoder *canonicalChatEncoder) updateMetadata(chunk *llm.Response) {
	if chunk == nil {
		return
	}
	if chunk.ID != "" {
		encoder.id = chunk.ID
	}
	if chunk.Model != "" {
		encoder.model = chunk.Model
	}
	if chunk.Created != 0 {
		encoder.created = chunk.Created
	}
}

func (encoder *canonicalChatEncoder) encode(event llm.Event) ([]*llm.Response, error) {
	if err := encoder.machine.Apply(event); err != nil {
		return nil, fmt.Errorf("encode canonical %s as Chat: %w", event.Kind, err)
	}
	choiceIndex := canonicalChoiceIndex(event.ItemRef)
	var responses []*llm.Response
	emitRole := func() {
		if encoder.roleEmitted[choiceIndex] {
			return
		}
		encoder.roleEmitted[choiceIndex] = true
		encoder.choices[choiceIndex] = true
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{Role: "assistant"}}))
	}

	switch event.Kind {
	case llm.EventKindResponseStarted, llm.EventKindResponseInProgress:
		return nil, nil
	case llm.EventKindItemAdded:
		emitRole()
		key, _ := event.ItemRef.StableKey()
		encoder.itemKinds[key] = event.Snapshot.Kind
		if event.Snapshot.Kind != llm.ItemKindToolCall {
			return responses, nil
		}
		call := event.Snapshot.ToolCall
		if call == nil || call.Kind != llm.ToolKindFunction {
			return nil, fmt.Errorf("Chat cannot encode unlowered tool kind %q", call.Kind)
		}
		toolIndex := encoder.toolIndex(key, choiceIndex, event.ItemRef.ToolIndex)
		encoder.sawTool[choiceIndex] = true
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{ToolCalls: []llm.ToolCall{{
			Index: toolIndex, ID: call.CallID, Type: llm.ToolTypeFunction,
			Function: llm.FunctionCall{Name: call.LogicalName, Namespace: call.Namespace},
		}}}}))
	case llm.EventKindTextDelta:
		emitRole()
		text := event.Delta.Text
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{Content: llm.MessageContent{Content: &text}}}))
	case llm.EventKindReasoningDelta:
		emitRole()
		text := event.Delta.Text
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{ReasoningContent: &text}}))
	case llm.EventKindRefusalDelta:
		emitRole()
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{Refusal: event.Delta.Text}}))
	case llm.EventKindToolInputDelta:
		emitRole()
		key, _ := event.ItemRef.StableKey()
		if encoder.itemKinds[key] != llm.ItemKindToolCall || event.Delta.ArgumentsJSON == "" {
			return nil, fmt.Errorf("Chat tool input delta is not lowered JSON for %q", key)
		}
		responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{ToolCalls: []llm.ToolCall{{
			Index: encoder.toolIndexes[key], Function: llm.FunctionCall{Arguments: event.Delta.ArgumentsJSON},
		}}}}))
	case llm.EventKindToolInputDone, llm.EventKindHostedStatus:
		return responses, nil
	case llm.EventKindItemDone:
		emitRole()
		if event.Snapshot.Kind == llm.ItemKindReasoning && event.Snapshot.Reasoning != nil && event.Snapshot.Reasoning.Signature != "" {
			signature := event.Snapshot.Reasoning.Signature
			responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{ReasoningSignature: &signature}}))
		}
		if event.Snapshot.Kind == llm.ItemKindMessage {
			annotations := canonicalChatAnnotations(event.Snapshot.Content)
			if len(annotations) > 0 {
				responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{Annotations: annotations}}))
			}
		}
		if event.Snapshot.Kind == llm.ItemKindHostedCall && event.Snapshot.HostedCall != nil && event.Snapshot.HostedCall.Result != nil {
			if content := canonicalHostedResultContent(event.Snapshot.HostedCall.Result.Content); len(content) > 0 {
				responses = append(responses, encoder.chunk(llm.Choice{Index: choiceIndex, Delta: &llm.Message{Content: llm.MessageContent{MultipleContent: content}}}))
			}
		}
	case llm.EventKindUsage:
		// Chat include_usage requires the usage-only chunk after the choices'
		// finish_reason but before [DONE]. Hold canonical usage until terminal.
		encoder.pendingUsage = event.Usage
	case llm.EventKindResponseCompleted, llm.EventKindResponseIncomplete, llm.EventKindResponseCancelled:
		if len(encoder.choices) == 0 {
			choiceIndex = 0
			emitRole()
		}
		choiceIndexes := make([]int, 0, len(encoder.choices))
		for index := range encoder.choices {
			choiceIndexes = append(choiceIndexes, index)
		}
		sort.Ints(choiceIndexes)
		finishChoices := make([]llm.Choice, 0, len(choiceIndexes))
		for _, index := range choiceIndexes {
			reason := "stop"
			if event.Kind == llm.EventKindResponseIncomplete {
				reason = "length"
			} else if event.Kind == llm.EventKindResponseCancelled {
				reason = "cancelled"
			} else if encoder.sawTool[index] {
				reason = "tool_calls"
			}
			finishChoices = append(finishChoices, llm.Choice{Index: index, Delta: &llm.Message{}, FinishReason: &reason})
		}
		responses = append(responses, encoder.chunk(finishChoices...))
		if encoder.pendingUsage != nil {
			responses = append(responses, &llm.Response{
				ID: encoder.id, Object: "chat.completion.chunk", Model: encoder.model, Created: encoder.created,
				Choices: []llm.Choice{}, Usage: encoder.pendingUsage,
			})
		}
		responses = append(responses, llm.DoneResponse)
	case llm.EventKindResponseFailed:
		return nil, event.Error
	case llm.EventKindError:
		return nil, event.Error
	default:
		return nil, fmt.Errorf("canonical event %q has no Chat stream encoding", event.Kind)
	}
	return responses, nil
}

func (encoder *canonicalChatEncoder) chunk(choices ...llm.Choice) *llm.Response {
	return &llm.Response{ID: encoder.id, Object: "chat.completion.chunk", Model: encoder.model, Created: encoder.created, Choices: choices}
}

func (encoder *canonicalChatEncoder) toolIndex(key string, choice int, preferred *int) int {
	if index, ok := encoder.toolIndexes[key]; ok {
		return index
	}
	index := encoder.nextToolIndex[choice]
	if preferred != nil {
		index = *preferred
	}
	encoder.toolIndexes[key] = index
	if index >= encoder.nextToolIndex[choice] {
		encoder.nextToolIndex[choice] = index + 1
	}
	return index
}

func canonicalChoiceIndex(ref llm.ItemRef) int {
	if ref.ChoiceIndex != nil {
		return *ref.ChoiceIndex
	}
	return 0
}

func canonicalChatAnnotations(content []llm.ContentBlock) []llm.Annotation {
	var annotations []llm.Annotation
	for index := range content {
		if content[index].Kind != llm.ContentKindCitation || content[index].Citation == nil {
			continue
		}
		citation := content[index].Citation
		annotations = append(annotations, llm.Annotation{Type: "url_citation", URLCitation: &llm.URLCitation{URL: citation.URL, Title: citation.Title}})
	}
	return annotations
}

func canonicalHostedResultContent(content []llm.ContentBlock) []llm.MessageContentPart {
	var parts []llm.MessageContentPart
	for index := range content {
		switch content[index].Kind {
		case llm.ContentKindText:
			text := content[index].Text
			parts = append(parts, llm.MessageContentPart{Type: "text", Text: &text})
		case llm.ContentKindImage:
			if content[index].Image != nil {
				image := *content[index].Image
				parts = append(parts, llm.MessageContentPart{Type: "image_url", ImageURL: &image})
			}
		}
	}
	return parts
}

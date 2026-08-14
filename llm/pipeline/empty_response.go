package pipeline

import (
	"errors"

	"github.com/looplj/axonhub/llm"
)

// ErrEmptyResponse indicates the response contains no meaningful content.
// This error triggers channel retry when empty response detection is enabled.
var ErrEmptyResponse = errors.New("empty response detected")

// ErrStreamFirstEventTimeout indicates a streaming response did not produce the first event in time.
var ErrStreamFirstEventTimeout = errors.New("stream first event timeout")

// ErrNonStreamResponseTimeout indicates a non-streaming response did not complete in time.
var ErrNonStreamResponseTimeout = errors.New("non-stream response timeout")

// ErrEmptyStreamChunks indicates an auto-upgraded streaming request produced no inbound chunks.
var ErrEmptyStreamChunks = errors.New("empty stream chunks")

// ErrEmptyAggregatedBody indicates inbound chunk aggregation produced an empty body.
var ErrEmptyAggregatedBody = errors.New("empty aggregated body")

func hasMessageContent(msg *llm.Message) bool {
	if msg == nil {
		return false
	}

	if msg.Content.Content != nil && *msg.Content.Content != "" {
		return true
	}

	if len(msg.Content.MultipleContent) > 0 {
		return true
	}

	if len(msg.ToolCalls) > 0 {
		return true
	}

	if msg.ReasoningContent != nil && *msg.ReasoningContent != "" {
		return true
	}

	if msg.Reasoning != nil && *msg.Reasoning != "" {
		return true
	}

	if msg.ReasoningSignature != nil && *msg.ReasoningSignature != "" {
		return true
	}

	if msg.Refusal != "" {
		return true
	}

	if msg.Audio != nil {
		return true
	}

	return false
}

// hasResponseContent checks if an llm.Response contains meaningful content.
func hasResponseContent(resp *llm.Response) bool {
	if resp == nil || resp == llm.DoneResponse || resp.Object == "[DONE]" {
		return false
	}

	if resp.Embedding != nil && len(resp.Embedding.Data) > 0 {
		return true
	}

	if resp.Rerank != nil && len(resp.Rerank.Results) > 0 {
		return true
	}

	if resp.Image != nil && len(resp.Image.Data) > 0 {
		return true
	}

	if resp.ImageStreamEvent != nil && (resp.ImageStreamEvent.B64JSON != "" || resp.ImageStreamEvent.URL != "") {
		return true
	}

	if resp.Video != nil &&
		(resp.Video.ID != "" || resp.Video.Status != "" || resp.Video.VideoURL != "" || resp.Video.Error != nil) {
		return true
	}

	if resp.Compact != nil && len(resp.Compact.Output) > 0 {
		return true
	}

	// A materialized incomplete/cancelled Responses terminal is client-visible
	// even with output=[]. Retrying it would create another provider attempt and
	// discard the explicit terminal state, usage, and incomplete_details. Failed
	// remains stricter: a bare failed status is malformed/empty unless it carries
	// the structured provider Error.
	switch resp.Status {
	case llm.ResponseStatusIncomplete, llm.ResponseStatusCancelled:
		return true
	case llm.ResponseStatusFailed:
		if resp.Error != nil {
			return true
		}
	}
	for index := range resp.Events {
		switch resp.Events[index].Kind {
		case llm.EventKindResponseFailed:
			if resp.Events[index].Error != nil {
				return true
			}
		case llm.EventKindResponseIncomplete, llm.EventKindResponseCancelled:
			// These are also client-visible lifecycle terminals. Retrying them
			// would manufacture another provider attempt after an explicit
			// incomplete/cancelled result rather than preserving its semantics.
			return true
		}
	}

	if resp.Speech != nil && len(resp.Speech.Audio) > 0 {
		return true
	}

	if resp.Transcription != nil && (resp.Transcription.Text != "" || len(resp.Transcription.Raw) > 0) {
		return true
	}

	// Only audio deltas count as content. A bare "speech.audio.done" event with
	// no audio chunks must still be treated as empty so empty-response detection
	// can retry instead of completing a request with audio_bytes=0.
	if resp.SpeechStreamEvent != nil && resp.SpeechStreamEvent.AudioBase64 != "" {
		return true
	}

	if resp.SpeechAudioChunk != nil && len(resp.SpeechAudioChunk.Audio) > 0 {
		return true
	}

	if resp.TranscriptionStreamEvent != nil && (resp.TranscriptionStreamEvent.Delta != "" || resp.TranscriptionStreamEvent.Text != "" || resp.TranscriptionStreamEvent.Type != "") {
		return true
	}

	if resp.Completion != nil {
		for _, choice := range resp.Completion.Choices {
			if choice.Text != "" {
				return true
			}
		}
	}

	for _, choice := range resp.Choices {
		if hasMessageContent(choice.Delta) || hasMessageContent(choice.Message) {
			return true
		}
	}

	// A gateway-owned Responses compaction is a complete, client-visible
	// continuation checkpoint even though it has no display text. Treat only the
	// fully typed opaque output as meaningful; a nil or empty checkpoint still
	// remains empty so retry/empty-response detection cannot bless malformed
	// lifecycle output.
	for index := range resp.Output {
		item := resp.Output[index]
		if item.Kind == llm.ItemKindCompaction && item.Compaction != nil && item.Compaction.EncryptedContent != "" {
			return true
		}
	}

	return false
}

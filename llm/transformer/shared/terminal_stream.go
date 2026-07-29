package shared

import (
	"errors"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

// ErrStreamIncomplete reports a transport EOF after at least one SSE event but
// before the provider protocol emitted its terminal event.
var ErrStreamIncomplete = errors.New("stream ended without terminal event")

// RequireTerminalEvent adds constant-space protocol completion validation to a
// raw provider stream. It never buffers or copies event payloads.
func RequireTerminalEvent(
	stream streams.Stream[*httpclient.StreamEvent],
	isTerminal func(*httpclient.StreamEvent) bool,
) streams.Stream[*httpclient.StreamEvent] {
	return &terminalEventStream{stream: stream, isTerminal: isTerminal}
}

type terminalEventStream struct {
	stream        streams.Stream[*httpclient.StreamEvent]
	isTerminal    func(*httpclient.StreamEvent) bool
	current       *httpclient.StreamEvent
	sawEvent      bool
	sawTerminal   bool
	incompleteErr error
}

func (stream *terminalEventStream) Next() bool {
	if stream.stream.Next() {
		event := stream.stream.Current()
		stream.current = event
		if event != nil && len(event.Data) > 0 {
			stream.sawEvent = true
			if stream.isTerminal != nil && stream.isTerminal(event) {
				stream.sawTerminal = true
			}
		}
		return true
	}
	if stream.stream.Err() == nil && stream.sawEvent && !stream.sawTerminal {
		stream.incompleteErr = ErrStreamIncomplete
	}
	return false
}

func (stream *terminalEventStream) Current() *httpclient.StreamEvent {
	return stream.current
}

func (stream *terminalEventStream) Err() error {
	if stream.incompleteErr != nil {
		return stream.incompleteErr
	}
	return stream.stream.Err()
}

func (stream *terminalEventStream) Close() error {
	return stream.stream.Close()
}

package shared

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

func TestRequireTerminalEvent(t *testing.T) {
	tests := []struct {
		name      string
		events    []*httpclient.StreamEvent
		wantError bool
	}{
		{name: "complete", events: []*httpclient.StreamEvent{{Data: []byte("chunk")}, {Data: []byte("done")}}},
		{name: "incomplete", events: []*httpclient.StreamEvent{{Data: []byte("chunk")}}, wantError: true},
		{name: "empty stream does not replace empty response detection"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			stream := RequireTerminalEvent(streams.SliceStream(test.events), func(event *httpclient.StreamEvent) bool {
				return string(event.Data) == "done"
			})
			_, err := streams.All(stream)
			if test.wantError {
				require.ErrorIs(t, err, ErrStreamIncomplete)
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestRequireTerminalEventPreservesSourceError(t *testing.T) {
	sourceErr := errors.New("source read failed")
	stream := RequireTerminalEvent(&terminalErrorStream{err: sourceErr}, nil)
	_, err := streams.All(stream)
	require.ErrorIs(t, err, sourceErr)
}

type terminalErrorStream struct{ err error }

func (*terminalErrorStream) Next() bool                       { return false }
func (*terminalErrorStream) Current() *httpclient.StreamEvent { return nil }
func (stream *terminalErrorStream) Err() error                { return stream.err }
func (*terminalErrorStream) Close() error                     { return nil }

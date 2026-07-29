package pipeline

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

type advancingCurrentStream struct {
	values []string
	index  int
}

func (s *advancingCurrentStream) Next() bool { return s.index < len(s.values) }

func (s *advancingCurrentStream) Current() string {
	value := s.values[s.index]
	s.index++
	return value
}

func (s *advancingCurrentStream) Err() error   { return nil }
func (s *advancingCurrentStream) Close() error { return nil }

func TestObservedStreamReadsEachUnderlyingCurrentExactlyOnce(t *testing.T) {
	source := &advancingCurrentStream{values: []string{"one", "two", "three", "four"}}
	stream := &observedStream[string]{
		ctx:    context.Background(),
		stream: source,
		stage:  StageClientStream,
		size:   func(value string) int64 { return int64(len(value)) },
	}

	var got []string
	for stream.Next() {
		got = append(got, stream.Current())
	}

	require.Equal(t, source.values, got)
	require.EqualValues(t, len(source.values), stream.events)
}

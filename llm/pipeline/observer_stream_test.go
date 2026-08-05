package pipeline

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type advancingCurrentStream struct {
	values []string
	index  int
}

func TestRecordProviderRoundKeepsOnlyBoundedSafeEvidence(t *testing.T) {
	RecordProviderRound(context.Background(), 1, time.Second, "completed", true)
	trace := newPipelineTrace(&benchmarkObserver{}, "openai/responses")
	ctx := withPipelineTrace(context.Background(), trace)
	RecordProviderRound(ctx, 0, time.Second, "completed", true)
	RecordProviderRound(ctx, 1, time.Second, "", true)
	for index := 1; index <= 7; index++ {
		RecordProviderRound(ctx, uint32(index), time.Duration(index)*time.Millisecond, "completed", index%2 == 0)
	}
	summary := emulationSummary(ctx)
	require.Len(t, summary.ProviderRounds, 6)
	require.EqualValues(t, 1, summary.ProviderRounds[0].RoundIndex)
	require.EqualValues(t, 1, summary.ProviderRounds[0].DurationMillis)
	require.False(t, summary.ProviderRounds[0].ProviderDispatched)
	require.EqualValues(t, 6, summary.ProviderRounds[5].RoundIndex)
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

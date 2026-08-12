package pipeline

import (
	"context"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
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

type conversionRestoreObservationRecorder struct {
	events []Observation
}

func (r *conversionRestoreObservationRecorder) Observe(_ context.Context, event Observation) {
	r.events = append(r.events, event)
}

type conversionRestoreDebugReporter struct {
	trace          *llm.ConversionDebugTrace
	summary        *llm.ConversionTraceSummary
	reportNilDebug bool
}

func (r *conversionRestoreDebugReporter) ConversionDebugFromRequest(*httpclient.Request) (*llm.ConversionDebugTrace, bool) {
	if r == nil {
		return nil, false
	}
	if r.reportNilDebug {
		return nil, true
	}
	if r.trace == nil {
		return nil, false
	}
	return r.trace.Clone(), true
}

func (r *conversionRestoreDebugReporter) ConversionDebugFromResponse(*llm.Response) (*llm.ConversionDebugTrace, bool) {
	return r.ConversionDebugFromRequest(nil)
}

func (r *conversionRestoreDebugReporter) ConversionSummaryFromRequest(*httpclient.Request) (llm.ConversionTraceSummary, bool) {
	if r == nil || r.summary == nil {
		return llm.ConversionTraceSummary{}, false
	}
	return *r.summary, true
}

func (r *conversionRestoreDebugReporter) ConversionSummaryFromResponse(*llm.Response) (llm.ConversionTraceSummary, bool) {
	return r.ConversionSummaryFromRequest(nil)
}

func TestConversionRestoreStreamPublishesCriticalReplacementBeforeEOF(t *testing.T) {
	ctx := context.Background()
	debugCtx := llm.WithConversionDebugTrace(ctx, []byte(t.Name()), llm.MaxConversionDebugActions)
	trace := llm.NewConversionDebugTrace(debugCtx, llm.MaxConversionDebugActions)
	require.NotNil(t, trace)
	for index := 0; index < llm.MaxConversionDebugActions; index++ {
		trace.Append(llm.ConversionActionTrace{
			Action: strconv.Itoa(index), Severity: llm.ConversionSeverityInfo,
		}, nil)
	}
	require.Len(t, trace.Actions, llm.MaxConversionDebugActions)

	recorder := &conversionRestoreObservationRecorder{}
	ctx = withPipelineTrace(ctx, newPipelineTrace(recorder, llm.APIFormatOpenAIResponse))
	stream := &conversionRestoreStream{
		ctx:           ctx,
		stream:        streams.SliceStream([]*llm.Response{{}, {}}),
		debugReporter: &conversionRestoreDebugReporter{trace: trace},
	}
	// The initial bounded trace is emitted on the first item.
	require.True(t, stream.Next())
	require.Len(t, recorder.events, 1)

	// This models a Responses provider stream blocker recorded during restore.
	// Append displaces one info entry but preserves the 64-action slice length.
	trace.Append(llm.ConversionActionTrace{
		Direction: llm.ConversionDirectionStream, ObjectID: "output[0]",
		Stage: llm.ConversionStageStreamRestore, Action: "unknown",
		SemanticClass: "agent_message_provider_output", Severity: llm.ConversionSeverityCritical,
	}, nil)
	require.Len(t, trace.Actions, llm.MaxConversionDebugActions)

	// A client encoder may fail immediately after receiving this second item and
	// never ask for EOF. The restore observation must nevertheless already carry
	// the critical action, and a repeat with no new sequence must not duplicate it.
	require.True(t, stream.Next())
	require.Len(t, recorder.events, 2)
	progress := recorder.events[1]
	require.Equal(t, StageConversionRestore, progress.Stage)
	require.NotNil(t, progress.ConversionDebug)
	require.Len(t, progress.ConversionDebug.Actions, llm.MaxConversionDebugActions)
	lastSeq := uint32(llm.MaxConversionDebugActions + 1)
	found := false
	for index := range progress.ConversionDebug.Actions {
		action := progress.ConversionDebug.Actions[index]
		if action.Seq == lastSeq {
			found = action.Severity == llm.ConversionSeverityCritical && action.SemanticClass == "agent_message_provider_output"
		}
	}
	require.True(t, found, "replacement critical action was not observed before EOF")
	stream.observeProgress()
	require.Len(t, recorder.events, 2, "unchanged trace must not emit duplicate progress")
}

func TestConversionRestoreStreamProgressHandlesUnavailableAndEmptyDebug(t *testing.T) {
	t.Parallel()

	if latestConversionActionSeq(nil) != 0 || latestConversionActionSeq(&llm.ConversionDebugTrace{}) != 0 {
		t.Fatal("empty debug trace must have zero action watermark")
	}
	// No reporter and no trace are both normal before conversion has recorded an
	// observable decision; neither state may emit a spurious restore event.
	stream := &conversionRestoreStream{}
	stream.observeProgress()
	recorder := &conversionRestoreObservationRecorder{}
	ctx := withPipelineTrace(context.Background(), newPipelineTrace(recorder, llm.APIFormatOpenAIResponse))
	stream = &conversionRestoreStream{ctx: ctx, debugReporter: &conversionRestoreDebugReporter{trace: &llm.ConversionDebugTrace{}}}
	stream.observeProgress()
	stream = &conversionRestoreStream{ctx: ctx, debugReporter: &conversionRestoreDebugReporter{reportNilDebug: true}}
	stream.observeProgress()
	require.Empty(t, recorder.events)
}

func TestConversionRestoreStreamProgressAttachesSummary(t *testing.T) {
	t.Parallel()

	debug := llm.NewRequiredConversionDebugTrace(1)
	debug.Append(llm.ConversionActionTrace{Action: "unknown", Severity: llm.ConversionSeverityCritical}, nil)
	summary := llm.ConversionTraceSummary{PlanVersion: 6, Unknown: 1}
	recorder := &conversionRestoreObservationRecorder{}
	ctx := withPipelineTrace(context.Background(), newPipelineTrace(recorder, llm.APIFormatOpenAIResponse))
	stream := &conversionRestoreStream{ctx: ctx, debugReporter: &conversionRestoreDebugReporter{trace: debug, summary: &summary}, reporter: &conversionRestoreDebugReporter{trace: debug, summary: &summary}}
	stream.observeProgress()
	require.Len(t, recorder.events, 1)
	require.NotNil(t, recorder.events[0].Conversion)
	require.Equal(t, summary, *recorder.events[0].Conversion)
}

func TestConversionRestoreStreamProgressSkipsMissingSummary(t *testing.T) {
	t.Parallel()

	debug := llm.NewRequiredConversionDebugTrace(1)
	debug.Append(llm.ConversionActionTrace{Action: "unknown", Severity: llm.ConversionSeverityCritical}, nil)
	recorder := &conversionRestoreObservationRecorder{}
	ctx := withPipelineTrace(context.Background(), newPipelineTrace(recorder, llm.APIFormatOpenAIResponse))
	reporter := &conversionRestoreDebugReporter{trace: debug}
	stream := &conversionRestoreStream{ctx: ctx, debugReporter: reporter, reporter: reporter}
	stream.observeProgress()
	require.Len(t, recorder.events, 1)
	require.Nil(t, recorder.events[0].Conversion)
}

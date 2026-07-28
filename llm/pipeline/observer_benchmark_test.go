package pipeline

import (
	"context"
	"sync/atomic"
	"testing"
)

type benchmarkObserver struct {
	events atomic.Int64
}

func (o *benchmarkObserver) Observe(_ context.Context, _ Observation) {
	o.events.Add(1)
}

func BenchmarkObservationDisabled(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		observeStage(ctx, StageOutboundTransform, observationStart(ctx), nil, observationData{})
	}
}

func BenchmarkObservationEnabled(b *testing.B) {
	observer := &benchmarkObserver{}
	ctx := withPipelineTrace(context.Background(), newPipelineTrace(observer, "openai/chat_completions"))
	traceFromContext(ctx).attempt.Store(1)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		observeStage(ctx, StageOutboundTransform, observationStart(ctx), nil, observationData{
			inputBytes:  128,
			outputBytes: 256,
		})
	}
	b.StopTimer()
	if got := observer.events.Load(); got != int64(b.N) {
		b.Fatalf("observed events = %d, want %d", got, b.N)
	}
}

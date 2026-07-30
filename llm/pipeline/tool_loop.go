package pipeline

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

var ErrCanonicalPassthrough = errors.New("canonical round cannot carry a raw passthrough response")

// CanonicalRoundTripper performs exactly one provider model round. It includes
// target transformation, raw attempt middleware, transport, and provider
// response decoding, but excludes public unified middleware and client wire
// encoding.
type CanonicalRoundTripper interface {
	TargetFormat() llm.APIFormat
	Complete(ctx context.Context, request *llm.Request) (*llm.Response, error)
	Stream(ctx context.Context, request *llm.Request) (streams.Stream[*llm.Response], error)
}

// ToolLoopController owns gateway-internal model rounds. Implementations must
// return only the public canonical response/stream; synthetic gateway calls
// and internal terminal events must be consumed inside the controller.
type ToolLoopController interface {
	Complete(ctx context.Context, request *llm.Request, rounds CanonicalRoundTripper) (*llm.Response, error)
	Stream(ctx context.Context, request *llm.Request, rounds CanonicalRoundTripper) (streams.Stream[*llm.Response], error)
}

type observedEmulationStream struct {
	ctx     context.Context
	stream  streams.Stream[*llm.Response]
	started time.Time
	once    sync.Once
}

func observeEmulationStream(ctx context.Context, started time.Time, stream streams.Stream[*llm.Response]) streams.Stream[*llm.Response] {
	return &observedEmulationStream{ctx: ctx, stream: stream, started: started}
}

func (stream *observedEmulationStream) Next() bool {
	hasNext := stream.stream.Next()
	if !hasNext {
		stream.observe(stream.stream.Err())
	}
	return hasNext
}

func (stream *observedEmulationStream) Current() *llm.Response { return stream.stream.Current() }

func (stream *observedEmulationStream) Err() error { return stream.stream.Err() }

func (stream *observedEmulationStream) Close() error {
	err := stream.stream.Close()
	stream.observe(errors.Join(stream.stream.Err(), err))
	return err
}

func (stream *observedEmulationStream) observe(err error) {
	stream.once.Do(func() {
		observeStage(stream.ctx, StageConversionEmulation, stream.started, err, observationData{emulation: emulationSummary(stream.ctx)})
	})
}

func WithToolLoopController(controller ToolLoopController) Option {
	return func(pipeline *pipeline) {
		pipeline.toolLoopController = controller
	}
}

type canonicalRoundTripper struct {
	pipeline *pipeline
}

func (rounds *canonicalRoundTripper) TargetFormat() llm.APIFormat {
	if rounds == nil || rounds.pipeline == nil || rounds.pipeline.Outbound == nil {
		return ""
	}
	return rounds.pipeline.Outbound.APIFormat()
}

func (rounds *canonicalRoundTripper) Complete(ctx context.Context, request *llm.Request) (*llm.Response, error) {
	if rounds == nil || rounds.pipeline == nil || request == nil {
		return nil, errors.New("canonical round requires pipeline and request")
	}
	prepared, err := rounds.pipeline.prepareProviderRound(ctx, request)
	if err != nil {
		return nil, err
	}
	if request.Stream != nil && *request.Stream {
		round, err := rounds.pipeline.streamRound(ctx, prepared.executor, prepared.request, 0)
		if err != nil {
			return nil, err
		}
		stream := round.publicStream()
		defer stream.Close()
		accumulator := llm.NewCanonicalResponseAccumulator()
		for stream.Next() {
			if err := accumulator.Observe(stream.Current()); err != nil {
				return nil, err
			}
		}
		if err := stream.Err(); err != nil {
			return nil, err
		}
		if !accumulator.IsTerminal() {
			return nil, errors.New("canonical provider stream ended without a terminal event")
		}
		return accumulator.Snapshot(), nil
	}

	round, err := rounds.pipeline.notStreamRound(ctx, prepared.executor, prepared.request)
	if err != nil {
		return nil, err
	}
	if round.passthrough != nil {
		_ = round.passthrough.Close()
		return nil, ErrCanonicalPassthrough
	}
	if rounds.pipeline.emptyResponseDetection && !hasResponseContent(round.response) {
		rounds.pipeline.applyRawErrorResponseMiddlewares(ctx, ErrEmptyResponse)
		return nil, ErrEmptyResponse
	}
	return round.response, nil
}

func (rounds *canonicalRoundTripper) Stream(ctx context.Context, request *llm.Request) (streams.Stream[*llm.Response], error) {
	if rounds == nil || rounds.pipeline == nil || request == nil {
		return nil, errors.New("canonical round requires pipeline and request")
	}
	prepared, err := rounds.pipeline.prepareProviderRound(ctx, request)
	if err != nil {
		return nil, err
	}
	round, err := rounds.pipeline.streamRound(ctx, prepared.executor, prepared.request, rounds.pipeline.streamFirstEventTimeout)
	if err != nil {
		return nil, fmt.Errorf("failed to stream canonical provider round: %w", err)
	}
	return round.publicStream(), nil
}

func isToolLoopRequest(request *llm.Request) bool {
	return request != nil && (request.RequestType == "" || request.RequestType == llm.RequestTypeChat)
}

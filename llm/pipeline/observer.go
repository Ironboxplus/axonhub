package pipeline

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

// Stage is a stable, bounded-cardinality pipeline lifecycle boundary.
// Stages describe processing mechanics only; provider, model, URL, payload,
// header, and credential values are deliberately absent.
type Stage string

const (
	StageInboundTransform          Stage = "inbound_transform"
	StageInboundMiddleware         Stage = "inbound_middleware"
	StageOutboundTransform         Stage = "outbound_transform"
	StageOutboundMiddleware        Stage = "outbound_middleware"
	StageProviderExchange          Stage = "provider_exchange"
	StageProviderStream            Stage = "provider_stream"
	StageRawResponseMiddleware     Stage = "raw_response_middleware"
	StageRawResponsePassthrough    Stage = "raw_response_passthrough"
	StageProviderResponseTransform Stage = "provider_response_transform"
	StageUnifiedResponseMiddleware Stage = "unified_response_middleware"
	StageResponseValidation        Stage = "response_validation"
	StageClientResponseTransform   Stage = "client_response_transform"
	StageClientResponseMiddleware  Stage = "client_response_middleware"
	StageProviderStreamTransform   Stage = "provider_stream_transform"
	StageUnifiedStream             Stage = "unified_stream"
	StageUnifiedStreamMiddleware   Stage = "unified_stream_middleware"
	StageStreamPrelude             Stage = "stream_prelude"
	StageClientStreamTransform     Stage = "client_stream_transform"
	StageClientStreamMiddleware    Stage = "client_stream_middleware"
	StageClientStream              Stage = "client_stream"
	StageStreamAggregate           Stage = "stream_aggregate"
	StageRetryDecision             Stage = "retry_decision"
)

type Outcome string

const (
	OutcomeSuccess  Outcome = "success"
	OutcomeFailure  Outcome = "failure"
	OutcomeCanceled Outcome = "canceled"
	OutcomeRetry    Outcome = "retry"
)

type ErrorClass string

const (
	ErrorClassNone       ErrorClass = ""
	ErrorClassCanceled   ErrorClass = "canceled"
	ErrorClassDeadline   ErrorClass = "deadline"
	ErrorClassTimeout    ErrorClass = "timeout"
	ErrorClassUpstream   ErrorClass = "upstream"
	ErrorClassTransform  ErrorClass = "transform"
	ErrorClassMiddleware ErrorClass = "middleware"
	ErrorClassEmpty      ErrorClass = "empty_response"
	ErrorClassStream     ErrorClass = "stream"
	ErrorClassUnknown    ErrorClass = "unknown"
)

type RetryKind string

const (
	RetryKindNone          RetryKind = ""
	RetryKindSameChannel   RetryKind = "same_channel"
	RetryKindSwitchChannel RetryKind = "switch_channel"
)

// Observation is the privacy-safe event contract emitted by Pipeline.
//
// It intentionally contains no request IDs, model names, URLs, headers,
// payloads, error text, or arbitrary attributes. Product integrations can
// correlate it with their own request/attempt trace through context while
// keeping secrets outside Axon's generic pipeline contract.
type Observation struct {
	Stage             Stage           `json:"stage"`
	Outcome           Outcome         `json:"outcome"`
	ErrorClass        ErrorClass      `json:"error_class,omitempty"`
	RetryKind         RetryKind       `json:"retry_kind,omitempty"`
	Attempt           int             `json:"attempt"`
	Duration          time.Duration   `json:"duration"`
	RequestType       llm.RequestType `json:"request_type,omitempty"`
	InboundAPIFormat  llm.APIFormat   `json:"inbound_api_format,omitempty"`
	OutboundAPIFormat llm.APIFormat   `json:"outbound_api_format,omitempty"`
	Stream            bool            `json:"stream"`
	StatusCode        int             `json:"status_code,omitempty"`
	InputBytes        int64           `json:"input_bytes,omitempty"`
	OutputBytes       int64           `json:"output_bytes,omitempty"`
	Events            int64           `json:"events,omitempty"`
}

// Observer consumes privacy-safe pipeline observations. Implementations must
// return quickly and must be safe for concurrent calls because terminal stream
// observations are emitted by the stream consumer. Pipeline contains observer
// panics so optional diagnostics can never fail provider traffic.
type Observer interface {
	Observe(ctx context.Context, event Observation)
}

type pipelineTraceContextKey struct{}

type pipelineTrace struct {
	observer          Observer
	attempt           atomic.Int64
	requestType       llm.RequestType
	inboundAPIFormat  llm.APIFormat
	outboundAPIFormat llm.APIFormat
	stream            bool
}

func newPipelineTrace(observer Observer, outboundAPIFormat llm.APIFormat) *pipelineTrace {
	if observer == nil {
		return nil
	}
	return &pipelineTrace{
		observer:          observer,
		outboundAPIFormat: outboundAPIFormat,
	}
}

func requestBodySize(request *httpclient.Request) int64 {
	if request == nil {
		return 0
	}
	if request.BodySource != nil {
		if size := request.BodySource.Size(); size > 0 {
			return size
		}
		return 0
	}
	return int64(len(request.Body))
}

func withPipelineTrace(ctx context.Context, trace *pipelineTrace) context.Context {
	if trace == nil {
		return ctx
	}
	return context.WithValue(ctx, pipelineTraceContextKey{}, trace)
}

func traceFromContext(ctx context.Context) *pipelineTrace {
	if ctx == nil {
		return nil
	}
	trace, _ := ctx.Value(pipelineTraceContextKey{}).(*pipelineTrace)
	return trace
}

func (t *pipelineTrace) setRequest(request *llm.Request) {
	if t == nil || request == nil {
		return
	}
	t.requestType = request.RequestType
	t.inboundAPIFormat = request.APIFormat
	t.stream = request.Stream != nil && *request.Stream
}

func (t *pipelineTrace) setOutboundAPIFormat(apiFormat llm.APIFormat) {
	if t == nil {
		return
	}
	t.outboundAPIFormat = apiFormat
}

func observationStart(ctx context.Context) time.Time {
	if traceFromContext(ctx) == nil {
		return time.Time{}
	}
	return time.Now()
}

type observationData struct {
	statusCode  int
	inputBytes  int64
	outputBytes int64
	events      int64
	retryKind   RetryKind
	outcome     Outcome
}

func observeStage(ctx context.Context, stage Stage, startedAt time.Time, err error, data observationData) {
	trace := traceFromContext(ctx)
	if trace == nil || trace.observer == nil {
		return
	}

	outcome := data.outcome
	if outcome == "" {
		switch {
		case err == nil:
			outcome = OutcomeSuccess
		case errors.Is(err, context.Canceled):
			outcome = OutcomeCanceled
		default:
			outcome = OutcomeFailure
		}
	}
	duration := time.Duration(0)
	if !startedAt.IsZero() {
		duration = time.Since(startedAt)
	}
	statusCode := data.statusCode
	if statusCode == 0 {
		if httpErr, ok := errors.AsType[*httpclient.Error](err); ok {
			statusCode = httpErr.StatusCode
		}
	}
	event := Observation{
		Stage:             stage,
		Outcome:           outcome,
		ErrorClass:        classifyObservationError(stage, err),
		RetryKind:         data.retryKind,
		Attempt:           int(trace.attempt.Load()),
		Duration:          duration,
		RequestType:       trace.requestType,
		InboundAPIFormat:  trace.inboundAPIFormat,
		OutboundAPIFormat: trace.outboundAPIFormat,
		Stream:            trace.stream,
		StatusCode:        statusCode,
		InputBytes:        data.inputBytes,
		OutputBytes:       data.outputBytes,
		Events:            data.events,
	}
	if event.Outcome == OutcomeCanceled && event.ErrorClass == ErrorClassNone {
		event.ErrorClass = ErrorClassCanceled
	}
	safeObserve(ctx, trace.observer, event)
}

func safeObserve(ctx context.Context, observer Observer, event Observation) {
	defer func() {
		if recover() != nil {
			// Do not include the recovered value: an observer panic can contain a
			// payload or credential supplied by a downstream sink.
			slog.ErrorContext(ctx, "pipeline observer panicked")
		}
	}()
	observer.Observe(ctx, event)
}

func classifyObservationError(stage Stage, err error) ErrorClass {
	if err == nil {
		return ErrorClassNone
	}
	switch {
	case errors.Is(err, context.Canceled):
		return ErrorClassCanceled
	case errors.Is(err, context.DeadlineExceeded):
		return ErrorClassDeadline
	case errors.Is(err, ErrStreamFirstEventTimeout), errors.Is(err, ErrNonStreamResponseTimeout):
		return ErrorClassTimeout
	case errors.Is(err, ErrEmptyResponse), errors.Is(err, ErrEmptyStreamChunks), errors.Is(err, ErrEmptyAggregatedBody):
		return ErrorClassEmpty
	}
	if _, ok := errors.AsType[*httpclient.Error](err); ok {
		return ErrorClassUpstream
	}
	switch stage {
	case StageInboundTransform, StageOutboundTransform, StageProviderResponseTransform,
		StageClientResponseTransform, StageProviderStreamTransform, StageClientStreamTransform,
		StageStreamAggregate:
		return ErrorClassTransform
	case StageInboundMiddleware, StageOutboundMiddleware, StageRawResponseMiddleware,
		StageUnifiedResponseMiddleware, StageClientResponseMiddleware,
		StageUnifiedStreamMiddleware, StageClientStreamMiddleware:
		return ErrorClassMiddleware
	case StageProviderStream, StageClientStream, StageStreamPrelude:
		return ErrorClassStream
	case StageProviderExchange:
		return ErrorClassUpstream
	default:
		return ErrorClassUnknown
	}
}

type observedStream[T any] struct {
	ctx       context.Context
	stream    streams.Stream[T]
	stage     Stage
	startedAt time.Time
	size      func(T) int64
	events    int64
	bytes     int64
	finished  bool
	once      sync.Once
}

func observeStream[T any](ctx context.Context, stage Stage, stream streams.Stream[T], size func(T) int64) streams.Stream[T] {
	if stream == nil || traceFromContext(ctx) == nil {
		return stream
	}
	return &observedStream[T]{
		ctx:       ctx,
		stream:    stream,
		stage:     stage,
		startedAt: time.Now(),
		size:      size,
	}
}

func (s *observedStream[T]) Next() bool {
	if s.stream.Next() {
		s.events++
		if s.size != nil {
			s.bytes += s.size(s.stream.Current())
		}
		return true
	}
	s.finished = true
	s.finish(s.stream.Err(), Outcome(""))
	return false
}

func (s *observedStream[T]) Current() T { return s.stream.Current() }
func (s *observedStream[T]) Err() error { return s.stream.Err() }

func (s *observedStream[T]) Close() error {
	err := s.stream.Close()
	outcome := OutcomeCanceled
	if s.finished {
		outcome = Outcome("")
	}
	s.finish(err, outcome)
	return err
}

func (s *observedStream[T]) finish(err error, outcome Outcome) {
	s.once.Do(func() {
		observeStage(s.ctx, s.stage, s.startedAt, err, observationData{
			outputBytes: s.bytes,
			events:      s.events,
			outcome:     outcome,
		})
	})
}

func rawStreamEventSize(event *httpclient.StreamEvent) int64 {
	if event == nil {
		return 0
	}
	return int64(len(event.LastEventID) + len(event.Type) + len(event.Data))
}

func unifiedStreamEventSize(response *llm.Response) int64 {
	if response == nil {
		return 0
	}
	var size int64
	if response.ImageStreamEvent != nil {
		size += int64(len(response.ImageStreamEvent.Type) + len(response.ImageStreamEvent.B64JSON) + len(response.ImageStreamEvent.URL))
	}
	for _, choice := range response.Choices {
		for _, message := range []*llm.Message{choice.Message, choice.Delta} {
			if message == nil {
				continue
			}
			if message.Content.Content != nil {
				size += int64(len(*message.Content.Content))
			}
			for _, part := range message.Content.MultipleContent {
				if part.Text != nil {
					size += int64(len(*part.Text))
				}
			}
			size += int64(len(message.Refusal))
			if message.ReasoningContent != nil {
				size += int64(len(*message.ReasoningContent))
			}
			if message.Reasoning != nil {
				size += int64(len(*message.Reasoning))
			}
			for _, call := range message.ToolCalls {
				size += int64(len(call.Function.Arguments))
				if call.ResponseCustomToolCall != nil {
					size += int64(len(call.ResponseCustomToolCall.Input))
				}
			}
		}
	}
	return size
}

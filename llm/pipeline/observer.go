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
	"github.com/looplj/axonhub/llm/transformer"
)

// Stage is a stable, bounded-cardinality pipeline lifecycle boundary.
// Stages describe processing mechanics only; provider, model, URL, payload,
// header, and credential values are deliberately absent.
type Stage string

const (
	StageInboundTransform          Stage = "inbound_transform"
	StageInboundMiddleware         Stage = "inbound_middleware"
	StageConversionPlan            Stage = "conversion_plan"
	StageConversionEmulation       Stage = "conversion_emulation"
	StageOutboundTransform         Stage = "outbound_transform"
	StageOutboundMiddleware        Stage = "outbound_middleware"
	StageProviderExchange          Stage = "provider_exchange"
	StageProviderStream            Stage = "provider_stream"
	StageRawResponseMiddleware     Stage = "raw_response_middleware"
	StageRawResponsePassthrough    Stage = "raw_response_passthrough"
	StageProviderResponseTransform Stage = "provider_response_transform"
	StageConversionRestore         Stage = "conversion_restore"
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
	ErrorClassEmulation  ErrorClass = "emulation"
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
	Stage             Stage                       `json:"stage"`
	Outcome           Outcome                     `json:"outcome"`
	ErrorClass        ErrorClass                  `json:"error_class,omitempty"`
	RetryKind         RetryKind                   `json:"retry_kind,omitempty"`
	Attempt           int                         `json:"attempt"`
	Duration          time.Duration               `json:"duration"`
	RequestType       llm.RequestType             `json:"request_type,omitempty"`
	InboundAPIFormat  llm.APIFormat               `json:"inbound_api_format,omitempty"`
	OutboundAPIFormat llm.APIFormat               `json:"outbound_api_format,omitempty"`
	Stream            bool                        `json:"stream"`
	StatusCode        int                         `json:"status_code,omitempty"`
	InputBytes        int64                       `json:"input_bytes,omitempty"`
	OutputBytes       int64                       `json:"output_bytes,omitempty"`
	Events            int64                       `json:"events,omitempty"`
	Conversion        *llm.ConversionTraceSummary `json:"conversion,omitempty"`
	ConversionDebug   *llm.ConversionDebugTrace   `json:"conversion_debug,omitempty"`
	Emulation         *llm.EmulationTraceSummary  `json:"emulation,omitempty"`
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
	observer                  Observer
	attempt                   atomic.Int64
	requestType               llm.RequestType
	inboundAPIFormat          llm.APIFormat
	outboundAPIFormat         llm.APIFormat
	stream                    bool
	emulationRounds           atomic.Uint32
	emulationCalls            atomic.Uint32
	emulationApprovals        atomic.Uint32
	emulationFailures         atomic.Uint32
	emulationLimitHit         atomic.Bool
	hostedWebSearchCalls      atomic.Uint32
	hostedWebFetchCalls       atomic.Uint32
	hostedFileSearchCalls     atomic.Uint32
	hostedCodeCalls           atomic.Uint32
	hostedShellCalls          atomic.Uint32
	hostedComputerCalls       atomic.Uint32
	hostedImageCalls          atomic.Uint32
	hostedToolSearchCalls     atomic.Uint32
	hostedOtherCalls          atomic.Uint32
	constraintCompiles        atomic.Uint32
	constraintValidations     atomic.Uint32
	constraintViolations      atomic.Uint32
	constraintRetries         atomic.Uint32
	constraintFallbacks       atomic.Uint32
	constraintCompileNanos    atomic.Int64
	constraintValidateNanos   atomic.Int64
	webSocketRequests         atomic.Uint32
	webSocketConnections      atomic.Uint32
	webSocketReuses           atomic.Uint32
	webSocketReconnects       atomic.Uint32
	webSocketIncrementalSends atomic.Uint32
	webSocketFullContextSends atomic.Uint32
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
	statusCode      int
	inputBytes      int64
	outputBytes     int64
	events          int64
	retryKind       RetryKind
	outcome         Outcome
	conversion      *llm.ConversionTraceSummary
	conversionDebug *llm.ConversionDebugTrace
	emulation       *llm.EmulationTraceSummary
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
	conversion := data.conversion
	if conversion != nil {
		cloned := *conversion
		attachWebSocketSummary(trace, &cloned)
		conversion = &cloned
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
		Conversion:        conversion,
		ConversionDebug:   data.conversionDebug,
		Emulation:         data.emulation,
	}
	if event.Outcome == OutcomeCanceled && event.ErrorClass == ErrorClassNone {
		event.ErrorClass = ErrorClassCanceled
	}
	safeObserve(ctx, trace.observer, event)
}

// RecordEmulationRound updates request-scoped, fixed-cardinality counters for
// a gateway-owned model round. It is a no-op when observation is disabled and
// never accepts payload-bearing values.
func RecordEmulationRound(ctx context.Context, toolCalls, approvals uint32) {
	trace := traceFromContext(ctx)
	if trace == nil {
		return
	}
	trace.emulationRounds.Add(1)
	trace.emulationCalls.Add(toolCalls)
	trace.emulationApprovals.Add(approvals)
}

func RecordEmulationFailure(ctx context.Context) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.emulationFailures.Add(1)
	}
}

func RecordEmulationLimit(ctx context.Context) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.emulationLimitHit.Store(true)
	}
}

// RecordHostedExecution records one gateway-owned executor attempt using a
// closed, fixed-cardinality capability set. It deliberately accepts no tool
// name, call ID, arguments, result, endpoint, or credential. The function is a
// no-op when observation is disabled and is called once per execution, never
// once per streaming event.
func RecordHostedExecution(ctx context.Context, kind llm.ToolKind) {
	trace := traceFromContext(ctx)
	if trace == nil {
		return
	}
	switch kind {
	case llm.ToolKindWebSearch:
		trace.hostedWebSearchCalls.Add(1)
	case llm.ToolKindWebFetch:
		trace.hostedWebFetchCalls.Add(1)
	case llm.ToolKindFileSearch:
		trace.hostedFileSearchCalls.Add(1)
	case llm.ToolKindCodeInterpreter, llm.ToolKindCodeExecution:
		trace.hostedCodeCalls.Add(1)
	case llm.ToolKindShell, llm.ToolKindLocalShell:
		trace.hostedShellCalls.Add(1)
	case llm.ToolKindComputer:
		trace.hostedComputerCalls.Add(1)
	case llm.ToolKindImageGeneration:
		trace.hostedImageCalls.Add(1)
	case llm.ToolKindToolSearch:
		trace.hostedToolSearchCalls.Add(1)
	default:
		trace.hostedOtherCalls.Add(1)
	}
}

// RecordCustomConstraintCompile records fixed-size timing for one grammar
// compilation. It deliberately accepts no grammar or tool identity.
func RecordCustomConstraintCompile(ctx context.Context, duration time.Duration) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.constraintCompiles.Add(1)
		trace.constraintCompileNanos.Add(max(duration.Nanoseconds(), 0))
	}
}

// RecordCustomConstraintValidation records one full-input grammar decision.
// A violation is expected correction-loop control flow, not an engine failure.
func RecordCustomConstraintValidation(ctx context.Context, duration time.Duration, valid bool) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.constraintValidations.Add(1)
		trace.constraintValidateNanos.Add(max(duration.Nanoseconds(), 0))
		if !valid {
			trace.constraintViolations.Add(1)
		}
	}
}

func RecordCustomConstraintRetry(ctx context.Context) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.constraintRetries.Add(1)
	}
}

func RecordCustomConstraintFallback(ctx context.Context) {
	if trace := traceFromContext(ctx); trace != nil {
		trace.constraintFallbacks.Add(1)
	}
}

// RecordResponsesWebSocketRequest records one prepared response.create send.
// It deliberately accepts only booleans, so session IDs, scopes, URLs, headers,
// previous response IDs, and input payloads cannot enter the trace contract.
// The function is a no-op when the caller did not configure an Observer.
func RecordResponsesWebSocketRequest(ctx context.Context, reused, reconnected, incremental, fullContext bool) {
	trace := traceFromContext(ctx)
	if trace == nil {
		return
	}
	trace.webSocketRequests.Add(1)
	if reused {
		trace.webSocketReuses.Add(1)
	} else {
		trace.webSocketConnections.Add(1)
	}
	if reconnected {
		trace.webSocketReconnects.Add(1)
	}
	if incremental {
		trace.webSocketIncrementalSends.Add(1)
	}
	if fullContext {
		trace.webSocketFullContextSends.Add(1)
	}
}

func attachWebSocketSummary(trace *pipelineTrace, summary *llm.ConversionTraceSummary) {
	if trace == nil || summary == nil {
		return
	}
	summary.WebSocketRequests = trace.webSocketRequests.Load()
	summary.WebSocketConnections = trace.webSocketConnections.Load()
	summary.WebSocketReuses = trace.webSocketReuses.Load()
	summary.WebSocketReconnects = trace.webSocketReconnects.Load()
	summary.WebSocketIncrementalSends = trace.webSocketIncrementalSends.Load()
	summary.WebSocketFullContextSends = trace.webSocketFullContextSends.Load()
}

func emulationSummary(ctx context.Context) *llm.EmulationTraceSummary {
	trace := traceFromContext(ctx)
	if trace == nil {
		return nil
	}
	return &llm.EmulationTraceSummary{
		InternalRounds:                trace.emulationRounds.Load(),
		ToolCalls:                     trace.emulationCalls.Load(),
		Approvals:                     trace.emulationApprovals.Load(),
		Failures:                      trace.emulationFailures.Load(),
		LimitHit:                      trace.emulationLimitHit.Load(),
		HostedWebSearchCalls:          trace.hostedWebSearchCalls.Load(),
		HostedWebFetchCalls:           trace.hostedWebFetchCalls.Load(),
		HostedFileSearchCalls:         trace.hostedFileSearchCalls.Load(),
		HostedCodeCalls:               trace.hostedCodeCalls.Load(),
		HostedShellCalls:              trace.hostedShellCalls.Load(),
		HostedComputerCalls:           trace.hostedComputerCalls.Load(),
		HostedImageCalls:              trace.hostedImageCalls.Load(),
		HostedToolSearchCalls:         trace.hostedToolSearchCalls.Load(),
		HostedOtherCalls:              trace.hostedOtherCalls.Load(),
		CustomConstraintCompiles:      trace.constraintCompiles.Load(),
		CustomConstraintValidations:   trace.constraintValidations.Load(),
		CustomConstraintViolations:    trace.constraintViolations.Load(),
		CustomConstraintRetries:       trace.constraintRetries.Load(),
		CustomConstraintFallbacks:     trace.constraintFallbacks.Load(),
		CustomConstraintCompileNanos:  trace.constraintCompileNanos.Load(),
		CustomConstraintValidateNanos: trace.constraintValidateNanos.Load(),
	}
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
	case StageConversionEmulation:
		return ErrorClassEmulation
	case StageInboundTransform, StageConversionPlan, StageOutboundTransform, StageProviderResponseTransform,
		StageConversionRestore,
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
	current   T
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

type conversionRestoreStream struct {
	ctx           context.Context
	stream        streams.Stream[*llm.Response]
	reporter      conversionSummaryReporter
	debugReporter conversionDebugReporter
	request       *httpclient.Request
	current       *llm.Response
	once          sync.Once
}

func observeConversionRestoreStream(
	ctx context.Context,
	outbound transformer.Outbound,
	request *httpclient.Request,
	stream streams.Stream[*llm.Response],
) streams.Stream[*llm.Response] {
	if stream == nil || traceFromContext(ctx) == nil {
		return stream
	}
	reporter, summaryOK := outboundCapability[conversionSummaryReporter](outbound)
	debugReporter, debugOK := outboundCapability[conversionDebugReporter](outbound)
	if !summaryOK && !debugOK {
		return stream
	}
	return &conversionRestoreStream{ctx: ctx, stream: stream, reporter: reporter, debugReporter: debugReporter, request: request}
}

func (s *conversionRestoreStream) Next() bool {
	if s.stream.Next() {
		s.current = s.stream.Current()
		return true
	}
	s.current = nil
	s.finish(s.stream.Err())
	return false
}

func (s *conversionRestoreStream) Current() *llm.Response { return s.current }
func (s *conversionRestoreStream) Err() error             { return s.stream.Err() }

func (s *conversionRestoreStream) Close() error {
	err := s.stream.Close()
	s.finish(err)
	return err
}

func (s *conversionRestoreStream) finish(err error) {
	s.once.Do(func() {
		data := observationData{}
		if s.reporter != nil {
			if summary, ok := s.reporter.ConversionSummaryFromRequest(s.request); ok {
				data.conversion = &summary
			}
		}
		if s.debugReporter != nil {
			if debug, ok := s.debugReporter.ConversionDebugFromRequest(s.request); ok {
				data.conversionDebug = debug
			}
		}
		if data.conversion != nil || data.conversionDebug != nil {
			observeStage(s.ctx, StageConversionRestore, time.Time{}, err, data)
		}
	})
}

func (s *observedStream[T]) Next() bool {
	if s.stream.Next() {
		s.current = s.stream.Current()
		s.events++
		if s.size != nil {
			s.bytes += s.size(s.current)
		}
		return true
	}
	var zero T
	s.current = zero
	s.finished = true
	s.finish(s.stream.Err(), Outcome(""))
	return false
}

func (s *observedStream[T]) Current() T { return s.current }
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

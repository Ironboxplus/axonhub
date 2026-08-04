package conversion

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
)

const (
	requestSessionMetadataKey  = "axon_conversion_session_v1"
	responseSummaryMetadataKey = "axon_conversion_summary_v1"
	responseDebugMetadataKey   = "axon_conversion_debug_v1"
)

// Outbound adds one provider-neutral semantic conversion boundary around an
// existing protocol encoder/decoder. It has no request-local mutable fields;
// the attempt session is carried by the concrete outbound HTTP request, which
// is already passed back to both non-streaming and streaming response paths.
type Outbound struct {
	wrapped      transformer.Outbound
	planner      *Planner
	continuation ContinuationBinding
}

var _ transformer.Outbound = (*Outbound)(nil)
var _ transformer.OutboundWrapper = (*Outbound)(nil)

func NewOutbound(wrapped transformer.Outbound, options ...OutboundOption) *Outbound {
	outbound := &Outbound{
		wrapped:      wrapped,
		planner:      NewPlanner(),
		continuation: DefaultArgumentContinuity,
	}
	for _, option := range options {
		option(outbound)
	}
	return outbound
}

// WithCapabilityProfile binds both candidate preflight and the actual
// attempt-local transform to the same effective channel capabilities. Hosts
// should merge protocol defaults and registered emulators once, then pass the
// resulting profile here instead of running a separate planner.
func WithCapabilityProfile(profile CapabilityProfile) OutboundOption {
	return func(outbound *Outbound) {
		outbound.planner = NewPlannerWithProfile(profile)
	}
}

// Preflight proves that this concrete target has a complete plan before any
// provider encoder is invoked. Orchestrators use it while filtering channel
// candidates; TransformRequest repeats the plan on the attempt-local clone.
func (o *Outbound) Preflight(request *llm.Request) (*Plan, error) {
	plan, err := o.planner.Plan(request, o.wrapped.APIFormat())
	if err == nil && plan != nil {
		err = plan.Validate()
	}
	return plan, err
}

func (o *Outbound) APIFormat() llm.APIFormat {
	return o.wrapped.APIFormat()
}

func (o *Outbound) UnwrapOutbound() transformer.Outbound {
	return o.wrapped
}

func (o *Outbound) TransformRequest(ctx context.Context, request *llm.Request) (*httpclient.Request, error) {
	trace := llm.ConversionTraceEnabled(ctx)
	prepared, plan, err := o.planner.preparePlan(request, o.wrapped.APIFormat(), trace)
	if plan != nil {
		plan.Debug = buildConversionDebugTrace(ctx, plan.Actions)
	}
	if err != nil {
		return nil, err
	}
	projected, err := projectCanonical(prepared, o.wrapped.APIFormat())
	if err != nil {
		return nil, err
	}
	projected = projectCompactEmulationInput(projected, plan)
	resolveContinuation(ctx, projected, o.continuation)
	lowered, session, err := lower(projected, plan, trace)
	if err != nil {
		return nil, err
	}
	lowered, err = prepareCompactEmulation(lowered, session)
	if err != nil {
		return nil, err
	}
	httpRequest, err := o.wrapped.TransformRequest(ctx, lowered)
	if err != nil {
		return nil, err
	}
	session.continuation = o.continuation
	setRequestSession(httpRequest, session)
	return httpRequest, nil
}

func normalizeRequestControlRequest(request *llm.Request) (*llm.Request, []llm.RequestControlAdjustment, error) {
	if request == nil {
		return nil, nil, nil
	}
	input, adjustments, err := llm.NormalizeRequestControls(request.Input)
	if err != nil || len(adjustments) == 0 {
		return request, adjustments, err
	}
	prepared := request.Clone()
	prepared.Input = input
	return prepared, adjustments, nil
}

func appendRequestControlActions(plan *Plan, adjustments []llm.RequestControlAdjustment) {
	if plan == nil || len(adjustments) == 0 {
		return
	}
	for index := range adjustments {
		adjustment := &adjustments[index]
		plan.Actions = append(plan.Actions, Action{
			Ref:             ObjectRef{Kind: ObjectRequestControl, ToolIndex: -1, ItemIndex: adjustment.FromIndex, ContentIndex: -1, MessageIndex: -1, ToolCallIndex: -1},
			DestinationPath: fmt.Sprintf("input[%d]", adjustment.ToIndex),
			Kind:            ActionLower, Strategy: StrategyRequestControl, Reason: ReasonProtocolConstraint, Reversible: true,
		})
	}
	planNanos := plan.Summary.PlanNanos
	plan.Summary = summarizePlan(plan.Source, plan.Target, plan.Actions, time.Time{})
	plan.Summary.PlanNanos = planNanos
}

func (o *Outbound) TransformResponse(ctx context.Context, response *httpclient.Response) (*llm.Response, error) {
	unified, err := o.wrapped.TransformResponse(ctx, response)
	if err != nil {
		return nil, err
	}
	session := sessionFromResponse(response)
	if session != nil && session.continuation == nil {
		session.continuation = o.continuation
	}
	unified = RestoreResponseContext(ctx, unified, session)
	return restoreCompactEmulation(unified, session), nil
}

func (o *Outbound) TransformStream(
	ctx context.Context,
	request *httpclient.Request,
	stream streams.Stream[*httpclient.StreamEvent],
) (streams.Stream[*llm.Response], error) {
	unified, err := o.wrapped.TransformStream(ctx, request, stream)
	if err != nil {
		return nil, err
	}
	session := SessionFromRequest(request)
	if session == nil {
		return unified, nil
	}
	if session.continuation == nil {
		session.continuation = o.continuation
	}
	restorer := newStreamRestorerContext(ctx, session)
	// Restoration is stateful. Map evaluates its mapper from Current(), and a
	// pipeline observer plus the source encoder may read Current more than once
	// for the same Next call. Cache the restored value in MapErr.Next so every
	// canonical chunk advances the ledger exactly once.
	restored := streams.MapErr(unified, func(response *llm.Response) (*llm.Response, error) {
		return restorer.restore(response), nil
	})
	return &observedConversionStream{stream: restored, session: session}, nil
}

type observedConversionStream struct {
	stream   streams.Stream[*llm.Response]
	session  *Session
	recorded bool
}

func (stream *observedConversionStream) Next() bool {
	hasNext := stream.stream.Next()
	if hasNext {
		response := stream.stream.Current()
		stream.recordTerminal(response)
	} else {
		stream.recordError()
	}
	return hasNext
}

func (stream *observedConversionStream) Current() *llm.Response { return stream.stream.Current() }

func (stream *observedConversionStream) Err() error {
	stream.recordError()
	return stream.stream.Err()
}

func (stream *observedConversionStream) Close() error { return stream.stream.Close() }

func (stream *observedConversionStream) recordTerminal(response *llm.Response) {
	if stream == nil || stream.session == nil || response == nil {
		return
	}
	for index := range response.Events {
		stream.session.recordTerminalEvent(response.Events[index].Kind)
	}
}

func (stream *observedConversionStream) recordError() {
	if stream == nil || stream.recorded || stream.stream == nil {
		return
	}
	stream.recorded = true
	var violation *llm.StreamInvariantError
	if errors.As(stream.stream.Err(), &violation) {
		stream.session.addStreamViolation(violation)
	}
}

func (o *Outbound) TransformError(ctx context.Context, err *httpclient.Error) *llm.ResponseError {
	return o.wrapped.TransformError(ctx, err)
}

func (o *Outbound) AggregateStreamChunks(
	ctx context.Context,
	request *httpclient.Request,
	chunks []*httpclient.StreamEvent,
) ([]byte, llm.ResponseMeta, error) {
	return o.wrapped.AggregateStreamChunks(ctx, request, chunks)
}

func setRequestSession(request *httpclient.Request, session *Session) {
	if request == nil || session == nil {
		return
	}
	if request.TransformerMetadata == nil {
		request.TransformerMetadata = make(map[string]any, 1)
	}
	request.TransformerMetadata[requestSessionMetadataKey] = session
}

func SessionFromRequest(request *httpclient.Request) *Session {
	if request == nil || request.TransformerMetadata == nil {
		return nil
	}
	session, _ := request.TransformerMetadata[requestSessionMetadataKey].(*Session)
	return session
}

func SummaryFromRequest(request *httpclient.Request) (llm.ConversionTraceSummary, bool) {
	session := SessionFromRequest(request)
	if session == nil {
		return llm.ConversionTraceSummary{}, false
	}
	return session.Summary(), true
}

func (o *Outbound) ConversionSummaryFromRequest(request *httpclient.Request) (llm.ConversionTraceSummary, bool) {
	return SummaryFromRequest(request)
}

func DebugTraceFromRequest(request *httpclient.Request) (*llm.ConversionDebugTrace, bool) {
	session := SessionFromRequest(request)
	if session == nil {
		return nil, false
	}
	trace := session.DebugTrace()
	return trace, trace != nil
}

func (o *Outbound) ConversionDebugFromRequest(request *httpclient.Request) (*llm.ConversionDebugTrace, bool) {
	return DebugTraceFromRequest(request)
}

func sessionFromResponse(response *httpclient.Response) *Session {
	if response == nil {
		return nil
	}
	return SessionFromRequest(response.Request)
}

func setResponseSummary(response *llm.Response, summary llm.ConversionTraceSummary) {
	// DoneResponse is a process-wide transport sentinel. It is immutable and
	// must never carry request-local state. Streaming summaries are read from
	// the request-local Session when the stream finishes, so attaching a
	// summary to each chunk (and especially to this sentinel) is unnecessary.
	if response == nil || response == llm.DoneResponse {
		return
	}
	if response.TransformerMetadata == nil {
		response.TransformerMetadata = make(map[string]any, 1)
	}
	response.TransformerMetadata[responseSummaryMetadataKey] = summary
}

func setResponseDebug(response *llm.Response, trace *llm.ConversionDebugTrace) {
	if response == nil || response == llm.DoneResponse || trace == nil {
		return
	}
	if response.TransformerMetadata == nil {
		response.TransformerMetadata = make(map[string]any, 1)
	}
	response.TransformerMetadata[responseDebugMetadataKey] = trace.Clone()
}

func SummaryFromResponse(response *llm.Response) (llm.ConversionTraceSummary, bool) {
	if response == nil || response.TransformerMetadata == nil {
		return llm.ConversionTraceSummary{}, false
	}
	summary, ok := response.TransformerMetadata[responseSummaryMetadataKey].(llm.ConversionTraceSummary)
	return summary, ok
}

func (o *Outbound) ConversionSummaryFromResponse(response *llm.Response) (llm.ConversionTraceSummary, bool) {
	return SummaryFromResponse(response)
}

func DebugTraceFromResponse(response *llm.Response) (*llm.ConversionDebugTrace, bool) {
	if response == nil || response.TransformerMetadata == nil {
		return nil, false
	}
	trace, ok := response.TransformerMetadata[responseDebugMetadataKey].(*llm.ConversionDebugTrace)
	if !ok || trace == nil {
		return nil, false
	}
	return trace.Clone(), true
}

func (o *Outbound) ConversionDebugFromResponse(response *llm.Response) (*llm.ConversionDebugTrace, bool) {
	return DebugTraceFromResponse(response)
}

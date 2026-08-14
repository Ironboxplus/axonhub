package pipeline

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

// Process executes the non-streaming LLM pipeline
// Steps: outbound transform -> HTTP request -> outbound response transform -> inbound response transform.
func (p *pipeline) notStream(
	ctx context.Context,
	executor Executor,
	request *httpclient.Request,
) (*httpclient.Response, error) {
	round, err := p.notStreamRound(ctx, executor, request)
	if err != nil {
		return nil, err
	}
	if round.passthrough != nil {
		return p.finalizePassthrough(ctx, round.passthrough)
	}
	return p.finalizeNotStream(ctx, round.response)
}

// nonStreamRound is the provider-facing half of a canonical exchange. It is
// deliberately separated from public response middleware and client encoding
// so gateway emulation can consume internal model rounds without persisting or
// exposing them as public response terminals.
type nonStreamRound struct {
	response    *llm.Response
	passthrough *httpclient.Response
}

func (p *pipeline) notStreamRound(
	ctx context.Context,
	executor Executor,
	request *httpclient.Request,
) (*nonStreamRound, error) {
	httpResp, err := p.notStreamProviderExchange(ctx, executor, request)
	if err != nil {
		return nil, err
	}
	return p.transformNotStreamProviderResponse(ctx, httpResp)
}

// notStreamProviderExchange owns the only provider-facing portion of a
// non-streaming round. Gateway callers may put a deadline around this exchange
// without accidentally cancelling response decoding, durable checkpoint Seal,
// client middleware, or lazy client SSE consumption after the provider has
// returned successfully.
func (p *pipeline) notStreamProviderExchange(
	ctx context.Context,
	executor Executor,
	request *httpclient.Request,
) (*httpclient.Response, error) {
	startedAt := observationStart(ctx)
	httpResp, err := executor.Do(ctx, request)
	statusCode := 0
	outputBytes := int64(0)
	if httpResp != nil {
		statusCode = httpResp.StatusCode
		outputBytes = int64(len(httpResp.Body))
	}
	observeStage(ctx, StageProviderExchange, startedAt, err, observationData{
		statusCode:  statusCode,
		inputBytes:  requestBodySize(request),
		outputBytes: outputBytes,
	})
	if err != nil {
		// Apply error response middlewares
		p.applyRawErrorResponseMiddlewares(ctx, err)

		if httpErr, ok := errors.AsType[*httpclient.Error](err); ok {
			return nil, WrapUpstreamError(p.Outbound.TransformError(ctx, httpErr))
		}

		return nil, WrapUpstreamError(fmt.Errorf("failed to do request: %w", err))
	}
	return httpResp, nil
}

// transformNotStreamProviderResponse runs the post-exchange provider response
// transforms. It deliberately accepts the original request context for forced
// inline-compaction streams: the provider timeout has ended, while response
// restore and durable gateway state persistence must still be cancellable only
// by the client request itself.
func (p *pipeline) transformNotStreamProviderResponse(ctx context.Context, httpResp *httpclient.Response) (*nonStreamRound, error) {
	// Apply raw response middlewares
	startedAt := observationStart(ctx)
	var err error
	httpResp, err = p.applyRawResponseMiddlewares(ctx, httpResp)
	statusCode := 0
	outputBytes := int64(0)
	if httpResp != nil {
		statusCode = httpResp.StatusCode
		outputBytes = int64(len(httpResp.Body))
	}
	observeStage(ctx, StageRawResponseMiddleware, startedAt, err, observationData{
		statusCode:  statusCode,
		outputBytes: outputBytes,
	})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)

		return nil, fmt.Errorf("failed to apply raw response middlewares: %w", err)
	}

	// Explicit large-body passthrough keeps successful image/media payloads out
	// of memory. Request transformation and raw middlewares have already run;
	// semantic response transformation is intentionally skipped because the
	// caller requested the provider's same-protocol wire representation.
	if httpResp != nil && httpResp.BodyStream != nil {
		return &nonStreamRound{passthrough: httpResp}, nil
	}

	startedAt = observationStart(ctx)
	llmResp, err := p.Outbound.TransformResponse(ctx, httpResp)
	observeStage(ctx, StageProviderResponseTransform, startedAt, err, observationData{
		statusCode: statusCode,
		inputBytes: outputBytes,
	})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)

		return nil, WrapUpstreamError(fmt.Errorf("failed to transform response: %w", err))
	}
	conversionData := observationData{}
	if reporter, ok := outboundCapability[conversionSummaryReporter](p.Outbound); ok {
		if summary, found := reporter.ConversionSummaryFromResponse(llmResp); found {
			conversionData.conversion = &summary
		}
	}
	if reporter, ok := outboundCapability[conversionDebugReporter](p.Outbound); ok {
		if debug, found := reporter.ConversionDebugFromResponse(llmResp); found {
			conversionData.conversionDebug = debug
		}
	}
	if conversionData.conversion != nil || conversionData.conversionDebug != nil {
		observeStage(ctx, StageConversionRestore, time.Time{}, nil, conversionData)
	}
	return &nonStreamRound{response: llmResp}, nil
}

func (p *pipeline) finalizeNotStream(ctx context.Context, llmResp *llm.Response) (*httpclient.Response, error) {
	var err error
	// Apply LLM response middlewares
	startedAt := observationStart(ctx)
	llmResp, err = p.applyLlmResponseMiddlewares(ctx, llmResp)
	observeStage(ctx, StageUnifiedResponseMiddleware, startedAt, err, observationData{})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)

		return nil, fmt.Errorf("failed to apply llm response middlewares: %w", err)
	}

	if p.emptyResponseDetection && !hasResponseContent(llmResp) {
		observeStage(ctx, StageResponseValidation, observationStart(ctx), ErrEmptyResponse, observationData{})
		p.applyRawErrorResponseMiddlewares(ctx, ErrEmptyResponse)

		return nil, ErrEmptyResponse
	}

	startedAt = observationStart(ctx)
	finalResp, err := p.Inbound.TransformResponse(ctx, llmResp)
	statusCode := 0
	outputBytes := int64(0)
	if finalResp != nil {
		statusCode = finalResp.StatusCode
		outputBytes = int64(len(finalResp.Body))
	}
	observeStage(ctx, StageClientResponseTransform, startedAt, err, observationData{
		statusCode:  statusCode,
		outputBytes: outputBytes,
	})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)

		return nil, fmt.Errorf("failed to transform final response: %w", err)
	}

	// Apply inbound raw response middlewares after final response transformation
	startedAt = observationStart(ctx)
	finalResp, err = p.applyInboundRawResponseMiddlewares(ctx, finalResp)
	statusCode = 0
	outputBytes = 0
	if finalResp != nil {
		statusCode = finalResp.StatusCode
		outputBytes = int64(len(finalResp.Body))
	}
	observeStage(ctx, StageClientResponseMiddleware, startedAt, err, observationData{
		statusCode:  statusCode,
		outputBytes: outputBytes,
	})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)

		return nil, fmt.Errorf("failed to apply inbound raw response middlewares: %w", err)
	}

	return finalResp, nil
}

// processForcedNonStreamingStream is the narrow bridge used by gateway-owned
// inline compaction. The provider sees an ordinary non-streaming summary
// request, while the original client stream contract is preserved by rendering
// the completed canonical compaction response through the client's normal SSE
// transformer. This path deliberately does not buffer an upstream SSE stream.
func (p *pipeline) processForcedNonStreamingStream(ctx context.Context, request *llm.Request) (*Result, error) {
	round, err := p.prepareProviderRound(ctx, request)
	if err != nil {
		return nil, err
	}
	timeoutCtx, cancel := p.withNonStreamTimeout(ctx)
	httpResponse, err := p.notStreamProviderExchange(timeoutCtx, round.executor, round.request)
	timedOut := p.isNonStreamTimeout(timeoutCtx)
	// The provider exchange is complete. Do not let its timeout context escape
	// into decoding, codec Seal, or the lazy client EventStream: callers consume
	// that stream after this function returns and must still receive
	// response.completed.
	cancel()
	if err != nil {
		if timedOut {
			return nil, ErrNonStreamResponseTimeout
		}
		return nil, err
	}
	providerRound, err := p.transformNotStreamProviderResponse(ctx, httpResponse)
	if err != nil {
		return nil, err
	}
	if providerRound == nil || providerRound.passthrough != nil || providerRound.response == nil {
		if providerRound != nil && providerRound.passthrough != nil {
			_ = providerRound.passthrough.Close()
		}
		return nil, fmt.Errorf("gateway inline compaction requires a materialized provider response")
	}

	// Preserve the normal non-stream response middleware boundary before the
	// response is rendered as client SSE. The restored response is then a single
	// canonical compaction output, never the provider's draft summary.
	startedAt := observationStart(ctx)
	response, err := p.applyLlmResponseMiddlewares(ctx, providerRound.response)
	observeStage(ctx, StageUnifiedResponseMiddleware, startedAt, err, observationData{})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, fmt.Errorf("failed to apply llm response middlewares: %w", err)
	}
	if p.emptyResponseDetection && !hasResponseContent(response) {
		observeStage(ctx, StageResponseValidation, observationStart(ctx), ErrEmptyResponse, observationData{})
		p.applyRawErrorResponseMiddlewares(ctx, ErrEmptyResponse)
		return nil, ErrEmptyResponse
	}
	events, err := llm.CanonicalEventsFromResponse(response)
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, fmt.Errorf("render inline compaction stream: %w", err)
	}
	streamResponse := *response
	streamResponse.Events = events
	clientStream, err := p.finalizeStream(ctx, streams.SliceStream([]*llm.Response{&streamResponse}))
	if err != nil {
		return nil, err
	}
	return &Result{Stream: true, EventStream: clientStream}, nil
}

func (p *pipeline) finalizePassthrough(ctx context.Context, response *httpclient.Response) (*httpclient.Response, error) {
	startedAt := observationStart(ctx)
	finalResponse, err := p.applyInboundRawResponseMiddlewares(ctx, response)
	statusCode := 0
	if response != nil {
		statusCode = response.StatusCode
	}
	observeStage(ctx, StageRawResponsePassthrough, startedAt, err, observationData{statusCode: statusCode})
	if err != nil {
		if response != nil {
			_ = response.Close()
		}
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, fmt.Errorf("failed to apply passthrough response middleware: %w", err)
	}
	return finalResponse, nil
}

func (p *pipeline) autoAggregateStream(
	ctx context.Context,
	executor Executor,
	request *httpclient.Request,
) (*httpclient.Response, error) {
	startedAt := observationStart(ctx)
	inboundStream, err := p.stream(ctx, executor, request, 0)
	if err != nil {
		observeStage(ctx, StageStreamAggregate, startedAt, err, observationData{})
		return nil, err
	}
	defer inboundStream.Close()

	chunks := make([]*httpclient.StreamEvent, 0, 8)
	for inboundStream.Next() {
		event := inboundStream.Current()
		if event != nil {
			chunks = append(chunks, event)
		}
	}

	if err := inboundStream.Err(); err != nil {
		observeStage(ctx, StageStreamAggregate, startedAt, err, observationData{
			events:      int64(len(chunks)),
			outputBytes: streamEventsSize(chunks),
		})
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, err
	}

	if len(chunks) == 0 {
		observeStage(ctx, StageStreamAggregate, startedAt, ErrEmptyStreamChunks, observationData{})
		p.applyRawErrorResponseMiddlewares(ctx, ErrEmptyStreamChunks)
		return nil, ErrEmptyStreamChunks
	}

	body, _, err := p.Inbound.AggregateStreamChunks(ctx, chunks)
	if err != nil {
		observeStage(ctx, StageStreamAggregate, startedAt, err, observationData{
			events:     int64(len(chunks)),
			inputBytes: streamEventsSize(chunks),
		})
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, err
	}

	if len(body) == 0 {
		observeStage(ctx, StageStreamAggregate, startedAt, ErrEmptyAggregatedBody, observationData{
			events:     int64(len(chunks)),
			inputBytes: streamEventsSize(chunks),
		})
		p.applyRawErrorResponseMiddlewares(ctx, ErrEmptyAggregatedBody)
		return nil, ErrEmptyAggregatedBody
	}

	resp := &httpclient.Response{
		StatusCode: http.StatusOK,
		Headers: http.Header{
			"Content-Type":  []string{"application/json"},
			"Cache-Control": []string{"no-cache"},
		},
		Body: body,
	}
	observeStage(ctx, StageStreamAggregate, startedAt, nil, observationData{
		events:      int64(len(chunks)),
		inputBytes:  streamEventsSize(chunks),
		outputBytes: int64(len(body)),
	})

	middlewareStartedAt := observationStart(ctx)
	resp, err = p.applyInboundRawResponseMiddlewares(ctx, resp)
	statusCode := 0
	outputBytes := int64(0)
	if resp != nil {
		statusCode = resp.StatusCode
		outputBytes = int64(len(resp.Body))
	}
	observeStage(ctx, StageClientResponseMiddleware, middlewareStartedAt, err, observationData{
		statusCode:  statusCode,
		outputBytes: outputBytes,
	})
	if err != nil {
		p.applyRawErrorResponseMiddlewares(ctx, err)
		return nil, fmt.Errorf("failed to apply inbound raw response middlewares: %w", err)
	}

	return resp, nil
}

func streamEventsSize(events []*httpclient.StreamEvent) int64 {
	var size int64
	for _, event := range events {
		size += rawStreamEventSize(event)
	}
	return size
}

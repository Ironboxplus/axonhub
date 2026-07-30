package pipeline

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
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

	// Apply raw response middlewares
	startedAt = observationStart(ctx)
	httpResp, err = p.applyRawResponseMiddlewares(ctx, httpResp)
	statusCode = 0
	outputBytes = 0
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

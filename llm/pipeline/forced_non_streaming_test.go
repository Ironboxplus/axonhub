package pipeline

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
)

func forcedCompactionProviderResponse() *llm.Response {
	return &llm.Response{
		ID: "resp_gateway", Status: llm.ResponseStatusCompleted,
		Output: []llm.Item{{
			Kind: llm.ItemKindMessage, ID: "msg_gateway", Role: llm.RoleAssistant,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "gateway checkpoint"}},
		}},
	}
}

type forcedCompactionTrackingBody struct {
	io.Reader
	closed bool
}

type forcedCompactionDebugOutbound struct {
	*mockOutbound
	debugCalls int
}

func (*forcedCompactionDebugOutbound) ConversionDebugFromRequest(*httpclient.Request) (*llm.ConversionDebugTrace, bool) {
	return nil, false
}

func (outbound *forcedCompactionDebugOutbound) ConversionDebugFromResponse(*llm.Response) (*llm.ConversionDebugTrace, bool) {
	outbound.debugCalls++
	return llm.NewRequiredConversionDebugTrace(1), true
}

func (body *forcedCompactionTrackingBody) Close() error {
	body.closed = true
	return nil
}

func forcedCompactionPipelineForTest(executor Executor, outbound transformer.Outbound, inbound *mockInbound, middlewares ...Middleware) *pipeline {
	return &pipeline{Executor: executor, Outbound: outbound, Inbound: inbound, middlewares: middlewares}
}

func forcedCompactionOutboundForTest(response func(*httpclient.Response) (*llm.Response, error)) *mockOutbound {
	return &mockOutbound{
		transformRequest: func(context.Context, *llm.Request) (*httpclient.Request, error) {
			return &httpclient.Request{Headers: http.Header{}}, nil
		},
		transformResponse: func(_ context.Context, raw *httpclient.Response) (*llm.Response, error) {
			return response(raw)
		},
	}
}

func TestProcessForcedNonStreamingStreamPreservesProviderDeadlineBoundary(t *testing.T) {
	t.Run("request and response transforms fail before any client stream", func(t *testing.T) {
		requestFailure := &mockOutbound{transformRequest: func(context.Context, *llm.Request) (*httpclient.Request, error) {
			return nil, errors.New("request transform failed")
		}}
		if _, err := forcedCompactionPipelineForTest(&mockExecutor{}, requestFailure, &mockInbound{}).processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("request transform failure was accepted")
		}

		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK}, nil
		}}
		responseFailure := forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return nil, errors.New("response transform failed")
		})
		if _, err := forcedCompactionPipelineForTest(executor, responseFailure, &mockInbound{}).processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("response transform failure was accepted")
		}
	})

	t.Run("provider completion restores and renders client stream", func(t *testing.T) {
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK}, nil
		}}
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return &llm.Response{ID: "resp_compaction", Status: llm.ResponseStatusCompleted, Output: []llm.Item{{Kind: llm.ItemKindCompaction, ID: "cmp_compaction", Compaction: &llm.CompactionItem{EncryptedContent: "gateway-opaque"}}}}, nil
		}), &mockInbound{})
		p.nonStreamTimeout = time.Millisecond

		result, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{})
		if err != nil || result == nil || !result.Stream || result.EventStream == nil {
			t.Fatalf("forced stream result=%#v err=%v", result, err)
		}
		if !result.EventStream.Next() || result.EventStream.Current() == nil {
			t.Fatal("restored client SSE was not consumable after provider timeout context was cancelled")
		}
		if err := result.EventStream.Close(); err != nil {
			t.Fatal(err)
		}
	})

	t.Run("post-provider debug evidence is observed", func(t *testing.T) {
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK}, nil
		}}
		outbound := &forcedCompactionDebugOutbound{mockOutbound: forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return forcedCompactionProviderResponse(), nil
		})}
		if reporter, ok := any(outbound).(conversionDebugReporter); !ok {
			t.Fatalf("debug outbound does not satisfy reporter: %T", outbound)
		} else if _, found := reporter.ConversionDebugFromResponse(forcedCompactionProviderResponse()); !found {
			t.Fatal("debug outbound did not report a trace")
		}
		outbound.debugCalls = 0
		result, err := forcedCompactionPipelineForTest(executor, outbound, &mockInbound{}).processForcedNonStreamingStream(context.Background(), &llm.Request{})
		if err != nil || result == nil || outbound.debugCalls != 1 {
			t.Fatalf("debug reporter result=%#v err=%v calls=%d", result, err, outbound.debugCalls)
		}
	})

	t.Run("provider deadline is classified before restore", func(t *testing.T) {
		executor := &mockExecutor{do: func(ctx context.Context, _ *httpclient.Request) (*httpclient.Response, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}}
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			t.Fatal("response transform must not run after provider timeout")
			return nil, nil
		}), &mockInbound{})
		p.nonStreamTimeout = time.Millisecond
		if _, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{}); !errors.Is(err, ErrNonStreamResponseTimeout) {
			t.Fatalf("provider timeout error=%v", err)
		}
	})

	t.Run("provider failure remains upstream failure", func(t *testing.T) {
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return nil, errors.New("provider unavailable")
		}}
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return forcedCompactionProviderResponse(), nil
		}), &mockInbound{})
		if _, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("provider failure was accepted")
		}
	})

	t.Run("passthrough response is not a compaction summary", func(t *testing.T) {
		body := &forcedCompactionTrackingBody{Reader: strings.NewReader("provider body")}
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK, BodyStream: body}, nil
		}}
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			t.Fatal("passthrough response must not be decoded as a summary")
			return nil, nil
		}), &mockInbound{})
		if _, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil || !body.closed {
			t.Fatalf("passthrough provider response was accepted/leaked: err=%v closed=%v", err, body.closed)
		}
	})

	t.Run("response middleware failure and empty response are both terminal", func(t *testing.T) {
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK}, nil
		}}
		middlewareFailure := newTrackingMiddleware("fail", &[]string{})
		middlewareFailure.shouldFailOnLlmResponse = true
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return forcedCompactionProviderResponse(), nil
		}), &mockInbound{}, middlewareFailure)
		if _, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("response middleware failure was accepted")
		}

		empty := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return &llm.Response{ID: "empty", Status: llm.ResponseStatusCompleted, Output: []llm.Item{{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{}}}}, nil
		}), &mockInbound{})
		empty.emptyResponseDetection = true
		if _, err := empty.processForcedNonStreamingStream(context.Background(), &llm.Request{}); !errors.Is(err, ErrEmptyResponse) {
			t.Fatalf("empty forced summary error=%v", err)
		}
	})

	t.Run("invalid canonical response and client stream transform both fail closed", func(t *testing.T) {
		executor := &mockExecutor{do: func(context.Context, *httpclient.Request) (*httpclient.Response, error) {
			return &httpclient.Response{StatusCode: http.StatusOK}, nil
		}}
		invalid := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return &llm.Response{ID: "invalid", Status: llm.ResponseStatusCompleted, Output: []llm.Item{{Kind: "future_union"}}}, nil
		}), &mockInbound{})
		if _, err := invalid.processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("invalid canonical response was rendered as a client SSE")
		}

		inboundFailure := &mockInbound{transformStream: func(context.Context, streams.Stream[*llm.Response]) (streams.Stream[*httpclient.StreamEvent], error) {
			return nil, errors.New("client stream transformer failed")
		}}
		p := forcedCompactionPipelineForTest(executor, forcedCompactionOutboundForTest(func(*httpclient.Response) (*llm.Response, error) {
			return forcedCompactionProviderResponse(), nil
		}), inboundFailure)
		if _, err := p.processForcedNonStreamingStream(context.Background(), &llm.Request{}); err == nil {
			t.Fatal("client stream transformer failure was accepted")
		}
	})
}

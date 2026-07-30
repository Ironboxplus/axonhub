package pipeline_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

type twoRoundController struct{}

func (twoRoundController) Complete(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (*llm.Response, error) {
	pipeline.RecordEmulationRound(ctx, 1, 0)
	if _, err := rounds.Complete(ctx, request.Clone()); err != nil {
		return nil, err
	}
	pipeline.RecordEmulationRound(ctx, 0, 0)
	return rounds.Complete(ctx, request.Clone())
}

func (twoRoundController) Stream(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (streams.Stream[*llm.Response], error) {
	pipeline.RecordEmulationRound(ctx, 1, 0)
	internal, err := rounds.Stream(ctx, request.Clone())
	if err != nil {
		return nil, err
	}
	for internal.Next() {
	}
	if err := internal.Err(); err != nil {
		_ = internal.Close()
		return nil, err
	}
	if err := internal.Close(); err != nil {
		return nil, err
	}
	pipeline.RecordEmulationRound(ctx, 0, 0)
	return rounds.Stream(ctx, request.Clone())
}

type emulationObservationRecorder struct {
	events []pipeline.Observation
}

func (recorder *emulationObservationRecorder) Observe(_ context.Context, event pipeline.Observation) {
	recorder.events = append(recorder.events, event)
}

type toolLoopCountingMiddleware struct {
	pipeline.DummyMiddleware
	rawRequests     atomic.Int64
	rawStreams      atomic.Int64
	publicResponses atomic.Int64
	publicStreams   atomic.Int64
}

func (middleware *toolLoopCountingMiddleware) Name() string { return "tool_loop_counter" }

func (middleware *toolLoopCountingMiddleware) OnOutboundRawRequest(_ context.Context, request *httpclient.Request) (*httpclient.Request, error) {
	middleware.rawRequests.Add(1)
	return request, nil
}

func (middleware *toolLoopCountingMiddleware) OnOutboundRawStream(_ context.Context, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*httpclient.StreamEvent], error) {
	middleware.rawStreams.Add(1)
	return stream, nil
}

func (middleware *toolLoopCountingMiddleware) OnOutboundLlmResponse(_ context.Context, response *llm.Response) (*llm.Response, error) {
	middleware.publicResponses.Add(1)
	return response, nil
}

func (middleware *toolLoopCountingMiddleware) OnOutboundLlmStream(_ context.Context, stream streams.Stream[*llm.Response]) (streams.Stream[*llm.Response], error) {
	middleware.publicStreams.Add(1)
	return stream, nil
}

func TestToolLoopPublishesOnlyFinalNonStreamRoundOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		text := "internal-hidden"
		if round == 2 {
			text = "public-final"
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-%d","object":"chat.completion","created":1785380000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":%q},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}`, round, text)
	}))
	t.Cleanup(provider.Close)

	middleware := &toolLoopCountingMiddleware{}
	result := runToolLoopPipeline(t, provider.URL, false, middleware)
	require.False(t, result.Stream)
	require.NotNil(t, result.Response)
	require.Contains(t, string(result.Response.Body), "public-final")
	require.NotContains(t, string(result.Response.Body), "internal-hidden")
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 2, middleware.rawRequests.Load())
	require.EqualValues(t, 1, middleware.publicResponses.Load())
	require.Zero(t, middleware.publicStreams.Load())
}

func TestToolLoopEmitsPayloadFreeEmulationEvidence(t *testing.T) {
	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		round := providerRounds.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"id":"chatcmpl-%d","object":"chat.completion","created":1785380000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`, round)
	}))
	t.Cleanup(provider.Close)

	recorder := &emulationObservationRecorder{}
	_ = runToolLoopPipelineWithObserver(t, provider.URL, false, &toolLoopCountingMiddleware{}, recorder)
	var observed *pipeline.Observation
	for index := range recorder.events {
		if recorder.events[index].Stage == pipeline.StageConversionEmulation {
			observed = &recorder.events[index]
			break
		}
	}
	require.NotNil(t, observed)
	require.NotNil(t, observed.Emulation)
	require.EqualValues(t, 2, observed.Emulation.InternalRounds)
	require.EqualValues(t, 1, observed.Emulation.ToolCalls)
	encoded, err := json.Marshal(observed)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), "provider-key")
	require.NotContains(t, string(encoded), "fixture-model")
}

func TestToolLoopPublishesOnlyFinalStreamRoundOverRealHTTP(t *testing.T) {
	t.Parallel()
	var providerRounds atomic.Int64
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		round := providerRounds.Add(1)
		text := "internal-hidden"
		if round == 2 {
			text = "public-visible"
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion.chunk\",\"created\":1785380000,\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":%q},\"finish_reason\":null}]}\n\n", round, text)
		_, _ = fmt.Fprintf(w, "data: {\"id\":\"chatcmpl-%d\",\"object\":\"chat.completion.chunk\",\"created\":1785380000,\"model\":\"fixture-model\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n", round)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(provider.Close)

	middleware := &toolLoopCountingMiddleware{}
	result := runToolLoopPipeline(t, provider.URL, true, middleware)
	require.True(t, result.Stream)
	require.NotNil(t, result.EventStream)
	defer result.EventStream.Close()
	var wire strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event != nil {
			wire.Write(event.Data)
		}
	}
	require.NoError(t, result.EventStream.Err())
	require.Contains(t, wire.String(), "public-visible")
	require.NotContains(t, wire.String(), "internal-hidden")
	require.EqualValues(t, 2, providerRounds.Load())
	require.EqualValues(t, 2, middleware.rawRequests.Load())
	require.EqualValues(t, 2, middleware.rawStreams.Load())
	require.EqualValues(t, 1, middleware.publicStreams.Load())
	require.Zero(t, middleware.publicResponses.Load())
}

func runToolLoopPipeline(t *testing.T, providerURL string, stream bool, middleware pipeline.Middleware) *pipeline.Result {
	return runToolLoopPipelineWithObserver(t, providerURL, stream, middleware, nil)
}

func runToolLoopPipelineWithObserver(t *testing.T, providerURL string, stream bool, middleware pipeline.Middleware, observer pipeline.Observer) *pipeline.Result {
	t.Helper()
	outbound, err := openai.NewOutboundTransformer(providerURL, "provider-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	t.Cleanup(cancel)
	body := fmt.Appendf(nil, `{"model":"fixture-model","messages":[{"role":"user","content":"hello"}],"stream":%t}`, stream)
	options := []pipeline.Option{pipeline.WithMiddlewares(middleware), pipeline.WithToolLoopController(twoRoundController{})}
	if observer != nil {
		options = append(options, pipeline.WithObserver(observer))
	}
	result, err := pipeline.NewFactory(executor).Pipeline(
		openai.NewInboundTransformer(), conversion.NewOutbound(outbound),
		options...,
	).Process(ctx, &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/chat/completions",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	return result
}

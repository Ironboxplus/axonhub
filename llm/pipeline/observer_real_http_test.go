package pipeline_test

import (
	"bytes"
	"context"
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	pipelinestream "github.com/looplj/axonhub/llm/pipeline/stream"
	"github.com/looplj/axonhub/llm/transformer/openai"
	responsestransformer "github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type observationRecorder struct {
	mu     sync.Mutex
	events []pipeline.Observation
}

func (r *observationRecorder) Observe(_ context.Context, event pipeline.Observation) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.events = append(r.events, event)
}

func (r *observationRecorder) snapshot() []pipeline.Observation {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]pipeline.Observation(nil), r.events...)
}

func TestPipelineObserverRealHTTPIsStageCompleteAndPayloadFree(t *testing.T) {
	const (
		apiKey           = "observer-real-http-secret-key"
		privatePrompt    = `inspect C:\Users\Arc\private-observer.txt`
		privateResponse  = "private provider response text"
		providerWaitTime = 15 * time.Millisecond
	)

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		require.Equal(t, "/v1/chat/completions", request.URL.Path)
		require.Equal(t, "Bearer "+apiKey, request.Header.Get("Authorization"))
		time.Sleep(providerWaitTime)
		w.Header().Set("Content-Type", "application/json")
		_, err := w.Write([]byte(`{
			"id":"chatcmpl_observer",
			"object":"chat.completion",
			"created":1785247203,
			"model":"observer-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"` + privateResponse + `"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}
		}`))
		require.NoError(t, err)
	}))
	t.Cleanup(provider.Close)

	outbound, err := openai.NewOutboundTransformer(provider.URL, apiKey)
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)

	recorder := &observationRecorder{}
	requestBody, err := json.Marshal(map[string]any{
		"model": "observer-model",
		"messages": []map[string]any{{
			"role":    "user",
			"content": privatePrompt,
		}},
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	result, err := pipeline.NewFactory(executor).
		Pipeline(
			openai.NewInboundTransformer(),
			outbound,
			pipeline.WithObserver(recorder),
		).
		Process(ctx, &httpclient.Request{
			Method: http.MethodPost,
			URL:    "/v1/chat/completions",
			Headers: http.Header{
				"Content-Type": []string{"application/json"},
			},
			Body: requestBody,
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.NotNil(t, result.Response)

	events := recorder.snapshot()
	require.NotEmpty(t, events)
	requireObservedStage(t, events, pipeline.StageInboundTransform)
	requireObservedStage(t, events, pipeline.StageOutboundTransform)
	providerEvent := requireObservedStage(t, events, pipeline.StageProviderExchange)
	require.Equal(t, pipeline.OutcomeSuccess, providerEvent.Outcome)
	require.GreaterOrEqual(t, providerEvent.Duration, providerWaitTime)
	require.Equal(t, http.StatusOK, providerEvent.StatusCode)
	require.Greater(t, providerEvent.OutputBytes, int64(0))
	requireObservedStage(t, events, pipeline.StageProviderResponseTransform)
	requireObservedStage(t, events, pipeline.StageClientResponseTransform)

	encoded, err := json.Marshal(events)
	require.NoError(t, err)
	for _, forbidden := range []string{apiKey, privatePrompt, privateResponse, "Authorization"} {
		require.NotContains(t, string(encoded), forbidden)
	}
}

func TestPipelineDebugLogsDoNotContainPayloadsOrCredentials(t *testing.T) {
	const (
		apiKey          = "debug-log-secret-key"
		privatePrompt   = "debug log private prompt"
		privateResponse = "debug log private response"
	)

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl_safe_log",
			"object":"chat.completion",
			"created":1785247203,
			"model":"safe-log-model",
			"choices":[{"index":0,"message":{"role":"assistant","content":"` + privateResponse + `"},"finish_reason":"stop"}]
		}`))
	}))
	t.Cleanup(provider.Close)

	var logs bytes.Buffer
	previousLogger := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	t.Cleanup(func() { slog.SetDefault(previousLogger) })

	outbound, err := openai.NewOutboundTransformer(provider.URL, apiKey)
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	body := []byte(`{"model":"safe-log-model","messages":[{"role":"user","content":"` + privatePrompt + `"}]}`)

	result, err := pipeline.NewFactory(executor).
		Pipeline(openai.NewInboundTransformer(), outbound).
		Process(context.Background(), &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/chat/completions",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    body,
		})
	require.NoError(t, err)
	require.NotNil(t, result)

	logOutput := logs.String()
	for _, forbidden := range []string{apiKey, privatePrompt, privateResponse, "Bearer " + apiKey} {
		if strings.Contains(logOutput, forbidden) {
			t.Fatalf("debug logs contain private payload or credential %q: %s", forbidden, logOutput)
		}
	}
}

func TestPipelineObserverAggregatesRealSSEWithoutPayloadEvents(t *testing.T) {
	const privateChunk = "stream payload must not enter observations"

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		chunks := []string{
			`{"id":"chatcmpl_stream_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{"role":"assistant","content":"` + privateChunk + `"},"finish_reason":null}]}`,
			`{"id":"chatcmpl_stream_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`[DONE]`,
		}
		for _, chunk := range chunks {
			_, err := w.Write([]byte("data: " + chunk + "\n\n"))
			require.NoError(t, err)
			flusher.Flush()
		}
	}))
	t.Cleanup(provider.Close)

	outbound, err := openai.NewOutboundTransformer(provider.URL, "stream-observer-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	recorder := &observationRecorder{}

	result, err := pipeline.NewFactory(executor).
		Pipeline(
			openai.NewInboundTransformer(),
			outbound,
			pipeline.WithObserver(recorder),
		).
		Process(context.Background(), &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/chat/completions",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"observer-model","stream":true,"messages":[{"role":"user","content":"hello"}]}`),
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.NotNil(t, result.EventStream)
	defer result.EventStream.Close()

	clientEvents := 0
	for result.EventStream.Next() {
		clientEvents++
	}
	require.NoError(t, result.EventStream.Err())
	require.Equal(t, 3, clientEvents)

	events := recorder.snapshot()
	providerStream := requireSingleObservedStage(t, events, pipeline.StageProviderStream)
	require.Equal(t, pipeline.OutcomeSuccess, providerStream.Outcome)
	require.Equal(t, int64(3), providerStream.Events)
	require.Greater(t, providerStream.OutputBytes, int64(0))
	unifiedStream := requireSingleObservedStage(t, events, pipeline.StageUnifiedStream)
	require.Equal(t, pipeline.OutcomeSuccess, unifiedStream.Outcome)
	require.GreaterOrEqual(t, unifiedStream.Events, int64(3))
	clientStream := requireSingleObservedStage(t, events, pipeline.StageClientStream)
	require.Equal(t, pipeline.OutcomeSuccess, clientStream.Outcome)
	require.Equal(t, int64(clientEvents), clientStream.Events)

	encoded, err := json.Marshal(events)
	require.NoError(t, err)
	require.NotContains(t, string(encoded), privateChunk)
}

func TestPipelineObserverPreservesEveryResponsesClientEventOverRealHTTP(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		require.Equal(t, "/v1/chat/completions", request.URL.Path)
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		flusher, ok := w.(http.Flusher)
		require.True(t, ok)
		chunks := []string{
			`{"id":"chatcmpl_responses_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
			`{"id":"chatcmpl_responses_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{"content":"alpha"},"finish_reason":null}]}`,
			`{"id":"chatcmpl_responses_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{"content":"-beta"},"finish_reason":null}]}`,
			`{"id":"chatcmpl_responses_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{"content":"-gamma"},"finish_reason":null}]}`,
			`{"id":"chatcmpl_responses_observer","object":"chat.completion.chunk","created":1785247203,"model":"observer-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
			`[DONE]`,
		}
		for _, chunk := range chunks {
			_, err := w.Write([]byte("data: " + chunk + "\n\n"))
			require.NoError(t, err)
			flusher.Flush()
		}
	}))
	t.Cleanup(provider.Close)

	outbound, err := openai.NewOutboundTransformer(provider.URL, "responses-observer-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	recorder := &observationRecorder{}

	result, err := pipeline.NewFactory(executor).
		Pipeline(
			responsestransformer.NewInboundTransformer(),
			outbound,
			pipeline.WithMiddlewares(pipelinestream.EnsureUsage()),
			pipeline.WithObserver(recorder),
			pipeline.WithEmptyResponseDetection(),
		).
		Process(context.Background(), &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"observer-model","stream":true,"input":"hello"}`),
		})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.NotNil(t, result.EventStream)
	defer result.EventStream.Close()

	var sequenceNumbers []int
	var text strings.Builder
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		require.NotNil(t, event)
		var envelope map[string]any
		require.NoError(t, json.Unmarshal(event.Data, &envelope))
		sequenceNumbers = append(sequenceNumbers, int(envelope["sequence_number"].(float64)))
		if envelope["type"] == "response.output_text.delta" {
			text.WriteString(envelope["delta"].(string))
		}
	}
	require.NoError(t, result.EventStream.Err())
	require.Equal(t, "alpha-beta-gamma", text.String())
	for i, sequenceNumber := range sequenceNumbers {
		require.Equal(t, i, sequenceNumber, "Responses client stream dropped or reordered an event")
	}
}

func TestPipelineObserverClassifiesRealHTTPFailureWithoutErrorBody(t *testing.T) {
	const privateErrorBody = "provider echoed a private prompt in this error"

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, privateErrorBody, http.StatusBadGateway)
	}))
	t.Cleanup(provider.Close)

	outbound, err := openai.NewOutboundTransformer(provider.URL, "failure-observer-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	recorder := &observationRecorder{}

	_, err = pipeline.NewFactory(executor).
		Pipeline(
			openai.NewInboundTransformer(),
			outbound,
			pipeline.WithObserver(recorder),
		).
		Process(context.Background(), &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/chat/completions",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    []byte(`{"model":"observer-model","messages":[{"role":"user","content":"hello"}]}`),
		})
	require.Error(t, err)

	events := recorder.snapshot()
	providerEvent := requireObservedStage(t, events, pipeline.StageProviderExchange)
	require.Equal(t, pipeline.OutcomeFailure, providerEvent.Outcome)
	require.Equal(t, pipeline.ErrorClassUpstream, providerEvent.ErrorClass)
	require.Equal(t, http.StatusBadGateway, providerEvent.StatusCode)
	encoded, marshalErr := json.Marshal(events)
	require.NoError(t, marshalErr)
	require.NotContains(t, string(encoded), privateErrorBody)
}

func requireObservedStage(t *testing.T, events []pipeline.Observation, stage pipeline.Stage) pipeline.Observation {
	t.Helper()
	for _, event := range events {
		if event.Stage == stage {
			return event
		}
	}
	t.Fatalf("stage %q not observed in %#v", stage, events)
	return pipeline.Observation{}
}

func requireSingleObservedStage(t *testing.T, events []pipeline.Observation, stage pipeline.Stage) pipeline.Observation {
	t.Helper()
	var matches []pipeline.Observation
	for _, event := range events {
		if event.Stage == stage {
			matches = append(matches, event)
		}
	}
	require.Len(t, matches, 1, "observations for stage %q: %#v", stage, matches)
	return matches[0]
}

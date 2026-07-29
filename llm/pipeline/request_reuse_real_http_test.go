package pipeline_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai"
)

type requestIdentityMiddleware struct {
	pipeline.DummyMiddleware
	request *llm.Request
}

func (m *requestIdentityMiddleware) Name() string { return "request_identity" }

func (m *requestIdentityMiddleware) OnInboundLlmRequest(_ context.Context, request *llm.Request) (*llm.Request, error) {
	m.request = request
	return request, nil
}

type requestIdentityOutbound struct {
	transformer.Outbound
	request *llm.Request
}

func (o *requestIdentityOutbound) TransformRequest(ctx context.Context, request *llm.Request) (*httpclient.Request, error) {
	o.request = request
	return o.Outbound.TransformRequest(ctx, request)
}

func TestSingleAttemptRequestReuseUsesPreparedGraphOverRealHTTP(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		require.Equal(t, "/v1/chat/completions", request.URL.Path)
		writer.Header().Set("Content-Type", "application/json")
		_, err := writer.Write([]byte(`{"id":"chatcmpl-reuse","object":"chat.completion","model":"reuse-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		require.NoError(t, err)
	}))
	t.Cleanup(provider.Close)

	for _, test := range []struct {
		name      string
		options   []pipeline.Option
		wantReuse bool
	}{
		{name: "safe default isolates outbound", wantReuse: false},
		{name: "explicit single attempt reuse", options: []pipeline.Option{pipeline.WithSingleAttemptRequestReuse()}, wantReuse: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			baseOutbound, err := openai.NewOutboundTransformer(provider.URL, "reuse-key")
			require.NoError(t, err)
			outbound := &requestIdentityOutbound{Outbound: baseOutbound}
			identity := &requestIdentityMiddleware{}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)

			options := append([]pipeline.Option{pipeline.WithMiddlewares(identity)}, test.options...)
			result, err := pipeline.NewFactory(executor).
				Pipeline(openai.NewInboundTransformer(), outbound, options...).
				Process(context.Background(), &httpclient.Request{
					Method:  http.MethodPost,
					URL:     "/v1/chat/completions",
					Headers: http.Header{"Content-Type": []string{"application/json"}},
					Body:    []byte(`{"model":"reuse-model","messages":[{"role":"user","content":"hello"}]}`),
				})
			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, identity.request)
			require.NotNil(t, outbound.request)
			if test.wantReuse {
				require.Same(t, identity.request, outbound.request)
			} else {
				require.NotSame(t, identity.request, outbound.request)
			}
		})
	}
}

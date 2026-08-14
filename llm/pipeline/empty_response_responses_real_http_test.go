package pipeline_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	responses "github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type alwaysRetryOutbound struct{ transformer.Outbound }

func (alwaysRetryOutbound) CanRetry(error) bool                   { return true }
func (alwaysRetryOutbound) PrepareForRetry(context.Context) error { return nil }

// These are real non-stream Responses exchanges. An explicit incomplete or
// cancelled terminal with output=[] is still a client-visible provider result:
// empty-response retry must neither issue another request nor erase status,
// usage, or incomplete_details.
func TestNonStreamResponsesTerminalWithoutOutputDoesNotRetryOverRealHTTP(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name string
		body string
	}{
		{name: "incomplete", body: `{"id":"resp_incomplete","object":"response","created_at":1,"model":"fixture","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[],"usage":{"input_tokens":3,"output_tokens":5,"total_tokens":8}}`},
		{name: "cancelled", body: `{"id":"resp_cancelled","object":"response","created_at":1,"model":"fixture","status":"cancelled","output":[],"usage":{"input_tokens":2,"output_tokens":0,"total_tokens":2}}`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			var hits atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				hits.Add(1)
				require.Equal(t, http.MethodPost, request.Method)
				require.Equal(t, "/v1/responses", request.URL.Path)
				_, _ = io.ReadAll(request.Body)
				writer.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(writer, test.body)
			}))
			t.Cleanup(provider.Close)

			outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
			require.NoError(t, err)
			retryable := alwaysRetryOutbound{Outbound: outbound}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)

			result, err := pipeline.NewFactory(executor).Pipeline(
				responses.NewInboundTransformer(), retryable,
				pipeline.WithRetry(0, 2, 0), pipeline.WithEmptyResponseDetection(),
			).Process(context.Background(), &httpclient.Request{
				Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
				Body: []byte(`{"model":"fixture","input":"continue"}`),
			})
			require.NoError(t, err)
			require.NotNil(t, result)
			require.NotNil(t, result.Response)
			require.Equal(t, int32(1), hits.Load())

			var payload map[string]json.RawMessage
			require.NoError(t, json.Unmarshal(result.Response.Body, &payload))
			require.JSONEq(t, `"`+test.name+`"`, string(payload["status"]))
			if test.name == "incomplete" {
				require.JSONEq(t, `{"reason":"max_output_tokens"}`, string(payload["incomplete_details"]))
			}
			var usage struct {
				TotalTokens int64 `json:"total_tokens"`
			}
			require.NoError(t, json.Unmarshal(payload["usage"], &usage))
			require.Positive(t, usage.TotalTokens)
		})
	}
}

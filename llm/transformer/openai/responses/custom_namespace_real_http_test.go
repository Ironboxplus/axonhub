package responses_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type malformedCanonicalUnionInbound struct {
	transformer.Inbound
}

func (inbound malformedCanonicalUnionInbound) TransformRequest(ctx context.Context, request *httpclient.Request) (*llm.Request, error) {
	canonical, err := inbound.Inbound.TransformRequest(ctx, request)
	if err != nil {
		return nil, err
	}
	canonical.ToolDefinitions = []llm.ToolDefinition{{
		Kind:        llm.ToolKindFunction,
		LogicalName: "lookup",
		Execution:   llm.ExecutionOwnerClient,
		Hosted:      &llm.HostedToolDefinition{Type: "function"},
	}}
	canonical.Tools = nil
	return canonical, nil
}

func TestResponsesNamespacedCustomCallAndResultContinueOverRealHTTP(t *testing.T) {
	t.Parallel()
	requestBody := []byte(`{
		"model":"fixture-model",
		"tools":[{"type":"namespace","name":"apps","tools":[{
			"type":"custom","name":"exec","description":"Run a command.",
			"format":{"type":"grammar","syntax":"lark","definition":"start: source"}
		}]}],
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"run command"}]},
			{"id":"ct_1","type":"custom_tool_call","call_id":"call_1","name":"exec","namespace":"apps","input":"pwd","status":"completed"},
			{"id":"ct_1_result","type":"custom_tool_call_output","call_id":"call_1","name":"exec","output":"/workspace"}
		]
	}`)

	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		hits.Add(1)
		body, err := io.ReadAll(request.Body)
		if err != nil {
			http.Error(writer, "read request", http.StatusBadRequest)
			return
		}
		var payload struct {
			Input []struct {
				Type      string `json:"type"`
				CallID    string `json:"call_id"`
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			http.Error(writer, "invalid request", http.StatusBadRequest)
			return
		}
		var calls, results int
		for _, item := range payload.Input {
			switch item.Type {
			case "custom_tool_call":
				if item.CallID != "call_1" || item.Name != "exec" || item.Namespace != "apps" {
					http.Error(writer, "custom call namespace was not preserved", http.StatusBadRequest)
					return
				}
				calls++
			case "custom_tool_call_output":
				if item.CallID != "call_1" || item.Name != "exec" {
					http.Error(writer, "custom result no longer binds the input call", http.StatusBadRequest)
					return
				}
				results++
			}
		}
		if calls != 1 || results != 1 {
			http.Error(writer, "custom call/result lifecycle was duplicated or lost", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(writer, `{"id":"resp_namespace_native","object":"response","created_at":1,"status":"completed","model":"fixture-model","output":[{"id":"ct_2","type":"custom_tool_call","call_id":"call_2","name":"exec","namespace":"apps","input":"whoami","status":"completed"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	result, err := pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(context.Background(), &httpclient.Request{
			Method: http.MethodPost, URL: "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: requestBody,
		})
	require.NoError(t, err)
	require.EqualValues(t, 1, hits.Load())
	require.NotNil(t, result)
	require.NotNil(t, result.Response)

	var client struct {
		Output []struct {
			Type      string `json:"type"`
			CallID    string `json:"call_id"`
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"output"`
	}
	require.NoError(t, json.Unmarshal(result.Response.Body, &client))
	require.Len(t, client.Output, 1)
	require.Equal(t, "custom_tool_call", client.Output[0].Type)
	require.Equal(t, "call_2", client.Output[0].CallID)
	require.Equal(t, "exec", client.Output[0].Name)
	require.Equal(t, "apps", client.Output[0].Namespace)
}

func TestResponsesMalformedCanonicalUnionDoesNotPanicOrReachProvider(t *testing.T) {
	t.Parallel()
	var hits atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
		hits.Add(1)
		http.Error(writer, "malformed canonical state must not reach provider", http.StatusInternalServerError)
	}))
	t.Cleanup(provider.Close)

	outbound, err := responses.NewOutboundTransformer(provider.URL, "fixture-key")
	require.NoError(t, err)
	executor := httpclient.NewHttpClientWithClient(provider.Client())
	t.Cleanup(executor.CloseIdleConnections)
	var pipelineErr error
	require.NotPanics(t, func() {
		_, pipelineErr = pipeline.NewFactory(executor).
			Pipeline(malformedCanonicalUnionInbound{Inbound: responses.NewInboundTransformer()}, conversion.NewOutbound(outbound)).
			Process(context.Background(), &httpclient.Request{
				Method: http.MethodPost, URL: "/v1/responses",
				Headers: http.Header{"Content-Type": []string{"application/json"}},
				Body:    []byte(`{"model":"fixture-model","input":"hello"}`),
			})
	})
	require.Error(t, pipelineErr)
	require.EqualValues(t, 0, hits.Load())
}

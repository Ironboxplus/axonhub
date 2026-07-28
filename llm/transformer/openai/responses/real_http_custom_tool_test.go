package responses_test

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

// TestResponsesCustomToolRoundTripOverRealHTTP exercises the production HTTP
// executor and SSE decoder. The upstream is deterministic, but it is a real TCP
// server rather than a pipeline.Executor mock.
func TestResponsesCustomToolRoundTripOverRealHTTP(t *testing.T) {
	t.Parallel()

	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			http.Error(w, "unexpected path: "+r.URL.Path, http.StatusNotFound)
			return
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			http.Error(w, "missing production bearer authentication", http.StatusUnauthorized)
			return
		}
		if got := r.Header.Get("Accept"); got != "text/event-stream" {
			http.Error(w, "expected SSE accept header, got "+got, http.StatusBadRequest)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read request: "+err.Error(), http.StatusBadRequest)
			return
		}
		if err := validateCustomToolContinuation(body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		streamResponsesFixture(t, w, "testdata/custom_tool.stream.jsonl")
	}))
	t.Cleanup(provider.Close)

	requestBody, err := os.ReadFile("testdata/custom_tool.request.json")
	require.NoError(t, err)

	inbound := responses.NewInboundTransformer()
	outbound, err := responses.NewOutboundTransformer(provider.URL, "test-key")
	require.NoError(t, err)

	executor := httpclient.NewHttpClientWithProxy(&httpclient.ProxyConfig{Type: httpclient.ProxyTypeDisabled})
	t.Cleanup(executor.CloseIdleConnections)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	result, err := pipeline.NewFactory(executor).Pipeline(inbound, outbound).Process(ctx, &httpclient.Request{
		Method: http.MethodPost,
		URL:    "/v1/responses",
		Headers: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body: requestBody,
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.NotNil(t, result.EventStream)
	defer result.EventStream.Close()

	var eventTypes []string
	foundCustomToolDone := false
	foundCustomInputDone := false
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		require.NotNil(t, event)
		eventTypes = append(eventTypes, event.Type)

		switch event.Type {
		case "response.custom_tool_call_input.done":
			foundCustomInputDone = true
		case "response.output_item.done":
			var decoded struct {
				Item struct {
					Type   string  `json:"type"`
					CallID string  `json:"call_id"`
					Name   string  `json:"name"`
					Input  *string `json:"input"`
				} `json:"item"`
			}
			require.NoError(t, json.Unmarshal(event.Data, &decoded))
			if decoded.Item.Type == "custom_tool_call" {
				foundCustomToolDone = true
				require.Equal(t, "call_patch_stream_001", decoded.Item.CallID)
				require.Equal(t, "apply_patch", decoded.Item.Name)
				require.NotNil(t, decoded.Item.Input)
				require.Contains(t, *decoded.Item.Input, "*** Begin Patch")
			}
		}
	}

	require.NoError(t, result.EventStream.Err())
	require.True(t, foundCustomInputDone, "event types: %v", eventTypes)
	require.True(t, foundCustomToolDone, "event types: %v", eventTypes)
	require.Contains(t, eventTypes, "response.completed")
}

func validateCustomToolContinuation(body []byte) error {
	var request struct {
		Input []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
		} `json:"input"`
		Tools []struct {
			Type string `json:"type"`
			Name string `json:"name"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		return &testValidationError{"invalid JSON request: " + err.Error()}
	}

	var hasCustomCall, hasCustomOutput, hasCustomDefinition bool
	for _, item := range request.Input {
		switch item.Type {
		case "custom_tool_call":
			hasCustomCall = item.CallID == "call_patch_001"
		case "custom_tool_call_output":
			hasCustomOutput = item.CallID == "call_patch_001"
		}
	}
	for _, tool := range request.Tools {
		if tool.Type == "custom" && tool.Name == "apply_patch" {
			hasCustomDefinition = true
		}
	}

	missing := make([]string, 0, 3)
	if !hasCustomCall {
		missing = append(missing, "custom_tool_call")
	}
	if !hasCustomOutput {
		missing = append(missing, "custom_tool_call_output")
	}
	if !hasCustomDefinition {
		missing = append(missing, "custom tool definition")
	}
	if len(missing) > 0 {
		return &testValidationError{"outbound request lost " + strings.Join(missing, ", ")}
	}
	return nil
}

func streamResponsesFixture(t *testing.T, w http.ResponseWriter, fixturePath string) {
	t.Helper()

	fixture, err := os.Open(fixturePath)
	if err != nil {
		t.Errorf("open SSE fixture: %v", err)
		return
	}
	defer fixture.Close()

	flusher, ok := w.(http.Flusher)
	if !ok {
		t.Error("test server does not support flushing")
		return
	}

	scanner := bufio.NewScanner(fixture)
	scanner.Buffer(make([]byte, 4096), 1<<20)
	for scanner.Scan() {
		var event struct {
			Type string `json:"Type"`
			Data string `json:"Data"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			t.Errorf("decode SSE fixture event: %v", err)
			return
		}
		if _, err := w.Write([]byte("event: " + event.Type + "\n")); err != nil {
			return
		}
		if _, err := w.Write([]byte("data: " + event.Data + "\n\n")); err != nil {
			return
		}
		flusher.Flush()
	}
	if err := scanner.Err(); err != nil {
		t.Errorf("scan SSE fixture: %v", err)
	}
}

type testValidationError struct {
	message string
}

func (e *testValidationError) Error() string {
	return e.message
}

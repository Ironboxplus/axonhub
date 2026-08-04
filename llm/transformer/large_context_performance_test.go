//go:build !race

package transformer_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

const (
	largeContextMessageCount = 425
	largeContextBodyBytes    = 1_490_000
	largeContextParseLimit   = 2 * time.Second
)

// TestLargeContextInboundTransformersStayWithinBudget covers the request shape
// that exposed a multi-second local-processing regression in production. The
// payloads are genuine wire documents for each public protocol; no transport or
// transformer interface is mocked.
func TestLargeContextInboundTransformersStayWithinBudget(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		transformer transformer.Inbound
		body        []byte
	}{
		{name: "chat_completions", transformer: openai.NewInboundTransformer(), body: largeChatRequest(t)},
		{name: "responses", transformer: responses.NewInboundTransformer(), body: largeResponsesRequest(t)},
		{name: "anthropic_messages", transformer: anthropic.NewInboundTransformer(), body: largeAnthropicRequest(t)},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if len(test.body) < largeContextBodyBytes {
				t.Fatalf("fixture is only %d bytes; want at least %d", len(test.body), largeContextBodyBytes)
			}

			startedAt := time.Now()
			request, err := test.transformer.TransformRequest(context.Background(), &httpclient.Request{
				Headers: http.Header{"Content-Type": []string{"application/json"}},
				Body:    test.body,
			})
			elapsed := time.Since(startedAt)
			if err != nil {
				t.Fatalf("transform %d-byte request: %v", len(test.body), err)
			}
			if got := len(request.Messages); got != largeContextMessageCount {
				t.Fatalf("message count = %d, want %d", got, largeContextMessageCount)
			}
			if elapsed > largeContextParseLimit {
				t.Fatalf("Axon inbound transform took %s for %d messages / %d bytes; limit %s", elapsed, len(request.Messages), len(test.body), largeContextParseLimit)
			}
			t.Logf("Axon inbound transform: messages=%d bytes=%d elapsed=%s", len(request.Messages), len(test.body), elapsed)
		})
	}
}

func largeChatRequest(t *testing.T) []byte {
	t.Helper()
	messages := make([]map[string]any, 0, largeContextMessageCount)
	for i := 0; i < largeContextMessageCount; i++ {
		messages = append(messages, map[string]any{
			"role":    alternatingRole(i),
			"content": largeContextText(i),
		})
	}
	return marshalLargeRequest(t, map[string]any{
		"model":    "large-context-test",
		"messages": messages,
		"stream":   true,
	})
}

func largeResponsesRequest(t *testing.T) []byte {
	t.Helper()
	input := make([]map[string]any, 0, largeContextMessageCount)
	for i := 0; i < largeContextMessageCount; i++ {
		input = append(input, map[string]any{
			"type": "message",
			"role": alternatingRole(i),
			"content": []map[string]any{{
				"type": "input_text",
				"text": largeContextText(i),
			}},
		})
	}
	return marshalLargeRequest(t, map[string]any{
		"model":  "large-context-test",
		"input":  input,
		"stream": true,
	})
}

func largeAnthropicRequest(t *testing.T) []byte {
	t.Helper()
	messages := make([]map[string]any, 0, largeContextMessageCount)
	for i := 0; i < largeContextMessageCount; i++ {
		messages = append(messages, map[string]any{
			"role": alternatingRole(i),
			"content": []map[string]any{{
				"type": "text",
				"text": largeContextText(i),
			}},
		})
	}
	return marshalLargeRequest(t, map[string]any{
		"model":      "large-context-test",
		"messages":   messages,
		"max_tokens": 1024,
		"stream":     true,
	})
}

func alternatingRole(index int) string {
	if index%2 == 0 {
		return "user"
	}
	return "assistant"
}

func largeContextText(index int) string {
	return fmt.Sprintf("message-%03d ", index) + strings.Repeat("context payload with unicode 路径 C:\\\\Users\\\\Arc\\\\octopus and email arc@example.com. ", 45)
}

func marshalLargeRequest(t *testing.T, value any) []byte {
	t.Helper()
	body, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal large request: %v", err)
	}
	return body
}

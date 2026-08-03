// axon-request-project re-encodes a captured OpenAI Responses request into a
// different public ingress protocol without contacting a provider.
//
// It is an acceptance utility, not a second conversion implementation: all
// decoding, canonical planning, lowering, and target encoding go through the
// production Axon pipeline and transformers.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

var errCaptured = errors.New("projected request captured")

type captureExecutor struct {
	body []byte
}

func (e *captureExecutor) capture(request *httpclient.Request) error {
	if request == nil {
		return errors.New("outbound transformer produced a nil request")
	}
	body := request.Body
	if len(request.JSONBody) > 0 {
		body = request.JSONBody
	}
	if len(body) == 0 {
		return errors.New("outbound transformer produced an empty request body")
	}
	e.body = append(e.body[:0], body...)
	return errCaptured
}

func (e *captureExecutor) Do(_ context.Context, request *httpclient.Request) (*httpclient.Response, error) {
	return nil, e.capture(request)
}

func (e *captureExecutor) DoStream(_ context.Context, request *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
	return nil, e.capture(request)
}

func targetOutbound(target string) (transformer.Outbound, error) {
	switch target {
	case "chat":
		return openai.NewOutboundTransformer("http://acceptance.invalid", "acceptance-key")
	case "responses":
		return responses.NewOutboundTransformer("http://acceptance.invalid", "acceptance-key")
	case "anthropic":
		return anthropic.NewOutboundTransformer("http://acceptance.invalid", "acceptance-key")
	default:
		return nil, fmt.Errorf("unsupported target protocol %q", target)
	}
}

func projectResponsesRequest(ctx context.Context, input []byte, target, model string) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(input, &payload); err != nil {
		return nil, fmt.Errorf("decode captured Responses request: %w", err)
	}
	if model != "" {
		payload["model"] = model
	}
	// The projection is later sent as an ordinary non-streaming HTTP request so
	// the acceptance driver can validate one complete JSON response per route.
	payload["stream"] = false
	input, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode normalized Responses request: %w", err)
	}

	outbound, err := targetOutbound(target)
	if err != nil {
		return nil, err
	}
	executor := &captureExecutor{}
	_, err = pipeline.NewFactory(executor).
		Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
		Process(ctx, &httpclient.Request{
			Method:  http.MethodPost,
			URL:     "/v1/responses",
			Headers: http.Header{"Content-Type": []string{"application/json"}},
			Body:    input,
		})
	if !errors.Is(err, errCaptured) {
		if err == nil {
			return nil, errors.New("projection pipeline returned without capturing an outbound request")
		}
		return nil, fmt.Errorf("project Responses request to %s: %w", target, err)
	}
	if !json.Valid(executor.body) {
		return nil, errors.New("projected request is not valid JSON")
	}
	return executor.body, nil
}

func run() error {
	inputPath := flag.String("input", "", "captured OpenAI Responses request JSON")
	outputPath := flag.String("output", "", "destination JSON file")
	target := flag.String("target", "", "chat, responses, or anthropic")
	model := flag.String("model", "", "model alias to place on the projected request")
	flag.Parse()
	if *inputPath == "" || *outputPath == "" || *target == "" {
		return errors.New("--input, --output, and --target are required")
	}
	input, err := os.ReadFile(*inputPath)
	if err != nil {
		return fmt.Errorf("read input: %w", err)
	}
	projected, err := projectResponsesRequest(context.Background(), input, *target, *model)
	if err != nil {
		return err
	}
	if err := os.WriteFile(*outputPath, projected, 0o600); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

func TestImageGenerationStreamingRoundTripsThroughRealHTTPPipeline(t *testing.T) {
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.URL.Path != "/v1/images/generations" {
			t.Errorf("provider path = %q", request.URL.Path)
		}
		if request.Header.Get("Authorization") != "Bearer provider-key" {
			t.Errorf("provider authorization was not finalized")
		}
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read provider request: %v", err)
			return
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("decode provider request: %v", err)
			return
		}
		if payload["model"] != "gpt-image-1" || payload["prompt"] != "draw a robust octopus" || payload["stream"] != true {
			t.Errorf("provider payload = %s", body)
		}

		writer.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprint(writer, "event: image_generation.partial_image\n")
		_, _ = fmt.Fprint(writer, `data: {"type":"image_generation.partial_image","b64_json":"cGFydGlhbA==","partial_image_index":0,"output_format":"png"}`+"\n\n")
		_, _ = fmt.Fprint(writer, "event: image_generation.completed\n")
		// Intentionally omit the final blank line. Real proxies can terminate
		// immediately after a valid final event; the decoder must not panic.
		_, _ = fmt.Fprint(writer, `data: {"type":"image_generation.completed","b64_json":"ZmluYWw=","output_format":"png","usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18}}`+"\n")
	}))
	defer provider.Close()

	clientRequest, err := http.NewRequest(http.MethodPost, "http://client.invalid/v1/images/generations", strings.NewReader(
		`{"model":"gpt-image-1","prompt":"draw a robust octopus","stream":true,"partial_images":1}`,
	))
	if err != nil {
		t.Fatalf("create client request: %v", err)
	}
	clientRequest.Header.Set("Content-Type", "application/json")
	raw, err := httpclient.ReadHTTPRequestWithLimit(clientRequest, 1<<20)
	if err != nil {
		t.Fatalf("read client request: %v", err)
	}
	outbound, err := NewOutboundTransformer(provider.URL+"/v1", "provider-key")
	if err != nil {
		t.Fatalf("create outbound: %v", err)
	}
	result, err := pipeline.NewFactory(httpclient.NewHttpClientWithClient(provider.Client(), httpclient.WithMaxSSEEventSize(1<<20))).
		Pipeline(NewImageGenerationInboundTransformer(), outbound, pipeline.WithEmptyResponseDetection()).
		Process(context.Background(), raw)
	if err != nil {
		t.Fatalf("process real image stream: %v", err)
	}
	if result == nil || !result.Stream || result.EventStream == nil {
		t.Fatalf("image stream result = %#v", result)
	}

	var rendered bytes.Buffer
	events := 0
	for result.EventStream.Next() {
		event := result.EventStream.Current()
		if event == nil {
			continue
		}
		events++
		rendered.Write(event.Data)
	}
	streamErr := result.EventStream.Err()
	if closeErr := result.EventStream.Close(); streamErr == nil {
		streamErr = closeErr
	}
	if streamErr != nil {
		t.Fatalf("consume image stream: %v", streamErr)
	}
	if events != 2 || !bytes.Contains(rendered.Bytes(), []byte(`"b64_json":"cGFydGlhbA=="`)) ||
		!bytes.Contains(rendered.Bytes(), []byte(`"b64_json":"ZmluYWw="`)) ||
		!bytes.Contains(rendered.Bytes(), []byte(`"input_tokens":11`)) {
		t.Fatalf("rendered image events=%d body=%s", events, rendered.Bytes())
	}
}

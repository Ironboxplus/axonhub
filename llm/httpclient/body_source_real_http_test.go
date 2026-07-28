package httpclient

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

type replayableTestBody struct {
	payload []byte
	opens   atomic.Int64
}

type trackedReadCloser struct {
	io.Reader
	closed *atomic.Bool
}

func (r *trackedReadCloser) Close() error {
	r.closed.Store(true)
	return nil
}

type closeTrackedBodySource struct {
	closed atomic.Bool
}

func (s *closeTrackedBodySource) Open(context.Context) (io.ReadCloser, error) {
	return &trackedReadCloser{Reader: strings.NewReader("body"), closed: &s.closed}, nil
}

func (s *closeTrackedBodySource) Size() int64 { return 4 }

func (s *replayableTestBody) Open(context.Context) (io.ReadCloser, error) {
	s.opens.Add(1)
	return io.NopCloser(bytes.NewReader(s.payload)), nil
}

func (s *replayableTestBody) Size() int64 { return int64(len(s.payload)) }

func TestReplayableBodyAndSuccessfulResponsePassthroughUseRealHTTP(t *testing.T) {
	source := &replayableTestBody{payload: []byte("large-multipart-body")}
	provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		body, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
			return
		}
		if !bytes.Equal(body, source.payload) {
			t.Errorf("provider body = %q, want %q", body, source.payload)
		}
		if request.ContentLength != int64(len(source.payload)) {
			t.Errorf("content length = %d, want %d", request.ContentLength, len(source.payload))
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"data":[{"b64_json":"`))
		if flusher, ok := writer.(http.Flusher); ok {
			flusher.Flush()
		}
		_, _ = writer.Write([]byte(`aGVsbG8="}]}`))
	}))
	defer provider.Close()

	request := &Request{
		Method:           http.MethodPost,
		URL:              provider.URL,
		Headers:          make(http.Header),
		BodySource:       source,
		ResponseBodyMode: ResponseBodyModeStream,
	}
	client := NewHttpClientWithClient(provider.Client())
	for attempt := int64(1); attempt <= 2; attempt++ {
		response, err := client.Do(context.Background(), request)
		if err != nil {
			t.Fatalf("real HTTP attempt %d: %v", attempt, err)
		}
		if len(response.Body) != 0 || response.BodyStream == nil {
			t.Fatalf("passthrough response = %#v", response)
		}
		body, err := io.ReadAll(response.BodyStream)
		if err != nil {
			t.Fatalf("read passthrough response: %v", err)
		}
		if err := response.Close(); err != nil {
			t.Fatalf("close passthrough response: %v", err)
		}
		if !bytes.Contains(body, []byte("aGVsbG8=")) {
			t.Fatalf("passthrough body = %s", body)
		}
	}
	if source.opens.Load() != 2 {
		t.Fatalf("body source opens = %d, want 2", source.opens.Load())
	}
}

func TestBuildHTTPRequestRejectsAmbiguousBodyRepresentations(t *testing.T) {
	_, err := BuildHttpRequest(context.Background(), &Request{
		Method:     http.MethodPost,
		URL:        "https://provider.invalid",
		Body:       []byte("buffered"),
		BodySource: &replayableTestBody{payload: []byte("source")},
	})
	if err == nil {
		t.Fatal("Body plus BodySource must be rejected")
	}
}

func TestBuildHTTPRequestClosesOpenedSourceWhenAuthenticationFails(t *testing.T) {
	source := &closeTrackedBodySource{}
	_, err := BuildHttpRequest(context.Background(), &Request{
		Method:     http.MethodPost,
		URL:        "https://provider.invalid",
		Headers:    make(http.Header),
		BodySource: source,
		Auth:       &AuthConfig{Type: AuthTypeBearer},
	})
	if err == nil {
		t.Fatal("empty bearer token unexpectedly accepted")
	}
	if !source.closed.Load() {
		t.Fatal("opened body source was not closed after request build failed")
	}
}

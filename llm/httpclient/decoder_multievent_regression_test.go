package httpclient

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

type panicBeforeEOFReadCloser struct{}

func (panicBeforeEOFReadCloser) Read([]byte) (int, error) { panic("test reader panic") }
func (panicBeforeEOFReadCloser) Close() error             { return nil }

func TestSSEDecoderConsumesMultipleEventsAndTerminalEventAtEOF(t *testing.T) {
	body := strings.Join([]string{
		`data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"hello"},"finish_reason":null}]}`,
		``,
		`data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	decoder := NewSSEDecoderWithMaxEventSize(context.Background(), io.NopCloser(strings.NewReader(body)), 1<<20)
	defer decoder.Close()

	var events []*StreamEvent
	for decoder.Next() {
		events = append(events, decoder.Current())
	}
	if err := decoder.Err(); err != nil {
		t.Fatalf("decode SSE: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("decoded event count = %d, want 3", len(events))
	}
	if string(events[0].Data) == "" || string(events[2].Data) != "[DONE]" {
		t.Fatalf("decoded events = %#v", events)
	}
}

func TestSSEDecoderDoesNotHidePanicsBeforePhysicalEOF(t *testing.T) {
	decoder := NewSSEDecoderWithMaxEventSize(context.Background(), panicBeforeEOFReadCloser{}, 1<<20)
	defer decoder.Close()

	if decoder.Next() {
		t.Fatal("panicking reader unexpectedly yielded an event")
	}
	if !errors.Is(decoder.Err(), errSSEDecoderPanic) {
		t.Fatalf("decoder error = %v, want errSSEDecoderPanic", decoder.Err())
	}
}

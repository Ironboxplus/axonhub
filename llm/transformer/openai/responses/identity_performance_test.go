package responses

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm/httpclient"
)

func BenchmarkResponsesIdentityNestedResidual512(b *testing.B) {
	benchmarkResponsesIdentityNestedResidual(b, 512)
}

func BenchmarkResponsesIdentityNestedResidualSizes(b *testing.B) {
	// These are ordinary message objects. The selected-union digest feature must
	// not make their 64->512 allocation slope worse.
	for _, messages := range []int{64, 512} {
		b.Run(strconv.Itoa(messages), func(b *testing.B) {
			benchmarkResponsesIdentityNestedResidual(b, messages)
		})
	}
}

func benchmarkResponsesIdentityNestedResidual(b *testing.B, messages int) {
	body := nestedResidualRequestBody(messages)
	inbound := NewInboundTransformer()
	outbound, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(body)))
	b.ResetTimer()
	for b.Loop() {
		if transformErr := runResponsesIdentityResidual(body, inbound, outbound); transformErr != nil {
			b.Fatal(transformErr)
		}
	}
}

func BenchmarkResponsesStreamIdentityResidual256(b *testing.B) {
	benchmarkResponsesStreamIdentityResidual(b, 256)
}

func BenchmarkResponsesStreamIdentityResidualSizes(b *testing.B) {
	for _, deltas := range []int{32, 128, 256} {
		b.Run(strconv.Itoa(deltas), func(b *testing.B) {
			benchmarkResponsesStreamIdentityResidual(b, deltas)
		})
	}
}

func benchmarkResponsesStreamIdentityResidual(b *testing.B, deltas int) {
	rawEvents := nestedResidualStreamEvents(deltas)
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if err := runResponsesStreamIdentityResidual(rawEvents); err != nil {
			b.Fatal(err)
		}
	}
}

func runResponsesIdentityResidual(body []byte, inbound *InboundTransformer, outbound *OutboundTransformer) error {
	canonical, err := inbound.TransformRequest(context.Background(), &httpclient.Request{Body: body})
	if err != nil {
		return err
	}
	_, err = outbound.TransformRequest(context.Background(), canonical)
	return err
}

func runResponsesStreamIdentityResidual(rawEvents [][]byte) error {
	decoder := newCanonicalStreamDecoder()
	encoder := newCanonicalStreamEncoder()
	source := &responsesInboundStream{
		ctx: context.Background(), transformerMetadata: make(map[string]any), aggregator: newStreamAggregator(),
	}
	for index := range rawEvents {
		var wire StreamEvent
		if err := wire.UnmarshalJSON(rawEvents[index]); err != nil {
			return err
		}
		events, err := decoder.decode(&wire)
		if err != nil {
			return err
		}
		for eventIndex := range events {
			if err := encoder.encode(source, events[eventIndex]); err != nil {
				return err
			}
		}
	}
	return nil
}

func nestedResidualRequestBody(messages int) []byte {
	var body strings.Builder
	body.Grow(messages * 180)
	body.WriteString(`{"model":"fixture-model","client_metadata":{"future":1},"input":[`)
	for index := 0; index < messages; index++ {
		if index > 0 {
			body.WriteByte(',')
		}
		body.WriteString(`{"type":"message","id":"msg_`)
		body.WriteString(strconv.Itoa(index))
		body.WriteString(`","role":"user","future_message":{"owner":`)
		body.WriteString(strconv.Itoa(index))
		body.WriteString(`},"content":[{"type":"input_text","text":"hello","future_content":{"owner":`)
		body.WriteString(strconv.Itoa(index))
		body.WriteString(`}}]}`)
	}
	body.WriteString(`]}`)
	return []byte(body.String())
}

func nestedResidualStreamEvents(deltas int) [][]byte {
	events := make([][]byte, 0, deltas+7)
	events = append(events,
		[]byte(`{"type":"response.created","response":{"id":"resp_perf","object":"response","created_at":1,"model":"fixture-model","status":"in_progress","output":[],"future_response":{"trace":1}}}`),
		[]byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_perf","type":"message","role":"assistant","status":"in_progress","content":[],"future_item":{"trace":1}}}`),
		[]byte(`{"type":"response.content_part.added","item_id":"msg_perf","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"future_part":{"trace":1}}}`),
	)
	for index := 0; index < deltas; index++ {
		events = append(events, []byte(fmt.Sprintf(
			`{"type":"response.output_text.delta","item_id":"msg_perf","output_index":0,"content_index":0,"delta":"x","future_delta":{"sequence":%d}}`,
			index,
		)))
	}
	text := strings.Repeat("x", deltas)
	events = append(events,
		[]byte(fmt.Sprintf(`{"type":"response.output_text.done","item_id":"msg_perf","output_index":0,"content_index":0,"text":%q}`, text)),
		[]byte(fmt.Sprintf(`{"type":"response.content_part.done","item_id":"msg_perf","output_index":0,"content_index":0,"part":{"type":"output_text","text":%q,"annotations":[],"future_part":{"trace":2}}}`, text)),
		[]byte(fmt.Sprintf(`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_perf","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}}`, text)),
		[]byte(fmt.Sprintf(`{"type":"response.completed","response":{"id":"resp_perf","object":"response","created_at":1,"model":"fixture-model","status":"completed","output":[{"id":"msg_perf","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":%q,"annotations":[]}]}]}}`, text)),
	)
	return events
}

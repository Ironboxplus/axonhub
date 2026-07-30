package conversion_test

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

const (
	documentDataURL = "data:application/pdf;base64,JVBERi0xLjQ="
	documentBase64  = "JVBERi0xLjQ="
	imageDataURL    = "data:image/png;base64,iVBORw0KGgo="
)

type documentProtocolCase struct {
	name       string
	format     llm.APIFormat
	path       string
	inbound    func() transformer.Inbound
	outbound   func(string) (transformer.Outbound, error)
	request    []byte
	response   string
	assertWire func(*testing.T, []byte)
}

func TestMixedTextImageDocumentRequestMatrixOverRealHTTP(t *testing.T) {
	t.Parallel()
	cases := []documentProtocolCase{
		{
			name: "chat", format: llm.APIFormatOpenAIChatCompletion, path: "/v1/chat/completions",
			inbound: func() transformer.Inbound { return openai.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:    []byte(`{"model":"fixture-model","messages":[{"role":"user","content":[{"type":"text","text":"inspect both"},{"type":"image_url","image_url":{"url":"` + imageDataURL + `"}},{"type":"file","file":{"file_data":"` + documentDataURL + `","filename":"report.pdf"}}]}]}`),
			response:   `{"id":"chat_doc","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
			assertWire: assertChatMixedDocumentWire,
		},
		{
			name: "responses", format: llm.APIFormatOpenAIResponse, path: "/v1/responses",
			inbound: func() transformer.Inbound { return responses.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:    []byte(`{"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"inspect both"},{"type":"input_image","image_url":"` + imageDataURL + `"},{"type":"input_file","file_data":"` + documentDataURL + `","filename":"report.pdf"}]}]}`),
			response:   `{"id":"resp_doc","object":"response","model":"fixture-model","status":"completed","output":[{"id":"msg_doc","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}`,
			assertWire: assertResponsesMixedDocumentWire,
		},
		{
			name: "anthropic", format: llm.APIFormatAnthropicMessage, path: "/v1/messages",
			inbound: func() transformer.Inbound { return anthropic.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:    []byte(`{"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"inspect both"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + documentBase64 + `"},"title":"report.pdf"}]}]}`),
			response:   `{"id":"msg_doc","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":8,"output_tokens":1}}`,
			assertWire: assertAnthropicMixedDocumentWire,
		},
	}

	runDocumentRequestMatrix(t, cases)
}

func TestFileIDDocumentRequestMatrixOverRealHTTP(t *testing.T) {
	t.Parallel()
	const fileID = "file_shared_123"
	cases := []documentProtocolCase{
		{
			name: "chat", format: llm.APIFormatOpenAIChatCompletion, path: "/v1/chat/completions",
			inbound: func() transformer.Inbound { return openai.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:  []byte(`{"model":"fixture-model","messages":[{"role":"user","content":[{"type":"text","text":"inspect file"},{"type":"file","file":{"file_id":"` + fileID + `","filename":"report.pdf"}}]}]}`),
			response: `{"id":"chat_file","object":"chat.completion","model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
			assertWire: func(t *testing.T, body []byte) {
				assertDocumentWireFields(t, body, `"type":"file"`, `"file_id":"`+fileID+`"`, `"filename":"report.pdf"`)
			},
		},
		{
			name: "responses", format: llm.APIFormatOpenAIResponse, path: "/v1/responses",
			inbound: func() transformer.Inbound { return responses.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:  []byte(`{"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"inspect file"},{"type":"input_file","file_id":"` + fileID + `","filename":"report.pdf"}]}]}`),
			response: `{"id":"resp_file","object":"response","model":"fixture-model","status":"completed","output":[{"id":"msg_file","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"ok"}]}]}`,
			assertWire: func(t *testing.T, body []byte) {
				assertDocumentWireFields(t, body, `"type":"input_file"`, `"file_id":"`+fileID+`"`, `"filename":"report.pdf"`)
			},
		},
		{
			name: "anthropic", format: llm.APIFormatAnthropicMessage, path: "/v1/messages",
			inbound: func() transformer.Inbound { return anthropic.NewInboundTransformer() },
			outbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			request:  []byte(`{"model":"fixture-model","max_tokens":64,"messages":[{"role":"user","content":[{"type":"text","text":"inspect file"},{"type":"document","source":{"type":"file","file_id":"` + fileID + `"},"title":"report.pdf"}]}]}`),
			response: `{"id":"msg_file","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":8,"output_tokens":1}}`,
			assertWire: func(t *testing.T, body []byte) {
				assertDocumentWireFields(t, body, `"type":"document"`, `"type":"file"`, `"file_id":"`+fileID+`"`, `"title":"report.pdf"`)
			},
		},
	}
	runDocumentRequestMatrix(t, cases)
}

func runDocumentRequestMatrix(t *testing.T, cases []documentProtocolCase) {
	t.Helper()
	for _, source := range cases {
		source := source
		for _, target := range cases {
			target := target
			t.Run(source.name+"_to_"+target.name, func(t *testing.T) {
				t.Parallel()
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
					if request.URL.Path != target.path {
						http.Error(writer, "unexpected path", http.StatusNotFound)
						return
					}
					body, err := io.ReadAll(request.Body)
					if err != nil {
						http.Error(writer, "read body", http.StatusBadRequest)
						return
					}
					target.assertWire(t, body)
					writer.Header().Set("Content-Type", "application/json")
					_, _ = io.WriteString(writer, target.response)
				}))
				t.Cleanup(provider.Close)
				outbound, err := target.outbound(provider.URL)
				if err != nil {
					t.Fatalf("create %s outbound: %v", target.name, err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				result, err := pipeline.NewFactory(executor).
					Pipeline(source.inbound(), conversion.NewOutbound(outbound)).
					Process(context.Background(), &httpclient.Request{
						Method: http.MethodPost, URL: source.path,
						Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: source.request,
					})
				if err != nil {
					t.Fatalf("%s -> %s mixed document conversion: %v", source.name, target.name, err)
				}
				if result == nil || result.Response == nil || !bytes.Contains(result.Response.Body, []byte("ok")) {
					t.Fatalf("%s -> %s client response = %#v", source.name, target.name, result)
				}
			})
		}
	}
}

func assertDocumentWireFields(t *testing.T, body []byte, fields ...string) {
	t.Helper()
	for _, expected := range fields {
		if !bytes.Contains(body, []byte(expected)) {
			t.Fatalf("document wire missing %s: %s", expected, body)
		}
	}
}

func assertChatMixedDocumentWire(t *testing.T, body []byte) {
	t.Helper()
	for _, expected := range []string{`"type":"text"`, `"type":"image_url"`, imageDataURL, `"type":"file"`, documentDataURL, `"filename":"report.pdf"`} {
		if !bytes.Contains(body, []byte(expected)) {
			t.Fatalf("Chat mixed document wire missing %s: %s", expected, body)
		}
	}
}

func assertResponsesMixedDocumentWire(t *testing.T, body []byte) {
	t.Helper()
	for _, expected := range []string{`"type":"input_text"`, `"type":"input_image"`, imageDataURL, `"type":"input_file"`, documentDataURL, `"filename":"report.pdf"`} {
		if !bytes.Contains(body, []byte(expected)) {
			t.Fatalf("Responses mixed document wire missing %s: %s", expected, body)
		}
	}
}

func assertAnthropicMixedDocumentWire(t *testing.T, body []byte) {
	t.Helper()
	for _, expected := range []string{`"type":"text"`, `"type":"image"`, `"type":"document"`, `"type":"base64"`, `"media_type":"application/pdf"`, documentBase64, `"title":"report.pdf"`} {
		if !bytes.Contains(body, []byte(expected)) {
			t.Fatalf("Anthropic mixed document wire missing %s: %s", expected, body)
		}
	}
	if bytes.Contains(body, []byte(documentDataURL)) {
		t.Fatalf("Anthropic document embedded the data URL wrapper instead of raw base64: %s", body)
	}
}

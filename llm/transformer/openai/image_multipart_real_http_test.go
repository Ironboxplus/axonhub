package openai

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
)

func TestSourceBackedImageMultipartRoundTripsThroughRealHTTPPipeline(t *testing.T) {
	tests := []struct {
		name       string
		path       string
		inbound    func() *ImageInboundTransformer
		withPrompt bool
		withMask   bool
	}{
		{name: "edit", path: "/v1/images/edits", inbound: NewImageEditInboundTransformer, withPrompt: true, withMask: true},
		{name: "variation", path: "/v1/images/variations", inbound: NewImageVariationInboundTransformer},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			imageBytes := bytes.Repeat([]byte("octopus-image-data-"), 64*1024)
			maskBytes := bytes.Repeat([]byte("octopus-mask-"), 1024)
			payload, contentType := buildSourceBackedImageMultipart(t, imageBytes, maskBytes, test.withPrompt, test.withMask)
			source := &countingImageBodySource{payload: payload}

			var providerCalls atomic.Int32
			provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				call := providerCalls.Add(1)
				if request.URL.Path != test.path {
					t.Errorf("provider path = %q, want %q", request.URL.Path, test.path)
				}
				if request.Header.Get("Authorization") != "Bearer provider-key" {
					t.Errorf("provider authorization was not finalized")
				}
				if request.ContentLength != -1 {
					t.Errorf("source-backed multipart content length = %d, want streamed unknown length", request.ContentLength)
				}

				fields, files, err := readProviderMultipart(request)
				if err != nil {
					t.Errorf("parse provider multipart: %v", err)
					return
				}
				if got := fields["model"]; len(got) != 1 || got[0] != "provider-image-model" {
					t.Errorf("provider model fields = %#v", got)
				}
				if got := fields["user"]; len(got) != 1 || got[0] != "masked-user" {
					t.Errorf("provider user fields = %#v", got)
				}
				if got := fields["vendor_option"]; len(got) != 2 || got[0] != "keep-one" || got[1] != "keep-two" {
					t.Errorf("provider extension fields = %#v", got)
				}
				if test.withPrompt {
					if got := fields["prompt"]; len(got) != 1 || got[0] != "masked-prompt" {
						t.Errorf("provider prompt fields = %#v", got)
					}
				} else if _, exists := fields["prompt"]; exists {
					t.Errorf("variation unexpectedly forwarded prompt")
				}
				assertFileHash(t, files["image"], imageBytes)
				if test.withMask {
					assertFileHash(t, files["mask"], maskBytes)
				}

				if call == 1 {
					writer.Header().Set("Content-Type", "application/json")
					writer.WriteHeader(http.StatusServiceUnavailable)
					_, _ = io.WriteString(writer, `{"error":{"message":"retry me"}}`)
					return
				}

				writer.Header().Set("Content-Type", "application/json")
				writer.WriteHeader(http.StatusOK)
				if flusher, ok := writer.(http.Flusher); ok {
					flusher.Flush()
				}
				// Ensure Process has returned before the payload becomes readable. This
				// catches accidental early cancellation of passthrough response bodies.
				time.Sleep(25 * time.Millisecond)
				_, _ = io.WriteString(writer, `{"created":123,"data":[{"b64_json":"cmVhbC1pbWFnZQ=="}],"usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}`)
			}))
			defer provider.Close()

			clientRequest, err := http.NewRequest(http.MethodPost, "http://client.invalid"+test.path, bytes.NewReader(payload))
			if err != nil {
				t.Fatalf("create client request: %v", err)
			}
			clientRequest.Header.Set("Content-Type", contentType)
			rawRequest, err := httpclient.NewRequestWithBodySource(clientRequest, source)
			if err != nil {
				t.Fatalf("capture source-backed request: %v", err)
			}

			outbound, err := NewOutboundTransformer(provider.URL+"/v1", "provider-key")
			if err != nil {
				t.Fatalf("create outbound: %v", err)
			}
			canonicalRewrite := pipeline.OnLlmRequest("canonical-image-rewrite", func(_ context.Context, request *llm.Request) (*llm.Request, error) {
				request.Model = "provider-image-model"
				request.Image.User = "masked-user"
				if test.withPrompt {
					request.Image.Prompt = "masked-prompt"
				}
				request.RawResponsePassthrough = true
				return request, nil
			})
			imagePipeline := pipeline.NewFactory(httpclient.NewHttpClientWithClient(provider.Client())).Pipeline(
				test.inbound(),
				outbound,
				pipeline.WithMiddlewares(canonicalRewrite),
				pipeline.WithResponseTimeouts(0, 2*time.Second),
			)

			if _, err := imagePipeline.Process(context.Background(), rawRequest); err == nil {
				t.Fatal("first real provider attempt unexpectedly succeeded")
			}
			result, err := imagePipeline.Process(context.Background(), rawRequest)
			if err != nil {
				t.Fatalf("second real provider attempt: %v", err)
			}
			if result == nil || result.Response == nil || result.Response.BodyStream == nil {
				t.Fatalf("passthrough result = %#v", result)
			}
			body, readErr := io.ReadAll(result.Response.BodyStream)
			closeErr := result.Response.Close()
			if readErr != nil || closeErr != nil {
				t.Fatalf("read/close passthrough response: read=%v close=%v", readErr, closeErr)
			}
			if !bytes.Contains(body, []byte(`"b64_json":"cmVhbC1pbWFnZQ=="`)) {
				t.Fatalf("passthrough response = %s", body)
			}
			if got := source.opens.Load(); got != 4 {
				t.Fatalf("body source opens = %d, want two validation scans plus two real transports", got)
			}
			if got := providerCalls.Load(); got != 2 {
				t.Fatalf("provider calls = %d, want 2", got)
			}
		})
	}
}

type countingImageBodySource struct {
	payload []byte
	opens   atomic.Int32
}

func (s *countingImageBodySource) Open(context.Context) (io.ReadCloser, error) {
	s.opens.Add(1)
	return io.NopCloser(bytes.NewReader(s.payload)), nil
}

func (s *countingImageBodySource) Size() int64 {
	return int64(len(s.payload))
}

func buildSourceBackedImageMultipart(t *testing.T, imageBytes, maskBytes []byte, withPrompt, withMask bool) ([]byte, string) {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	writeField := func(name, value string) {
		if err := writer.WriteField(name, value); err != nil {
			t.Fatalf("write multipart field %s: %v", name, err)
		}
	}
	writeField("model", "client-image-model")
	writeField("user", "private-user")
	writeField("vendor_option", "keep-one")
	writeField("vendor_option", "keep-two")
	if withPrompt {
		writeField("prompt", `edit C:\private\octopus.png`)
	}
	writeImageTestPart(t, writer, "image", "octopus.png", imageBytes)
	if withMask {
		writeImageTestPart(t, writer, "mask", "mask.png", maskBytes)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close multipart input: %v", err)
	}
	return body.Bytes(), writer.FormDataContentType()
}

func writeImageTestPart(t *testing.T, writer *multipart.Writer, name, filename string, data []byte) {
	t.Helper()
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name=%q; filename=%q`, name, filename))
	header.Set("Content-Type", "image/png")
	part, err := writer.CreatePart(header)
	if err != nil {
		t.Fatalf("create image part: %v", err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatalf("write image part: %v", err)
	}
}

type providerFile struct {
	filename string
	hash     string
	size     int64
}

func readProviderMultipart(request *http.Request) (map[string][]string, map[string]providerFile, error) {
	reader, err := request.MultipartReader()
	if err != nil {
		return nil, nil, err
	}
	fields := make(map[string][]string)
	files := make(map[string]providerFile)
	for {
		part, err := reader.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, err
		}
		if part.FileName() == "" {
			value, readErr := io.ReadAll(part)
			_ = part.Close()
			if readErr != nil {
				return nil, nil, readErr
			}
			fields[part.FormName()] = append(fields[part.FormName()], string(value))
			continue
		}
		hash := sha256.New()
		size, copyErr := io.Copy(hash, part)
		_ = part.Close()
		if copyErr != nil {
			return nil, nil, copyErr
		}
		files[part.FormName()] = providerFile{filename: part.FileName(), hash: hex.EncodeToString(hash.Sum(nil)), size: size}
	}
	return fields, files, nil
}

func assertFileHash(t *testing.T, actual providerFile, expected []byte) {
	t.Helper()
	hash := sha256.Sum256(expected)
	if actual.size != int64(len(expected)) || actual.hash != hex.EncodeToString(hash[:]) || !strings.HasSuffix(actual.filename, ".png") {
		t.Fatalf("provider file = %#v, expected size=%d hash=%s", actual, len(expected), hex.EncodeToString(hash[:]))
	}
}

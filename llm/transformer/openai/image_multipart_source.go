package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"strconv"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer"
)

type imageMultipartKind uint8

const (
	imageMultipartEdit imageMultipartKind = iota + 1
	imageMultipartVariation
)

type imageMultipartField struct {
	name  string
	value string
}

// multipartRewriteBodySource keeps large image files outside the unified
// semantic model. Each transport attempt reopens the original bounded source,
// copies file/extension parts verbatim, and writes canonical semantic fields
// after model mapping and request middleware have run.
type multipartRewriteBodySource struct {
	source         httpclient.BodySource
	inputBoundary  string
	outputBoundary string
	fields         []imageMultipartField
	knownFields    map[string]struct{}
}

func newMultipartRewriteBodySource(
	source httpclient.BodySource,
	inputBoundary string,
	fields []imageMultipartField,
	knownFields map[string]struct{},
) *multipartRewriteBodySource {
	boundaryWriter := multipart.NewWriter(io.Discard)
	outputBoundary := boundaryWriter.Boundary()
	_ = boundaryWriter.Close()

	return &multipartRewriteBodySource{
		source:         source,
		inputBoundary:  inputBoundary,
		outputBoundary: outputBoundary,
		fields:         append([]imageMultipartField(nil), fields...),
		knownFields:    knownFields,
	}
}

func (s *multipartRewriteBodySource) Open(ctx context.Context) (io.ReadCloser, error) {
	input, err := s.source.Open(ctx)
	if err != nil {
		return nil, err
	}

	reader, writer := io.Pipe()
	go func() {
		err := s.rewrite(ctx, input, writer)
		closeErr := input.Close()
		if err == nil {
			err = closeErr
		}
		_ = writer.CloseWithError(err)
	}()
	return reader, nil
}

func (s *multipartRewriteBodySource) Size() int64 {
	// Replacing semantic fields changes the encoded length. Unknown length lets
	// net/http stream it with chunked transfer encoding instead of buffering it.
	return -1
}

func (s *multipartRewriteBodySource) contentType() string {
	return mime.FormatMediaType("multipart/form-data", map[string]string{"boundary": s.outputBoundary})
}

func (s *multipartRewriteBodySource) rewrite(ctx context.Context, input io.Reader, output io.Writer) error {
	reader := multipart.NewReader(&contextReader{ctx: ctx, reader: input}, s.inputBoundary)
	writer := multipart.NewWriter(output)
	if err := writer.SetBoundary(s.outputBoundary); err != nil {
		return fmt.Errorf("set multipart boundary: %w", err)
	}

	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("read multipart part: %w", err)
		}

		_, replace := s.knownFields[part.FormName()]
		if part.FileName() == "" && replace {
			_, err = io.Copy(io.Discard, part)
			_ = part.Close()
			if err != nil {
				return fmt.Errorf("discard replaced multipart field: %w", err)
			}
			continue
		}

		header := cloneMIMEHeader(part.Header)
		destination, createErr := writer.CreatePart(header)
		if createErr != nil {
			_ = part.Close()
			return fmt.Errorf("create multipart part: %w", createErr)
		}
		_, copyErr := io.Copy(destination, part)
		_ = part.Close()
		if copyErr != nil {
			return fmt.Errorf("copy multipart part: %w", copyErr)
		}
	}

	for _, field := range s.fields {
		if err := writer.WriteField(field.name, field.value); err != nil {
			return fmt.Errorf("write multipart field %s: %w", field.name, err)
		}
	}
	if err := writer.Close(); err != nil {
		return fmt.Errorf("close multipart body: %w", err)
	}
	return nil
}

func cloneMIMEHeader(source textproto.MIMEHeader) textproto.MIMEHeader {
	cloned := make(textproto.MIMEHeader, len(source))
	for key, values := range source {
		cloned[key] = append([]string(nil), values...)
	}
	return cloned
}

func (t *OutboundTransformer) buildSourceBackedImageRequest(
	request *llm.Request,
	apiKey string,
	kind imageMultipartKind,
) (*httpclient.Request, error) {
	if request == nil || request.Image == nil || request.RawRequest == nil || request.RawRequest.BodySource == nil {
		return nil, fmt.Errorf("%w: replayable image body is required", transformer.ErrInvalidRequest)
	}
	if kind == imageMultipartEdit {
		if strings.TrimSpace(request.Image.Prompt) == "" {
			return nil, fmt.Errorf("%w: prompt is required for image editing", transformer.ErrInvalidRequest)
		}
		if len(request.Image.Images) == 0 {
			return nil, fmt.Errorf("%w: at least one image is required for image editing", transformer.ErrInvalidRequest)
		}
	} else if len(request.Image.Images) != 1 {
		return nil, fmt.Errorf("%w: exactly one image is required for image variations", transformer.ErrInvalidRequest)
	}

	inputBoundary, err := multipartBoundary(request.RawRequest.Headers.Get("Content-Type"))
	if err != nil {
		return nil, err
	}
	fields, known := canonicalImageMultipartFields(request, kind)
	source := newMultipartRewriteBodySource(request.RawRequest.BodySource, inputBoundary, fields, known)

	path := "/images/edits"
	if kind == imageMultipartVariation {
		path = "/images/variations"
	}
	url := t.config.BaseURL + path
	if t.config.EndpointPath != "" {
		url = t.config.BaseURL + t.config.EndpointPath
	}

	summary, err := json.Marshal(map[string]any{
		"model":       request.Model,
		"prompt":      request.Image.Prompt,
		"image_count": len(request.Image.Images),
		"has_mask":    kind == imageMultipartEdit && hasMultipartMask(request.RawRequest.JSONBody),
		"note":        "binary image data omitted",
	})
	if err != nil {
		return nil, fmt.Errorf("marshal multipart summary: %w", err)
	}

	headers := make(http.Header)
	headers.Set("Content-Type", source.contentType())
	headers.Set("Accept", "application/json")
	httpRequest := &httpclient.Request{
		Method:      http.MethodPost,
		URL:         url,
		Headers:     headers,
		ContentType: source.contentType(),
		BodySource:  source,
		JSONBody:    summary,
		Auth: &httpclient.AuthConfig{
			Type:   httpclient.AuthTypeBearer,
			APIKey: apiKey,
		},
	}
	if request.RawResponsePassthrough {
		httpRequest.ResponseBodyMode = httpclient.ResponseBodyModeStream
	}
	return httpRequest, nil
}

func multipartBoundary(contentType string) (string, error) {
	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil || !strings.HasPrefix(strings.ToLower(mediaType), "multipart/") {
		return "", fmt.Errorf("%w: invalid multipart content-type", transformer.ErrInvalidRequest)
	}
	boundary := strings.TrimSpace(params["boundary"])
	if boundary == "" {
		return "", fmt.Errorf("%w: missing multipart boundary", transformer.ErrInvalidRequest)
	}
	return boundary, nil
}

func canonicalImageMultipartFields(request *llm.Request, kind imageMultipartKind) ([]imageMultipartField, map[string]struct{}) {
	knownNames := []string{
		"model", "prompt", "n", "size", "response_format", "user", "stream",
	}
	if kind == imageMultipartEdit {
		knownNames = append(knownNames,
			"quality", "background", "output_format", "output_compression", "input_fidelity", "partial_images",
		)
	}
	known := make(map[string]struct{}, len(knownNames))
	for _, name := range knownNames {
		known[name] = struct{}{}
	}

	fields := make([]imageMultipartField, 0, len(knownNames))
	appendString := func(name, value string) {
		if value != "" {
			fields = append(fields, imageMultipartField{name: name, value: value})
		}
	}
	appendInt := func(name string, value *int64) {
		if value != nil {
			fields = append(fields, imageMultipartField{name: name, value: strconv.FormatInt(*value, 10)})
		}
	}

	appendString("model", request.Model)
	if kind == imageMultipartEdit {
		appendString("prompt", request.Image.Prompt)
	}
	appendInt("n", request.Image.N)
	appendString("size", request.Image.Size)
	appendString("user", request.Image.User)

	responseFormat := request.Image.ResponseFormat
	if kind == imageMultipartEdit && supportsImageEditResponseFormat(request.Model) && responseFormat == "" {
		responseFormat = "b64_json"
	}
	if kind == imageMultipartVariation && supportsImageVariationResponseFormat(request.Model) && responseFormat == "" {
		responseFormat = "b64_json"
	}
	if (kind == imageMultipartEdit && supportsImageEditResponseFormat(request.Model)) ||
		(kind == imageMultipartVariation && supportsImageVariationResponseFormat(request.Model)) {
		appendString("response_format", responseFormat)
	}

	if kind == imageMultipartEdit {
		appendString("quality", request.Image.Quality)
		appendString("background", request.Image.Background)
		appendString("output_format", request.Image.OutputFormat)
		appendInt("output_compression", request.Image.OutputCompression)
		appendString("input_fidelity", request.Image.InputFidelity)
		appendInt("partial_images", request.Image.PartialImages)
	}
	return fields, known
}

func hasMultipartMask(jsonBody []byte) bool {
	if len(jsonBody) == 0 {
		return false
	}
	var body map[string]json.RawMessage
	if json.Unmarshal(jsonBody, &body) != nil {
		return false
	}
	_, ok := body["mask"]
	return ok
}

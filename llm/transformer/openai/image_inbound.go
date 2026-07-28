package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/internal/pkg/xurl"
	"github.com/looplj/axonhub/llm/streams"
	transformer "github.com/looplj/axonhub/llm/transformer"
)

const (
	defaultMaxImageFileSize   = 50 * 1024 * 1024
	maxImageCount             = 16
	maxImageBodySize          = defaultMaxImageFileSize*maxImageCount + 16*1024*1024
	maxImageSemanticFieldSize = 1 * 1024 * 1024
)

var maxImageFileSize = initMaxImageFileSize()

func initMaxImageFileSize() int {
	if v := os.Getenv("AXONHUB_MAX_IMAGE_FILE_SIZE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}

	return defaultMaxImageFileSize
}

var allowedImageTypes = []string{
	"image/png",
	"image/jpeg",
	"image/gif",
	"image/webp",
}

// ImageGenerationRequest represents the request structure for image generation API.
type ImageGenerationRequest struct {
	Prompt            string `json:"prompt"`
	Model             string `json:"model"`
	N                 *int64 `json:"n,omitempty"`
	Quality           string `json:"quality,omitempty"`
	ResponseFormat    string `json:"response_format,omitempty"`
	Size              string `json:"size,omitempty"`
	Style             string `json:"style,omitempty"`
	User              string `json:"user,omitempty"`
	Background        string `json:"background,omitempty"`
	OutputFormat      string `json:"output_format,omitempty"`
	OutputCompression *int64 `json:"output_compression,omitempty"`
	Moderation        string `json:"moderation,omitempty"`
	PartialImages     *int64 `json:"partial_images,omitempty"`
	Stream            bool   `json:"stream,omitempty"`
}

type ImageInboundTransformer struct {
	apiFormat llm.APIFormat
}

func NewImageGenerationInboundTransformer() *ImageInboundTransformer {
	return &ImageInboundTransformer{
		apiFormat: llm.APIFormatOpenAIImageGeneration,
	}
}

func NewImageEditInboundTransformer() *ImageInboundTransformer {
	return &ImageInboundTransformer{
		apiFormat: llm.APIFormatOpenAIImageEdit,
	}
}

func NewImageVariationInboundTransformer() *ImageInboundTransformer {
	return &ImageInboundTransformer{
		apiFormat: llm.APIFormatOpenAIImageVariation,
	}
}

// APIFormat returns the API format of the transformer.
func (t *ImageInboundTransformer) APIFormat() llm.APIFormat {
	return t.apiFormat
}

func (t *ImageInboundTransformer) TransformRequest(ctx context.Context, httpReq *httpclient.Request) (*llm.Request, error) {
	if httpReq == nil {
		return nil, fmt.Errorf("%w: http request is nil", transformer.ErrInvalidRequest)
	}

	if len(httpReq.Body) == 0 && httpReq.BodySource == nil {
		return nil, fmt.Errorf("%w: request body is empty", transformer.ErrInvalidRequest)
	}

	//nolint:exhaustive // Only image-related API formats are handled here
	switch t.apiFormat {
	case llm.APIFormatOpenAIImageGeneration:
		return t.transformGenerationRequest(httpReq)
	case llm.APIFormatOpenAIImageEdit:
		return t.transformEditRequest(ctx, httpReq)
	case llm.APIFormatOpenAIImageVariation:
		return t.transformVariationRequest(ctx, httpReq)
	default:
		return nil, fmt.Errorf("%w: unknown image api format: %s", transformer.ErrInvalidRequest, t.apiFormat)
	}
}

func (t *ImageInboundTransformer) TransformResponse(ctx context.Context, llmResp *llm.Response) (*httpclient.Response, error) {
	if llmResp == nil || llmResp.Image == nil {
		return nil, fmt.Errorf("%w: image response is nil", transformer.ErrInvalidResponse)
	}

	created := llmResp.Created
	if created == 0 {
		created = time.Now().Unix()
	}

	oaiResp := ImagesResponse{
		Created: created,
		Data:    make([]ImageData, 0),
	}

	img := llmResp.Image
	oaiResp.Created = img.Created
	oaiResp.Background = img.Background
	oaiResp.OutputFormat = img.OutputFormat
	oaiResp.Quality = img.Quality
	oaiResp.Size = img.Size

	if llmResp.Usage != nil {
		oaiResp.Usage = &ImagesResponseUsage{
			InputTokens:  llmResp.Usage.PromptTokens,
			OutputTokens: llmResp.Usage.CompletionTokens,
			TotalTokens:  llmResp.Usage.TotalTokens,
		}
		if llmResp.Usage.PromptTokensDetails != nil {
			oaiResp.Usage.InputTokensDetails = &ImagesResponseUsageInputTokensDetails{
				ImageTokens:  llmResp.Usage.PromptTokensDetails.ImageTokens,
				TextTokens:   llmResp.Usage.PromptTokensDetails.TextTokens,
				CachedTokens: llmResp.Usage.PromptTokensDetails.CachedTokens,
			}
		}

		if llmResp.Usage.CompletionTokensDetails != nil {
			oaiResp.Usage.OutputTokensDetails = &ImagesResponseUsageOutputTokensDetails{
				ReasoningTokens: llmResp.Usage.CompletionTokensDetails.ReasoningTokens,
			}
		}
	}

	for _, data := range img.Data {
		oaiResp.Data = append(oaiResp.Data, ImageData{
			B64JSON:       data.B64JSON,
			URL:           data.URL,
			RevisedPrompt: data.RevisedPrompt,
		})
	}

	body, err := json.Marshal(oaiResp)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal image response: %w", err)
	}

	return &httpclient.Response{
		StatusCode: http.StatusOK,
		Body:       body,
		Headers: http.Header{
			"Content-Type":  []string{"application/json"},
			"Cache-Control": []string{"no-cache"},
		},
	}, nil
}

func (t *ImageInboundTransformer) TransformStream(ctx context.Context, stream streams.Stream[*llm.Response]) (streams.Stream[*httpclient.StreamEvent], error) {
	if t.apiFormat != llm.APIFormatOpenAIImageGeneration {
		return nil, fmt.Errorf("%w: only image generation supports streaming", transformer.ErrInvalidRequest)
	}
	return streams.NoNil(streams.MapErr(stream, renderImageStreamEvent)), nil
}

func (t *ImageInboundTransformer) TransformError(ctx context.Context, rawErr error) *httpclient.Error {
	chatInbound := NewInboundTransformer()
	return chatInbound.TransformError(ctx, rawErr)
}

func (t *ImageInboundTransformer) AggregateStreamChunks(ctx context.Context, chunks []*httpclient.StreamEvent) ([]byte, llm.ResponseMeta, error) {
	if t.apiFormat != llm.APIFormatOpenAIImageGeneration {
		return nil, llm.ResponseMeta{}, fmt.Errorf("%w: only image generation supports streaming", transformer.ErrInvalidRequest)
	}
	response := ImagesResponse{Data: make([]ImageData, 0, 1)}
	for _, chunk := range chunks {
		if chunk == nil || bytes.HasPrefix(bytes.TrimSpace(chunk.Data), []byte("[DONE]")) {
			continue
		}
		var event imageStreamWireEvent
		if err := json.Unmarshal(chunk.Data, &event); err != nil {
			return nil, llm.ResponseMeta{}, fmt.Errorf("failed to decode image stream chunk: %w", err)
		}
		if event.Type == "" {
			event.Type = chunk.Type
		}
		if !strings.HasSuffix(event.Type, ".completed") {
			continue
		}
		response.Created = event.Created
		response.Background = event.Background
		response.OutputFormat = event.OutputFormat
		response.Quality = event.Quality
		response.Size = event.Size
		response.Usage = event.Usage
		if event.B64JSON != "" || event.URL != "" {
			response.Data = append(response.Data, ImageData{B64JSON: event.B64JSON, URL: event.URL})
		}
	}
	if len(response.Data) == 0 {
		return nil, llm.ResponseMeta{}, fmt.Errorf("%w: image stream did not contain a completed image", transformer.ErrInvalidResponse)
	}
	body, err := json.Marshal(response)
	if err != nil {
		return nil, llm.ResponseMeta{}, fmt.Errorf("failed to aggregate image stream: %w", err)
	}
	return body, llm.ResponseMeta{Usage: imageUsageToLLM(response.Usage), Completed: true}, nil
}

func (t *ImageInboundTransformer) transformGenerationRequest(httpReq *httpclient.Request) (*llm.Request, error) {
	if httpReq.BodySource != nil {
		return nil, fmt.Errorf("%w: generations requires a buffered JSON body", transformer.ErrInvalidRequest)
	}
	contentType := strings.ToLower(httpReq.Headers.Get("Content-Type"))
	if !strings.Contains(contentType, "application/json") {
		return nil, fmt.Errorf("%w: generations requires application/json", transformer.ErrInvalidRequest)
	}

	var genReq ImageGenerationRequest

	if err := json.Unmarshal(httpReq.Body, &genReq); err != nil {
		return nil, fmt.Errorf("%w: failed to decode generation request: %w", transformer.ErrInvalidRequest, err)
	}

	model := genReq.Model
	if model == "" {
		model = "dall-e-2"
	}

	if genReq.Prompt == "" {
		return nil, fmt.Errorf("%w: prompt is required", transformer.ErrInvalidRequest)
	}

	imageReq := &llm.ImageRequest{
		Prompt:            genReq.Prompt,
		N:                 genReq.N,
		Size:              genReq.Size,
		Quality:           genReq.Quality,
		ResponseFormat:    genReq.ResponseFormat,
		User:              genReq.User,
		Background:        genReq.Background,
		OutputFormat:      genReq.OutputFormat,
		OutputCompression: genReq.OutputCompression,
		Moderation:        genReq.Moderation,
		PartialImages:     genReq.PartialImages,
		Style:             genReq.Style,
	}

	llmReq := &llm.Request{
		Model:       model,
		Modalities:  []string{"image"},
		Stream:      lo.ToPtr(genReq.Stream),
		RawRequest:  httpReq,
		RequestType: llm.RequestTypeImage,
		APIFormat:   t.apiFormat,
		Image:       imageReq,
	}

	return llmReq, nil
}

func (t *ImageInboundTransformer) transformEditRequest(ctx context.Context, httpReq *httpclient.Request) (*llm.Request, error) {
	formData, err := parseMultipartRequest(ctx, httpReq)
	if err != nil {
		return nil, err
	}

	if strings.EqualFold(strings.TrimSpace(formData.Fields["stream"]), "true") {
		return nil, fmt.Errorf("%w: image edit does not support streaming", transformer.ErrInvalidRequest)
	}

	prompt := strings.TrimSpace(formData.Fields["prompt"])
	if prompt == "" {
		return nil, fmt.Errorf("%w: prompt is required for image edits", transformer.ErrInvalidRequest)
	}

	model := strings.TrimSpace(formData.Fields["model"])
	if model == "" {
		model = "dall-e-2"
	}

	if len(formData.Images) == 0 {
		return nil, fmt.Errorf("%w: at least one image is required for edits", transformer.ErrInvalidRequest)
	}

	// Build JSONBody for logging: replace binary image data with metadata descriptions.
	if jsonBody, err := buildMultipartJSONBody(formData.Fields, formData.Images, formData.Mask); err == nil {
		httpReq.JSONBody = jsonBody
	}

	// Extract image data for ImageRequest
	images := make([][]byte, 0, len(formData.Images))
	for _, img := range formData.Images {
		images = append(images, img.Data)
	}

	var mask []byte
	if formData.Mask != nil {
		mask = formData.Mask.Data
	}

	user := strings.TrimSpace(formData.Fields["user"])

	imageReq := &llm.ImageRequest{
		Prompt:            prompt,
		Images:            images,
		Mask:              mask,
		N:                 parseOptionalInt64(formData.Fields["n"]),
		Size:              strings.TrimSpace(formData.Fields["size"]),
		Quality:           strings.TrimSpace(formData.Fields["quality"]),
		ResponseFormat:    strings.TrimSpace(formData.Fields["response_format"]),
		User:              user,
		Background:        strings.TrimSpace(formData.Fields["background"]),
		OutputFormat:      strings.TrimSpace(formData.Fields["output_format"]),
		OutputCompression: parseOptionalInt64(formData.Fields["output_compression"]),
		InputFidelity:     strings.TrimSpace(formData.Fields["input_fidelity"]),
		PartialImages:     parseOptionalInt64(formData.Fields["partial_images"]),
	}

	llmReq := &llm.Request{
		Model:       model,
		Modalities:  []string{"image"},
		Stream:      lo.ToPtr(false),
		RawRequest:  httpReq,
		RequestType: llm.RequestTypeImage,
		APIFormat:   t.apiFormat,
		Image:       imageReq,
	}

	return llmReq, nil
}

func (t *ImageInboundTransformer) transformVariationRequest(ctx context.Context, httpReq *httpclient.Request) (*llm.Request, error) {
	formData, err := parseMultipartRequest(ctx, httpReq)
	if err != nil {
		return nil, err
	}

	if strings.EqualFold(strings.TrimSpace(formData.Fields["stream"]), "true") {
		return nil, fmt.Errorf("%w: image variation does not support streaming", transformer.ErrInvalidRequest)
	}

	if strings.TrimSpace(formData.Fields["prompt"]) != "" {
		return nil, fmt.Errorf("%w: prompt is not allowed for variations", transformer.ErrInvalidRequest)
	}

	model := strings.TrimSpace(formData.Fields["model"])
	if model == "" {
		model = "dall-e-2"
	}

	if len(formData.Images) == 0 {
		return nil, fmt.Errorf("%w: image is required for variations", transformer.ErrInvalidRequest)
	}

	if len(formData.Images) > 1 {
		return nil, fmt.Errorf("%w: variations supports a single image", transformer.ErrInvalidRequest)
	}

	// Build JSONBody for logging: replace binary image data with metadata descriptions.
	if jsonBody, err := buildMultipartJSONBody(formData.Fields, formData.Images, nil); err == nil {
		httpReq.JSONBody = jsonBody
	}

	user := strings.TrimSpace(formData.Fields["user"])

	imageReq := &llm.ImageRequest{
		Images:         [][]byte{formData.Images[0].Data},
		N:              parseOptionalInt64(formData.Fields["n"]),
		Size:           strings.TrimSpace(formData.Fields["size"]),
		ResponseFormat: strings.TrimSpace(formData.Fields["response_format"]),
		User:           user,
	}

	llmReq := &llm.Request{
		Model:       model,
		Modalities:  []string{"image"},
		Stream:      lo.ToPtr(false),
		RawRequest:  httpReq,
		RequestType: llm.RequestTypeImage,
		APIFormat:   t.apiFormat,
		Image:       imageReq,
	}

	return llmReq, nil
}

type multipartFile struct {
	Filename     string
	ContentType  string
	Data         []byte
	Size         int64
	SourceBacked bool
}

type imageFormData struct {
	Images []multipartFile
	Mask   *multipartFile
	Fields map[string]string
}

func parseMultipartRequest(ctx context.Context, httpReq *httpclient.Request) (*imageFormData, error) {
	if httpReq.BodySource != nil && len(httpReq.Body) > 0 {
		return nil, fmt.Errorf("%w: request body and body source are mutually exclusive", transformer.ErrInvalidRequest)
	}
	if len(httpReq.Body) > maxImageBodySize || httpReq.BodySource != nil && httpReq.BodySource.Size() > int64(maxImageBodySize) {
		return nil, fmt.Errorf("%w: request body too large", transformer.ErrInvalidRequest)
	}

	contentType := httpReq.Headers.Get("Content-Type")

	mediaType, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid content-type", transformer.ErrInvalidRequest)
	}

	if !strings.HasPrefix(strings.ToLower(mediaType), "multipart/") {
		return nil, fmt.Errorf("%w: expected multipart/form-data", transformer.ErrInvalidRequest)
	}

	boundary := params["boundary"]
	if boundary == "" {
		return nil, fmt.Errorf("%w: missing boundary in content-type", transformer.ErrInvalidRequest)
	}

	var (
		body         io.ReadCloser
		sourceBacked bool
	)
	if httpReq.BodySource != nil {
		body, err = httpReq.BodySource.Open(ctx)
		if err != nil {
			return nil, fmt.Errorf("%w: failed to open multipart body", transformer.ErrInvalidRequest)
		}
		sourceBacked = true
	} else {
		body = io.NopCloser(bytes.NewReader(httpReq.Body))
	}
	defer body.Close()

	// Bound unknown-size sources as well as each individual part. The context
	// wrapper makes an initial validation scan of a disk-backed upload stop
	// promptly when its client disconnects.
	limitedBody := &io.LimitedReader{R: &contextReader{ctx: ctx, reader: body}, N: int64(maxImageBodySize) + 1}
	reader := multipart.NewReader(limitedBody, boundary)

	formData := &imageFormData{
		Fields: map[string]string{},
	}

	imageCount := 0

	for {
		part, err := reader.NextPart()
		if errors.Is(err, io.EOF) {
			break
		}

		if err != nil {
			return nil, fmt.Errorf("%w: failed to read multipart", transformer.ErrInvalidRequest)
		}

		fieldName := part.FormName()
		filename := part.FileName()

		if filename == "" {
			fieldLimit := int64(maxImageFileSize)
			if sourceBacked {
				if !isImageSemanticField(fieldName) {
					if _, err := io.Copy(io.Discard, part); err != nil {
						return nil, fmt.Errorf("%w: failed to read multipart extension field", transformer.ErrInvalidRequest)
					}
					continue
				}
				fieldLimit = maxImageSemanticFieldSize
			}
			value, err := io.ReadAll(io.LimitReader(part, fieldLimit+1))
			if err != nil {
				return nil, fmt.Errorf("%w: failed to read multipart field", transformer.ErrInvalidRequest)
			}

			if int64(len(value)) > fieldLimit {
				return nil, fmt.Errorf("%w: multipart field too large", transformer.ErrInvalidRequest)
			}

			formData.Fields[fieldName] = string(value)

			continue
		}

		imageCount++
		if imageCount > maxImageCount {
			return nil, fmt.Errorf("%w: too many images", transformer.ErrInvalidRequest)
		}

		contentType := strings.TrimSpace(part.Header.Get("Content-Type"))

		fileLimit := int64(maxImageFileSize)
		if sourceBacked {
			// Replayable sources are expected to be bounded by their owner (and are
			// additionally bounded by maxImageBodySize above). Allow Octopus' legacy
			// single-file uploads up to that existing request limit without buffering.
			fileLimit = int64(maxImageBodySize)
		}
		data, prefix, size, err := readMultipartFile(part, fileLimit, !sourceBacked)
		if err != nil {
			return nil, fmt.Errorf("%w: failed to read multipart file", transformer.ErrInvalidRequest)
		}

		if size > fileLimit {
			return nil, fmt.Errorf("%w: file too large", transformer.ErrInvalidRequest)
		}

		if contentType == "" {
			contentType = http.DetectContentType(prefix)
		}

		if !isAllowedImageType(contentType) {
			return nil, fmt.Errorf("%w: unsupported image type", transformer.ErrInvalidRequest)
		}

		file := multipartFile{
			Filename:     filename,
			ContentType:  contentType,
			Data:         data,
			Size:         size,
			SourceBacked: sourceBacked,
		}

		switch fieldName {
		case "image", "image[]":
			formData.Images = append(formData.Images, file)
		case "mask":
			formData.Mask = &file
		default:
		}
	}
	if _, err := io.Copy(io.Discard, limitedBody); err != nil {
		return nil, fmt.Errorf("%w: failed to finish multipart body", transformer.ErrInvalidRequest)
	}
	if limitedBody.N <= 0 {
		return nil, fmt.Errorf("%w: request body too large", transformer.ErrInvalidRequest)
	}

	return formData, nil
}

func isImageSemanticField(name string) bool {
	switch name {
	case "model", "prompt", "n", "quality", "response_format", "size", "style", "user",
		"background", "output_format", "output_compression", "moderation", "partial_images",
		"input_fidelity", "stream":
		return true
	default:
		return false
	}
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r *contextReader) Read(p []byte) (int, error) {
	select {
	case <-r.ctx.Done():
		return 0, r.ctx.Err()
	default:
		return r.reader.Read(p)
	}
}

func readMultipartFile(part io.Reader, limit int64, retain bool) (data, prefix []byte, size int64, err error) {
	limited := &io.LimitedReader{R: part, N: limit + 1}
	var destination io.Writer = io.Discard
	var body bytes.Buffer
	if retain {
		destination = &body
	}

	prefixWriter := &limitedPrefixWriter{remaining: 512}
	size, err = io.Copy(io.MultiWriter(destination, prefixWriter), limited)
	if err != nil {
		return nil, nil, size, err
	}
	if retain {
		data = body.Bytes()
	}
	return data, prefixWriter.bytes, size, nil
}

type limitedPrefixWriter struct {
	bytes     []byte
	remaining int
}

func (w *limitedPrefixWriter) Write(p []byte) (int, error) {
	if w.remaining > 0 {
		n := min(len(p), w.remaining)
		w.bytes = append(w.bytes, p[:n]...)
		w.remaining -= n
	}
	return len(p), nil
}

// buildMultipartJSONBody builds a JSON representation of a multipart/form-data request
// suitable for logging. Buffered bodies retain the legacy data-URL shape used
// by the trace UI; source-backed bodies contain metadata only and never copy
// binary payloads into JSONBody.
func buildMultipartJSONBody(fields map[string]string, images []multipartFile, mask *multipartFile) ([]byte, error) {
	body := make(map[string]any, len(fields)+2)

	for k, v := range fields {
		if v != "" {
			body[k] = v
		}
	}

	switch len(images) {
	case 1:
		body["image"] = multipartFileToDataURL(images[0])
	case 0:
		// no image
	default:
		urls := make([]string, len(images))
		for i, img := range images {
			urls[i] = multipartFileToDataURL(img)
		}

		body["image"] = urls
	}

	if mask != nil {
		body["mask"] = multipartFileToDataURL(*mask)
	}

	return json.Marshal(body)
}

func multipartFileToDataURL(f multipartFile) string {
	if f.SourceBacked {
		return fmt.Sprintf("image:type=%s;size=%d", f.ContentType, f.Size)
	}
	// Use xurl.BuildDataURL (single exact-size concat) instead of fmt.Sprintf to
	// avoid the printer's doubling-growth buffer churn on large base64 data.
	return xurl.BuildDataURL(f.ContentType, base64.StdEncoding.EncodeToString(f.Data), true)
}

func isAllowedImageType(contentType string) bool {
	for _, allowed := range allowedImageTypes {
		if strings.EqualFold(contentType, allowed) {
			return true
		}
	}

	return false
}

func parseOptionalInt64(s string) *int64 {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil
	}

	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return nil
	}

	return &v
}

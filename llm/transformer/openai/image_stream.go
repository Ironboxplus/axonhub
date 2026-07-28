package openai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

type imageStreamWireEvent struct {
	Type              string               `json:"type"`
	B64JSON           string               `json:"b64_json,omitempty"`
	URL               string               `json:"url,omitempty"`
	PartialImageIndex *int64               `json:"partial_image_index,omitempty"`
	Created           int64                `json:"created,omitempty"`
	Background        string               `json:"background,omitempty"`
	OutputFormat      string               `json:"output_format,omitempty"`
	Quality           string               `json:"quality,omitempty"`
	Size              string               `json:"size,omitempty"`
	Usage             *ImagesResponseUsage `json:"usage,omitempty"`
}

func transformImageStreamChunk(request *httpclient.Request, event *httpclient.StreamEvent) (*llm.Response, error) {
	if event == nil {
		return nil, nil
	}
	if bytes.HasPrefix(bytes.TrimSpace(event.Data), []byte("[DONE]")) {
		return llm.DoneResponse, nil
	}
	if streamErr := parseStreamErrorEvent(event); streamErr != nil {
		return nil, streamErr
	}
	var wire imageStreamWireEvent
	if err := json.Unmarshal(event.Data, &wire); err != nil {
		return nil, fmt.Errorf("failed to decode image stream event: %w", err)
	}
	if wire.Type == "" {
		wire.Type = event.Type
	}
	if !strings.HasPrefix(wire.Type, "image_generation.") {
		return nil, nil
	}
	model := "image-generation"
	if request != nil && request.TransformerMetadata != nil {
		if configuredModel, ok := request.TransformerMetadata["model"].(string); ok && configuredModel != "" {
			model = configuredModel
		}
	}
	return &llm.Response{
		Object:      wire.Type,
		Created:     wire.Created,
		Model:       model,
		RequestType: llm.RequestTypeImage,
		APIFormat:   llm.APIFormatOpenAIImageGeneration,
		Usage:       imageUsageToLLM(wire.Usage),
		ImageStreamEvent: &llm.ImageStreamEvent{
			Type:              wire.Type,
			B64JSON:           wire.B64JSON,
			URL:               wire.URL,
			PartialImageIndex: wire.PartialImageIndex,
			Created:           wire.Created,
			Background:        wire.Background,
			OutputFormat:      wire.OutputFormat,
			Quality:           wire.Quality,
			Size:              wire.Size,
		},
	}, nil
}

func renderImageStreamEvent(response *llm.Response) (*httpclient.StreamEvent, error) {
	if response == nil {
		return nil, nil
	}
	if response == llm.DoneResponse || response.Object == "[DONE]" {
		return &httpclient.StreamEvent{Data: []byte("[DONE]")}, nil
	}
	if response.ImageStreamEvent == nil {
		return nil, nil
	}
	imageEvent := response.ImageStreamEvent
	wire := imageStreamWireEvent{
		Type:              imageEvent.Type,
		B64JSON:           imageEvent.B64JSON,
		URL:               imageEvent.URL,
		PartialImageIndex: imageEvent.PartialImageIndex,
		Created:           imageEvent.Created,
		Background:        imageEvent.Background,
		OutputFormat:      imageEvent.OutputFormat,
		Quality:           imageEvent.Quality,
		Size:              imageEvent.Size,
		Usage:             llmUsageToImage(response.Usage),
	}
	if wire.Type == "" {
		wire.Type = response.Object
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("failed to encode image stream event: %w", err)
	}
	return &httpclient.StreamEvent{Type: wire.Type, Data: body}, nil
}

func imageUsageToLLM(usage *ImagesResponseUsage) *llm.Usage {
	if usage == nil {
		return nil
	}
	result := &llm.Usage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      usage.TotalTokens,
	}
	if usage.InputTokensDetails != nil {
		result.PromptTokensDetails = &llm.PromptTokensDetails{
			ImageTokens:  usage.InputTokensDetails.ImageTokens,
			TextTokens:   usage.InputTokensDetails.TextTokens,
			CachedTokens: usage.InputTokensDetails.CachedTokens,
		}
	}
	if usage.OutputTokensDetails != nil {
		result.CompletionTokensDetails = &llm.CompletionTokensDetails{
			ReasoningTokens: usage.OutputTokensDetails.ReasoningTokens,
		}
	}
	return result
}

func llmUsageToImage(usage *llm.Usage) *ImagesResponseUsage {
	if usage == nil {
		return nil
	}
	result := &ImagesResponseUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
		TotalTokens:  usage.TotalTokens,
	}
	if usage.PromptTokensDetails != nil {
		result.InputTokensDetails = &ImagesResponseUsageInputTokensDetails{
			ImageTokens:  usage.PromptTokensDetails.ImageTokens,
			TextTokens:   usage.PromptTokensDetails.TextTokens,
			CachedTokens: usage.PromptTokensDetails.CachedTokens,
		}
	}
	if usage.CompletionTokensDetails != nil {
		result.OutputTokensDetails = &ImagesResponseUsageOutputTokensDetails{
			ReasoningTokens: usage.CompletionTokensDetails.ReasoningTokens,
		}
	}
	return result
}

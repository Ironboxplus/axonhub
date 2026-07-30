package hosted

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// httpExecutorContract is the semantic half of a hosted HTTP executor. The
// transport owns endpoint policy, credentials, limits, and status handling;
// each contract owns one capability's typed arguments, configuration, and
// result union. Raw JSON remains in the envelope only to preserve future
// provider fields after the typed contract has validated known behavior.
type httpExecutorContract interface {
	Parameters() json.RawMessage
	ValidateConfiguration(json.RawMessage) error
	ValidateArguments(json.RawMessage) error
	ResultContent(json.RawMessage) ([]llm.ContentBlock, error)
	AllowsSafetyMetadata() bool
}

func newHTTPExecutorContract(kind llm.ToolKind) (httpExecutorContract, error) {
	switch kind {
	case llm.ToolKindFileSearch:
		return fileSearchHTTPContract{}, nil
	case llm.ToolKindCodeInterpreter, llm.ToolKindCodeExecution:
		return codeHTTPContract{}, nil
	case llm.ToolKindShell:
		return shellHTTPContract{}, nil
	case llm.ToolKindImageGeneration:
		return imageGenerationHTTPContract{}, nil
	case llm.ToolKindComputer:
		return computerHTTPContract{}, nil
	default:
		return nil, fmt.Errorf("hosted HTTP executor does not support %q", kind)
	}
}

type FileSearchArguments struct {
	Query   string   `json:"query,omitempty"`
	Queries []string `json:"queries,omitempty"`
}

type FileSearchConfiguration struct {
	Type           string          `json:"type,omitempty"`
	VectorStoreIDs []string        `json:"vector_store_ids,omitempty"`
	MaxNumResults  *int64          `json:"max_num_results,omitempty"`
	Filters        json.RawMessage `json:"filters,omitempty"`
	RankingOptions json.RawMessage `json:"ranking_options,omitempty"`
}

type FileSearchHit struct {
	FileID     string          `json:"file_id,omitempty"`
	Filename   string          `json:"filename,omitempty"`
	Score      *float64        `json:"score,omitempty"`
	Text       string          `json:"text,omitempty"`
	Attributes json.RawMessage `json:"attributes,omitempty"`
}

type fileSearchHTTPContract struct{}

func (fileSearchHTTPContract) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"queries":{"type":"array","items":{"type":"string"}}},"anyOf":[{"required":["query"]},{"required":["queries"]}],"additionalProperties":true}`)
}

func (fileSearchHTTPContract) ValidateConfiguration(raw json.RawMessage) error {
	var configuration FileSearchConfiguration
	return decodeOptionalObject(raw, &configuration)
}

func (fileSearchHTTPContract) ValidateArguments(raw json.RawMessage) error {
	var arguments FileSearchArguments
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return err
	}
	if strings.TrimSpace(arguments.Query) == "" && !hasNonEmptyString(arguments.Queries) {
		return errors.New("query or queries is required")
	}
	return nil
}

func (fileSearchHTTPContract) ResultContent(raw json.RawMessage) ([]llm.ContentBlock, error) {
	var hits []FileSearchHit
	if err := json.Unmarshal(raw, &hits); err != nil {
		return nil, fmt.Errorf("decode ranked hits: %w", err)
	}
	return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(raw)}}, nil
}

func (fileSearchHTTPContract) AllowsSafetyMetadata() bool { return false }

type CodeArguments struct {
	Code        string `json:"code"`
	ContainerID string `json:"container_id,omitempty"`
}

type CodeConfiguration struct {
	Type      string          `json:"type,omitempty"`
	Container json.RawMessage `json:"container,omitempty"`
}

type CodeOutput struct {
	Type         string `json:"type"`
	Logs         string `json:"logs,omitempty"`
	ImageURL     string `json:"image_url,omitempty"`
	DataBase64   string `json:"data_base64,omitempty"`
	OutputFormat string `json:"output_format,omitempty"`
	FileID       string `json:"file_id,omitempty"`
	Filename     string `json:"filename,omitempty"`
}

type codeHTTPContract struct{}

func (codeHTTPContract) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"code":{"type":"string"},"container_id":{"type":"string"}},"required":["code"],"additionalProperties":true}`)
}

func (codeHTTPContract) ValidateConfiguration(raw json.RawMessage) error {
	var configuration CodeConfiguration
	return decodeOptionalObject(raw, &configuration)
}

func (codeHTTPContract) ValidateArguments(raw json.RawMessage) error {
	var arguments CodeArguments
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return err
	}
	if strings.TrimSpace(arguments.Code) == "" {
		return errors.New("code is required")
	}
	return nil
}

func (codeHTTPContract) ResultContent(raw json.RawMessage) ([]llm.ContentBlock, error) {
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("decode outputs: %w", err)
	}
	content := make([]llm.ContentBlock, 0, len(parts))
	for index := range parts {
		var output CodeOutput
		if err := json.Unmarshal(parts[index], &output); err != nil {
			return nil, fmt.Errorf("decode output %d: %w", index, err)
		}
		switch output.Type {
		case "logs":
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindText, Text: output.Logs})
		case "image":
			imageURL, err := typedImageURL(output.ImageURL, output.DataBase64, output.OutputFormat)
			if err != nil {
				return nil, fmt.Errorf("output %d: %w", index, err)
			}
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: imageURL}})
		case "file":
			if output.FileID == "" {
				return nil, fmt.Errorf("output %d file_id is required", index)
			}
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{
				SourceType: llm.DocumentSourceFile, FileID: output.FileID, Filename: output.Filename,
			}})
		default:
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindUnknown, UnknownRaw: append(json.RawMessage(nil), parts[index]...)})
		}
	}
	return content, nil
}

func (codeHTTPContract) AllowsSafetyMetadata() bool { return false }

type ShellArguments struct {
	Commands        []string `json:"commands"`
	TimeoutMS       *int64   `json:"timeout_ms,omitempty"`
	MaxOutputLength *int64   `json:"max_output_length,omitempty"`
}

type ShellOutput struct {
	Stdout  string          `json:"stdout,omitempty"`
	Stderr  string          `json:"stderr,omitempty"`
	Outcome json.RawMessage `json:"outcome,omitempty"`
}

type shellHTTPContract struct{}

func (shellHTTPContract) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"commands":{"type":"array","items":{"type":"string"}},"timeout_ms":{"type":"integer","minimum":0},"max_output_length":{"type":"integer","minimum":0}},"required":["commands"],"additionalProperties":true}`)
}

func (shellHTTPContract) ValidateConfiguration(raw json.RawMessage) error {
	var configuration struct {
		Type string `json:"type,omitempty"`
	}
	return decodeOptionalObject(raw, &configuration)
}

func (shellHTTPContract) ValidateArguments(raw json.RawMessage) error {
	var arguments ShellArguments
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return err
	}
	if !hasNonEmptyString(arguments.Commands) {
		return errors.New("commands is required")
	}
	return nil
}

func (shellHTTPContract) ResultContent(raw json.RawMessage) ([]llm.ContentBlock, error) {
	var outputs []ShellOutput
	if err := json.Unmarshal(raw, &outputs); err != nil {
		return nil, fmt.Errorf("decode command outputs: %w", err)
	}
	return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(raw)}}, nil
}

func (shellHTTPContract) AllowsSafetyMetadata() bool { return false }

type ImageGenerationArguments struct {
	Prompt     string `json:"prompt"`
	Action     string `json:"action,omitempty"`
	Size       string `json:"size,omitempty"`
	Quality    string `json:"quality,omitempty"`
	Background string `json:"background,omitempty"`
}

type ImageGenerationConfiguration struct {
	Type              string `json:"type,omitempty"`
	Action            string `json:"action,omitempty"`
	Background        string `json:"background,omitempty"`
	InputFidelity     string `json:"input_fidelity,omitempty"`
	OutputCompression *int64 `json:"output_compression,omitempty"`
	OutputFormat      string `json:"output_format,omitempty"`
	PartialImages     *int64 `json:"partial_images,omitempty"`
	Quality           string `json:"quality,omitempty"`
	Size              string `json:"size,omitempty"`
}

type ImageGenerationOutput struct {
	ImageURL      string `json:"image_url,omitempty"`
	DataBase64    string `json:"data_base64,omitempty"`
	OutputFormat  string `json:"output_format,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

type imageGenerationHTTPContract struct{}

func (imageGenerationHTTPContract) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"prompt":{"type":"string"},"action":{"type":"string"},"size":{"type":"string"},"quality":{"type":"string"},"background":{"type":"string"}},"required":["prompt"],"additionalProperties":true}`)
}

func (imageGenerationHTTPContract) ValidateConfiguration(raw json.RawMessage) error {
	var configuration ImageGenerationConfiguration
	return decodeOptionalObject(raw, &configuration)
}

func (imageGenerationHTTPContract) ValidateArguments(raw json.RawMessage) error {
	var arguments ImageGenerationArguments
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return err
	}
	if strings.TrimSpace(arguments.Prompt) == "" {
		return errors.New("prompt is required")
	}
	return nil
}

func (imageGenerationHTTPContract) ResultContent(raw json.RawMessage) ([]llm.ContentBlock, error) {
	var output ImageGenerationOutput
	if err := json.Unmarshal(raw, &output); err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	imageURL, err := typedImageURL(output.ImageURL, output.DataBase64, output.OutputFormat)
	if err != nil {
		return nil, err
	}
	return []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: imageURL}}}, nil
}

func (imageGenerationHTTPContract) AllowsSafetyMetadata() bool { return false }

type ComputerArguments struct {
	Type    string            `json:"type"`
	X       *int64            `json:"x,omitempty"`
	Y       *int64            `json:"y,omitempty"`
	Button  string            `json:"button,omitempty"`
	Keys    []string          `json:"keys,omitempty"`
	Text    string            `json:"text,omitempty"`
	ScrollX *int64            `json:"scroll_x,omitempty"`
	ScrollY *int64            `json:"scroll_y,omitempty"`
	Actions []json.RawMessage `json:"actions,omitempty"`
}

type ComputerConfiguration struct {
	Type          string `json:"type,omitempty"`
	DisplayWidth  *int64 `json:"display_width,omitempty"`
	DisplayHeight *int64 `json:"display_height,omitempty"`
	Environment   string `json:"environment,omitempty"`
}

type ComputerScreenshot struct {
	Type     string `json:"type"`
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
}

type computerHTTPContract struct{}

func (computerHTTPContract) Parameters() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"type":{"type":"string"},"x":{"type":"integer"},"y":{"type":"integer"},"button":{"type":"string"},"keys":{"type":"array","items":{"type":"string"}},"text":{"type":"string"},"scroll_x":{"type":"integer"},"scroll_y":{"type":"integer"},"actions":{"type":"array","items":{"type":"object","additionalProperties":true}}},"required":["type"],"additionalProperties":true}`)
}

func (computerHTTPContract) ValidateConfiguration(raw json.RawMessage) error {
	var configuration ComputerConfiguration
	return decodeOptionalObject(raw, &configuration)
}

func (computerHTTPContract) ValidateArguments(raw json.RawMessage) error {
	var arguments ComputerArguments
	if err := json.Unmarshal(raw, &arguments); err != nil {
		return err
	}
	if strings.TrimSpace(arguments.Type) == "" {
		return errors.New("action type is required")
	}
	return nil
}

func (computerHTTPContract) ResultContent(raw json.RawMessage) ([]llm.ContentBlock, error) {
	var screenshot ComputerScreenshot
	if err := json.Unmarshal(raw, &screenshot); err != nil {
		return nil, fmt.Errorf("decode screenshot: %w", err)
	}
	if screenshot.Type != "computer_screenshot" {
		return nil, errors.New("computer_screenshot type is required")
	}
	if screenshot.ImageURL != "" {
		return []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: screenshot.ImageURL}}}, nil
	}
	if screenshot.FileID == "" {
		return nil, errors.New("image_url or file_id is required")
	}
	return []llm.ContentBlock{{Kind: llm.ContentKindText, Text: string(raw)}}, nil
}

func (computerHTTPContract) AllowsSafetyMetadata() bool { return true }

func decodeOptionalObject(raw json.RawMessage, target any) error {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, target); err != nil {
		return err
	}
	return nil
}

func hasNonEmptyString(values []string) bool {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return true
		}
	}
	return false
}

func typedImageURL(imageURL, dataBase64, outputFormat string) (string, error) {
	if imageURL != "" {
		return imageURL, nil
	}
	if dataBase64 == "" {
		return "", errors.New("image_url or data_base64 is required")
	}
	if outputFormat == "" {
		outputFormat = "png"
	}
	return "data:image/" + outputFormat + ";base64," + dataBase64, nil
}

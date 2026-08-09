// Package responses implements the response API for OpenAI.
package responses

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xjson"
	"github.com/looplj/axonhub/llm/transformer"
)

// ImageGeneration is a permissive structure to carry image generation tool
// parameters. It mirrors the OpenRouter/OpenAI Responses API fields we care
// about, but is intentionally loose to allow forward-compatibility.
type ImageGeneration struct {
	llm.ImageGeneration

	Type  string `json:"type"`
	Model string `json:"model"`
}

type Tool struct {
	// Raw preserves native Responses tool configuration fields that Axon does
	// not interpret. Only provider/client built-ins populate it; ordinary
	// function and custom tools continue to use the typed representation.
	Raw json.RawMessage `json:"-"`
	// Residual contains future fields not owned by the typed Tool model. Unlike
	// Raw, it is overlaid by current canonical fields during marshaling so an
	// edited definition cannot be silently replaced by its stale source object.
	Residual json.RawMessage `json:"-"`
	// Any of "function", "image_generation", "custom", "web_search", "namespace", "mcp".
	Type        string `json:"type,omitempty"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Execution   string `json:"execution,omitempty"`

	// This field is from variant [FunctionTool].
	Parameters map[string]any `json:"parameters,omitempty"`
	// This field is from variant [FunctionTool].
	Strict *bool `json:"strict,omitempty"`

	// Tools holds sub-tools when Type is "namespace".
	Tools []Tool `json:"tools,omitempty"`

	// This field is for custom tool format definition.
	Format *CustomToolFormat `json:"format,omitempty"`

	// These fields are for web search.
	Filters      *WebSearchFilters      `json:"filters,omitempty"`
	UserLocation *WebSearchUserLocation `json:"user_location,omitempty"`

	// This field is for ImageGeneration
	Action string `json:"action,omitempty"`
	// This field is for ImageGeneration
	Background string `json:"background,omitempty"`
	// This field is for ImageGeneration
	InputFidelity string `json:"input_fidelity,omitempty"`
	// This field is for ImageGeneration
	InputImageMask map[string]any `json:"input_image_mask,omitempty"`
	// This field is for ImageGeneration
	Model string `json:"model,omitempty"`
	// This field is for ImageGeneration
	Moderation string `json:"moderation,omitempty"`
	// This field is for ImageGeneration
	OutputCompression *int64 `json:"output_compression,omitempty"`
	// This field is for ImageGeneration
	OutputFormat string `json:"output_format,omitempty"`
	// This field is for ImageGeneration
	PartialImages *int64 `json:"partial_images,omitempty"`
	// This field is for ImageGeneration
	Quality string `json:"quality,omitempty"`
	// This field is for ImageGeneration
	Size string `json:"size,omitempty"`

	// These fields are for remote MCP tools. Authorization and Headers are
	// extracted into llm.ToolExecutionSecrets immediately after decoding.
	ServerLabel       string            `json:"server_label,omitempty"`
	ServerDescription string            `json:"server_description,omitempty"`
	ServerURL         string            `json:"server_url,omitempty"`
	ConnectorID       string            `json:"connector_id,omitempty"`
	TunnelID          string            `json:"tunnel_id,omitempty"`
	Authorization     string            `json:"authorization,omitempty"`
	Headers           map[string]string `json:"headers,omitempty"`
	DeferLoading      *bool             `json:"defer_loading,omitempty"`
	AllowedCallers    []string          `json:"allowed_callers,omitempty"`
	AllowedTools      json.RawMessage   `json:"allowed_tools,omitempty"`
	RequireApproval   json.RawMessage   `json:"require_approval,omitempty"`

	// Native built-in configuration. Raw remains authoritative on identity
	// routes, while these fields provide the portable subset used by encoders.
	VectorStoreIDs []string        `json:"vector_store_ids,omitempty"`
	MaxNumResults  *int64          `json:"max_num_results,omitempty"`
	RankingOptions json.RawMessage `json:"ranking_options,omitempty"`
	Container      json.RawMessage `json:"container,omitempty"`
	DisplayWidth   *int64          `json:"display_width,omitempty"`
	DisplayHeight  *int64          `json:"display_height,omitempty"`
	Environment    string          `json:"environment,omitempty"`
}

func (tool *Tool) UnmarshalJSON(data []byte) error {
	type toolWire Tool
	var wire toolWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*tool = Tool(wire)
	tool.Residual = responseToolResidual(data)
	switch tool.Type {
	case "file_search", "code_interpreter", "computer_use_preview", "computer", "shell", "apply_patch":
		tool.Raw = append(json.RawMessage(nil), data...)
	}
	return nil
}

func (tool Tool) MarshalJSON() ([]byte, error) {
	type toolWire Tool
	raw, err := json.Marshal(toolWire(tool))
	if err != nil {
		return nil, err
	}
	// Function schemas are a required wire union member. A map together with
	// omitempty used to erase both a valid empty schema ({}) and malformed
	// nil/missing state. Keep the field explicit so the final wire contract can
	// distinguish and reject null while preserving a legal empty object.
	if tool.Type == "function" && len(tool.Parameters) == 0 {
		parameters, err := json.Marshal(tool.Parameters)
		if err != nil {
			return nil, err
		}
		if len(raw) == 0 || raw[len(raw)-1] != '}' {
			return nil, fmt.Errorf("function tool must marshal to an object")
		}
		raw = append(raw[:len(raw)-1], ',')
		raw = append(raw, `"parameters":`...)
		raw = append(raw, parameters...)
		raw = append(raw, '}')
	}
	return mergeResidualObject(raw, tool.Residual)
}

type WebSearchFilters struct {
	AllowedDomains []string `json:"allowed_domains,omitempty"`
}

type WebSearchUserLocation struct {
	Type     string `json:"type,omitempty"`
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// CustomToolFormat represents the format definition for a custom tool.
type CustomToolFormat struct {
	// Type is the format type, e.g. "grammar".
	Type string `json:"type"`
	// Syntax is the grammar syntax, e.g. "lark".
	Syntax string `json:"syntax,omitempty"`
	// Definition is the grammar definition string.
	Definition string `json:"definition,omitempty"`
}

// Request is a struct for OpenAI Responses API creation.
// Reference: github.com/openai/openai-go/v2/responses.ResponseNewParams.
type Request struct {
	Model string `json:"model"`

	// A system (or developer) message inserted into the model's context.
	Instructions string `json:"instructions"`

	Temperature *float64 `json:"temperature,omitempty"`

	// Input can be a string prompt or an array of input items.
	Input Input `json:"input"`
	// Tools includes the function/image_generation/web_search/custom tools.
	Tools []Tool `json:"tools,omitzero"`
	// Parallel tool calls preference.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// Whether to run the model response in the background.
	Background *bool `json:"background,omitempty"`

	Stream           *bool             `json:"stream,omitempty"`
	Store            *bool             `json:"store,omitempty"`
	ServiceTier      *string           `json:"service_tier,omitempty"`
	SafetyIdentifier *string           `json:"safety_identifier,omitempty"`
	User             *string           `json:"user,omitempty"`
	Metadata         map[string]string `json:"metadata,omitempty"`
	MaxOutputTokens  *int64            `json:"max_output_tokens,omitempty"`
	MaxToolCalls     *int64            `json:"max_tool_calls,omitempty"`
	Text             *TextOptions      `json:"text,omitempty"`

	// Specify additional output data to include in the model response.
	// e.g., "file_search_call.results", "message.input_image.image_url", "reasoning.encrypted_content"
	Include []string `json:"include,omitempty"`

	// The unique ID of the previous response to the model for multi-turn conversations.
	PreviousResponseID *string `json:"previous_response_id,omitempty"`

	// Reference to a prompt template and its variables.
	// TODO
	// Prompt *Prompt `json:"prompt,omitempty"`

	// Used by OpenAI to cache responses for similar requests.
	PromptCacheKey *string `json:"prompt_cache_key,omitempty"`

	// The retention policy for the prompt cache. Any of "in-memory", "24h".
	PromptCacheRetention *string `json:"prompt_cache_retention,omitempty"`

	// Configuration options for reasoning models.
	Reasoning *Reasoning `json:"reasoning,omitempty"`

	// Options for streaming responses.
	StreamOptions *StreamOptions `json:"stream_options,omitempty"`

	// How the model should select which tool to use.
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	// The truncation strategy. Any of "auto", "disabled".
	Truncation *string `json:"truncation,omitempty"`

	// The conversation that this response belongs to.
	Conversation *Conversation `json:"conversation,omitempty"`

	// An integer between 0 and 20 specifying the number of most likely tokens to return.
	TopLogprobs *int64 `json:"top_logprobs,omitempty"`

	// Nucleus sampling parameter.
	TopP *float64 `json:"top_p,omitempty"`
}

// Prompt represents a reference to a prompt template.
type Prompt struct {
	ID        string            `json:"id"`
	Version   *string           `json:"version,omitempty"`
	Variables map[string]string `json:"variables,omitempty"`
}

// Reasoning represents configuration options for reasoning models.
type Reasoning struct {
	// The reasoning context scope requested by internal Responses features.
	Context string `json:"context,omitempty"`
	// The effort level for reasoning. Any of "low", "medium", "high".
	Effort string `json:"effort,omitempty"`
	// Whether to generate a summary of the reasoning. Any of "auto", "concise", "detailed".
	GenerateSummary string `json:"generate_summary,omitempty"`
	// The summary type. Any of "auto", "concise", "detailed".
	Summary string `json:"summary,omitempty"`
	// Maximum number of reasoning tokens.
	MaxTokens *int64 `json:"max_tokens,omitempty"`
}

// StreamOptions represents options for streaming responses.
type StreamOptions struct {
	// When true, stream obfuscation will be enabled.
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

// ToolChoice represents how the model should select which tool to use (for requests).
type ToolChoice struct {
	Residual json.RawMessage `json:"-"`
	// Mode can be "none", "auto", "required".
	Mode *string `json:"mode,omitempty"`
	// Type for specific tool choice. Any of "function", "file_search", "web_search", "shell" etc.
	Type *string `json:"type,omitempty"`
	// Name of the function for function tool choice.
	Name *string `json:"name,omitempty"`

	// Allow multiple tools to be selected.
	Tools []ToolOption `json:"tools,omitempty"`
}

type ToolOption struct {
	Residual    json.RawMessage `json:"-"`
	Type        string          `json:"type"`
	Name        string          `json:"name,omitempty"`
	ServerLabel string          `json:"server_label,omitempty"`
}

func (option *ToolOption) UnmarshalJSON(data []byte) error {
	type optionWire ToolOption
	var wire optionWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*option = ToolOption(wire)
	option.Residual = jsonObjectResidual(data, responsesToolOptionFields)
	return nil
}

func (option ToolOption) MarshalJSON() ([]byte, error) {
	type optionWire ToolOption
	typed, _ := json.Marshal(optionWire(option))
	return mergeResidualObject(typed, option.Residual)
}

type ToolChoiceAlias ToolChoice

func (t *ToolChoice) UnmarshalJSON(data []byte) error {
	mode, err := xjson.To[string](data)
	if err == nil {
		t.Mode = &mode
		t.Residual = nil
		return nil
	}

	tc, err := xjson.To[ToolChoiceAlias](data)
	if err == nil {
		*t = ToolChoice(tc)
		t.Residual = responseToolChoiceResidual(data)
		return nil
	}

	return errors.New("invalid tool choice type")
}

func (t *ToolChoice) MarshalJSON() ([]byte, error) {
	if t.Mode != nil && t.Type == nil && t.Name == nil && len(t.Tools) == 0 {
		return json.Marshal(*t.Mode)
	}

	// For other cases, marshal as object
	raw, _ := json.Marshal(&struct {
		Mode  *string      `json:"mode,omitempty"`
		Type  *string      `json:"type,omitempty"`
		Name  *string      `json:"name,omitempty"`
		Tools []ToolOption `json:"tools,omitempty"`
	}{
		Mode:  t.Mode,
		Type:  t.Type,
		Name:  t.Name,
		Tools: t.Tools,
	})
	return mergeResidualObject(raw, t.Residual)
}

// ResponseToolChoice represents tool_choice in responses, which can be a string or object.
type ResponseToolChoice struct {
	// String value when tool_choice is a simple string like "auto", "none", "required".
	// StringValue and ObjectValue are mutually exclusive representations of tool_choice.
	// If both are populated, StringValue takes precedence during marshaling.
	StringValue string
	// Object value when tool_choice is an object.
	ObjectValue *ToolChoice
}

func (r *ResponseToolChoice) UnmarshalJSON(data []byte) error {
	// Try to unmarshal as string first
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		r.StringValue = str
		r.ObjectValue = nil

		return nil
	}

	// Try to unmarshal as object
	var obj ToolChoice
	if err := json.Unmarshal(data, &obj); err == nil {
		r.StringValue = ""
		r.ObjectValue = &obj

		return nil
	}

	return fmt.Errorf("tool_choice must be a string or object")
}

func (r ResponseToolChoice) MarshalJSON() ([]byte, error) {
	if r.StringValue != "" {
		return json.Marshal(r.StringValue)
	}

	if r.ObjectValue != nil {
		return json.Marshal(r.ObjectValue)
	}

	return []byte("null"), nil
}

// Conversation represents the conversation context for requests.
type Conversation struct {
	// The conversation ID.
	ID *string `json:"id,omitempty"`
}

// ResponseConversation represents the conversation context in responses.
type ResponseConversation struct {
	// The unique ID of the conversation.
	ID string `json:"id"`
}

// ResponseIncompleteDetails contains details about why the response is incomplete.
type ResponseIncompleteDetails struct {
	// The reason why the response is incomplete.
	// Any of "max_output_tokens", "content_filter".
	Reason string `json:"reason"`
}

// ResponseReasoning represents reasoning configuration in responses.
type ResponseReasoning struct {
	// Constrains effort on reasoning for reasoning models.
	// Any of "none", "minimal", "low", "medium", "high", "xhigh".
	Effort string `json:"effort,omitempty"`
	// A summary of the reasoning performed by the model.
	// Any of "auto", "concise", "detailed".
	Summary string `json:"summary,omitempty"`
	// Deprecated: use Summary instead.
	GenerateSummary string `json:"generate_summary,omitempty"`
}

type TextOptions struct {
	// An object specifying the format that the model must output.
	// Configuring { "type": "json_schema" } enables Structured Outputs, which ensures the model will match your supplied JSON schema. Learn more in the Structured Outputs
	// guide.
	// The default format is { "type": "text" } with no additional options.
	// { "type": "json_object" } is also supported.
	Format *TextFormat `json:"format,omitempty"`

	// The verbosity of the response. Any of "low", "medium", "high".
	Verbosity *string `json:"verbosity,omitempty"`
}

// TextFormat specifies the format that the model must output.
type TextFormat struct {
	// The type of the format. Any of "text", "json_object", "json_schema".
	Type string `json:"type,omitempty"`
	// The name of the schema (for json_schema type).
	Name string `json:"name,omitempty"`
	// The description of the schema (for json_schema type).
	Description string `json:"description,omitempty"`
	// The JSON schema (for json_schema type).
	Schema json.RawMessage `json:"schema,omitempty"`
	// Whether to enforce strict schema adherence (for json_schema type).
	Strict *bool `json:"strict,omitempty"`
}

type Input struct {
	// Text and Items are mutually exclusive representations of the same input payload.
	// If both are populated, Text takes precedence during marshaling.
	Text  *string
	Items []Item
}

func (i *Input) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		i.Text = &text
		i.Items = nil

		return nil
	}

	var items []Item
	if err := json.Unmarshal(data, &items); err == nil {
		i.Text = nil
		i.Items = items

		return nil
	}

	return fmt.Errorf("invalid input: %w", transformer.ErrInvalidRequest)
}

func (i Input) MarshalJSON() ([]byte, error) {
	if i.Text != nil {
		return json.Marshal(i.Text)
	}

	return json.Marshal(i.Items)
}

type Annotation struct {
	Residual json.RawMessage `json:"-"`
	// Type is the type of annotation, e.g., "url_citation".
	Type string `json:"type,omitempty"`
	// StartIndex is the start offset of the annotated span in the output text.
	StartIndex *int64 `json:"start_index,omitempty"`
	// EndIndex is the end offset of the annotated span in the output text.
	EndIndex *int64 `json:"end_index,omitempty"`
	// URLCitation contains URL citation details when Type is "url_citation".
	URLCitation *URLCitation `json:"url_citation,omitempty"`
}

func (a *Annotation) UnmarshalJSON(data []byte) error {
	type rawAnnotation Annotation

	var raw struct {
		rawAnnotation

		URL   *string `json:"url,omitempty"`
		Title *string `json:"title,omitempty"`
	}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*a = Annotation(raw.rawAnnotation)
	a.Residual = jsonObjectResidual(data, responsesAnnotationFields)
	if a.URLCitation == nil && (raw.URL != nil || raw.Title != nil) {
		a.URLCitation = &URLCitation{}
		if raw.URL != nil {
			a.URLCitation.URL = *raw.URL
		}
		if raw.Title != nil {
			a.URLCitation.Title = *raw.Title
		}
	}

	return nil
}

func (a Annotation) MarshalJSON() ([]byte, error) {
	type rawAnnotation Annotation
	wire := rawAnnotation(a)
	var source map[string]json.RawMessage
	_ = json.Unmarshal(a.Residual, &source)
	_, hadURL := source["url"]
	_, hadTitle := source["title"]
	if a.URLCitation != nil && (hadURL || hadTitle) {
		wire.URLCitation = nil
		typed, _ := json.Marshal(struct {
			rawAnnotation
			URL   string `json:"url,omitempty"`
			Title string `json:"title,omitempty"`
		}{rawAnnotation: wire, URL: a.URLCitation.URL, Title: a.URLCitation.Title})
		return mergeResidualObject(typed, a.Residual)
	}
	typed, err := json.Marshal(wire)
	if err != nil {
		return nil, err
	}
	return mergeResidualObject(typed, a.Residual)
}

// URLCitation represents a URL-based citation.
type URLCitation struct {
	Residual json.RawMessage `json:"-"`
	// URL is the citation URL.
	URL string `json:"url,omitempty"`
	// Title is the title of the cited source.
	Title string `json:"title,omitempty"`
}

func (citation *URLCitation) UnmarshalJSON(data []byte) error {
	type citationWire URLCitation
	var wire citationWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*citation = URLCitation(wire)
	citation.Residual = jsonObjectResidual(data, responsesCitationFields)
	return nil
}

func (citation URLCitation) MarshalJSON() ([]byte, error) {
	type citationWire URLCitation
	typed, _ := json.Marshal(citationWire(citation))
	return mergeResidualObject(typed, citation.Residual)
}

const responsesWebSearchCallsTransformerMetadataKey = "openai_responses_web_search_calls"
const responsesReasoningItemTransformerMetadataKey = "openai_responses_reasoning_item"

type responsesReasoningItemMetadata struct {
	ID   string `json:"id,omitempty"`
	Done bool   `json:"done,omitempty"`
}

type WebSearchSource struct {
	Residual json.RawMessage `json:"-"`
	Type     string          `json:"type,omitempty"`
	URL      string          `json:"url,omitempty"`
	Title    string          `json:"title,omitempty"`
}

func (source *WebSearchSource) UnmarshalJSON(data []byte) error {
	type sourceWire WebSearchSource
	var wire sourceWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*source = WebSearchSource(wire)
	source.Residual = jsonObjectResidual(data, responsesWebSourceFields)
	return nil
}

func (source WebSearchSource) MarshalJSON() ([]byte, error) {
	type sourceWire WebSearchSource
	typed, _ := json.Marshal(sourceWire(source))
	return mergeResidualObject(typed, source.Residual)
}

type WebSearchAction struct {
	Type    string            `json:"type,omitempty"`
	Query   string            `json:"query,omitempty"`
	Queries []string          `json:"queries,omitempty"`
	Sources []WebSearchSource `json:"sources,omitempty"`
}

type LocalShellAction struct {
	Type             string            `json:"type"`
	Command          []string          `json:"command"`
	TimeoutMS        *uint64           `json:"timeout_ms,omitempty"`
	WorkingDirectory *string           `json:"working_directory,omitempty"`
	Env              map[string]string `json:"env,omitempty"`
	User             *string           `json:"user,omitempty"`
}

type ComputerSafetyCheck struct {
	Residual json.RawMessage `json:"-"`
	ID       string          `json:"id,omitempty"`
	Code     string          `json:"code,omitempty"`
	Message  string          `json:"message,omitempty"`
}

func (check *ComputerSafetyCheck) UnmarshalJSON(data []byte) error {
	type checkWire ComputerSafetyCheck
	var wire checkWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*check = ComputerSafetyCheck(wire)
	check.Residual = jsonObjectResidual(data, responsesSafetyCheckFields)
	return nil
}

func (check ComputerSafetyCheck) MarshalJSON() ([]byte, error) {
	type checkWire ComputerSafetyCheck
	typed, _ := json.Marshal(checkWire(check))
	return mergeResidualObject(typed, check.Residual)
}

type ComputerScreenshot struct {
	Residual json.RawMessage `json:"-"`
	Type     string          `json:"type"`
	FileID   string          `json:"file_id,omitempty"`
	ImageURL string          `json:"image_url,omitempty"`
}

func (screenshot *ComputerScreenshot) UnmarshalJSON(data []byte) error {
	type screenshotWire ComputerScreenshot
	var wire screenshotWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*screenshot = ComputerScreenshot(wire)
	screenshot.Residual = jsonObjectResidual(data, responsesScreenshotFields)
	return nil
}

func (screenshot ComputerScreenshot) MarshalJSON() ([]byte, error) {
	type screenshotWire ComputerScreenshot
	typed, _ := json.Marshal(screenshotWire(screenshot))
	return mergeResidualObject(typed, screenshot.Residual)
}

// ItemAction is the polymorphic "action" field of an output item.
// ImageGenerationAction and WebSearch are mutually exclusive;
// if both are set, ImageGenerationAction takes precedence during marshaling.
type ItemAction struct {
	Residual json.RawMessage
	// ImageGenerationAction holds the bare-string action for image_generation_call items
	// (e.g. "generate", "edit").
	ImageGenerationAction string
	// LocalShell holds the structured action for local_shell_call items.
	LocalShell *LocalShellAction
	// WebSearch holds the structured action for web_search_call items.
	WebSearch *WebSearchAction
}

// NewImageGenerationAction creates an ItemAction with a bare-string action value.
func NewImageGenerationAction(action string) *ItemAction {
	return &ItemAction{ImageGenerationAction: action}
}

// NewWebSearchAction creates an ItemAction with a structured WebSearchAction value.
func NewWebSearchAction(action *WebSearchAction) *ItemAction {
	return &ItemAction{WebSearch: action}
}

func NewLocalShellAction(action *LocalShellAction) *ItemAction {
	return &ItemAction{LocalShell: action}
}

// IsImageGeneration reports whether this action represents an image_generation_call string action.
func (a *ItemAction) IsImageGeneration() bool {
	return a != nil && a.ImageGenerationAction != ""
}

// IsWebSearch reports whether this action represents a web_search_call structured action.
func (a *ItemAction) IsWebSearch() bool {
	return a != nil && a.WebSearch != nil
}

func (a *ItemAction) UnmarshalJSON(data []byte) error {
	// Try string form first (image_generation_call).
	var str string
	if err := json.Unmarshal(data, &str); err == nil {
		a.ImageGenerationAction = str
		a.LocalShell = nil
		a.WebSearch = nil
		a.Residual = nil

		return nil
	}

	var discriminator struct {
		Command json.RawMessage `json:"command"`
	}
	if err := json.Unmarshal(data, &discriminator); err == nil && len(discriminator.Command) > 0 {
		var action LocalShellAction
		if err := json.Unmarshal(data, &action); err != nil {
			return err
		}
		a.ImageGenerationAction = ""
		a.LocalShell = &action
		a.WebSearch = nil
		a.Residual = responseItemActionResidual(data)
		return nil
	}

	// Then object form (web_search_call).
	var obj WebSearchAction
	if err := json.Unmarshal(data, &obj); err == nil {
		a.ImageGenerationAction = ""
		a.LocalShell = nil
		a.WebSearch = &obj
		a.Residual = responseItemActionResidual(data)

		return nil
	}

	return fmt.Errorf("action must be a string or object")
}

func (a ItemAction) MarshalJSON() ([]byte, error) {
	if a.ImageGenerationAction != "" {
		return json.Marshal(a.ImageGenerationAction)
	}
	var typed json.RawMessage
	var err error
	if a.LocalShell != nil {
		typed, err = json.Marshal(a.LocalShell)
	} else if a.WebSearch != nil {
		typed, err = json.Marshal(a.WebSearch)
	} else {
		return []byte("null"), nil
	}
	if err != nil {
		return nil, err
	}
	return mergeResidualObject(typed, a.Residual)
}

// Item is a unified structure for both input and output items in the Responses API.
// This follows the openai-go pattern where input and output items share the same structure.
// Reference: github.com/openai/openai-go/v3/responses.ResponseOutputItemUnion.
type Item struct {
	// Residual contains source fields not owned by the typed wire model. Custom
	// marshaling overlays the current typed object on this residual so canonical
	// edits win while future fields remain attached to their original object.
	Residual json.RawMessage `json:"-"`

	// The ID of the item, generated by the server.
	ID string `json:"id,omitempty"`

	// Any of "message", "input_text", "input_image", "input_audio", "output_text", "compaction", "compaction_summary",
	// "function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output", "image_generation_call", "reasoning", "reasoning_text".
	Type string `json:"type,omitempty"`

	// The annotations of the text output.
	Annotations []Annotation `json:"annotations,omitzero"`

	// Any of "system", "user", "assistant", "developer".
	Role string `json:"role,omitempty"`

	// The content of the message. Can be string or array of content items.
	// For input items: string or array of input items.
	// For output message items: array of ContentItem (stored as []Item with Type="output_text").
	Content *Input `json:"content,omitempty"`

	// Status of the item.
	// Any of "in_progress", "completed", "incomplete".
	Status *string `json:"status,omitempty"`

	// The URL of the image url or base64 encoded image, for input_image type.
	ImageURL *string `json:"image_url,omitempty"`

	// The detail of the image. high, low, or auto, for input_image type.
	Detail *string `json:"detail,omitempty"`

	// Document input fields for input_file. Exactly one source field is set.
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	FileURL  string `json:"file_url,omitempty"`
	Filename string `json:"filename,omitempty"`

	// Text for output_text/input_text type.
	Text *string `json:"text,omitempty"`

	// Image generation fields

	// Background for image generated, e.g: opaque
	Background *string `json:"background,omitempty"`
	// Output format for image generated, e.g: png
	OutputFormat *string `json:"output_format,omitempty"`
	// Quality for image generated, e.g: low
	Quality *string `json:"quality,omitempty"`
	// Size for image generated, e.g: 1024x1024
	Size *string `json:"size,omitempty"`

	// Result for image_generation_call type.
	Result        *string `json:"result,omitempty"`
	RevisedPrompt string  `json:"revised_prompt,omitempty"`

	// Function call fields
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Namespace string `json:"namespace,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Execution string `json:"execution,omitempty"`

	// Custom tool call fields (for type="custom_tool_call")
	// Input is the freeform input text generated by the model for custom tool calls.
	Input *string `json:"input,omitempty"`

	// Output for function_call_output/custom_tool_call_output type.
	Output *Input `json:"output,omitempty"`
	// ComputerOutput is the object-shaped output branch used by
	// computer_call_output. Item's custom JSON codec owns the shared key.
	ComputerOutput *ComputerScreenshot `json:"-"`

	// MCP lifecycle fields.
	ServerLabel string          `json:"server_label,omitempty"`
	Tools       []MCPListedTool `json:"tools,omitempty"`
	// AdditionalTools is the typed tool catalog carried by a Responses Lite
	// input item. It shares the wire key "tools" with MCP discovery results,
	// so Item's custom JSON codec selects the correct union branch by Type.
	AdditionalTools   []Tool  `json:"-"`
	ApprovalRequestID string  `json:"approval_request_id,omitempty"`
	Approve           *bool   `json:"approve,omitempty"`
	Reason            string  `json:"reason,omitempty"`
	Error             *string `json:"error,omitempty"`

	// Reasoning fields (for type="reasoning")
	// Reasoning summary content - array of summary text items.
	Summary []ReasoningSummary `json:"summary,omitempty"`
	// Reasoning text content - array of reasoning text items.
	ReasoningContent []ReasoningContent `json:"reasoning_content,omitempty"`
	// The encrypted content of the reasoning item.
	EncryptedContent *string `json:"encrypted_content,omitempty"`

	// Action is the polymorphic "action" field: web_search_call uses an object,
	// image_generation_call uses a bare string. See ItemAction.
	Action *ItemAction `json:"action,omitempty"`
	// ComputerAction is the unmodified object-shaped computer action. Keeping
	// it as JSON avoids losing new action variants while canonical code still
	// treats it as behavioral input.
	ComputerAction           json.RawMessage       `json:"-"`
	Operation                json.RawMessage       `json:"operation,omitempty"`
	PendingSafetyChecks      []ComputerSafetyCheck `json:"pending_safety_checks,omitempty"`
	AcknowledgedSafetyChecks []ComputerSafetyCheck `json:"acknowledged_safety_checks,omitempty"`

	// Provider-hosted file-search/code-interpreter lifecycle fields.
	Queries         []string        `json:"queries,omitempty"`
	Results         json.RawMessage `json:"results,omitempty"`
	Code            *string         `json:"code,omitempty"`
	ContainerID     *string         `json:"container_id,omitempty"`
	Outputs         json.RawMessage `json:"outputs,omitempty"`
	RawOutput       json.RawMessage `json:"-"`
	MaxOutputLength *int64          `json:"max_output_length,omitempty"`

	// Compaction fields (for type="compaction")
	// The identifier of the actor that created the item.
	CreatedBy *string `json:"created_by,omitempty"`
}

func (item *Item) UnmarshalJSON(data []byte) error {
	type itemAlias Item
	raw := struct {
		itemAlias
		Arguments json.RawMessage `json:"arguments"`
		Action    json.RawMessage `json:"action"`
		Output    json.RawMessage `json:"output"`
	}{}

	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	*item = Item(raw.itemAlias)
	item.Residual = responseItemResidual(data)
	if len(raw.Action) > 0 && !bytes.Equal(raw.Action, []byte("null")) {
		if item.Type == "computer_call" || item.Type == "shell_call" {
			// raw.Action came from a successful outer JSON decode and is therefore
			// already syntactically valid. Keep it byte-for-byte for future variants.
			item.ComputerAction = append(json.RawMessage(nil), raw.Action...)
		} else {
			var action ItemAction
			if err := json.Unmarshal(raw.Action, &action); err != nil {
				return err
			}
			item.Action = &action
		}
	}
	if len(raw.Output) > 0 && !bytes.Equal(raw.Output, []byte("null")) {
		if item.Type == "computer_call_output" {
			var output ComputerScreenshot
			if err := json.Unmarshal(raw.Output, &output); err != nil {
				return err
			}
			item.ComputerOutput = &output
		} else if item.Type == "shell_call_output" {
			// raw.Output is valid JSON by construction; its provider-owned shape is
			// intentionally not narrowed here.
			item.RawOutput = append(json.RawMessage(nil), raw.Output...)
		} else {
			var output Input
			if err := json.Unmarshal(raw.Output, &output); err != nil {
				return err
			}
			item.Output = &output
		}
	}
	if item.Type == "additional_tools" || item.Type == "tool_search_output" {
		var additional struct {
			Tools []Tool `json:"tools"`
		}
		if err := json.Unmarshal(data, &additional); err != nil {
			return err
		}
		item.AdditionalTools = additional.Tools
		item.Tools = nil
	}
	if len(raw.Arguments) == 0 || bytes.Equal(raw.Arguments, []byte("null")) {
		return nil
	}

	var arguments string
	if err := json.Unmarshal(raw.Arguments, &arguments); err == nil {
		item.Arguments = arguments
		return nil
	}

	var compacted bytes.Buffer
	_ = json.Compact(&compacted, raw.Arguments)
	item.Arguments = compacted.String()

	return nil
}

// MarshalJSON omits summary for non-reasoning items and forces an empty array for reasoning items.
func (item Item) MarshalJSON() ([]byte, error) {
	raw, err := item.marshalTypedJSON()
	if err != nil {
		return nil, err
	}
	return mergeResidualObject(raw, item.Residual)
}

func (item Item) marshalTypedJSON() ([]byte, error) {
	type itemAlias Item

	if item.Type == "computer_call" || item.Type == "shell_call" {
		action := item.ComputerAction
		if len(action) == 0 {
			action = json.RawMessage(`{}`)
		}
		if !json.Valid(action) {
			return nil, fmt.Errorf("computer_call action must be valid JSON")
		}
		return json.Marshal(struct {
			itemAlias
			Action json.RawMessage `json:"action"`
		}{itemAlias: itemAlias(item), Action: action})
	}

	if item.Type == "computer_call_output" {
		return json.Marshal(struct {
			itemAlias
			Output *ComputerScreenshot `json:"output"`
		}{itemAlias: itemAlias(item), Output: item.ComputerOutput})
	}

	if item.Type == "shell_call_output" {
		output := item.RawOutput
		if len(output) == 0 {
			output = json.RawMessage(`[]`)
		}
		if !json.Valid(output) {
			return nil, fmt.Errorf("shell_call_output output must be valid JSON")
		}
		return json.Marshal(struct {
			itemAlias
			Output json.RawMessage `json:"output"`
		}{itemAlias: itemAlias(item), Output: output})
	}

	if item.Type == "additional_tools" || item.Type == "tool_search_output" {
		return json.Marshal(struct {
			itemAlias
			Tools []Tool `json:"tools"`
		}{itemAlias: itemAlias(item), Tools: item.AdditionalTools})
	}

	if item.Type == "tool_search_call" {
		arguments := json.RawMessage(item.Arguments)
		if len(arguments) == 0 {
			arguments = json.RawMessage(`{}`)
		}
		if !json.Valid(arguments) {
			return nil, fmt.Errorf("tool_search_call arguments must be valid JSON")
		}
		return json.Marshal(struct {
			itemAlias
			Arguments json.RawMessage `json:"arguments"`
		}{itemAlias: itemAlias(item), Arguments: arguments})
	}

	if item.Type == "function_call" || item.Type == "mcp_call" || item.Type == "mcp_approval_request" {
		type functionCallItem struct {
			itemAlias

			Arguments string `json:"arguments"`
		}

		return json.Marshal(functionCallItem{
			itemAlias: itemAlias(item),
			Arguments: item.Arguments,
		})
	}

	if item.Type == "mcp_list_tools" {
		tools := item.Tools
		if tools == nil {
			tools = []MCPListedTool{}
		}
		return json.Marshal(struct {
			itemAlias
			Tools []MCPListedTool `json:"tools"`
		}{itemAlias: itemAlias(item), Tools: tools})
	}

	if item.Type == "custom_tool_call" {
		type customToolCallItem struct {
			itemAlias

			InputStr string `json:"input"`
		}

		inputStr := ""
		if item.Input != nil {
			inputStr = *item.Input
		}

		return json.Marshal(customToolCallItem{
			itemAlias: itemAlias(item),
			InputStr:  inputStr,
		})
	}

	if item.Type == "compaction" {
		type compactionItem struct {
			itemAlias

			EncryptedContent string `json:"encrypted_content"`
		}

		encContent := ""
		if item.EncryptedContent != nil {
			encContent = *item.EncryptedContent
		}

		return json.Marshal(compactionItem{
			itemAlias:        itemAlias(item),
			EncryptedContent: encContent,
		})
	}

	if item.Type != "reasoning" {
		item.Summary = nil
		return json.Marshal(itemAlias(item))
	}

	// Ensure reasoning items always include summary, even if empty.
	type reasoningItem struct {
		itemAlias

		Summary []ReasoningSummary `json:"summary"`
	}

	summary := item.Summary
	if summary == nil {
		summary = []ReasoningSummary{}
	}

	return json.Marshal(reasoningItem{
		itemAlias: itemAlias(item),
		Summary:   summary,
	})
}

// MCPListedTool is the Responses wire representation of an MCP discovery
// result. Raw JSON retains arbitrary JSON Schema and MCP annotation objects
// without routing them through a provider-private sidecar.
type MCPListedTool struct {
	Residual    json.RawMessage `json:"-"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Annotations json.RawMessage `json:"annotations,omitempty"`
}

func (tool *MCPListedTool) UnmarshalJSON(data []byte) error {
	type toolWire MCPListedTool
	var wire toolWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*tool = MCPListedTool(wire)
	tool.Residual = jsonObjectResidual(data, responsesMCPListedToolFields)
	return nil
}

func (tool MCPListedTool) MarshalJSON() ([]byte, error) {
	type toolWire MCPListedTool
	typed, err := json.Marshal(toolWire(tool))
	if err != nil {
		return nil, err
	}
	return mergeResidualObject(typed, tool.Residual)
}

// isOutputMessageContent checks if Content.Items contains output message content items.
func (item Item) isOutputMessageContent() bool {
	if item.Content == nil || len(item.Content.Items) == 0 {
		return false
	}

	for _, ci := range item.Content.Items {
		if ci.Type == "output_text" {
			return true
		}
	}

	return false
}

// GetContentItems returns the content items as []ContentItem for output message items.
// This is a helper method to access content items stored in Content.Items.
func (item Item) GetContentItems() []ContentItem {
	if item.Content == nil || len(item.Content.Items) == 0 {
		return nil
	}

	result := make([]ContentItem, 0, len(item.Content.Items))
	for _, ci := range item.Content.Items {
		text := ""
		if ci.Text != nil {
			text = *ci.Text
		}

		result = append(result, ContentItem{
			Type:        ci.Type,
			Text:        text,
			Annotations: append([]Annotation(nil), ci.Annotations...),
		})
	}

	return result
}

// SetContentItems sets the content items from []ContentItem for output message items.
// This is a helper method to store content items in Content.Items.
func (item *Item) SetContentItems(items []ContentItem) {
	if len(items) == 0 {
		return
	}

	contentItems := make([]Item, 0, len(items))
	for _, ci := range items {
		contentItems = append(contentItems, Item{
			Type:        ci.Type,
			Text:        &ci.Text,
			Annotations: append([]Annotation(nil), ci.Annotations...),
		})
	}

	item.Content = &Input{Items: contentItems}
}

// ReasoningSummary represents a summary text from the model.
type ReasoningSummary struct {
	Residual json.RawMessage `json:"-"`
	// A summary of the reasoning output from the model.
	Text string `json:"text"`
	// The type of the object. Always "summary_text".
	Type string `json:"type"`
}

func (summary *ReasoningSummary) UnmarshalJSON(data []byte) error {
	type summaryWire ReasoningSummary
	var wire summaryWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*summary = ReasoningSummary(wire)
	summary.Residual = jsonObjectResidual(data, responsesSummaryFields)
	return nil
}

func (summary ReasoningSummary) MarshalJSON() ([]byte, error) {
	type summaryWire ReasoningSummary
	typed, _ := json.Marshal(summaryWire(summary))
	return mergeResidualObject(typed, summary.Residual)
}

// ReasoningContent represents reasoning text from the model.
type ReasoningContent struct {
	Residual json.RawMessage `json:"-"`
	// The reasoning text from the model.
	Text string `json:"text"`
	// The type of the reasoning text. Always "reasoning_text".
	Type string `json:"type"`
}

func (content *ReasoningContent) UnmarshalJSON(data []byte) error {
	type contentWire ReasoningContent
	var wire contentWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*content = ReasoningContent(wire)
	content.Residual = jsonObjectResidual(data, responsesReasoningFields)
	return nil
}

func (content ReasoningContent) MarshalJSON() ([]byte, error) {
	type contentWire ReasoningContent
	typed, _ := json.Marshal(contentWire(content))
	return mergeResidualObject(typed, content.Residual)
}

type Response struct {
	// Residual contains provider response fields not owned by the typed model.
	// Current canonical fields overlay it during an identity projection.
	Residual json.RawMessage `json:"-"`
	// The object type of this resource - always set to "response".
	Object string `json:"object"`
	// Unique identifier for this Response.
	ID string `json:"id"`
	// An error object returned when the model fails to generate a Response.
	Error *Error `json:"error,omitempty"`
	// Unix timestamp (in seconds) of when this Response was created.
	CreatedAt int64 `json:"created_at"`
	// Model ID used to generate the response.
	Model string `json:"model"`
	// An array of content items generated by the model.
	Output []Item `json:"output"`

	// Status of the response.
	// Any of "completed", "failed", "in_progress", "canceled", "queued", or "incomplete".
	Status *string `json:"status,omitempty"`

	// Details about why the response is incomplete.
	IncompleteDetails *ResponseIncompleteDetails `json:"incomplete_details,omitempty"`

	// A system (or developer) message inserted into the model's context.
	Instructions *string `json:"instructions,omitempty"`

	// Set of key-value pairs that can be attached to an object.
	Metadata map[string]string `json:"metadata,omitempty"`

	// Whether to allow the model to run tool calls in parallel.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// What sampling temperature to use, between 0 and 2.
	Temperature *float64 `json:"temperature,omitempty"`

	// How the model should select which tool (or tools) to use.
	ToolChoice *ResponseToolChoice `json:"tool_choice,omitempty"`

	// An array of tools the model may call while generating a response.
	Tools []Tool `json:"tools,omitempty"`

	// An alternative to sampling with temperature, called nucleus sampling.
	TopP *float64 `json:"top_p,omitempty"`

	// Whether to run the model response in the background.
	Background *bool `json:"background,omitempty"`

	// The conversation that this response belongs to.
	Conversation *ResponseConversation `json:"conversation,omitempty"`

	// An upper bound for the number of tokens that can be generated for a response.
	MaxOutputTokens *int64 `json:"max_output_tokens,omitempty"`

	// The maximum number of total calls to built-in tools that can be processed.
	MaxToolCalls *int64 `json:"max_tool_calls,omitempty"`

	// The unique ID of the previous response to the model.
	PreviousResponseID *string `json:"previous_response_id,omitempty"`

	// Reference to a prompt template and its variables.
	Prompt *Prompt `json:"prompt,omitempty"`

	// Used by OpenAI to cache responses for similar requests.
	PromptCacheKey *string `json:"prompt_cache_key,omitempty"`

	// The retention policy for the prompt cache. Any of "in-memory", "24h".
	PromptCacheRetention *string `json:"prompt_cache_retention,omitempty"`

	// Configuration options for reasoning models.
	Reasoning *ResponseReasoning `json:"reasoning,omitempty"`

	// A stable identifier used to help detect users that may be violating usage policies.
	SafetyIdentifier *string `json:"safety_identifier,omitempty"`

	// Specifies the processing type used for serving the request.
	// Any of "auto", "default", "flex", "scale", "priority".
	ServiceTier *string `json:"service_tier,omitempty"`

	// Configuration options for a text response from the model.
	Text *TextOptions `json:"text,omitempty"`

	// An integer between 0 and 20 specifying the number of most likely tokens to return.
	TopLogprobs *int64 `json:"top_logprobs,omitempty"`

	// The truncation strategy to use for the model response. Any of "auto", "disabled".
	Truncation *string `json:"truncation,omitempty"`

	// Represents token usage details.
	Usage *Usage `json:"usage,omitempty"`

	// A stable identifier for your end-users (deprecated, use safety_identifier).
	User *string `json:"user,omitempty"`
}

func (response *Response) UnmarshalJSON(data []byte) error {
	type responseWire Response
	var wire responseWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*response = Response(wire)
	response.Residual = responseResponseResidual(data)
	return nil
}

func (response Response) MarshalJSON() ([]byte, error) {
	type responseWire Response
	raw, err := json.Marshal(responseWire(response))
	if err != nil {
		return nil, err
	}
	return mergeResidualObject(raw, response.Residual)
}

type ContentItem struct {
	Type        string       `json:"type"`
	Text        string       `json:"text,omitempty"`
	Annotations []Annotation `json:"annotations,omitempty"`
}

type Error struct {
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message"`
}

type rawJSONSchema struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
	Strict      *bool           `json:"strict,omitempty"`
}

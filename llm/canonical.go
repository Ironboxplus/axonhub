package llm

import (
	"encoding/json"
	"errors"
	"fmt"
)

type ItemKind string

const (
	ItemKindMessage             ItemKind = "message"
	ItemKindToolCall            ItemKind = "tool_call"
	ItemKindToolResult          ItemKind = "tool_result"
	ItemKindHostedCall          ItemKind = "hosted_tool_call"
	ItemKindMCPListTools        ItemKind = "mcp_list_tools"
	ItemKindMCPApprovalRequest  ItemKind = "mcp_approval_request"
	ItemKindMCPApprovalResponse ItemKind = "mcp_approval_response"
	ItemKindMCPCall             ItemKind = "mcp_call"
	ItemKindReasoning           ItemKind = "reasoning"
	ItemKindCompaction          ItemKind = "compaction"
	ItemKindUnknown             ItemKind = "unknown"
)

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

type ItemStatus string

const (
	ItemStatusInProgress ItemStatus = "in_progress"
	ItemStatusCompleted  ItemStatus = "completed"
	ItemStatusIncomplete ItemStatus = "incomplete"
	ItemStatusFailed     ItemStatus = "failed"
)

type ResponseStatus string

const (
	ResponseStatusQueued     ResponseStatus = "queued"
	ResponseStatusInProgress ResponseStatus = "in_progress"
	ResponseStatusCompleted  ResponseStatus = "completed"
	ResponseStatusIncomplete ResponseStatus = "incomplete"
	ResponseStatusFailed     ResponseStatus = "failed"
	ResponseStatusCancelled  ResponseStatus = "cancelled"
)

type ContentKind string

const (
	ContentKindText     ContentKind = "text"
	ContentKindRefusal  ContentKind = "refusal"
	ContentKindImage    ContentKind = "image"
	ContentKindAudio    ContentKind = "audio"
	ContentKindDocument ContentKind = "document"
	ContentKindCitation ContentKind = "citation"
	ContentKindUnknown  ContentKind = "unknown"
)

type ToolKind string

const (
	ToolKindFunction          ToolKind = "function"
	ToolKindCustom            ToolKind = "custom"
	ToolKindWebSearch         ToolKind = "web_search"
	ToolKindWebFetch          ToolKind = "web_fetch"
	ToolKindFileSearch        ToolKind = "file_search"
	ToolKindMCP               ToolKind = "mcp"
	ToolKindToolSearch        ToolKind = "tool_search"
	ToolKindCodeInterpreter   ToolKind = "code_interpreter"
	ToolKindCodeExecution     ToolKind = "code_execution"
	ToolKindShell             ToolKind = "shell"
	ToolKindLocalShell        ToolKind = "local_shell"
	ToolKindApplyPatch        ToolKind = "apply_patch"
	ToolKindComputer          ToolKind = "computer"
	ToolKindImageGeneration   ToolKind = "image_generation"
	ToolKindAnthropicServer   ToolKind = "anthropic_server"
	ToolKindUnknownBehavioral ToolKind = "unknown_behavioral"
)

type ExecutionOwner string

const (
	ExecutionOwnerClient   ExecutionOwner = "client"
	ExecutionOwnerProvider ExecutionOwner = "provider"
	ExecutionOwnerGateway  ExecutionOwner = "gateway"
)

type ToolCallStatus string

const (
	ToolCallStatusInProgress ToolCallStatus = "in_progress"
	ToolCallStatusCompleted  ToolCallStatus = "completed"
	ToolCallStatusIncomplete ToolCallStatus = "incomplete"
	ToolCallStatusFailed     ToolCallStatus = "failed"
)

type ToolResultStatus string

const (
	ToolResultStatusCompleted ToolResultStatus = "completed"
	ToolResultStatusFailed    ToolResultStatus = "failed"
)

// Item is the ordered canonical unit shared by request input and response
// output. Exactly one typed payload branch is allowed for every Kind.
type Item struct {
	Kind   ItemKind   `json:"kind"`
	ID     string     `json:"id,omitempty"`
	Role   Role       `json:"role,omitempty"`
	Status ItemStatus `json:"status,omitempty"`

	Content             []ContentBlock       `json:"content,omitempty"`
	ToolCall            *ToolInvocation      `json:"tool_call,omitempty"`
	ToolResult          *ToolResult          `json:"tool_result,omitempty"`
	HostedCall          *HostedToolCall      `json:"hosted_call,omitempty"`
	MCPListTools        *MCPListTools        `json:"mcp_list_tools,omitempty"`
	MCPApprovalRequest  *MCPApprovalRequest  `json:"mcp_approval_request,omitempty"`
	MCPApprovalResponse *MCPApprovalResponse `json:"mcp_approval_response,omitempty"`
	MCPCall             *MCPCall             `json:"mcp_call,omitempty"`
	Reasoning           *ReasoningItem       `json:"reasoning,omitempty"`
	Compaction          *CompactionItem      `json:"compaction,omitempty"`
	Unknown             *UnknownItem         `json:"unknown,omitempty"`
	ProtocolHints       ProtocolHints        `json:"protocol_hints,omitempty"`
}

// ProtocolHints may preserve non-behavioral source layout hints. Any field
// that changes model or tool behavior belongs in a typed canonical field.
type ProtocolHints struct {
	SourceFormat APIFormat `json:"source_format,omitempty"`
	SourceType   string    `json:"source_type,omitempty"`
	SourceRole   Role      `json:"source_role,omitempty"`
	Ordinal      int       `json:"ordinal,omitempty"`
}

type ContentBlock struct {
	Kind       ContentKind     `json:"kind"`
	ID         string          `json:"id,omitempty"`
	Text       string          `json:"text,omitempty"`
	StartIndex *int64          `json:"start_index,omitempty"`
	EndIndex   *int64          `json:"end_index,omitempty"`
	Image      *ImageURL       `json:"image,omitempty"`
	Audio      *InputAudio     `json:"audio,omitempty"`
	Document   *DocumentURL    `json:"document,omitempty"`
	Citation   *URLCitation    `json:"citation,omitempty"`
	UnknownRaw json.RawMessage `json:"unknown_raw,omitempty"`
}

type ToolDefinition struct {
	Kind        ToolKind `json:"kind"`
	LogicalName string   `json:"logical_name"`
	Description string   `json:"description,omitempty"`
	// DeferLoading keeps tool-discovery visibility separate from the tool's
	// executable schema. A deferred definition is part of the request catalog
	// but must not be exposed to a target model until a tool-search result
	// activates it. This is behavioral state, not a protocol hint.
	DeferLoading  *bool                 `json:"defer_loading,omitempty"`
	Function      *FunctionDefinition   `json:"function,omitempty"`
	Freeform      *FreeformDefinition   `json:"freeform,omitempty"`
	Hosted        *HostedToolDefinition `json:"hosted,omitempty"`
	MCP           *MCPDefinition        `json:"mcp,omitempty"`
	Execution     ExecutionOwner        `json:"execution"`
	ProtocolHints ProtocolHints         `json:"protocol_hints,omitempty"`
}

type FunctionDefinition struct {
	Parameters json.RawMessage `json:"parameters,omitempty"`
	Strict     *bool           `json:"strict,omitempty"`
	Namespace  string          `json:"namespace,omitempty"`
}

type FreeformDefinition struct {
	Format     string `json:"format,omitempty"`
	Syntax     string `json:"syntax,omitempty"`
	Definition string `json:"definition,omitempty"`
	Namespace  string `json:"namespace,omitempty"`
}

type HostedToolDefinition struct {
	Type    string `json:"type"`
	Version string `json:"version,omitempty"`
	// WebSearch is the provider-neutral capability subset shared by Responses
	// and Anthropic. Configuration keeps source-private fields for identity
	// routes; cross-protocol encoders must read this typed view.
	WebSearch     *WebSearch      `json:"web_search,omitempty"`
	WebFetch      *WebFetch       `json:"web_fetch,omitempty"`
	Configuration json.RawMessage `json:"configuration,omitempty"`
}

// WebFetch is the provider-neutral subset of Anthropic's versioned web-fetch
// definitions. Version-specific fields that do not affect another protocol's
// behavior remain in HostedToolDefinition.Configuration.
type WebFetch struct {
	MaxUses           *int64   `json:"max_uses,omitempty"`
	AllowedDomains    []string `json:"allowed_domains,omitempty"`
	BlockedDomains    []string `json:"blocked_domains,omitempty"`
	CitationsEnabled  *bool    `json:"citations_enabled,omitempty"`
	MaxContentTokens  *int64   `json:"max_content_tokens,omitempty"`
	UseCache          *bool    `json:"use_cache,omitempty"`
	ResponseInclusion string   `json:"response_inclusion,omitempty"`
}

type MCPDefinition struct {
	ServerLabel       string             `json:"server_label"`
	ServerDescription string             `json:"server_description,omitempty"`
	ServerURL         string             `json:"server_url,omitempty"`
	ConnectorID       string             `json:"connector_id,omitempty"`
	TunnelID          string             `json:"tunnel_id,omitempty"`
	DeferLoading      *bool              `json:"defer_loading,omitempty"`
	AllowedCallers    []string           `json:"allowed_callers,omitempty"`
	AllowedTools      *MCPToolFilter     `json:"allowed_tools,omitempty"`
	RequireApproval   *MCPApprovalPolicy `json:"require_approval,omitempty"`
}

// MCPToolFilter is the protocol-neutral form shared by MCP discovery and
// protocol encoders. Current Responses uses an allow-list array; ReadOnly is
// a discovery-time constraint over MCP readOnlyHint annotations and must not
// be emitted as a provider-private object on a target that lacks that field.
// A nil ReadOnly means the source did not constrain the MCP annotation.
type MCPToolFilter struct {
	ToolNames []string `json:"tool_names,omitempty"`
	ReadOnly  *bool    `json:"read_only,omitempty"`
}

// HasMCPReadOnlyFilter reports whether a request needs MCP discovery before
// it can be projected to a wire that only accepts explicit tool-name lists.
// It is intentionally request-scoped: a route must not choose the gateway for
// ordinary portable MCP definitions merely because another request did.
func (request *Request) HasMCPReadOnlyFilter() bool {
	if request == nil {
		return false
	}
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if definition.Kind == ToolKindMCP && definition.MCP != nil &&
			definition.MCP.AllowedTools != nil && definition.MCP.AllowedTools.ReadOnly != nil {
			return true
		}
	}
	return false
}

type MCPApprovalMode string

const (
	MCPApprovalAlways MCPApprovalMode = "always"
	MCPApprovalNever  MCPApprovalMode = "never"
)

// MCPApprovalPolicy preserves either the simple always/never mode or the two
// approval filters supported by Responses. Mode and the filter pair are
// mutually exclusive.
type MCPApprovalPolicy struct {
	Mode   MCPApprovalMode `json:"mode,omitempty"`
	Always *MCPToolFilter  `json:"always,omitempty"`
	Never  *MCPToolFilter  `json:"never,omitempty"`
}

type ToolInvocation struct {
	Kind                ToolKind          `json:"kind"`
	ID                  string            `json:"id,omitempty"`
	CallID              string            `json:"call_id"`
	LogicalName         string            `json:"logical_name"`
	Namespace           string            `json:"namespace,omitempty"`
	ArgumentsJSON       json.RawMessage   `json:"arguments_json,omitempty"`
	ArgumentsText       string            `json:"arguments_text,omitempty"`
	InputText           string            `json:"input_text,omitempty"`
	Status              ToolCallStatus    `json:"status,omitempty"`
	Execution           ExecutionOwner    `json:"execution,omitempty"`
	ProviderData        json.RawMessage   `json:"provider_data,omitempty"`
	PendingSafetyChecks []ToolSafetyCheck `json:"pending_safety_checks,omitempty"`
}

type ToolResult struct {
	Kind                     ToolKind          `json:"kind"`
	CallID                   string            `json:"call_id"`
	LogicalName              string            `json:"logical_name,omitempty"`
	Content                  []ContentBlock    `json:"content,omitempty"`
	StructuredContent        json.RawMessage   `json:"structured_content,omitempty"`
	DiscoveredTools          []ToolDefinition  `json:"discovered_tools,omitempty"`
	Execution                ExecutionOwner    `json:"execution,omitempty"`
	IsError                  bool              `json:"is_error,omitempty"`
	Status                   ToolResultStatus  `json:"status,omitempty"`
	ProviderData             json.RawMessage   `json:"provider_data,omitempty"`
	AcknowledgedSafetyChecks []ToolSafetyCheck `json:"acknowledged_safety_checks,omitempty"`
}

// ToolSafetyCheck carries provider-requested confirmations for client-owned
// computer actions. Keeping it canonical prevents protocol conversion from
// hiding a required safety decision inside opaque provider data.
type ToolSafetyCheck struct {
	ID      string `json:"id,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

type HostedToolCall struct {
	Invocation ToolInvocation `json:"invocation"`
	Result     *ToolResult    `json:"result,omitempty"`
}

// MCPDiscoveredTool is the provider-neutral discovery result shared by native
// Responses MCP and the gateway MCP client. Responses currently exposes only
// Name, Description, InputSchema, and Annotations; the remaining MCP-native
// fields prevent the gateway execution path from needing a second model.
type MCPDiscoveredTool struct {
	Name         string          `json:"name"`
	Title        string          `json:"title,omitempty"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"input_schema"`
	OutputSchema json.RawMessage `json:"output_schema,omitempty"`
	Annotations  json.RawMessage `json:"annotations,omitempty"`
	Meta         json.RawMessage `json:"meta,omitempty"`
}

type MCPListTools struct {
	ServerLabel string              `json:"server_label"`
	Tools       []MCPDiscoveredTool `json:"tools,omitempty"`
	Error       string              `json:"error,omitempty"`
}

type MCPApprovalRequest struct {
	ServerLabel   string          `json:"server_label"`
	LogicalName   string          `json:"logical_name"`
	ArgumentsJSON json.RawMessage `json:"arguments_json,omitempty"`
	ArgumentsText string          `json:"arguments_text,omitempty"`
}

type MCPApprovalResponse struct {
	ApprovalRequestID string `json:"approval_request_id"`
	Approve           bool   `json:"approve"`
	Reason            string `json:"reason,omitempty"`
}

type MCPCallStatus string

const (
	MCPCallStatusCalling    MCPCallStatus = "calling"
	MCPCallStatusInProgress MCPCallStatus = "in_progress"
	MCPCallStatusCompleted  MCPCallStatus = "completed"
	MCPCallStatusIncomplete MCPCallStatus = "incomplete"
	MCPCallStatusFailed     MCPCallStatus = "failed"
)

// MCPCall deliberately remains distinct from ToolInvocation/ToolResult.
// Responses models a provider-owned MCP invocation and its terminal output as
// one item, with approval correlation and a dedicated stream lifecycle.
type MCPCall struct {
	ServerLabel       string          `json:"server_label"`
	LogicalName       string          `json:"logical_name"`
	ApprovalRequestID string          `json:"approval_request_id,omitempty"`
	ArgumentsJSON     json.RawMessage `json:"arguments_json,omitempty"`
	ArgumentsText     string          `json:"arguments_text,omitempty"`
	Output            string          `json:"output,omitempty"`
	Error             string          `json:"error,omitempty"`
	Status            MCPCallStatus   `json:"status,omitempty"`
}

type CompactionItem struct {
	EncryptedContent string `json:"encrypted_content,omitempty"`
	CreatedBy        string `json:"created_by,omitempty"`
}

type UnknownItem struct {
	Type       string          `json:"type"`
	Raw        json.RawMessage `json:"raw"`
	Behavioral bool            `json:"behavioral"`
}

type ConversationRef struct {
	ID       string `json:"id"`
	ParentID string `json:"parent_id,omitempty"`
}

type LifecycleOptions struct {
	// Pointers preserve the difference between a client explicitly selecting
	// false and leaving the protocol default unspecified. A conversation store
	// must make that policy decision; protocol adapters must not guess it.
	Background *bool `json:"background,omitempty"`
	Store      *bool `json:"store,omitempty"`
}

func ValidateItems(items []Item) error {
	if err := ValidateItemStructure(items); err != nil {
		return err
	}
	calls := make(map[string]ToolKind)
	results := make(map[string]struct{})
	for index := range items {
		item := &items[index]
		switch item.Kind {
		case ItemKindToolCall:
			if _, duplicate := calls[item.ToolCall.CallID]; duplicate {
				return fmt.Errorf("item %d: duplicate tool call_id %q", index, item.ToolCall.CallID)
			}
			calls[item.ToolCall.CallID] = item.ToolCall.Kind
		case ItemKindToolResult:
			callKind, found := calls[item.ToolResult.CallID]
			if !found {
				return fmt.Errorf("item %d: tool result references unknown call_id %q", index, item.ToolResult.CallID)
			}
			if _, duplicate := results[item.ToolResult.CallID]; duplicate {
				return fmt.Errorf("item %d: duplicate result for call_id %q", index, item.ToolResult.CallID)
			}
			if item.ToolResult.Kind != callKind {
				return fmt.Errorf("item %d: result kind %q does not match call kind %q", index, item.ToolResult.Kind, callKind)
			}
			results[item.ToolResult.CallID] = struct{}{}
		}
	}
	return nil
}

// ValidateItemStructure checks typed union and field invariants without
// requiring the request to contain the earlier turn that introduced every
// call_id. Decoders use this boundary; candidate planning uses ValidateItems
// when a complete local lifecycle is required.
func ValidateItemStructure(items []Item) error {
	for index := range items {
		if err := items[index].Validate(); err != nil {
			return fmt.Errorf("item %d: %w", index, err)
		}
	}
	return nil
}

func (item *Item) Validate() error {
	if item == nil {
		return errors.New("nil item")
	}
	branches := 0
	if len(item.Content) > 0 {
		branches++
	}
	for _, present := range []bool{
		item.ToolCall != nil,
		item.ToolResult != nil,
		item.HostedCall != nil,
		item.MCPListTools != nil,
		item.MCPApprovalRequest != nil,
		item.MCPApprovalResponse != nil,
		item.MCPCall != nil,
		item.Reasoning != nil,
		item.Compaction != nil,
		item.Unknown != nil,
	} {
		if present {
			branches++
		}
	}
	if branches != 1 {
		return fmt.Errorf("kind %q has %d payload branches, want exactly one", item.Kind, branches)
	}

	switch item.Kind {
	case ItemKindMessage:
		if len(item.Content) == 0 || item.Role == "" {
			return errors.New("message requires role and content")
		}
		for index := range item.Content {
			if err := item.Content[index].Validate(); err != nil {
				return fmt.Errorf("content %d: %w", index, err)
			}
		}
	case ItemKindToolCall:
		if item.ToolCall == nil {
			return errors.New("tool_call payload is missing")
		}
		return item.ToolCall.Validate()
	case ItemKindToolResult:
		if item.ToolResult == nil {
			return errors.New("tool_result payload is missing")
		}
		return item.ToolResult.Validate()
	case ItemKindHostedCall:
		if item.HostedCall == nil {
			return errors.New("hosted_tool_call payload is missing")
		}
		if err := item.HostedCall.Invocation.Validate(); err != nil {
			return err
		}
		if item.HostedCall.Invocation.Execution != ExecutionOwnerProvider {
			return errors.New("hosted_tool_call invocation must be provider-owned")
		}
		if item.HostedCall.Result == nil {
			return nil
		}
		if err := item.HostedCall.Result.Validate(); err != nil {
			return err
		}
		if item.HostedCall.Result.Kind != item.HostedCall.Invocation.Kind {
			return fmt.Errorf("hosted result kind %q does not match invocation kind %q", item.HostedCall.Result.Kind, item.HostedCall.Invocation.Kind)
		}
		if item.HostedCall.Result.CallID != item.HostedCall.Invocation.CallID {
			return fmt.Errorf("hosted result call_id %q does not match invocation call_id %q", item.HostedCall.Result.CallID, item.HostedCall.Invocation.CallID)
		}
		return nil
	case ItemKindMCPListTools:
		if item.MCPListTools == nil {
			return errors.New("mcp_list_tools payload is missing")
		}
		if item.ID == "" {
			return errors.New("mcp_list_tools requires an item id")
		}
		return item.MCPListTools.Validate()
	case ItemKindMCPApprovalRequest:
		if item.MCPApprovalRequest == nil {
			return errors.New("mcp_approval_request payload is missing")
		}
		if item.ID == "" {
			return errors.New("mcp_approval_request requires an item id")
		}
		return item.MCPApprovalRequest.Validate()
	case ItemKindMCPApprovalResponse:
		if item.MCPApprovalResponse == nil {
			return errors.New("mcp_approval_response payload is missing")
		}
		return item.MCPApprovalResponse.Validate()
	case ItemKindMCPCall:
		if item.MCPCall == nil {
			return errors.New("mcp_call payload is missing")
		}
		if item.ID == "" {
			return errors.New("mcp_call requires an item id")
		}
		return item.MCPCall.Validate()
	case ItemKindReasoning:
		if item.Reasoning == nil {
			return errors.New("reasoning payload is missing")
		}
	case ItemKindCompaction:
		if item.Compaction == nil {
			return errors.New("compaction payload is missing")
		}
	case ItemKindUnknown:
		if item.Unknown == nil || item.Unknown.Type == "" || !json.Valid(item.Unknown.Raw) {
			return errors.New("unknown item requires source type and valid raw JSON")
		}
	default:
		return fmt.Errorf("unknown item kind %q", item.Kind)
	}
	return nil
}

func (content *ContentBlock) Validate() error {
	if content == nil {
		return errors.New("nil content")
	}
	switch content.Kind {
	case ContentKindText, ContentKindRefusal:
		return nil
	case ContentKindImage:
		if content.Image == nil {
			return errors.New("image content is missing image payload")
		}
	case ContentKindAudio:
		if content.Audio == nil {
			return errors.New("audio content is missing audio payload")
		}
	case ContentKindDocument:
		if content.Document == nil {
			return errors.New("document content is missing document payload")
		}
		if err := content.Document.Validate(); err != nil {
			return fmt.Errorf("invalid document content: %w", err)
		}
	case ContentKindCitation:
		if content.Citation == nil {
			return errors.New("citation content is missing citation payload")
		}
	case ContentKindUnknown:
		if !json.Valid(content.UnknownRaw) {
			return errors.New("unknown content requires valid raw JSON")
		}
	default:
		return fmt.Errorf("unknown content kind %q", content.Kind)
	}
	return nil
}

func (document *DocumentURL) Validate() error {
	if document == nil {
		return errors.New("document payload is missing")
	}
	sourceType := document.SourceType
	if sourceType == "" && document.URL != "" && document.Data == "" && document.FileID == "" && len(document.Content) == 0 {
		sourceType = DocumentSourceURL
	}
	sources := 0
	if document.URL != "" {
		sources++
	}
	if document.Data != "" {
		sources++
	}
	if document.FileID != "" {
		sources++
	}
	if len(document.Content) > 0 {
		sources++
	}
	if sources != 1 {
		return errors.New("document requires exactly one source payload")
	}
	switch sourceType {
	case DocumentSourceBase64:
		if document.Data == "" {
			return errors.New("base64 document requires data")
		}
	case DocumentSourceURL:
		if document.URL == "" {
			return errors.New("URL document requires url")
		}
	case DocumentSourceFile:
		if document.FileID == "" {
			return errors.New("file document requires file_id")
		}
	case DocumentSourceText:
		if document.Data == "" {
			return errors.New("text document requires data")
		}
	case DocumentSourceContent:
		if len(document.Content) == 0 || !json.Valid(document.Content) {
			return errors.New("content document requires valid JSON content")
		}
	default:
		return fmt.Errorf("unsupported document source type %q", sourceType)
	}
	return nil
}

func (call *ToolInvocation) Validate() error {
	if call == nil || call.Kind == "" || call.CallID == "" || call.LogicalName == "" {
		return errors.New("tool invocation requires kind, call_id, and logical_name")
	}
	if len(call.ArgumentsJSON) > 0 && !json.Valid(call.ArgumentsJSON) {
		return errors.New("tool invocation arguments_json is invalid")
	}
	if len(call.ArgumentsJSON) > 0 && call.ArgumentsText != "" {
		return errors.New("tool invocation cannot contain both arguments_json and arguments_text")
	}
	if len(call.ProviderData) > 0 && !json.Valid(call.ProviderData) {
		return errors.New("tool invocation provider_data is invalid JSON")
	}
	return nil
}

func (result *ToolResult) Validate() error {
	if result == nil || result.Kind == "" || result.CallID == "" {
		return errors.New("tool result requires kind and call_id")
	}
	for index := range result.Content {
		if err := result.Content[index].Validate(); err != nil {
			return fmt.Errorf("content %d: %w", index, err)
		}
	}
	if result.Kind == ToolKindToolSearch {
		if result.Execution == "" {
			return errors.New("tool search result requires execution owner")
		}
		for index := range result.DiscoveredTools {
			if err := result.DiscoveredTools[index].Validate(); err != nil {
				return fmt.Errorf("discovered tool %d: %w", index, err)
			}
		}
	} else if len(result.DiscoveredTools) > 0 {
		return errors.New("only tool search results may contain discovered tools")
	}
	if len(result.ProviderData) > 0 && !json.Valid(result.ProviderData) {
		return errors.New("tool result provider_data is invalid JSON")
	}
	if len(result.StructuredContent) > 0 && !json.Valid(result.StructuredContent) {
		return errors.New("tool result structured_content is invalid JSON")
	}
	return nil
}

func (list *MCPListTools) Validate() error {
	if list == nil || list.ServerLabel == "" {
		return errors.New("MCP tool list requires server_label")
	}
	for index := range list.Tools {
		tool := &list.Tools[index]
		if tool.Name == "" || len(tool.InputSchema) == 0 || !json.Valid(tool.InputSchema) {
			return fmt.Errorf("MCP discovered tool %d requires name and valid input_schema", index)
		}
		for field, value := range map[string]json.RawMessage{
			"output_schema": tool.OutputSchema,
			"annotations":   tool.Annotations,
			"meta":          tool.Meta,
		} {
			if len(value) > 0 && !json.Valid(value) {
				return fmt.Errorf("MCP discovered tool %d has invalid %s", index, field)
			}
		}
	}
	return nil
}

func (request *MCPApprovalRequest) Validate() error {
	if request == nil || request.ServerLabel == "" || request.LogicalName == "" {
		return errors.New("MCP approval request requires server_label and logical_name")
	}
	return validateMCPArguments(request.ArgumentsJSON, request.ArgumentsText)
}

func (response *MCPApprovalResponse) Validate() error {
	if response == nil || response.ApprovalRequestID == "" {
		return errors.New("MCP approval response requires approval_request_id")
	}
	return nil
}

func (call *MCPCall) Validate() error {
	if call == nil || call.ServerLabel == "" || call.LogicalName == "" {
		return errors.New("MCP call requires server_label and logical_name")
	}
	if err := validateMCPArguments(call.ArgumentsJSON, call.ArgumentsText); err != nil {
		return err
	}
	switch call.Status {
	case "", MCPCallStatusCalling, MCPCallStatusInProgress, MCPCallStatusCompleted,
		MCPCallStatusIncomplete, MCPCallStatusFailed:
		return nil
	default:
		return fmt.Errorf("unsupported MCP call status %q", call.Status)
	}
}

func validateMCPArguments(argumentsJSON json.RawMessage, argumentsText string) error {
	if len(argumentsJSON) > 0 && !json.Valid(argumentsJSON) {
		return errors.New("MCP arguments_json is invalid")
	}
	if len(argumentsJSON) > 0 && argumentsText != "" {
		return errors.New("MCP item cannot contain both arguments_json and arguments_text")
	}
	return nil
}

func (definition *ToolDefinition) Validate() error {
	if definition == nil || definition.Kind == "" || definition.LogicalName == "" || definition.Execution == "" {
		return errors.New("tool definition requires kind, logical_name, and execution owner")
	}
	branches := 0
	for _, present := range []bool{definition.Function != nil, definition.Freeform != nil, definition.Hosted != nil, definition.MCP != nil} {
		if present {
			branches++
		}
	}
	if branches != 1 {
		return fmt.Errorf("tool definition kind %q has %d payload branches, want exactly one", definition.Kind, branches)
	}
	if definition.Function != nil && len(definition.Function.Parameters) > 0 && !json.Valid(definition.Function.Parameters) {
		return errors.New("function parameters are invalid JSON")
	}
	if definition.Hosted != nil && len(definition.Hosted.Configuration) > 0 && !json.Valid(definition.Hosted.Configuration) {
		return errors.New("hosted tool configuration is invalid JSON")
	}
	if definition.MCP != nil {
		mcp := definition.MCP
		locations := 0
		for _, present := range []bool{mcp.ServerURL != "", mcp.ConnectorID != "", mcp.TunnelID != ""} {
			if present {
				locations++
			}
		}
		if mcp.ServerLabel == "" || locations != 1 {
			return errors.New("MCP tool requires server_label and exactly one server location")
		}
		if policy := mcp.RequireApproval; policy != nil {
			filtered := policy.Always != nil || policy.Never != nil
			if policy.Mode != "" && filtered {
				return errors.New("MCP approval mode and filters are mutually exclusive")
			}
			if policy.Mode == "" && !filtered {
				return errors.New("MCP approval policy requires a mode or filter")
			}
			if policy.Mode != "" && policy.Mode != MCPApprovalAlways && policy.Mode != MCPApprovalNever {
				return fmt.Errorf("unsupported MCP approval mode %q", policy.Mode)
			}
		}
	}
	return nil
}

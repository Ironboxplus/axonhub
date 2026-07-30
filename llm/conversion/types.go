package conversion

import (
	"errors"
	"fmt"
	"time"

	"github.com/looplj/axonhub/llm"
)

const PlanVersion uint32 = 4

var ErrIncompletePlan = errors.New("semantic conversion plan is incomplete")

type ActionKind string

const (
	ActionNative  ActionKind = "native"
	ActionLower   ActionKind = "lower"
	ActionEmulate ActionKind = "emulate"
	ActionOpaque  ActionKind = "opaque"
	ActionUnknown ActionKind = "unknown"
)

type ObjectKind string

const (
	ObjectToolDefinition ObjectKind = "tool_definition"
	ObjectToolChoice     ObjectKind = "tool_choice"
	ObjectToolCall       ObjectKind = "tool_call"
	ObjectToolResult     ObjectKind = "tool_result"
	ObjectInputItem      ObjectKind = "input_item"
	ObjectContentBlock   ObjectKind = "content_block"
	ObjectProviderData   ObjectKind = "provider_data"
	ObjectCompaction     ObjectKind = "compaction"
)

type StrategyID string

const (
	StrategyNative              StrategyID = "native"
	StrategyCustomAsFunction    StrategyID = "custom_as_function"
	StrategyReasoningProject    StrategyID = "reasoning_projection"
	StrategyCitationProject     StrategyID = "citation_projection"
	StrategyHostedProject       StrategyID = "hosted_lifecycle_projection"
	StrategyHostedGateway       StrategyID = "hosted_gateway"
	StrategyClientToolAsFunc    StrategyID = "client_tool_as_function"
	StrategyMCPGateway          StrategyID = "mcp_gateway"
	StrategyAllowedTools        StrategyID = "allowed_tools_projection"
	StrategyIdentifierNormalize StrategyID = "identifier_normalization"
	StrategySchemaNormalize     StrategyID = "schema_normalization"
	StrategyOpaqueSidecar       StrategyID = "opaque_sidecar"
	StrategyCompactAsChat       StrategyID = "compact_as_chat"
	StrategyUnavailable         StrategyID = "unavailable"
)

type ReasonCode string

const (
	ReasonTargetNative       ReasonCode = "target_native"
	ReasonTargetFunctionOnly ReasonCode = "target_function_only"
	ReasonSemanticProjection ReasonCode = "semantic_projection"
	ReasonGatewayExecution   ReasonCode = "gateway_execution"
	ReasonProviderPrivate    ReasonCode = "provider_private"
	ReasonSameProtocolOpaque ReasonCode = "same_protocol_opaque"
	ReasonProtocolConstraint ReasonCode = "protocol_constraint"
	ReasonTargetNoCompact    ReasonCode = "target_no_compact"
	ReasonNoStrategy         ReasonCode = "no_strategy"
)

type ObjectRef struct {
	Kind          ObjectKind
	ToolIndex     int
	ItemIndex     int
	ContentIndex  int
	MessageIndex  int
	ToolCallIndex int
}

type Action struct {
	Ref        ObjectRef
	Kind       ActionKind
	Strategy   StrategyID
	Reason     ReasonCode
	Reversible bool
}

type CapabilityProfile struct {
	ID            string
	APIFormat     llm.APIFormat
	NativeTools   ToolCapabilitySet
	EmulatedTools ToolCapabilitySet
}

type ToolCapabilitySet uint32

const (
	CapabilityFunctionTool ToolCapabilitySet = 1 << iota
	CapabilityCustomTool
	CapabilityWebSearchTool
	CapabilityImageGenerationTool
	CapabilityMCPTool
	CapabilityLocalShellTool
	CapabilityToolSearch
	CapabilityWebFetchTool
	CapabilityFileSearchTool
	CapabilityCodeExecutionTool
	CapabilityComputerTool
	CapabilityShellTool
	CapabilityApplyPatchTool
	CapabilityAnthropicServerTool
)

func (set ToolCapabilitySet) Supports(capability ToolCapabilitySet) bool {
	return capability != 0 && set&capability == capability
}

func CapabilityForToolKind(kind llm.ToolKind) ToolCapabilitySet {
	return capabilityForToolKind(string(kind))
}

func ProfileFor(format llm.APIFormat) (CapabilityProfile, bool) {
	switch format {
	case llm.APIFormatOpenAIChatCompletion:
		return CapabilityProfile{
			ID: "openai-chat/v1", APIFormat: format,
			NativeTools: CapabilityFunctionTool,
		}, true
	case llm.APIFormatOpenAIResponse:
		return CapabilityProfile{
			ID: "openai-responses/v1", APIFormat: format,
			NativeTools: CapabilityFunctionTool | CapabilityCustomTool | CapabilityWebSearchTool | CapabilityImageGenerationTool |
				CapabilityMCPTool | CapabilityLocalShellTool | CapabilityToolSearch | CapabilityFileSearchTool |
				CapabilityCodeExecutionTool | CapabilityComputerTool | CapabilityShellTool | CapabilityApplyPatchTool,
		}, true
	case llm.APIFormatAnthropicMessage:
		return CapabilityProfile{
			ID: "anthropic-messages/v1", APIFormat: format,
			NativeTools: CapabilityFunctionTool | CapabilityWebSearchTool | CapabilityWebFetchTool | CapabilityCodeExecutionTool | CapabilityToolSearch,
		}, true
	default:
		return CapabilityProfile{}, false
	}
}

type Plan struct {
	Source  llm.APIFormat
	Target  CapabilityProfile
	Actions []Action
	Summary llm.ConversionTraceSummary
	Debug   *llm.ConversionDebugTrace
}

func (p *Plan) Complete() bool {
	return p != nil && p.Summary.Complete
}

func (p *Plan) Validate() error {
	if p == nil {
		return fmt.Errorf("%w: nil plan", ErrIncompletePlan)
	}
	if !p.Complete() {
		return fmt.Errorf("%w: %s to %s", ErrIncompletePlan, p.Source, p.Target.APIFormat)
	}
	return nil
}

func summarizePlan(source llm.APIFormat, target CapabilityProfile, actions []Action, startedAt time.Time) llm.ConversionTraceSummary {
	summary := llm.ConversionTraceSummary{
		SourceFormat: source,
		TargetFormat: target.APIFormat,
		ProfileID:    target.ID,
		PlanVersion:  PlanVersion,
		Complete:     true,
	}
	if !startedAt.IsZero() {
		summary.PlanNanos = time.Since(startedAt).Nanoseconds()
	}
	for _, action := range actions {
		switch action.Kind {
		case ActionNative:
			summary.Native++
		case ActionLower:
			summary.Lowered++
		case ActionEmulate:
			summary.Emulated++
		case ActionOpaque:
			summary.Opaque++
		case ActionUnknown:
			summary.Unknown++
			summary.Complete = false
		}
	}
	return summary
}

package llm

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"sync"
)

type conversionTraceContextKey struct{}

type conversionDebugTraceContextKey struct{}

const (
	// DefaultConversionDebugActions bounds sampled action evidence unless the
	// caller asks for a smaller limit.
	DefaultConversionDebugActions = 32
	// MaxConversionDebugActions is a hard privacy and storage bound. Callers
	// cannot raise it through configuration.
	MaxConversionDebugActions = 64
	// DefaultRequiredConversionActions bounds evidence that is retained even
	// when full debug sampling is disabled. Required evidence contains only
	// compatibility changes and failures, never native pass-through actions.
	DefaultRequiredConversionActions = 32
)

type conversionDebugTraceConfig struct {
	key        [sha256.Size]byte
	maxActions int
}

// WithConversionTrace enables optional conversion timing and response summary
// attachment for the current pipeline execution.
func WithConversionTrace(ctx context.Context) context.Context {
	return context.WithValue(ctx, conversionTraceContextKey{}, true)
}

func ConversionTraceEnabled(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	enabled, _ := ctx.Value(conversionTraceContextKey{}).(bool)
	return enabled
}

// WithConversionDebugTrace enables bounded action-level conversion evidence.
// The request-scoped key is immediately reduced to a one-way digest and is
// used only to make logical object identifiers stable within this request but
// unlinkable across requests. The raw key is never retained or emitted.
func WithConversionDebugTrace(ctx context.Context, requestScopedKey []byte, maxActions int) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if maxActions <= 0 {
		maxActions = DefaultConversionDebugActions
	}
	if maxActions > MaxConversionDebugActions {
		maxActions = MaxConversionDebugActions
	}
	return context.WithValue(ctx, conversionDebugTraceContextKey{}, conversionDebugTraceConfig{
		key:        sha256.Sum256(requestScopedKey),
		maxActions: maxActions,
	})
}

// ConversionDirection identifies which side of the canonical boundary made a
// sampled decision. It is intentionally a closed, low-cardinality value.
type ConversionDirection string

const (
	ConversionDirectionRequest  ConversionDirection = "request"
	ConversionDirectionResponse ConversionDirection = "response"
	ConversionDirectionStream   ConversionDirection = "stream"
)

// ConversionEvidenceMode distinguishes always-on compatibility/failure
// evidence from sampled full traces. Required mode intentionally omits the
// request-scoped logical hash; ObjectID is already a payload-free structural
// location such as input[3].content[1].
type ConversionEvidenceMode string

const (
	ConversionEvidenceRequired ConversionEvidenceMode = "required"
	ConversionEvidenceSampled  ConversionEvidenceMode = "sampled"
)

// ConversionEvidenceStage is a closed, user-facing processing boundary. It
// deliberately avoids internal Go function names.
type ConversionEvidenceStage string

const (
	ConversionStageRequestPlanning  ConversionEvidenceStage = "request_planning"
	ConversionStageRequestTransform ConversionEvidenceStage = "request_transform"
	ConversionStageResponseRestore  ConversionEvidenceStage = "response_restore"
	ConversionStageStreamRestore    ConversionEvidenceStage = "stream_restore"
)

// ConversionEvidenceResult describes what happened to one canonical object or
// field at the stage above.
type ConversionEvidenceResult string

const (
	ConversionResultNative      ConversionEvidenceResult = "native"
	ConversionResultLowered     ConversionEvidenceResult = "lowered"
	ConversionResultEmulated    ConversionEvidenceResult = "emulated"
	ConversionResultOpaque      ConversionEvidenceResult = "opaque"
	ConversionResultUnknown     ConversionEvidenceResult = "unknown"
	ConversionResultRestored    ConversionEvidenceResult = "restored"
	ConversionResultRestoreMiss ConversionEvidenceResult = "restore_miss"
	ConversionResultNormalized  ConversionEvidenceResult = "normalized"
	ConversionResultRepaired    ConversionEvidenceResult = "repaired"
)

type ConversionEvidenceSeverity string

const (
	ConversionSeverityInfo     ConversionEvidenceSeverity = "info"
	ConversionSeverityWarning  ConversionEvidenceSeverity = "warning"
	ConversionSeverityCritical ConversionEvidenceSeverity = "critical"
)

// ConversionActionTrace explains one sampled conversion decision without
// carrying payloads, names, call IDs, schemas, arguments, results, or URLs.
type ConversionActionTrace struct {
	Seq             uint32                     `json:"seq"`
	Direction       ConversionDirection        `json:"direction"`
	ObjectKind      string                     `json:"object_kind"`
	ObjectID        string                     `json:"object_id"`
	FieldPath       string                     `json:"field_path"`
	DestinationPath string                     `json:"destination_path,omitempty"`
	Stage           ConversionEvidenceStage    `json:"stage"`
	Action          string                     `json:"action"`
	Strategy        string                     `json:"strategy"`
	Reason          string                     `json:"reason"`
	Result          ConversionEvidenceResult   `json:"result"`
	Severity        ConversionEvidenceSeverity `json:"severity"`
	Reversible      bool                       `json:"reversible"`
	LogicalIDHash   string                     `json:"logical_id_hash,omitempty"`
}

// ConversionDebugTrace is emitted only for sampled requests. Actions are
// always bounded by MaxConversionDebugActions; Truncated preserves evidence
// that additional decisions existed without retaining them.
type ConversionDebugTrace struct {
	Mode      ConversionEvidenceMode  `json:"mode"`
	Actions   []ConversionActionTrace `json:"actions,omitempty"`
	Truncated uint32                  `json:"truncated,omitempty"`

	key           [sha256.Size]byte
	maxActions    int
	hashLogicalID bool
	mu            sync.Mutex
}

// NewConversionDebugTrace returns nil unless the caller explicitly enabled
// sampled action tracing in the context. Allocation therefore stays entirely
// off the default request path.
func NewConversionDebugTrace(ctx context.Context, expectedActions int) *ConversionDebugTrace {
	if ctx == nil {
		return nil
	}
	config, ok := ctx.Value(conversionDebugTraceContextKey{}).(conversionDebugTraceConfig)
	if !ok || config.maxActions <= 0 {
		return nil
	}
	capacity := min(expectedActions, config.maxActions)
	if capacity < 0 {
		capacity = 0
	}
	return &ConversionDebugTrace{
		Mode:          ConversionEvidenceSampled,
		Actions:       make([]ConversionActionTrace, 0, capacity),
		key:           config.key,
		maxActions:    config.maxActions,
		hashLogicalID: true,
	}
}

// NewRequiredConversionDebugTrace creates a bounded trace for compatibility
// changes and failures that must remain observable even when sampling is off.
// Callers must still avoid invoking it for native-only requests.
func NewRequiredConversionDebugTrace(expectedActions int) *ConversionDebugTrace {
	maxActions := DefaultRequiredConversionActions
	capacity := min(expectedActions, maxActions)
	if capacity < 0 {
		capacity = 0
	}
	return &ConversionDebugTrace{
		Mode:       ConversionEvidenceRequired,
		Actions:    make([]ConversionActionTrace, 0, capacity),
		maxActions: maxActions,
	}
}

// Append records a payload-free decision. logicalRef must be a structural
// reference (indexes/kinds), never a raw tool name or provider call ID.
func (trace *ConversionDebugTrace) Append(action ConversionActionTrace, logicalRef []byte) {
	if trace == nil {
		return
	}
	trace.mu.Lock()
	defer trace.mu.Unlock()
	if len(trace.Actions) >= trace.maxActions {
		trace.Truncated++
		return
	}
	action.Seq = uint32(len(trace.Actions) + 1)
	if trace.hashLogicalID {
		hasher := hmac.New(sha256.New, trace.key[:])
		_, _ = hasher.Write(logicalRef)
		digest := hasher.Sum(nil)
		var logicalHash [16]byte
		hex.Encode(logicalHash[:], digest[:8])
		action.LogicalIDHash = string(logicalHash[:])
	}
	trace.Actions = append(trace.Actions, action)
}

// Clone returns a persistence-safe snapshot with no request-scoped hash key.
func (trace *ConversionDebugTrace) Clone() *ConversionDebugTrace {
	if trace == nil {
		return nil
	}
	trace.mu.Lock()
	defer trace.mu.Unlock()
	return &ConversionDebugTrace{
		Mode:      trace.Mode,
		Actions:   append([]ConversionActionTrace(nil), trace.Actions...),
		Truncated: trace.Truncated,
	}
}

// ConversionTraceSummary is the privacy-safe, fixed-size evidence emitted for
// one semantic conversion attempt. It deliberately contains no model names,
// tool names, call IDs, schemas, prompts, arguments, or results.
type ConversionTraceSummary struct {
	SourceFormat APIFormat `json:"source_format"`
	TargetFormat APIFormat `json:"target_format"`
	ProfileID    string    `json:"profile_id"`
	PlanVersion  uint32    `json:"plan_version"`

	Native   uint32 `json:"native"`
	Lowered  uint32 `json:"lowered"`
	Emulated uint32 `json:"emulated"`
	Opaque   uint32 `json:"opaque"`
	Unknown  uint32 `json:"unknown"`

	RestoreMiss           uint32    `json:"restore_miss"`
	CustomInputsRepaired  uint32    `json:"custom_inputs_repaired"`
	IdentifiersNormalized uint32    `json:"identifiers_normalized"`
	SchemasNormalized     uint32    `json:"schemas_normalized"`
	StreamViolations      uint32    `json:"stream_violations"`
	LastStreamViolation   string    `json:"last_stream_violation,omitempty"`
	LastStreamEvent       EventKind `json:"last_stream_event,omitempty"`
	TerminalEvent         EventKind `json:"terminal_event,omitempty"`
	Complete              bool      `json:"complete"`

	// Responses WebSocket transport evidence is counted once per logical
	// request, never per delta. It contains no URL, session, tenant, header, or
	// payload data and therefore stays fixed-size on long streams.
	WebSocketRequests         uint32 `json:"websocket_requests,omitempty"`
	WebSocketConnections      uint32 `json:"websocket_connections,omitempty"`
	WebSocketReuses           uint32 `json:"websocket_reuses,omitempty"`
	WebSocketReconnects       uint32 `json:"websocket_reconnects,omitempty"`
	WebSocketIncrementalSends uint32 `json:"websocket_incremental_sends,omitempty"`
	WebSocketFullContextSends uint32 `json:"websocket_full_context_sends,omitempty"`

	PlanNanos    int64 `json:"plan_nanos"`
	LowerNanos   int64 `json:"lower_nanos"`
	RestoreNanos int64 `json:"restore_nanos"`
}

// EmulationTraceSummary is the fixed-size, payload-free evidence emitted for
// one gateway-owned tool loop. Tool names, IDs, arguments, outputs, endpoints,
// and credentials are deliberately absent.
type EmulationTraceSummary struct {
	InternalRounds                uint32               `json:"internal_rounds"`
	ProviderRoundAttempts         uint32               `json:"provider_round_attempts,omitempty"`
	ProviderRoundsCompleted       uint32               `json:"provider_rounds_completed,omitempty"`
	ProviderRounds                []ProviderRoundTrace `json:"provider_rounds,omitempty"`
	ToolCalls                     uint32               `json:"tool_calls"`
	Approvals                     uint32               `json:"approvals"`
	Failures                      uint32               `json:"failures"`
	LimitHit                      bool                 `json:"limit_hit"`
	HostedWebSearchCalls          uint32               `json:"hosted_web_search_calls,omitempty"`
	HostedWebFetchCalls           uint32               `json:"hosted_web_fetch_calls,omitempty"`
	HostedFileSearchCalls         uint32               `json:"hosted_file_search_calls,omitempty"`
	HostedCodeCalls               uint32               `json:"hosted_code_calls,omitempty"`
	HostedShellCalls              uint32               `json:"hosted_shell_calls,omitempty"`
	HostedComputerCalls           uint32               `json:"hosted_computer_calls,omitempty"`
	HostedImageCalls              uint32               `json:"hosted_image_calls,omitempty"`
	HostedToolSearchCalls         uint32               `json:"hosted_tool_search_calls,omitempty"`
	HostedOtherCalls              uint32               `json:"hosted_other_calls,omitempty"`
	HostedBudgetRejections        uint32               `json:"hosted_budget_rejections,omitempty"`
	RepeatedHostedInvocations     uint32               `json:"repeated_hosted_invocations,omitempty"`
	StopReason                    string               `json:"stop_reason,omitempty"`
	CustomConstraintCompiles      uint32               `json:"custom_constraint_compiles"`
	CustomConstraintValidations   uint32               `json:"custom_constraint_validations"`
	CustomConstraintViolations    uint32               `json:"custom_constraint_violations"`
	CustomConstraintRetries       uint32               `json:"custom_constraint_retries"`
	CustomConstraintFallbacks     uint32               `json:"custom_constraint_fallbacks"`
	CustomConstraintCompileNanos  int64                `json:"custom_constraint_compile_nanos"`
	CustomConstraintValidateNanos int64                `json:"custom_constraint_validate_nanos"`
}

// ProviderRoundTrace is a bounded, payload-free record of one provider round
// inside a gateway-owned tool loop. It intentionally excludes model names,
// endpoints, request content, tool arguments, results, and credentials.
type ProviderRoundTrace struct {
	RoundIndex         uint32 `json:"round_index"`
	DurationMillis     int64  `json:"duration_ms"`
	Outcome            string `json:"outcome"`
	ProviderDispatched bool   `json:"provider_dispatched"`
}

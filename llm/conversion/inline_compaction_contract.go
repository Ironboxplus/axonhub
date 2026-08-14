package conversion

import (
	"context"
	"net/http"

	"github.com/looplj/axonhub/llm"
)

// CompactionStateCodec is supplied by the authenticated host. Seal must
// durably persist the state under the host's tenant/scope/TTL/replay policy and
// return a short, opaque, authenticated reference. Open must return owned=false
// for a token it did not mint in this scope, and an error for a malformed or
// expired owned token. Axon never owns a process-local state map or a secret.
//
// Implementations must not log state or token bytes. Axon records only bounded
// byte counts and a one-way digest in conversion evidence.
type CompactionStateCodec interface {
	Seal(context.Context, CompactionState) (string, error)
	Open(context.Context, string) (state CompactionState, owned bool, err error)
}

// CompactionStateCodecFailure lets the host classify its own durable-store
// boundary without leaking driver/SQL/remote error text. Unknown codec errors
// intentionally default to unavailable (503), never to a client-bad-token
// diagnosis.
type CompactionStateCodecFailureCode string

const (
	CompactionStateCodecFailureInvalid     CompactionStateCodecFailureCode = "invalid"
	CompactionStateCodecFailureUnavailable CompactionStateCodecFailureCode = "unavailable"
)

type CompactionStateCodecFailure interface {
	error
	CompactionStateCodecFailureCode() CompactionStateCodecFailureCode
}

// CompactionState is the provider-visible continuation Axon restores after an
// authenticated gateway checkpoint. Retained carries only Codex's safe
// user/developer/system input messages (including actual bounded input_image
// blocks). Continuation is one verified plaintext canonical message generated
// by the non-tool summary round. Tool/MCP/reasoning/provider-private history
// is deliberately never re-injected after compaction.
//
// A host may retain richer source evidence in its own protected storage, but
// that is not part of Axon's injectable continuation contract.
type CompactionState struct {
	Version      uint32     `json:"version"`
	Generation   uint32     `json:"generation"`
	Retained     []llm.Item `json:"retained"`
	Continuation llm.Item   `json:"continuation"`
}

const (
	inlineCompactionStateVersion = 1
	// MaxInlineCompactionCheckpointTokenBytes is the maximum opaque checkpoint
	// reference accepted before any codec Open or provider dispatch. Hosts use
	// this same contract for privacy-safe pre-open admission; tokens are durable
	// references, never serialized gateway state.
	MaxInlineCompactionCheckpointTokenBytes = 64 << 10
	maxInlineCompactionStateBytes           = 4 << 20
	maxInlineCompactionTokenBytes           = MaxInlineCompactionCheckpointTokenBytes
	maxInlineCompactionRetainedItemBytes    = 1 << 20
	maxInlineCompactionProjectedText        = 64 << 10
	maxInlineCompactionSummaryVisibleBytes  = 512 << 10
	maxInlineCompactionVisibleTextBytes     = 64 << 10
	inlineCompactionImageBudgetUnit         = 1
)

const inlineCompactionMaxOutputTokens int64 = 2048

const inlineCompactionContinuationSourceGroup = "axon_inline_compaction_continuation_v1"

type InlineCompactionErrorCode string

const (
	InlineCompactionCodecMissing     InlineCompactionErrorCode = "checkpoint_codec_missing"
	InlineCompactionCodecUnavailable InlineCompactionErrorCode = "checkpoint_codec_unavailable"
	InlineCompactionForeignToken     InlineCompactionErrorCode = "foreign_checkpoint"
	InlineCompactionInvalidToken     InlineCompactionErrorCode = "checkpoint_invalid"
	InlineCompactionOversize         InlineCompactionErrorCode = "checkpoint_too_large"
	InlineCompactionInvalidState     InlineCompactionErrorCode = "checkpoint_state_invalid"
	InlineCompactionUnsafeInput      InlineCompactionErrorCode = "checkpoint_input_unsafe"
	InlineCompactionUnsafeOutput     InlineCompactionErrorCode = "checkpoint_output_unsafe"
)

// InlineCompactionError is stable, typed, and intentionally excludes opaque
// token bytes, provider data, canonical history, and summary contents.
type InlineCompactionError struct {
	Code InlineCompactionErrorCode
	Err  error
}

func (err *InlineCompactionError) Error() string {
	if err == nil || err.Code == "" {
		return "inline compaction error"
	}
	return "inline compaction " + string(err.Code)
}

func (err *InlineCompactionError) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.Err
}

// SafeDiagnostic makes an actionable but payload-free failure available to
// observability. Unknown codes intentionally return zero diagnostics.
func (err *InlineCompactionError) SafeDiagnostic() llm.ErrorDiagnostic {
	if err == nil {
		return llm.ErrorDiagnostic{}
	}
	const component = "inline_compaction"
	switch err.Code {
	case InlineCompactionCodecMissing:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "Gateway compaction storage is not configured.", StatusCode: http.StatusServiceUnavailable}
	case InlineCompactionCodecUnavailable:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "Gateway compaction storage is temporarily unavailable.", StatusCode: http.StatusServiceUnavailable}
	case InlineCompactionForeignToken:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The compaction checkpoint is not owned by this gateway.", StatusCode: http.StatusBadRequest}
	case InlineCompactionInvalidToken:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The compaction checkpoint is invalid or expired.", StatusCode: http.StatusBadRequest}
	case InlineCompactionOversize:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The compaction checkpoint exceeds the gateway limit.", StatusCode: http.StatusBadRequest}
	case InlineCompactionInvalidState:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The gateway compaction state cannot be restored safely.", StatusCode: http.StatusInternalServerError}
	case InlineCompactionUnsafeInput:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The request contains state that cannot be compacted safely.", StatusCode: http.StatusBadRequest}
	case InlineCompactionUnsafeOutput:
		return llm.ErrorDiagnostic{Component: component, Code: string(err.Code), Message: "The summary provider returned no safe plaintext continuation.", StatusCode: http.StatusBadGateway}
	default:
		return llm.ErrorDiagnostic{}
	}
}

type inlineCompactionState struct {
	codec        CompactionStateCodec
	retained     []llm.Item
	generation   uint32
	drops        inlineCompactionDrops
	stateBytes   uint32
	sourceBytes  uint32
	sourceDigest string
}

type inlineCompactionDrops struct {
	Reasoning              uint32
	HostedProjected        uint32
	Private                uint32
	Structured             uint32
	RetainedTruncated      uint32
	RetainedTruncatedBytes uint32
	SummaryTruncated       uint32
	SummaryTruncatedBytes  uint32
	DocumentProjected      uint32
	DocumentTruncated      uint32
	DocumentTruncatedBytes uint32
	DocumentSidecars       uint32
	SourceSidecars         uint32
}

type inlineCompactionRetentionStats struct {
	Truncated      uint32
	TruncatedBytes uint32
}

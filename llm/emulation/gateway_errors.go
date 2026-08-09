package emulation

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/pipeline"
)

var (
	ErrGatewayWallTimeExceeded          = errors.New("gateway-hosted tool execution exceeded its total time budget")
	ErrGatewayProviderRoundTimeExceeded = errors.New("gateway provider round exceeded its time budget")
	ErrRequestDeadlineBudgetExhausted   = errors.New("request deadline left insufficient time for another gateway continuation round")
	ErrHostedMaxUses                    = errors.New("hosted tool execution limit exceeded")
	ErrRepeatedHostedInvocation         = errors.New("provider repeated a hosted tool call without making progress")
)

type GatewayStopReason string

const (
	GatewayStopWallTime           GatewayStopReason = "gateway_wall_time_exceeded"
	GatewayStopProviderRoundTime  GatewayStopReason = "gateway_provider_round_timeout"
	GatewayStopRequestDeadline    GatewayStopReason = "request_deadline_exceeded"
	GatewayStopHostedMaxUses      GatewayStopReason = "hosted_max_uses"
	GatewayStopRepeatedInvocation GatewayStopReason = "hosted_repeated_invocation"
	GatewayStopRequiredTool       GatewayStopReason = "required_gateway_tool_call_missing"
)

const (
	providerRoundOutcomeCompleted       = "completed"
	providerRoundOutcomeProviderTimeout = "provider_timeout"
	providerRoundOutcomeRequestDeadline = "request_deadline"
	providerRoundOutcomeGatewayWall     = "gateway_wall_time"
	providerRoundOutcomeCanceled        = "canceled"
	providerRoundOutcomeTransportError  = "transport_error"
)

type gatewayFailure struct {
	reason   GatewayStopReason
	sentinel error
	cause    error
}

func (failure *gatewayFailure) Error() string {
	if failure == nil || failure.sentinel == nil {
		return "gateway tool execution failed"
	}
	return failure.sentinel.Error()
}

func (failure *gatewayFailure) Unwrap() []error {
	if failure == nil {
		return nil
	}
	causes := make([]error, 0, 2)
	if failure.sentinel != nil {
		causes = append(causes, failure.sentinel)
	}
	if failure.cause != nil && failure.cause != failure.sentinel {
		causes = append(causes, failure.cause)
	}
	return causes
}

func (failure *gatewayFailure) SafeDiagnostic() llm.ErrorDiagnostic {
	if failure == nil {
		return llm.ErrorDiagnostic{}
	}
	switch failure.reason {
	case GatewayStopWallTime:
		return llm.ErrorDiagnostic{
			Component: "gateway", Code: "gateway_wall_time_exceeded",
			Message: "Gateway-hosted tool execution exceeded its total time budget.", StatusCode: http.StatusGatewayTimeout,
		}
	case GatewayStopProviderRoundTime:
		return llm.ErrorDiagnostic{
			Component: "gateway", Code: "gateway_provider_round_timeout",
			Message: "A gateway continuation round exceeded its provider time budget.", StatusCode: http.StatusGatewayTimeout,
		}
	case GatewayStopRequestDeadline:
		return llm.ErrorDiagnostic{
			Component: "request", Code: "request_deadline_exceeded",
			Message: "The request deadline left insufficient time for another gateway continuation round.", StatusCode: http.StatusGatewayTimeout,
		}
	case GatewayStopHostedMaxUses:
		return llm.ErrorDiagnostic{
			Component: "gateway", Code: "hosted_max_uses",
			Message: "The hosted tool reached its execution limit.", StatusCode: http.StatusBadGateway,
		}
	case GatewayStopRepeatedInvocation:
		return llm.ErrorDiagnostic{
			Component: "gateway", Code: "hosted_repeated_invocation",
			Message: "The provider repeated a hosted tool call without making progress.", StatusCode: http.StatusBadGateway,
		}
	case GatewayStopRequiredTool:
		return llm.ErrorDiagnostic{
			Component: "gateway", Code: "required_gateway_tool_call_missing",
			Message: "The provider did not call the required gateway tool.", StatusCode: http.StatusBadGateway,
		}
	default:
		return llm.ErrorDiagnostic{}
	}
}

func newGatewayFailure(reason GatewayStopReason, sentinel, cause error) error {
	return &gatewayFailure{reason: reason, sentinel: sentinel, cause: cause}
}

func normalizeGatewayContextError(ctx context.Context, err error) error {
	if err == nil || ctx == nil || !errors.Is(context.Cause(ctx), ErrGatewayWallTimeExceeded) {
		return err
	}
	pipeline.RecordEmulationStop(ctx, string(GatewayStopWallTime))
	return newGatewayFailure(
		GatewayStopWallTime,
		ErrGatewayWallTimeExceeded,
		errors.Join(err, context.DeadlineExceeded),
	)
}

func normalizeProviderRoundContextError(ctx context.Context, err error) error {
	if err == nil || ctx == nil {
		return err
	}
	switch cause := context.Cause(ctx); {
	case errors.Is(cause, ErrGatewayProviderRoundTimeExceeded):
		pipeline.RecordEmulationStop(ctx, string(GatewayStopProviderRoundTime))
		return newGatewayFailure(
			GatewayStopProviderRoundTime,
			ErrGatewayProviderRoundTimeExceeded,
			errors.Join(err, context.DeadlineExceeded),
		)
	case errors.Is(cause, ErrGatewayWallTimeExceeded):
		pipeline.RecordEmulationStop(ctx, string(GatewayStopWallTime))
		return newGatewayFailure(
			GatewayStopWallTime,
			ErrGatewayWallTimeExceeded,
			errors.Join(err, context.DeadlineExceeded),
		)
	case errors.Is(cause, ErrRequestDeadlineBudgetExhausted):
		pipeline.RecordEmulationStop(ctx, string(GatewayStopRequestDeadline))
		return newGatewayFailure(
			GatewayStopRequestDeadline,
			ErrRequestDeadlineBudgetExhausted,
			errors.Join(err, context.DeadlineExceeded),
		)
	default:
		return err
	}
}

func deadlineBudgetFailure(ctx context.Context, effectiveDeadline, gatewayDeadline time.Time) error {
	if cause := context.Cause(ctx); cause != nil {
		return normalizeGatewayContextError(ctx, cause)
	}
	if ownsGatewayDeadline(effectiveDeadline, gatewayDeadline) {
		pipeline.RecordEmulationStop(ctx, string(GatewayStopWallTime))
		return newGatewayFailure(GatewayStopWallTime, ErrGatewayWallTimeExceeded, context.DeadlineExceeded)
	}
	pipeline.RecordEmulationStop(ctx, string(GatewayStopRequestDeadline))
	return newGatewayFailure(
		GatewayStopRequestDeadline, ErrRequestDeadlineBudgetExhausted, context.DeadlineExceeded,
	)
}

func ownsGatewayDeadline(effectiveDeadline, gatewayDeadline time.Time) bool {
	return !gatewayDeadline.IsZero() && effectiveDeadline.Equal(gatewayDeadline)
}

func providerRoundOutcome(err error) string {
	switch {
	case err == nil:
		return providerRoundOutcomeCompleted
	case errors.Is(err, ErrGatewayProviderRoundTimeExceeded):
		return providerRoundOutcomeProviderTimeout
	case errors.Is(err, ErrGatewayWallTimeExceeded):
		return providerRoundOutcomeGatewayWall
	case errors.Is(err, ErrRequestDeadlineBudgetExhausted), errors.Is(err, context.DeadlineExceeded):
		return providerRoundOutcomeRequestDeadline
	case errors.Is(err, context.Canceled):
		return providerRoundOutcomeCanceled
	default:
		return providerRoundOutcomeTransportError
	}
}

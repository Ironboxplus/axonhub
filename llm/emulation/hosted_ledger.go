package emulation

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"strings"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/pipeline"
)

// hostedExecutionLedger is request-local. It enforces both source protocol
// max_uses and deployment-owned per-capability limits before any executor can
// leave the process. It stores only in-memory hashes of typed invocations;
// arguments and results never enter observation or durable state.
type hostedExecutionLedger struct {
	operatorLimits map[llm.ToolKind]int
	used           map[string]int
	seen           map[[32]byte]struct{}
}

func newHostedExecutionLedger(operatorLimits map[llm.ToolKind]int) *hostedExecutionLedger {
	return &hostedExecutionLedger{
		operatorLimits: operatorLimits,
		used:           make(map[string]int),
		seen:           make(map[[32]byte]struct{}),
	}
}

func (ledger *hostedExecutionLedger) reserve(ctx context.Context, calls []gatewayHostedCall) error {
	if ledger == nil || len(calls) == 0 {
		return nil
	}
	used := make(map[string]int, len(ledger.used))
	for key, count := range ledger.used {
		used[key] = count
	}
	seen := make(map[[32]byte]struct{}, len(ledger.seen)+len(calls))
	for fingerprint := range ledger.seen {
		seen[fingerprint] = struct{}{}
	}
	for index := range calls {
		call := calls[index]
		fingerprint := hostedInvocationFingerprint(call.binding, *call.item.ToolCall)
		if _, duplicate := seen[fingerprint]; duplicate {
			pipeline.RecordRepeatedHostedInvocation(ctx)
			pipeline.RecordEmulationStop(ctx, string(GatewayStopRepeatedInvocation))
			return newGatewayFailure(GatewayStopRepeatedInvocation, ErrRepeatedHostedInvocation, nil)
		}
		key := hostedDefinitionKey(call.binding.Definition)
		if limit := ledger.effectiveLimit(call.binding.Definition); limit > 0 && used[key] >= limit {
			pipeline.RecordHostedBudgetRejection(ctx)
			pipeline.RecordEmulationStop(ctx, string(GatewayStopHostedMaxUses))
			return newGatewayFailure(GatewayStopHostedMaxUses, ErrHostedMaxUses, nil)
		}
		seen[fingerprint] = struct{}{}
		used[key]++
	}
	ledger.used = used
	ledger.seen = seen
	return nil
}

func (ledger *hostedExecutionLedger) allows(definition llm.ToolDefinition) bool {
	if ledger == nil {
		return true
	}
	limit := ledger.effectiveLimit(definition)
	return limit <= 0 || ledger.used[hostedDefinitionKey(definition)] < limit
}

func (ledger *hostedExecutionLedger) effectiveLimit(definition llm.ToolDefinition) int {
	operatorLimit := 0
	if ledger != nil && ledger.operatorLimits != nil {
		operatorLimit = ledger.operatorLimits[definition.Kind]
	}
	sourceLimit := 0
	if definition.Hosted != nil && definition.Hosted.WebSearch != nil && definition.Hosted.WebSearch.MaxUses != nil && *definition.Hosted.WebSearch.MaxUses > 0 {
		sourceLimit = int(*definition.Hosted.WebSearch.MaxUses)
	}
	if operatorLimit <= 0 {
		return sourceLimit
	}
	if sourceLimit <= 0 || operatorLimit < sourceLimit {
		return operatorLimit
	}
	return sourceLimit
}

func hostedDefinitionKey(definition llm.ToolDefinition) string {
	return string(definition.Kind) + "\x00" + definition.LogicalName
}

func hostedInvocationFingerprint(binding hosted.Binding, invocation llm.ToolInvocation) [32]byte {
	canonical := canonicalHostedArguments(invocation)
	material := make([]byte, 0, len(binding.Definition.Kind)+len(binding.Definition.LogicalName)+len(canonical)+2)
	material = append(material, binding.Definition.Kind...)
	material = append(material, 0)
	material = append(material, binding.Definition.LogicalName...)
	material = append(material, 0)
	material = append(material, canonical...)
	return sha256.Sum256(material)
}

func canonicalHostedArguments(invocation llm.ToolInvocation) []byte {
	if len(invocation.ArgumentsJSON) > 0 {
		var value any
		if json.Unmarshal(invocation.ArgumentsJSON, &value) == nil {
			if canonical, err := json.Marshal(value); err == nil {
				return canonical
			}
		}
		return append([]byte(nil), invocation.ArgumentsJSON...)
	}
	if invocation.ArgumentsText != "" {
		return []byte(strings.TrimSpace(invocation.ArgumentsText))
	}
	return []byte(strings.TrimSpace(invocation.InputText))
}

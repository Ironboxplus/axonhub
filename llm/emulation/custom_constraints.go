package emulation

import (
	"context"
	"errors"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/emulation/freeform"
	"github.com/looplj/axonhub/llm/pipeline"
)

const customConstraintFeedback = "Custom tool input violated its required grammar. Call the same tool again with one complete input that matches the grammar exactly."

type customConstraintRegistry struct {
	validators map[string]*freeform.Validator
}

// NeedsCustomConstraintEmulation reports whether a target protocol loses a
// client-owned grammar constraint and therefore needs the gateway validator.
func NeedsCustomConstraintEmulation(request *llm.Request, target llm.APIFormat) bool {
	if request == nil || target == llm.APIFormatOpenAIResponse {
		return false
	}
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind == llm.ToolKindCustom && definition.Execution == llm.ExecutionOwnerClient &&
			definition.Freeform != nil && definition.Freeform.Format == "grammar" {
			return true
		}
	}
	return false
}

func compileCustomConstraints(ctx context.Context, request *llm.Request, target llm.APIFormat) (*customConstraintRegistry, error) {
	registry := &customConstraintRegistry{validators: make(map[string]*freeform.Validator)}
	if !NeedsCustomConstraintEmulation(request, target) {
		return registry, nil
	}
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindCustom || definition.Execution != llm.ExecutionOwnerClient ||
			definition.Freeform == nil || definition.Freeform.Format != "grammar" {
			continue
		}
		if _, duplicate := registry.validators[definition.LogicalName]; duplicate {
			_ = registry.Close(context.WithoutCancel(ctx))
			return nil, errors.New("duplicate constrained custom tool identity")
		}
		startedAt := time.Now()
		validator, err := freeform.Compile(ctx, definition.Freeform)
		pipeline.RecordCustomConstraintCompile(ctx, time.Since(startedAt))
		if err != nil {
			_ = registry.Close(context.WithoutCancel(ctx))
			return nil, err
		}
		registry.validators[definition.LogicalName] = validator
	}
	return registry, nil
}

func (registry *customConstraintRegistry) Has(logicalName string) bool {
	if registry == nil {
		return false
	}
	_, ok := registry.validators[logicalName]
	return ok
}

func (registry *customConstraintRegistry) Validate(ctx context.Context, call *llm.ToolInvocation) (bool, bool, error) {
	if registry == nil || call == nil || call.Kind != llm.ToolKindCustom {
		return false, false, nil
	}
	validator, constrained := registry.validators[call.LogicalName]
	if !constrained {
		return false, false, nil
	}
	startedAt := time.Now()
	valid, err := validator.Validate(ctx, call.InputText)
	pipeline.RecordCustomConstraintValidation(ctx, time.Since(startedAt), valid && err == nil)
	return true, valid, err
}

func (registry *customConstraintRegistry) Close(ctx context.Context) error {
	if registry == nil {
		return nil
	}
	errorsSeen := make([]error, 0)
	for name, validator := range registry.validators {
		if err := validator.Close(ctx); err != nil {
			errorsSeen = append(errorsSeen, err)
		}
		delete(registry.validators, name)
	}
	return errors.Join(errorsSeen...)
}

func customConstraintResult(call *llm.ToolInvocation) llm.Item {
	return llm.Item{
		Kind: llm.ItemKindToolResult, Status: llm.ItemStatusFailed,
		ToolResult: &llm.ToolResult{
			Kind: llm.ToolKindCustom, CallID: call.CallID, LogicalName: call.LogicalName,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: customConstraintFeedback}},
			IsError: true, Status: llm.ToolResultStatusFailed,
		},
	}
}

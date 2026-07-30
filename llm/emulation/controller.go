package emulation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/emulation/mcp"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
)

const (
	defaultMaxRounds                  = 8
	defaultMaxToolCalls               = 64
	defaultMaxParallel                = 8
	defaultMaxCustomConstraintRetries = 2
)

var (
	ErrLoopLimit     = errors.New("gateway tool loop limit exceeded")
	ErrApprovalState = errors.New("invalid gateway approval state")
)

type ControllerConfig struct {
	MCP                        mcp.RegistryConfig
	Hosted                     hosted.Config
	ContinuePauseTurns         bool
	DefaultRequireApproval     bool
	MaxRounds                  int
	MaxToolCalls               int
	MaxParallelCalls           int
	MaxWallTime                time.Duration
	EnableCustomConstraints    bool
	MaxCustomConstraintRetries int
}

// Controller owns provider-internal MCP rounds. It never recursively invokes
// an HTTP handler: every model attempt goes through CanonicalRoundTripper so
// candidate conversion, raw middleware, transport, and response restoration
// retain their normal attempt boundaries.
type Controller struct {
	config ControllerConfig
}

func NewController(config ControllerConfig) (*Controller, error) {
	if err := validateHostedConfig(config.Hosted); err != nil {
		return nil, fmt.Errorf("create gateway controller: %w", err)
	}
	mcpConfigured := controllerMCPConfigured(config)
	hostedConfigured := len(config.Hosted.SyntheticNameKey) >= 32
	if !mcpConfigured && !hostedConfigured && !config.ContinuePauseTurns && !config.EnableCustomConstraints {
		return nil, fmt.Errorf("create gateway controller: %w", mcp.ErrRegistryConfig)
	}
	if config.MaxRounds <= 0 {
		config.MaxRounds = defaultMaxRounds
	}
	if config.MaxToolCalls <= 0 {
		config.MaxToolCalls = defaultMaxToolCalls
	}
	if config.MaxParallelCalls <= 0 {
		config.MaxParallelCalls = defaultMaxParallel
	}
	if config.MaxCustomConstraintRetries <= 0 {
		config.MaxCustomConstraintRetries = defaultMaxCustomConstraintRetries
	}
	return &Controller{config: config}, nil
}

func (controller *Controller) Complete(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (*llm.Response, error) {
	if controller == nil || request == nil || rounds == nil {
		return nil, errors.New("gateway tool loop requires controller, request, and round tripper")
	}
	if !needsMCPEmulation(request, rounds.TargetFormat()) && !hasHostedDefinitionsForTarget(request, rounds.TargetFormat()) &&
		!controller.needsPauseTurnContinuation(request, rounds.TargetFormat()) &&
		!(controller.config.EnableCustomConstraints && NeedsCustomConstraintEmulation(request, rounds.TargetFormat())) {
		return rounds.Complete(ctx, request)
	}
	return controller.completeMCP(ctx, request, rounds)
}

func (controller *Controller) Stream(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (streams.Stream[*llm.Response], error) {
	if controller == nil || request == nil || rounds == nil {
		return nil, errors.New("gateway tool loop requires controller, request, and round tripper")
	}
	if !needsMCPEmulation(request, rounds.TargetFormat()) && !hasHostedDefinitionsForTarget(request, rounds.TargetFormat()) &&
		!controller.needsPauseTurnContinuation(request, rounds.TargetFormat()) &&
		!(controller.config.EnableCustomConstraints && NeedsCustomConstraintEmulation(request, rounds.TargetFormat())) {
		return rounds.Stream(ctx, request)
	}
	return controller.streamMCP(ctx, request, rounds)
}

func (controller *Controller) completeMCP(ctx context.Context, request *llm.Request, rounds pipeline.CanonicalRoundTripper) (*llm.Response, error) {
	loopCtx := ctx
	cancel := func() {}
	if controller.config.MaxWallTime > 0 {
		loopCtx, cancel = context.WithTimeout(ctx, controller.config.MaxWallTime)
	}
	defer cancel()
	gatewayRequest, err := projectGatewayAllowedTools(request, rounds.TargetFormat())
	if err != nil {
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	constraints, err := compileCustomConstraints(loopCtx, gatewayRequest, rounds.TargetFormat())
	if err != nil {
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	defer func() { _ = constraints.Close(context.WithoutCancel(loopCtx)) }()

	registry := mcp.NewEmptyRegistry(controller.config.MCP.SyntheticNameKey)
	if needsMCPEmulation(gatewayRequest, rounds.TargetFormat()) {
		registry, err = mcp.DiscoverRegistry(loopCtx, gatewayRequest, controller.config.MCP)
		if err != nil {
			pipeline.RecordEmulationFailure(loopCtx)
			return nil, err
		}
	}
	defer func() { _ = registry.Close(context.WithoutCancel(loopCtx)) }()
	hostedRegistry, err := hosted.DiscoverRegistry(gatewayRequest, controller.config.Hosted, func(definition llm.ToolDefinition) bool {
		return emulateHostedDefinition(gatewayRequest.APIFormat, rounds.TargetFormat(), definition)
	})
	if err != nil {
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	prepared := gatewayRequest.Clone()
	resumed := []llm.Item(nil)
	if needsMCPEmulation(gatewayRequest, rounds.TargetFormat()) {
		prepared, resumed, err = controller.lowerHistory(loopCtx, gatewayRequest, registry)
		if err != nil {
			pipeline.RecordEmulationFailure(loopCtx)
			return nil, err
		}
	}
	prepared, err = lowerHostedHistory(prepared, hostedRegistry)
	if err != nil {
		pipeline.RecordEmulationFailure(loopCtx)
		return nil, err
	}
	publicItems := uniqueCurrentListItems(request.Input, registry.ListItems())
	publicItems = append(publicItems, resumed...)
	var billedUsage *llm.Usage
	totalCalls := len(resumed)
	resumedCalls := len(resumed)

	constraintRetries := 0
	for round := 0; round < controller.config.MaxRounds; round++ {
		response, roundErr := rounds.Complete(loopCtx, prepared)
		if roundErr != nil {
			pipeline.RecordEmulationFailure(loopCtx)
			return nil, roundErr
		}
		billedUsage = addUsage(billedUsage, response.Usage)
		processed, processErr := controller.processRound(
			loopCtx, registry, hostedRegistry, constraints, response, controller.config.MaxToolCalls-totalCalls,
			constraintRetries < controller.config.MaxCustomConstraintRetries,
		)
		if processErr != nil {
			if errors.Is(processErr, ErrLoopLimit) {
				pipeline.RecordEmulationLimit(loopCtx)
			} else {
				pipeline.RecordEmulationFailure(loopCtx)
			}
			return nil, processErr
		}
		totalCalls += processed.gatewayCalls
		if processed.constraintRetry {
			constraintRetries++
			pipeline.RecordCustomConstraintRetry(loopCtx)
		}
		billedUsage = addUsage(billedUsage, usageForServerTools(processed.serverToolUsage))
		if controller.shouldContinuePauseTurn(prepared, rounds.TargetFormat(), response, processed) {
			pipeline.RecordEmulationRound(loopCtx, uint32(processed.gatewayCalls+resumedCalls), uint32(processed.approvals))
			resumedCalls = 0
			publicItems = append(publicItems, publicRoundItems(response.Output, processed.replacements, registry, hostedRegistry)...)
			appendProviderRound(prepared, response.Output, processed.results)
			refreshRegistryDefinitions(prepared, registry)
			refreshHostedDefinitions(prepared, hostedRegistry)
			continue
		}
		if processed.gatewayCalls == 0 {
			pipeline.RecordEmulationRound(loopCtx, uint32(resumedCalls), 0)
			return publicResponse(response, publicItems, nil, registry, hostedRegistry, billedUsage), nil
		}
		pipeline.RecordEmulationRound(loopCtx, uint32(processed.gatewayCalls+resumedCalls), uint32(processed.approvals))
		resumedCalls = 0

		if processed.boundary {
			return publicResponse(response, publicItems, processed.replacements, registry, hostedRegistry, billedUsage), nil
		}
		publicItems = append(publicItems, publicRoundItems(response.Output, processed.replacements, registry, hostedRegistry)...)

		appendProviderRound(prepared, response.Output, processed.results)
		refreshRegistryDefinitions(prepared, registry)
		refreshHostedDefinitions(prepared, hostedRegistry)
	}
	pipeline.RecordEmulationLimit(loopCtx)
	return nil, ErrLoopLimit
}

func (controller *Controller) needsPauseTurnContinuation(request *llm.Request, target llm.APIFormat) bool {
	return controller != nil && controller.config.ContinuePauseTurns && request != nil &&
		request.APIFormat != llm.APIFormatAnthropicMessage && target == llm.APIFormatAnthropicMessage
}

func (controller *Controller) shouldContinuePauseTurn(request *llm.Request, target llm.APIFormat, response *llm.Response, processed *processedRound) bool {
	if !controller.needsPauseTurnContinuation(request, target) || response == nil || response.TerminalReason != "pause_turn" ||
		processed == nil || processed.boundary || processed.gatewayCalls != 0 {
		return false
	}
	// pause_turn is a provider-owned continuation boundary. A client-owned
	// function call still belongs to the caller and must never be swallowed.
	for index := range response.Output {
		if response.Output[index].Kind == llm.ItemKindToolCall {
			return false
		}
	}
	return true
}

func appendProviderRound(request *llm.Request, output []llm.Item, results map[string]llm.Item) {
	if request == nil {
		return
	}
	for index := range output {
		item := output[index]
		request.Input = append(request.Input, llm.CloneCanonicalItem(item))
		if item.Kind == llm.ItemKindToolCall && item.ToolCall != nil {
			if result, ok := results[item.ToolCall.CallID]; ok {
				request.Input = append(request.Input, llm.CloneCanonicalItem(result))
			}
		}
	}
}

func projectGatewayAllowedTools(request *llm.Request, target llm.APIFormat) (*llm.Request, error) {
	if request == nil || target == llm.APIFormatOpenAIResponse || request.ToolChoice == nil || request.ToolChoice.AllowedTools == nil {
		return request, nil
	}
	return conversion.ProjectAllowedTools(request)
}

type gatewayCall struct {
	item    llm.Item
	binding mcp.Binding
}

type gatewayHostedCall struct {
	item    llm.Item
	binding hosted.Binding
}

type callExecution struct {
	call   gatewayCall
	result llm.Item
	public llm.Item
}

type processedRound struct {
	gatewayCalls    int
	approvals       int
	boundary        bool
	constraintRetry bool
	serverToolUsage llm.ServerToolUsage
	replacements    map[string]llm.Item
	results         map[string]llm.Item
}

func (controller *Controller) processRound(
	ctx context.Context,
	registry *mcp.Registry,
	hostedRegistry *hosted.Registry,
	constraints *customConstraintRegistry,
	response *llm.Response,
	remainingCalls int,
	allowConstraintRetry bool,
) (*processedRound, error) {
	gatewayCalls, hostedCalls, clientCalls := classifyCalls(response.Output, registry, hostedRegistry)
	violations, err := validateCustomConstraintCalls(ctx, constraints, response.Output)
	if err != nil {
		return nil, err
	}
	if len(gatewayCalls)+len(hostedCalls) > remainingCalls {
		return nil, ErrLoopLimit
	}
	processed := &processedRound{
		gatewayCalls: len(gatewayCalls) + len(hostedCalls),
		replacements: make(map[string]llm.Item, len(gatewayCalls)+len(hostedCalls)),
		results:      make(map[string]llm.Item, len(gatewayCalls)+len(hostedCalls)),
	}
	for index := range hostedCalls {
		addHostedRequestUsage(&processed.serverToolUsage, hostedCalls[index].binding.Definition.Kind)
	}
	if len(gatewayCalls)+len(hostedCalls) == 0 && len(violations) == 0 {
		return processed, nil
	}
	if response.Status != "" && response.Status != llm.ResponseStatusCompleted {
		for _, call := range gatewayCalls {
			processed.replacements[call.item.ToolCall.CallID] = interruptedMCPCall(registry, call, response.Status)
		}
		for _, call := range hostedCalls {
			processed.replacements[call.item.ToolCall.CallID] = interruptedHostedCall(call, response.Status)
		}
		recordCustomConstraintFallbacks(ctx, violations)
		processed.boundary = true
		return processed, nil
	}

	approvals := make(map[string]llm.Item)
	executable := make([]gatewayCall, 0, len(gatewayCalls))
	for _, call := range gatewayCalls {
		if call.binding.ApprovalRequired(controller.config.DefaultRequireApproval) {
			arguments, argumentsErr := executableArguments(call.item.ToolCall)
			if argumentsErr != nil {
				executable = append(executable, call)
				continue
			}
			approvalID := registry.ApprovalID(call.binding, call.item.ToolCall.CallID, arguments)
			approvals[call.item.ToolCall.CallID] = approvalItem(approvalID, call)
			continue
		}
		executable = append(executable, call)
	}
	processed.approvals = len(approvals)
	executions := controller.executeCalls(ctx, registry, executable)
	for _, execution := range executions {
		callID := execution.call.item.ToolCall.CallID
		processed.replacements[callID] = execution.public
		processed.results[callID] = execution.result
	}
	hostedExecutions := controller.executeHostedCalls(ctx, hostedRegistry, hostedCalls)
	for _, execution := range hostedExecutions {
		callID := execution.call.item.ToolCall.CallID
		processed.replacements[callID] = execution.public
		processed.results[callID] = execution.result
	}
	for callID, approval := range approvals {
		processed.replacements[callID] = approval
	}
	canRetryConstraints := len(violations) > 0 && allowConstraintRetry && len(approvals) == 0 &&
		clientCalls == len(violations) && processed.gatewayCalls+len(violations) <= remainingCalls
	if canRetryConstraints {
		for index := range violations {
			call := violations[index]
			processed.results[call.CallID] = customConstraintResult(call)
			// A zero replacement suppresses only this invalid internal attempt.
			// The corrected call from the next model round remains public.
			processed.replacements[call.CallID] = llm.Item{}
		}
		processed.gatewayCalls += len(violations)
		processed.constraintRetry = true
	} else {
		recordCustomConstraintFallbacks(ctx, violations)
	}
	processed.boundary = len(approvals) > 0 || clientCalls > len(violations) || len(violations) > 0 && !canRetryConstraints
	return processed, nil
}

func validateCustomConstraintCalls(
	ctx context.Context, constraints *customConstraintRegistry, items []llm.Item,
) ([]*llm.ToolInvocation, error) {
	violations := make([]*llm.ToolInvocation, 0)
	for index := range items {
		item := &items[index]
		if item.Kind != llm.ItemKindToolCall || item.ToolCall == nil {
			continue
		}
		constrained, valid, err := constraints.Validate(ctx, item.ToolCall)
		if err != nil {
			return nil, err
		}
		if constrained && !valid {
			violations = append(violations, item.ToolCall)
		}
	}
	return violations, nil
}

func recordCustomConstraintFallbacks(ctx context.Context, violations []*llm.ToolInvocation) {
	for range violations {
		pipeline.RecordCustomConstraintFallback(ctx)
	}
}

func addHostedRequestUsage(usage *llm.ServerToolUsage, kind llm.ToolKind) {
	if usage == nil {
		return
	}
	switch kind {
	case llm.ToolKindWebSearch:
		usage.WebSearchRequests++
	case llm.ToolKindWebFetch:
		usage.WebFetchRequests++
	case llm.ToolKindCodeInterpreter, llm.ToolKindCodeExecution:
		usage.CodeExecutionRequests++
	case llm.ToolKindToolSearch:
		usage.ToolSearchRequests++
	}
}

func usageForServerTools(usage llm.ServerToolUsage) *llm.Usage {
	if usage == (llm.ServerToolUsage{}) {
		return nil
	}
	return &llm.Usage{ServerToolUsage: &usage}
}

func interruptedHostedCall(call gatewayHostedCall, status llm.ResponseStatus) llm.Item {
	itemStatus := llm.ItemStatusIncomplete
	callStatus := llm.ToolCallStatusIncomplete
	if status == llm.ResponseStatusFailed || status == llm.ResponseStatusCancelled {
		itemStatus = llm.ItemStatusFailed
		callStatus = llm.ToolCallStatusFailed
	}
	invocation := *call.item.ToolCall
	invocation.Kind = call.binding.Definition.Kind
	invocation.LogicalName = call.binding.Definition.LogicalName
	invocation.Execution = llm.ExecutionOwnerProvider
	invocation.Status = callStatus
	return llm.Item{
		Kind: llm.ItemKindHostedCall, ID: invocation.ID, Role: llm.RoleAssistant, Status: itemStatus,
		HostedCall: &llm.HostedToolCall{Invocation: invocation},
	}
}

func classifyCalls(items []llm.Item, registry *mcp.Registry, hostedRegistry *hosted.Registry) ([]gatewayCall, []gatewayHostedCall, int) {
	gateway := make([]gatewayCall, 0)
	hostedCalls := make([]gatewayHostedCall, 0)
	client := 0
	for index := range items {
		item := items[index]
		if item.Kind != llm.ItemKindToolCall || item.ToolCall == nil {
			continue
		}
		binding, ok := registry.Binding(item.ToolCall.LogicalName)
		if !ok {
			if hostedBinding, hostedOK := hostedRegistry.Binding(item.ToolCall.LogicalName); hostedOK {
				hostedCalls = append(hostedCalls, gatewayHostedCall{item: llm.CloneCanonicalItem(item), binding: hostedBinding})
				continue
			}
			client++
			continue
		}
		gateway = append(gateway, gatewayCall{item: llm.CloneCanonicalItem(item), binding: binding})
	}
	return gateway, hostedCalls, client
}

type hostedCallExecution struct {
	call   gatewayHostedCall
	result llm.Item
	public llm.Item
}

func (controller *Controller) executeHostedCalls(ctx context.Context, registry *hosted.Registry, calls []gatewayHostedCall) []hostedCallExecution {
	results := make([]hostedCallExecution, len(calls))
	semaphore := make(chan struct{}, controller.config.MaxParallelCalls)
	var wait sync.WaitGroup
	for index := range calls {
		index := index
		wait.Add(1)
		go func() {
			defer wait.Done()
			defer func() {
				if recover() != nil {
					pipeline.RecordEmulationFailure(ctx)
					slog.ErrorContext(ctx, "hosted executor panicked")
					results[index] = failedHostedExecution(calls[index], errors.New("hosted executor panicked"))
				}
			}()
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				results[index] = failedHostedExecution(calls[index], ctx.Err())
				return
			}
			results[index] = executeHostedCall(ctx, registry, calls[index])
		}()
	}
	wait.Wait()
	return results
}

func executeHostedCall(ctx context.Context, registry *hosted.Registry, call gatewayHostedCall) hostedCallExecution {
	invocation := *call.item.ToolCall
	invocation.Kind = call.binding.Definition.Kind
	invocation.LogicalName = call.binding.Definition.LogicalName
	invocation.Execution = llm.ExecutionOwnerProvider
	pipeline.RecordHostedExecution(ctx, invocation.Kind)
	var result *llm.ToolResult
	var err error
	if call.binding.Definition.Kind == llm.ToolKindToolSearch && call.binding.Executor == nil {
		if registry == nil {
			err = errors.New("built-in tool search registry is missing")
		} else {
			result, err = registry.Search(call.binding, invocation)
		}
	} else if executor, ok := call.binding.Executor.(hosted.LifecycleExecutor); ok {
		var execution *hosted.LifecycleExecution
		execution, err = executor.ExecuteLifecycle(ctx, call.binding.Definition, invocation)
		if err == nil {
			if execution == nil {
				err = errors.New("hosted lifecycle executor returned no execution")
			} else {
				invocation.PendingSafetyChecks = append([]llm.ToolSafetyCheck(nil), execution.PendingSafetyChecks...)
				result = execution.Result
			}
		}
	} else if call.binding.Executor != nil {
		result, err = call.binding.Executor.Execute(ctx, call.binding.Definition, invocation)
	} else {
		err = errors.New("hosted executor is missing")
	}
	if err != nil {
		pipeline.RecordEmulationFailure(ctx)
		return failedHostedExecution(call, err)
	}
	if result == nil {
		return failedHostedExecution(call, errors.New("hosted executor returned no result"))
	}
	result.Kind = call.binding.Definition.Kind
	result.CallID = invocation.CallID
	result.LogicalName = call.binding.Definition.LogicalName
	if result.Status == "" {
		result.Status = llm.ToolResultStatusCompleted
	}
	publicStatus := llm.ItemStatusCompleted
	if result.IsError || result.Status == llm.ToolResultStatusFailed {
		publicStatus = llm.ItemStatusFailed
	}
	publicResult := llm.CloneCanonicalItem(llm.Item{Kind: llm.ItemKindToolResult, ToolResult: result}).ToolResult
	return hostedCallExecution{
		call: call,
		result: llm.Item{
			Kind: llm.ItemKindToolResult, Status: publicStatus,
			ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindFunction, CallID: invocation.CallID, LogicalName: call.binding.SyntheticName,
				Content: publicResult.Content, IsError: result.IsError, Status: result.Status,
			},
		},
		public: llm.Item{
			Kind: llm.ItemKindHostedCall, ID: invocation.ID, Role: llm.RoleAssistant, Status: publicStatus,
			HostedCall: &llm.HostedToolCall{Invocation: invocation, Result: publicResult},
		},
	}
}

func failedHostedExecution(call gatewayHostedCall, err error) hostedCallExecution {
	message := "hosted tool execution failed"
	if err != nil {
		message = err.Error()
	}
	invocation := *call.item.ToolCall
	invocation.Kind = call.binding.Definition.Kind
	invocation.LogicalName = call.binding.Definition.LogicalName
	invocation.Execution = llm.ExecutionOwnerProvider
	result := &llm.ToolResult{
		Kind: invocation.Kind, CallID: invocation.CallID, LogicalName: invocation.LogicalName,
		Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: message}},
		IsError: true, Status: llm.ToolResultStatusFailed,
	}
	return hostedCallExecution{
		call: call,
		result: llm.Item{
			Kind: llm.ItemKindToolResult, Status: llm.ItemStatusFailed,
			ToolResult: &llm.ToolResult{
				Kind: llm.ToolKindFunction, CallID: invocation.CallID, LogicalName: call.binding.SyntheticName,
				Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: message}}, IsError: true, Status: llm.ToolResultStatusFailed,
			},
		},
		public: llm.Item{
			Kind: llm.ItemKindHostedCall, ID: invocation.ID, Role: llm.RoleAssistant, Status: llm.ItemStatusFailed,
			HostedCall: &llm.HostedToolCall{Invocation: invocation, Result: result},
		},
	}
}

func (controller *Controller) executeCalls(ctx context.Context, registry *mcp.Registry, calls []gatewayCall) []callExecution {
	results := make([]callExecution, len(calls))
	semaphore := make(chan struct{}, controller.config.MaxParallelCalls)
	var wait sync.WaitGroup
	for index := range calls {
		index := index
		wait.Add(1)
		go func() {
			defer wait.Done()
			defer func() {
				if recover() != nil {
					pipeline.RecordEmulationFailure(ctx)
					slog.ErrorContext(ctx, "MCP executor panicked")
					results[index] = failedExecution(registry, calls[index], errors.New("MCP executor panicked"))
				}
			}()
			select {
			case semaphore <- struct{}{}:
				defer func() { <-semaphore }()
			case <-ctx.Done():
				pipeline.RecordEmulationFailure(ctx)
				results[index] = failedExecution(registry, calls[index], ctx.Err())
				return
			}
			results[index] = executeCall(ctx, registry, calls[index])
		}()
	}
	wait.Wait()
	return results
}

func executeCall(ctx context.Context, registry *mcp.Registry, call gatewayCall) callExecution {
	arguments, err := executableArguments(call.item.ToolCall)
	if err != nil {
		pipeline.RecordEmulationFailure(ctx)
		return failedExecution(registry, call, err)
	}
	if call.binding.IsDiscovery() {
		result, tools, searchErr := registry.Search(call.binding, arguments)
		if searchErr != nil {
			pipeline.RecordEmulationFailure(ctx)
			return failedExecution(registry, call, searchErr)
		}
		encoded, encodeErr := json.Marshal(result)
		if encodeErr != nil {
			pipeline.RecordEmulationFailure(ctx)
			return failedExecution(registry, call, errors.New("encode MCP tool search result"))
		}
		return callExecution{
			call: call, result: toolResultItem(call, string(encoded), false),
			public: registry.DiscoveryListItem(call.binding, call.item.ToolCall.CallID, tools),
		}
	}
	result, err := registry.Call(ctx, call.binding, arguments)
	if err != nil {
		pipeline.RecordEmulationFailure(ctx)
		return failedExecution(registry, call, err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		pipeline.RecordEmulationFailure(ctx)
		return failedExecution(registry, call, errors.New("encode MCP result"))
	}
	failed := result.IsError
	status := llm.ItemStatusCompleted
	mcpStatus := llm.MCPCallStatusCompleted
	if failed {
		pipeline.RecordEmulationFailure(ctx)
		status = llm.ItemStatusFailed
		mcpStatus = llm.MCPCallStatusFailed
	}
	return callExecution{
		call:   call,
		result: toolResultItem(call, string(encoded), failed),
		public: llm.Item{
			Kind: llm.ItemKindMCPCall, ID: registry.MCPCallItemID(call.item.ToolCall.CallID), Status: status,
			MCPCall: &llm.MCPCall{
				ServerLabel: call.binding.ServerLabel, LogicalName: call.binding.Tool.Name,
				ArgumentsJSON: arguments, Output: string(encoded), Status: mcpStatus,
			},
		},
	}
}

func failedExecution(registry *mcp.Registry, call gatewayCall, err error) callExecution {
	message := "MCP tool execution failed"
	if err != nil {
		message = err.Error()
	}
	execution := callExecution{
		call:   call,
		result: toolResultItem(call, message, true),
		public: llm.Item{
			Kind: llm.ItemKindMCPCall, ID: registry.MCPCallItemID(call.item.ToolCall.CallID), Status: llm.ItemStatusFailed,
			MCPCall: &llm.MCPCall{
				ServerLabel: call.binding.ServerLabel, LogicalName: call.binding.Tool.Name,
				Error: message, Status: llm.MCPCallStatusFailed,
			},
		},
	}
	if call.binding.IsDiscovery() {
		execution.public = registry.DiscoveryListFailure(call.binding, call.item.ToolCall.CallID, message)
		return execution
	}
	copyInvocationArguments(execution.public.MCPCall, call.item.ToolCall)
	return execution
}

func interruptedMCPCall(registry *mcp.Registry, call gatewayCall, status llm.ResponseStatus) llm.Item {
	itemStatus := llm.ItemStatusIncomplete
	mcpStatus := llm.MCPCallStatusIncomplete
	errorMessage := ""
	if status == llm.ResponseStatusFailed || status == llm.ResponseStatusCancelled {
		itemStatus = llm.ItemStatusFailed
		mcpStatus = llm.MCPCallStatusFailed
		errorMessage = "MCP tool call was not executed because the provider round did not complete"
	}
	item := llm.Item{
		Kind: llm.ItemKindMCPCall, ID: registry.MCPCallItemID(call.item.ToolCall.CallID), Status: itemStatus,
		MCPCall: &llm.MCPCall{
			ServerLabel: call.binding.ServerLabel, LogicalName: call.binding.Tool.Name,
			Error: errorMessage, Status: mcpStatus,
		},
	}
	copyInvocationArguments(item.MCPCall, call.item.ToolCall)
	return item
}

func toolResultItem(call gatewayCall, output string, failed bool) llm.Item {
	itemStatus := llm.ItemStatusCompleted
	resultStatus := llm.ToolResultStatusCompleted
	if failed {
		itemStatus = llm.ItemStatusFailed
		resultStatus = llm.ToolResultStatusFailed
	}
	return llm.Item{
		Kind: llm.ItemKindToolResult, Status: itemStatus,
		ToolResult: &llm.ToolResult{
			Kind: llm.ToolKindFunction, CallID: call.item.ToolCall.CallID,
			LogicalName: call.item.ToolCall.LogicalName, IsError: failed, Status: resultStatus,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: output}},
		},
	}
}

func executableArguments(call *llm.ToolInvocation) (json.RawMessage, error) {
	if call == nil {
		return nil, errors.New("MCP tool invocation payload is missing")
	}
	if len(call.ArgumentsJSON) > 0 {
		if !json.Valid(call.ArgumentsJSON) {
			return nil, errors.New("MCP tool arguments are invalid JSON")
		}
		return append(json.RawMessage(nil), call.ArgumentsJSON...), nil
	}
	if call.ArgumentsText == "" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid([]byte(call.ArgumentsText)) {
		return nil, errors.New("MCP tool arguments are invalid JSON")
	}
	return append(json.RawMessage(nil), call.ArgumentsText...), nil
}

func copyInvocationArguments(target *llm.MCPCall, source *llm.ToolInvocation) {
	if target == nil || source == nil {
		return
	}
	target.ArgumentsJSON = append(json.RawMessage(nil), source.ArgumentsJSON...)
	target.ArgumentsText = source.ArgumentsText
}

func approvalItem(approvalID string, call gatewayCall) llm.Item {
	item := llm.Item{
		Kind: llm.ItemKindMCPApprovalRequest, ID: approvalID, Status: llm.ItemStatusCompleted,
		MCPApprovalRequest: &llm.MCPApprovalRequest{
			ServerLabel: call.binding.ServerLabel, LogicalName: call.binding.Tool.Name,
		},
	}
	if call.item.ToolCall != nil {
		item.MCPApprovalRequest.ArgumentsJSON = append(json.RawMessage(nil), call.item.ToolCall.ArgumentsJSON...)
		item.MCPApprovalRequest.ArgumentsText = call.item.ToolCall.ArgumentsText
	}
	return item
}

func publicResponse(source *llm.Response, prefix []llm.Item, replacements map[string]llm.Item, registry *mcp.Registry, hostedRegistry *hosted.Registry, usage *llm.Usage) *llm.Response {
	response := *source
	response.Output = llm.CloneCanonicalItems(prefix)
	response.Output = append(response.Output, publicRoundItems(source.Output, replacements, registry, hostedRegistry)...)
	response.Usage = usage
	if response.Status == "" {
		response.Status = llm.ResponseStatusCompleted
	}
	return &response
}

func publicRoundItems(source []llm.Item, replacements map[string]llm.Item, registry *mcp.Registry, hostedRegistry *hosted.Registry) []llm.Item {
	result := make([]llm.Item, 0, len(source))
	for index := range source {
		item := source[index]
		if item.Kind == llm.ItemKindToolCall && item.ToolCall != nil {
			if replacement, ok := replacements[item.ToolCall.CallID]; ok {
				if replacement.Kind != "" {
					result = append(result, llm.CloneCanonicalItem(replacement))
				}
				continue
			}
			_, mcpGateway := registry.Binding(item.ToolCall.LogicalName)
			_, hostedGateway := hostedRegistry.Binding(item.ToolCall.LogicalName)
			if mcpGateway || hostedGateway {
				if replacement, ok := replacements[item.ToolCall.CallID]; ok {
					result = append(result, llm.CloneCanonicalItem(replacement))
				}
				continue
			}
		}
		result = append(result, llm.CloneCanonicalItem(item))
	}
	return result
}

func refreshRegistryDefinitions(request *llm.Request, registry *mcp.Registry) {
	if request == nil || registry == nil {
		return
	}
	definitions := request.ToolDefinitions[:0]
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if _, owned := registry.Binding(definition.LogicalName); owned {
			continue
		}
		definitions = append(definitions, definition)
	}
	request.ToolDefinitions = append(definitions, registry.FunctionDefinitions()...)
}

func hasMCPDefinitions(request *llm.Request) bool {
	if request == nil {
		return false
	}
	for index := range request.ToolDefinitions {
		if request.ToolDefinitions[index].Kind == llm.ToolKindMCP {
			return true
		}
	}
	return false
}

func needsMCPEmulation(request *llm.Request, target llm.APIFormat) bool {
	return target != llm.APIFormatOpenAIResponse && hasMCPDefinitions(request)
}

func hasHostedDefinitionsForTarget(request *llm.Request, target llm.APIFormat) bool {
	if request == nil {
		return false
	}
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if emulateHostedDefinition(request.APIFormat, target, definition) {
			return true
		}
	}
	return false
}

func emulateHostedDefinition(source, target llm.APIFormat, definition llm.ToolDefinition) bool {
	if definition.Execution != llm.ExecutionOwnerProvider || definition.Hosted == nil || source == target {
		return false
	}
	return !conversion.HostedToolNativeEquivalent(definition, target)
}

func uniqueCurrentListItems(input, lists []llm.Item) []llm.Item {
	seen := make(map[string]struct{})
	for index := range input {
		if input[index].MCPListTools != nil {
			seen[input[index].MCPListTools.ServerLabel] = struct{}{}
		}
	}
	result := make([]llm.Item, 0, len(lists))
	for index := range lists {
		label := lists[index].MCPListTools.ServerLabel
		if _, exists := seen[label]; !exists {
			result = append(result, llm.CloneCanonicalItem(lists[index]))
		}
	}
	return result
}

func addUsage(total, current *llm.Usage) *llm.Usage {
	if current == nil {
		return total
	}
	if total == nil {
		return cloneUsage(current)
	}
	total.PromptTokens += current.PromptTokens
	total.CompletionTokens += current.CompletionTokens
	total.TotalTokens += current.TotalTokens
	total.Recovered = total.Recovered || current.Recovered
	addServerToolUsage(total, current)
	addPromptTokenDetails(total, current)
	addCompletionTokenDetails(total, current)
	total.PromptModalityTokenDetails = addModalityTokenDetails(total.PromptModalityTokenDetails, current.PromptModalityTokenDetails)
	total.CompletionModalityTokenDetails = addModalityTokenDetails(total.CompletionModalityTokenDetails, current.CompletionModalityTokenDetails)
	return total
}

func cloneUsage(source *llm.Usage) *llm.Usage {
	if source == nil {
		return nil
	}
	clone := *source
	if source.PromptTokensDetails != nil {
		details := *source.PromptTokensDetails
		clone.PromptTokensDetails = &details
	}
	if source.CompletionTokensDetails != nil {
		details := *source.CompletionTokensDetails
		clone.CompletionTokensDetails = &details
	}
	if source.ServerToolUsage != nil {
		details := *source.ServerToolUsage
		clone.ServerToolUsage = &details
	}
	clone.PromptModalityTokenDetails = append([]llm.ModalityTokenCount(nil), source.PromptModalityTokenDetails...)
	clone.CompletionModalityTokenDetails = append([]llm.ModalityTokenCount(nil), source.CompletionModalityTokenDetails...)
	return &clone
}

func addServerToolUsage(total, current *llm.Usage) {
	if current.ServerToolUsage == nil {
		return
	}
	if total.ServerToolUsage == nil {
		total.ServerToolUsage = &llm.ServerToolUsage{}
	}
	total.ServerToolUsage.WebSearchRequests += current.ServerToolUsage.WebSearchRequests
	total.ServerToolUsage.WebFetchRequests += current.ServerToolUsage.WebFetchRequests
	total.ServerToolUsage.CodeExecutionRequests += current.ServerToolUsage.CodeExecutionRequests
	total.ServerToolUsage.ToolSearchRequests += current.ServerToolUsage.ToolSearchRequests
}

func addPromptTokenDetails(total, current *llm.Usage) {
	if current.PromptTokensDetails == nil {
		return
	}
	if total.PromptTokensDetails == nil {
		total.PromptTokensDetails = &llm.PromptTokensDetails{}
	}
	total.PromptTokensDetails.AudioTokens += current.PromptTokensDetails.AudioTokens
	total.PromptTokensDetails.CachedTokens += current.PromptTokensDetails.CachedTokens
	total.PromptTokensDetails.WriteCachedTokens += current.PromptTokensDetails.WriteCachedTokens
	total.PromptTokensDetails.WriteCached5MinTokens += current.PromptTokensDetails.WriteCached5MinTokens
	total.PromptTokensDetails.WriteCached1HourTokens += current.PromptTokensDetails.WriteCached1HourTokens
	total.PromptTokensDetails.ImageTokens += current.PromptTokensDetails.ImageTokens
	total.PromptTokensDetails.TextTokens += current.PromptTokensDetails.TextTokens
}

func addCompletionTokenDetails(total, current *llm.Usage) {
	if current.CompletionTokensDetails == nil {
		return
	}
	if total.CompletionTokensDetails == nil {
		total.CompletionTokensDetails = &llm.CompletionTokensDetails{}
	}
	total.CompletionTokensDetails.AudioTokens += current.CompletionTokensDetails.AudioTokens
	total.CompletionTokensDetails.ReasoningTokens += current.CompletionTokensDetails.ReasoningTokens
	total.CompletionTokensDetails.AcceptedPredictionTokens += current.CompletionTokensDetails.AcceptedPredictionTokens
	total.CompletionTokensDetails.RejectedPredictionTokens += current.CompletionTokensDetails.RejectedPredictionTokens
}

func addModalityTokenDetails(total, current []llm.ModalityTokenCount) []llm.ModalityTokenCount {
	indices := make(map[string]int, len(total))
	for index := range total {
		indices[total[index].Modality] = index
	}
	for index := range current {
		position, exists := indices[current[index].Modality]
		if exists {
			total[position].TokenCount += current[index].TokenCount
			continue
		}
		indices[current[index].Modality] = len(total)
		total = append(total, current[index])
	}
	return total
}

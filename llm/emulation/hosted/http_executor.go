package hosted

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/looplj/axonhub/llm"
)

const defaultHTTPExecutorResponseLimit int64 = 8 << 20

// HTTPExecutorConfig binds one canonical hosted capability to a
// protocol-neutral HTTP service. The service sees canonical arguments and the
// native definition configuration, never an OpenAI/Anthropic wire request.
type HTTPExecutorConfig struct {
	Kind             llm.ToolKind
	Endpoint         string
	Client           *http.Client
	Headers          http.Header
	EndpointPolicy   EndpointPolicy
	MaxResponseBytes int64
	Parameters       json.RawMessage
}

type HTTPExecutor struct {
	kind       llm.ToolKind
	contract   httpExecutorContract
	endpoint   *url.URL
	client     *http.Client
	headers    http.Header
	limit      int64
	parameters json.RawMessage
}

type HTTPExecutorRequest struct {
	Kind                llm.ToolKind          `json:"kind"`
	LogicalName         string                `json:"logical_name"`
	Arguments           json.RawMessage       `json:"arguments"`
	Configuration       json.RawMessage       `json:"configuration,omitempty"`
	PendingSafetyChecks []llm.ToolSafetyCheck `json:"pending_safety_checks,omitempty"`
}

type HTTPExecutorResponse struct {
	Result                   json.RawMessage       `json:"result,omitempty"`
	Error                    string                `json:"error,omitempty"`
	IsError                  bool                  `json:"is_error,omitempty"`
	Status                   llm.ToolResultStatus  `json:"status,omitempty"`
	PendingSafetyChecks      []llm.ToolSafetyCheck `json:"pending_safety_checks,omitempty"`
	AcknowledgedSafetyChecks []llm.ToolSafetyCheck `json:"acknowledged_safety_checks,omitempty"`
}

func NewHTTPExecutor(config HTTPExecutorConfig) (*HTTPExecutor, error) {
	if config.Kind == "" {
		return nil, errors.New("hosted HTTP executor requires a tool kind")
	}
	contract, err := newHTTPExecutorContract(config.Kind)
	if err != nil {
		return nil, err
	}
	endpoint, client, err := policyHTTPClient("hosted HTTP executor", config.Endpoint, config.Client, config.EndpointPolicy)
	if err != nil {
		return nil, err
	}
	parameters := append(json.RawMessage(nil), config.Parameters...)
	if len(parameters) == 0 {
		parameters = contract.Parameters()
	}
	if !json.Valid(parameters) {
		return nil, errors.New("hosted HTTP executor parameters are invalid JSON")
	}
	limit := config.MaxResponseBytes
	if limit <= 0 {
		limit = defaultHTTPExecutorResponseLimit
	}
	headers := config.Headers.Clone()
	if headers == nil {
		headers = make(http.Header)
	}
	return &HTTPExecutor{
		kind: config.Kind, contract: contract, endpoint: endpoint, client: client, headers: headers,
		limit: limit, parameters: parameters,
	}, nil
}

func (executor *HTTPExecutor) Kind() llm.ToolKind {
	if executor == nil {
		return ""
	}
	return executor.kind
}

func (executor *HTTPExecutor) Function(definition llm.ToolDefinition) (llm.FunctionDefinition, error) {
	if executor == nil || definition.Kind != executor.kind || definition.Execution != llm.ExecutionOwnerProvider || definition.Hosted == nil {
		return llm.FunctionDefinition{}, errors.New("hosted HTTP executor received an incompatible definition")
	}
	return llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), executor.parameters...)}, nil
}

func (executor *HTTPExecutor) Execute(ctx context.Context, definition llm.ToolDefinition, invocation llm.ToolInvocation) (*llm.ToolResult, error) {
	execution, err := executor.ExecuteLifecycle(ctx, definition, invocation)
	if err != nil || execution == nil {
		return nil, err
	}
	return execution.Result, nil
}

func (executor *HTTPExecutor) ExecuteLifecycle(ctx context.Context, definition llm.ToolDefinition, invocation llm.ToolInvocation) (*LifecycleExecution, error) {
	if executor == nil || executor.endpoint == nil {
		return nil, errors.New("hosted HTTP executor is not initialized")
	}
	if definition.Kind != executor.kind || invocation.Kind != executor.kind || definition.Hosted == nil {
		return nil, errors.New("hosted HTTP executor received another tool kind")
	}
	if err := executor.contract.ValidateConfiguration(definition.Hosted.Configuration); err != nil {
		return nil, fmt.Errorf("hosted %s configuration: %w", executor.kind, err)
	}
	arguments := append(json.RawMessage(nil), invocation.ArgumentsJSON...)
	if len(arguments) == 0 && invocation.ArgumentsText != "" {
		arguments = json.RawMessage(invocation.ArgumentsText)
	}
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	if !json.Valid(arguments) {
		return nil, errors.New("hosted HTTP executor invocation requires JSON arguments")
	}
	if err := executor.contract.ValidateArguments(arguments); err != nil {
		return nil, fmt.Errorf("hosted %s arguments: %w", executor.kind, err)
	}
	payload := HTTPExecutorRequest{
		Kind: executor.kind, LogicalName: definition.LogicalName, Arguments: arguments,
		Configuration:       append(json.RawMessage(nil), definition.Hosted.Configuration...),
		PendingSafetyChecks: append([]llm.ToolSafetyCheck(nil), invocation.PendingSafetyChecks...),
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode hosted %s request: %w", executor.kind, err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, executor.endpoint.String(), bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create hosted %s request: %w", executor.kind, err)
	}
	request.Header = executor.headers.Clone()
	request.Header.Set("Content-Type", "application/json")
	response, err := executor.client.Do(request)
	if err != nil {
		return nil, fmt.Errorf("execute hosted %s request: %w", executor.kind, err)
	}
	defer response.Body.Close()
	responseBody, err := io.ReadAll(io.LimitReader(response.Body, executor.limit+1))
	if err != nil {
		return nil, fmt.Errorf("read hosted %s response: %w", executor.kind, err)
	}
	if int64(len(responseBody)) > executor.limit {
		return nil, fmt.Errorf("hosted %s response exceeds limit", executor.kind)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("hosted %s service returned HTTP %d", executor.kind, response.StatusCode)
	}
	var output HTTPExecutorResponse
	if err := json.Unmarshal(responseBody, &output); err != nil {
		return nil, fmt.Errorf("decode hosted %s response: %w", executor.kind, err)
	}
	if len(output.Result) > 0 && !json.Valid(output.Result) {
		return nil, fmt.Errorf("hosted %s service returned invalid result JSON", executor.kind)
	}
	if output.Status != "" && output.Status != llm.ToolResultStatusCompleted && output.Status != llm.ToolResultStatusFailed {
		return nil, fmt.Errorf("hosted %s service returned invalid status %q", executor.kind, output.Status)
	}
	if !executor.contract.AllowsSafetyMetadata() &&
		(len(output.PendingSafetyChecks) > 0 || len(output.AcknowledgedSafetyChecks) > 0) {
		return nil, fmt.Errorf("hosted %s service returned computer safety metadata", executor.kind)
	}
	status := output.Status
	if status == "" {
		status = llm.ToolResultStatusCompleted
	}
	result := &llm.ToolResult{
		Kind: executor.kind, CallID: invocation.CallID, LogicalName: definition.LogicalName,
		Execution: llm.ExecutionOwnerGateway, Status: status,
		IsError:                  output.IsError || output.Error != "" || status == llm.ToolResultStatusFailed,
		AcknowledgedSafetyChecks: append([]llm.ToolSafetyCheck(nil), output.AcknowledgedSafetyChecks...),
	}
	if result.IsError {
		result.Status = llm.ToolResultStatusFailed
	}
	if output.Error != "" {
		result.Content = append(result.Content, llm.ContentBlock{Kind: llm.ContentKindText, Text: output.Error})
	}
	if len(output.Result) > 0 && string(output.Result) != "null" {
		result.StructuredContent = append(json.RawMessage(nil), output.Result...)
		content, contentErr := executor.contract.ResultContent(output.Result)
		if contentErr != nil {
			return nil, fmt.Errorf("hosted %s result: %w", executor.kind, contentErr)
		}
		result.Content = append(result.Content, content...)
	}
	return &LifecycleExecution{
		Result: result, PendingSafetyChecks: append([]llm.ToolSafetyCheck(nil), output.PendingSafetyChecks...),
	}, nil
}

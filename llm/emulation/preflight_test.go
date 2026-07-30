package emulation

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/emulation/hosted"
	"github.com/looplj/axonhub/llm/emulation/mcp"
)

func TestControllerPreflightMergesOnlyConfiguredEmulators(t *testing.T) {
	t.Parallel()
	webRequest := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindWebSearch, LogicalName: "web_search", Execution: llm.ExecutionOwnerProvider,
			Hosted: &llm.HostedToolDefinition{Type: "web_search", WebSearch: &llm.WebSearch{}},
		}},
	}
	webController, err := NewController(ControllerConfig{Hosted: hosted.Config{
		SyntheticNameKey: []byte(strings.Repeat("preflight-hosted-key-", 2)),
		Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: preflightHostedExecutor{}},
	}})
	if err != nil {
		t.Fatalf("new hosted controller: %v", err)
	}
	webPlan, err := webController.Preflight(webRequest, llm.APIFormatOpenAIChatCompletion)
	if err != nil || !webPlan.Complete() || webPlan.Summary.Emulated != 1 || webPlan.Actions[0].Strategy != conversion.StrategyHostedGateway {
		t.Fatalf("hosted preflight = %#v, err=%v", webPlan, err)
	}

	mcpOnlyController, err := NewController(ControllerConfig{MCP: mcp.RegistryConfig{
		SyntheticNameKey: []byte(strings.Repeat("preflight-mcp-key-", 2)),
		EndpointPolicy:   func(*url.URL) error { return nil },
	}})
	if err != nil {
		t.Fatalf("new MCP controller: %v", err)
	}
	missingPlan, missingErr := mcpOnlyController.Preflight(webRequest, llm.APIFormatOpenAIChatCompletion)
	if !errors.Is(missingErr, conversion.ErrIncompletePlan) || missingPlan == nil || missingPlan.Summary.Unknown != 1 {
		t.Fatalf("missing hosted executor preflight = %#v, err=%v", missingPlan, missingErr)
	}

	mcpRequest := &llm.Request{
		APIFormat: llm.APIFormatOpenAIResponse,
		ToolDefinitions: []llm.ToolDefinition{{
			Kind: llm.ToolKindMCP, LogicalName: "inventory", Execution: llm.ExecutionOwnerProvider,
			MCP: &llm.MCPDefinition{ServerLabel: "inventory", ServerURL: "https://mcp.example.test/rpc"},
		}},
	}
	mcpPlan, err := mcpOnlyController.Preflight(mcpRequest, llm.APIFormatAnthropicMessage)
	if err != nil || !mcpPlan.Complete() || mcpPlan.Summary.Emulated != 1 || mcpPlan.Actions[0].Strategy != conversion.StrategyMCPGateway {
		t.Fatalf("MCP preflight = %#v, err=%v", mcpPlan, err)
	}
}

func TestControllerRejectsMismatchedHostedExecutorRegistration(t *testing.T) {
	t.Parallel()
	_, err := NewController(ControllerConfig{Hosted: hosted.Config{
		SyntheticNameKey: []byte(strings.Repeat("mismatched-hosted-key-", 2)),
		Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebFetch: preflightHostedExecutor{}},
	}})
	if !errors.Is(err, hosted.ErrConfig) {
		t.Fatalf("mismatched hosted executor error = %v", err)
	}
}

type preflightHostedExecutor struct{}

func (preflightHostedExecutor) Kind() llm.ToolKind { return llm.ToolKindWebSearch }

func (preflightHostedExecutor) Function(llm.ToolDefinition) (llm.FunctionDefinition, error) {
	return llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)}, nil
}

func (preflightHostedExecutor) Execute(context.Context, llm.ToolDefinition, llm.ToolInvocation) (*llm.ToolResult, error) {
	return &llm.ToolResult{}, nil
}

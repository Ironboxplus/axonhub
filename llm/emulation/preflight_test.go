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
	forcedWebController, err := NewController(ControllerConfig{
		Hosted:             webController.config.Hosted,
		ForceHostedGateway: conversion.CapabilityWebSearchTool,
	})
	if err != nil {
		t.Fatalf("new forced Responses hosted controller: %v", err)
	}
	forcedWebPlan, err := forcedWebController.Preflight(webRequest, llm.APIFormatOpenAIResponse)
	if err != nil || !forcedWebPlan.Complete() || forcedWebPlan.Summary.Emulated != 1 ||
		forcedWebPlan.Actions[0].Strategy != conversion.StrategyHostedGateway ||
		!forcedWebController.needsHostedEmulation(webRequest, llm.APIFormatOpenAIResponse) {
		t.Fatalf("forced Responses hosted preflight = %#v, err=%v", forcedWebPlan, err)
	}
	if webController.needsHostedEmulation(webRequest, llm.APIFormatOpenAIResponse) {
		t.Fatal("same-protocol hosted tool was forced without attempt admission policy")
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
	forcedMissingController, err := NewController(ControllerConfig{
		MCP: mcpOnlyController.config.MCP, ForceHostedGateway: conversion.CapabilityWebSearchTool,
	})
	if err != nil {
		t.Fatalf("new missing forced hosted controller: %v", err)
	}
	forcedMissingPlan, forcedMissingErr := forcedMissingController.Preflight(webRequest, llm.APIFormatOpenAIResponse)
	if !errors.Is(forcedMissingErr, conversion.ErrIncompletePlan) || forcedMissingPlan == nil || forcedMissingPlan.Summary.Unknown != 1 {
		t.Fatalf("missing forced hosted executor preflight = %#v, err=%v", forcedMissingPlan, forcedMissingErr)
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

	responsesGatewayController, err := NewController(ControllerConfig{
		MCP: mcpOnlyController.config.MCP, ForceMCPGateway: true,
	})
	if err != nil {
		t.Fatalf("new forced Responses MCP controller: %v", err)
	}
	responsesPlan, err := responsesGatewayController.Preflight(mcpRequest, llm.APIFormatOpenAIResponse)
	if err != nil || !responsesPlan.Complete() || responsesPlan.Summary.Emulated != 1 ||
		responsesPlan.Actions[0].Strategy != conversion.StrategyMCPGateway {
		t.Fatalf("forced Responses MCP preflight = %#v, err=%v", responsesPlan, err)
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

func TestControllerCapabilityProfileDefensiveBoundaries(t *testing.T) {
	t.Parallel()
	future := llm.APIFormat("future/protocol")
	if profile, ok := (&Controller{}).CapabilityProfile(future); ok || profile.ID != "" {
		t.Fatalf("unknown capability profile = %#v, ok=%v", profile, ok)
	}
	var nilController *Controller
	if profile, ok := nilController.CapabilityProfile(llm.APIFormatOpenAIResponse); !ok || profile.ID == "" {
		t.Fatalf("nil controller base profile = %#v, ok=%v", profile, ok)
	}
	invalidExecutorController := &Controller{config: ControllerConfig{Hosted: hosted.Config{
		SyntheticNameKey: []byte(strings.Repeat("profile-invalid-executor-", 2)),
		Executors:        map[llm.ToolKind]hosted.Executor{llm.ToolKindWebSearch: nil},
	}}}
	profile, ok := invalidExecutorController.CapabilityProfile(llm.APIFormatOpenAIResponse)
	if !ok || profile.EmulatedTools.Supports(conversion.CapabilityWebSearchTool) {
		t.Fatalf("nil executor was advertised by profile = %#v, ok=%v", profile, ok)
	}
	if invalidExecutorController.needsHostedEmulation(nil, llm.APIFormatOpenAIResponse) {
		t.Fatal("nil request requires hosted emulation")
	}
	if plan, err := (&Controller{}).Preflight(&llm.Request{APIFormat: llm.APIFormatOpenAIResponse}, future); err == nil || plan == nil {
		t.Fatalf("unknown target preflight = %#v, err=%v", plan, err)
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

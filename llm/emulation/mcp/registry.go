package mcp

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sort"
	"strings"
	"sync"

	"github.com/looplj/axonhub/llm"
)

const (
	defaultMaxDiscoveryPages  = 32
	defaultMaxDiscoveredTools = 1024
	syntheticNamePrefix       = "axon_mcp_"
)

var (
	ErrRegistryConfig         = errors.New("invalid MCP registry configuration")
	ErrSyntheticNameCollision = errors.New("MCP synthetic tool name collision")
	ErrDiscoveryLimit         = errors.New("MCP discovery limit exceeded")
)

// EndpointResolver resolves connector_id and tunnel_id definitions to a
// Streamable HTTP endpoint owned by the deployment. Direct server_url tools do
// not call the resolver. Returned endpoint values never enter registry errors.
type EndpointResolver func(context.Context, llm.MCPDefinition) (string, error)

// EndpointPolicyFactory selects the policy for one MCP definition after its
// executable endpoint has been resolved. Deployments use this to keep direct
// server_url trust separate from connector/tunnel trust; a shared host must
// never let a caller-provided URL inherit deployment-owned network access.
type EndpointPolicyFactory func(llm.MCPDefinition, string) (EndpointPolicy, error)

type RegistryConfig struct {
	HTTPClient            *http.Client
	EndpointPolicy        EndpointPolicy
	EndpointPolicyFactory EndpointPolicyFactory
	EndpointResolver      EndpointResolver
	SyntheticNameKey      []byte
	MaxDiscoveryPages     int
	MaxDiscoveredTools    int
}

// Binding is request-owned and contains the typed identity needed to execute a
// generated function call without parsing or trusting its synthetic name.
type Binding struct {
	SyntheticName string
	ServerLabel   string
	Tool          llm.MCPDiscoveredTool
	Approval      *llm.MCPApprovalPolicy
	Deferred      bool
	Discovery     bool
	client        *Client
}

type Registry struct {
	mu          sync.RWMutex
	bindings    map[string]Binding
	byIdentity  map[string]string
	definitions []llm.ToolDefinition
	lists       []llm.Item
	clients     []*Client
	nameKey     []byte
	active      map[string]struct{}
	hasDeferred bool
}

func DiscoverRegistry(ctx context.Context, request *llm.Request, config RegistryConfig) (*Registry, error) {
	if request == nil || config.EndpointPolicy == nil && config.EndpointPolicyFactory == nil || len(config.SyntheticNameKey) < 32 {
		return nil, fmt.Errorf("%w: endpoint policy and a 32-byte synthetic-name key are required", ErrRegistryConfig)
	}
	maxPages := config.MaxDiscoveryPages
	if maxPages <= 0 {
		maxPages = defaultMaxDiscoveryPages
	}
	maxTools := config.MaxDiscoveredTools
	if maxTools <= 0 {
		maxTools = defaultMaxDiscoveredTools
	}

	reserved := make(map[string]struct{}, len(request.ToolDefinitions))
	for index := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindMCP {
			reserved[definition.LogicalName] = struct{}{}
		}
	}
	registry := &Registry{
		bindings: make(map[string]Binding), byIdentity: make(map[string]string),
		nameKey: append([]byte(nil), config.SyntheticNameKey...), active: make(map[string]struct{}),
	}
	discoveredCount := 0
	failed := true
	defer func() {
		if failed {
			_ = registry.Close(context.WithoutCancel(ctx))
		}
	}()

	for definitionIndex := range request.ToolDefinitions {
		definition := &request.ToolDefinitions[definitionIndex]
		if definition.Kind != llm.ToolKindMCP {
			continue
		}
		if err := definition.Validate(); err != nil {
			return nil, fmt.Errorf("discover MCP definition %d: %w", definitionIndex, err)
		}
		endpoint, err := resolveRegistryEndpoint(ctx, *definition.MCP, config.EndpointResolver)
		if err != nil {
			return nil, fmt.Errorf("resolve MCP definition %d: %w", definitionIndex, err)
		}
		endpointPolicy := config.EndpointPolicy
		if config.EndpointPolicyFactory != nil {
			endpointPolicy, err = config.EndpointPolicyFactory(*definition.MCP, endpoint)
			if err != nil {
				return nil, fmt.Errorf("select MCP endpoint policy for definition %d: %w", definitionIndex, err)
			}
		}
		if endpointPolicy == nil {
			return nil, fmt.Errorf("select MCP endpoint policy for definition %d: %w", definitionIndex, ErrRegistryConfig)
		}
		secrets := llm.MCPConnectionSecrets{}
		if request.ToolExecutionSecrets != nil {
			secrets = request.ToolExecutionSecrets.MCP[definition.MCP.ServerLabel]
		}
		client, err := NewClient(Config{
			Endpoint: endpoint, HTTPClient: config.HTTPClient, EndpointPolicy: endpointPolicy, Secrets: secrets,
		})
		if err != nil {
			return nil, fmt.Errorf("create MCP client for definition %d: %w", definitionIndex, err)
		}
		registry.clients = append(registry.clients, client)

		tools, err := discoverAllTools(ctx, client, maxPages, maxTools-discoveredCount)
		if err != nil {
			return nil, fmt.Errorf("discover MCP tools for definition %d: %w", definitionIndex, err)
		}
		discoveredCount += len(tools)
		list := llm.Item{
			Kind: llm.ItemKindMCPListTools, ID: syntheticItemID(config.SyntheticNameKey, "list", definition.MCP.ServerLabel),
			Status:       llm.ItemStatusCompleted,
			MCPListTools: &llm.MCPListTools{ServerLabel: definition.MCP.ServerLabel},
		}
		deferred := definition.MCP.DeferLoading != nil && *definition.MCP.DeferLoading
		for toolIndex := range tools {
			tool := tools[toolIndex]
			if !matchesToolFilter(definition.MCP.AllowedTools, tool) {
				continue
			}
			canonical, err := canonicalDiscoveredTool(tool)
			if err != nil {
				return nil, fmt.Errorf("decode MCP discovered tool %d for definition %d: %w", toolIndex, definitionIndex, err)
			}
			name := syntheticToolName(config.SyntheticNameKey, definition.MCP.ServerLabel, tool.Name)
			if _, collision := reserved[name]; collision {
				return nil, fmt.Errorf("%w: client definition occupies reserved generated name", ErrSyntheticNameCollision)
			}
			if _, duplicate := registry.bindings[name]; duplicate {
				return nil, fmt.Errorf("%w: duplicate generated binding", ErrSyntheticNameCollision)
			}
			binding := Binding{
				SyntheticName: name, ServerLabel: definition.MCP.ServerLabel, Tool: canonical,
				Approval: cloneApprovalPolicy(definition.MCP.RequireApproval), Deferred: deferred,
				client: client,
			}
			registry.bindings[name] = binding
			registry.byIdentity[bindingIdentity(binding.ServerLabel, binding.Tool.Name)] = name
			list.MCPListTools.Tools = append(list.MCPListTools.Tools, canonical)
			if !binding.Deferred {
				registry.definitions = append(registry.definitions, binding.functionDefinition())
				registry.active[binding.SyntheticName] = struct{}{}
			} else {
				registry.hasDeferred = true
			}
		}
		if deferred && len(list.MCPListTools.Tools) > 0 {
			searchName := syntheticName(config.SyntheticNameKey, "search", definition.MCP.ServerLabel)
			if _, collision := reserved[searchName]; collision {
				return nil, fmt.Errorf("%w: client definition occupies reserved discovery name", ErrSyntheticNameCollision)
			}
			if _, duplicate := registry.bindings[searchName]; duplicate {
				return nil, fmt.Errorf("%w: duplicate discovery binding", ErrSyntheticNameCollision)
			}
			search := Binding{
				SyntheticName: searchName, ServerLabel: definition.MCP.ServerLabel,
				Tool: llm.MCPDiscoveredTool{
					Name: "tool_search", Description: "Search and load tools from the deferred MCP server",
					InputSchema: deferredSearchSchema(),
				},
				Deferred: true, Discovery: true, client: client,
			}
			registry.bindings[searchName] = search
			registry.definitions = append(registry.definitions, search.functionDefinition())
		}
		if !deferred && len(list.MCPListTools.Tools) > 0 {
			registry.lists = append(registry.lists, list)
		}
		if discoveredCount > maxTools {
			return nil, ErrDiscoveryLimit
		}
	}
	failed = false
	return registry, nil
}

// NewEmptyRegistry lets the shared gateway loop run hosted executors when a
// request has no MCP definitions. It carries no clients or bindings.
func NewEmptyRegistry(nameKey []byte) *Registry {
	return &Registry{
		bindings: make(map[string]Binding), byIdentity: make(map[string]string),
		nameKey: append([]byte(nil), nameKey...), active: make(map[string]struct{}),
	}
}

func (registry *Registry) BindingFor(serverLabel, toolName string) (Binding, bool) {
	if registry == nil {
		return Binding{}, false
	}
	name, ok := registry.byIdentity[bindingIdentity(serverLabel, toolName)]
	if !ok {
		return Binding{}, false
	}
	return registry.Binding(name)
}

func (binding Binding) IsDiscovery() bool { return binding.Discovery }

func (registry *Registry) HasDeferred() bool {
	return registry != nil && registry.hasDeferred
}

func (registry *Registry) ApprovalID(binding Binding, callID string, arguments json.RawMessage) string {
	if registry == nil {
		return ""
	}
	return "approval_" + opaqueToken(registry.nameKey, "approval", binding.ServerLabel, binding.Tool.Name, callID, string(arguments))
}

func (registry *Registry) SyntheticCallID(approvalID string) string {
	if registry == nil {
		return ""
	}
	return "call_" + opaqueToken(registry.nameKey, "approval_call", approvalID)
}

func (registry *Registry) MCPCallItemID(callID string) string {
	if registry == nil {
		return ""
	}
	return "mcp_" + opaqueToken(registry.nameKey, "call_item", callID)
}

func (registry *Registry) Binding(syntheticName string) (Binding, bool) {
	if registry == nil {
		return Binding{}, false
	}
	binding, ok := registry.bindings[syntheticName]
	return binding, ok
}

func (registry *Registry) FunctionDefinitions() []llm.ToolDefinition {
	if registry == nil {
		return nil
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	return (&llm.Request{ToolDefinitions: registry.definitions}).Clone().ToolDefinitions
}

func (registry *Registry) ListItems() []llm.Item {
	if registry == nil {
		return nil
	}
	return llm.CloneCanonicalItems(registry.lists)
}

func (registry *Registry) Call(ctx context.Context, binding Binding, arguments json.RawMessage) (CallToolResult, error) {
	if registry == nil || binding.client == nil {
		return CallToolResult{}, fmt.Errorf("%w: binding has no client", ErrRegistryConfig)
	}
	current, ok := registry.bindings[binding.SyntheticName]
	if !ok || current.ServerLabel != binding.ServerLabel || current.Tool.Name != binding.Tool.Name || current.client != binding.client || current.Discovery != binding.Discovery {
		return CallToolResult{}, fmt.Errorf("%w: binding is not owned by this registry", ErrRegistryConfig)
	}
	if binding.Discovery {
		result, _, err := registry.Search(binding, arguments)
		return result, err
	}
	return binding.client.CallTool(ctx, binding.Tool.Name, arguments)
}

func (registry *Registry) Search(binding Binding, arguments json.RawMessage) (CallToolResult, []llm.MCPDiscoveredTool, error) {
	if registry == nil || !binding.Discovery {
		return CallToolResult{}, nil, fmt.Errorf("%w: binding is not a discovery function", ErrRegistryConfig)
	}
	var request struct {
		Query     string   `json:"query"`
		ToolNames []string `json:"tool_names"`
		Limit     int      `json:"limit"`
	}
	if len(arguments) == 0 || !json.Valid(arguments) {
		return CallToolResult{}, nil, errors.New("MCP tool search requires valid JSON arguments")
	}
	if err := json.Unmarshal(arguments, &request); err != nil {
		return CallToolResult{}, nil, errors.New("decode MCP tool search arguments")
	}
	if request.Limit <= 0 {
		request.Limit = 8
	}
	if request.Limit > 32 {
		request.Limit = 32
	}

	type candidate struct {
		binding Binding
		score   int
	}
	wantedNames := make(map[string]struct{}, len(request.ToolNames))
	for _, name := range request.ToolNames {
		wantedNames[strings.ToLower(name)] = struct{}{}
	}
	terms := strings.Fields(strings.ToLower(request.Query))
	candidates := make([]candidate, 0)
	for _, current := range registry.bindings {
		if current.Discovery || !current.Deferred || current.ServerLabel != binding.ServerLabel {
			continue
		}
		score := 0
		if _, wanted := wantedNames[strings.ToLower(current.Tool.Name)]; wanted {
			score += 1000
		}
		haystack := strings.ToLower(current.Tool.Name + " " + current.Tool.Title + " " + current.Tool.Description)
		for _, term := range terms {
			if strings.Contains(haystack, term) {
				score++
			}
		}
		if len(wantedNames) > 0 || len(terms) > 0 {
			if score == 0 {
				continue
			}
		}
		candidates = append(candidates, candidate{binding: current, score: score})
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].score != candidates[j].score {
			return candidates[i].score > candidates[j].score
		}
		return candidates[i].binding.Tool.Name < candidates[j].binding.Tool.Name
	})
	if len(candidates) > request.Limit {
		candidates = candidates[:request.Limit]
	}
	selected := make([]llm.MCPDiscoveredTool, 0, len(candidates))
	registry.mu.Lock()
	for _, candidate := range candidates {
		selected = append(selected, candidate.binding.Tool)
		if _, active := registry.active[candidate.binding.SyntheticName]; active {
			continue
		}
		registry.active[candidate.binding.SyntheticName] = struct{}{}
		registry.definitions = append(registry.definitions, candidate.binding.functionDefinition())
	}
	registry.mu.Unlock()

	payload := struct {
		ServerLabel string                  `json:"server_label"`
		Tools       []llm.MCPDiscoveredTool `json:"tools"`
	}{ServerLabel: binding.ServerLabel, Tools: selected}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return CallToolResult{}, nil, errors.New("encode MCP tool search result")
	}
	return CallToolResult{
		Content: []Content{{Type: "text", Text: string(encoded)}}, StructuredContent: encoded,
	}, selected, nil
}

func (registry *Registry) DiscoveryListItem(binding Binding, callID string, tools []llm.MCPDiscoveredTool) llm.Item {
	return llm.Item{
		Kind:   llm.ItemKindMCPListTools,
		ID:     "mcp_" + opaqueToken(registry.nameKey, "search_list", binding.ServerLabel, callID),
		Status: llm.ItemStatusCompleted,
		MCPListTools: &llm.MCPListTools{
			ServerLabel: binding.ServerLabel,
			Tools:       append([]llm.MCPDiscoveredTool(nil), tools...),
		},
	}
}

func (registry *Registry) DiscoveryListFailure(binding Binding, callID, message string) llm.Item {
	return llm.Item{
		Kind:   llm.ItemKindMCPListTools,
		ID:     "mcp_" + opaqueToken(registry.nameKey, "search_list", binding.ServerLabel, callID),
		Status: llm.ItemStatusFailed,
		MCPListTools: &llm.MCPListTools{
			ServerLabel: binding.ServerLabel,
			Error:       message,
		},
	}
}

// ApprovalRequired evaluates the Responses always/never policy against the
// discovered MCP annotations. defaultRequired is used only when the request
// omitted a policy or neither explicit filter matched.
func (binding Binding) ApprovalRequired(defaultRequired bool) bool {
	if binding.Discovery {
		return false
	}
	policy := binding.Approval
	if policy == nil {
		return defaultRequired
	}
	switch policy.Mode {
	case llm.MCPApprovalAlways:
		return true
	case llm.MCPApprovalNever:
		return false
	}
	tool := Tool{Name: binding.Tool.Name, Annotations: binding.Tool.Annotations}
	if policy.Always != nil && matchesToolFilter(policy.Always, tool) {
		return true
	}
	if policy.Never != nil && matchesToolFilter(policy.Never, tool) {
		return false
	}
	return defaultRequired
}

func (registry *Registry) Close(ctx context.Context) error {
	if registry == nil {
		return nil
	}
	var result error
	for _, client := range registry.clients {
		result = errors.Join(result, client.Close(ctx))
	}
	registry.clients = nil
	return result
}

func (binding Binding) functionDefinition() llm.ToolDefinition {
	return llm.ToolDefinition{
		Kind: llm.ToolKindFunction, LogicalName: binding.SyntheticName,
		Description: binding.Tool.Description,
		Function:    &llm.FunctionDefinition{Parameters: append(json.RawMessage(nil), binding.Tool.InputSchema...)},
		Execution:   llm.ExecutionOwnerGateway,
	}
}

func deferredSearchSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"},"tool_names":{"type":"array","items":{"type":"string"}},"limit":{"type":"integer","minimum":1,"maximum":32}},"additionalProperties":false}`)
}

func resolveRegistryEndpoint(ctx context.Context, definition llm.MCPDefinition, resolver EndpointResolver) (string, error) {
	if definition.ServerURL != "" {
		return definition.ServerURL, nil
	}
	if resolver == nil {
		return "", fmt.Errorf("%w: connector or tunnel resolver is required", ErrRegistryConfig)
	}
	endpoint, err := resolver(ctx, definition)
	if err != nil {
		return "", fmt.Errorf("endpoint resolver failed")
	}
	if endpoint == "" {
		return "", fmt.Errorf("%w: endpoint resolver returned no endpoint", ErrRegistryConfig)
	}
	return endpoint, nil
}

func discoverAllTools(ctx context.Context, client *Client, maxPages, maxTools int) ([]Tool, error) {
	if maxTools < 0 {
		return nil, ErrDiscoveryLimit
	}
	tools := make([]Tool, 0)
	cursor := ""
	seen := make(map[string]struct{})
	for page := 0; page < maxPages; page++ {
		result, err := client.ListTools(ctx, cursor)
		if err != nil {
			return nil, err
		}
		if len(tools)+len(result.Tools) > maxTools {
			return nil, ErrDiscoveryLimit
		}
		tools = append(tools, result.Tools...)
		if result.NextCursor == "" {
			return tools, nil
		}
		if _, duplicate := seen[result.NextCursor]; duplicate {
			return nil, fmt.Errorf("MCP discovery returned a repeated cursor")
		}
		seen[result.NextCursor] = struct{}{}
		cursor = result.NextCursor
	}
	return nil, ErrDiscoveryLimit
}

func matchesToolFilter(filter *llm.MCPToolFilter, tool Tool) bool {
	if filter == nil {
		return true
	}
	if len(filter.ToolNames) > 0 && !slices.Contains(filter.ToolNames, tool.Name) {
		return false
	}
	if filter.ReadOnly == nil {
		return true
	}
	return toolReadOnly(tool) == *filter.ReadOnly
}

func toolReadOnly(tool Tool) bool {
	if len(tool.Annotations) == 0 {
		return false
	}
	var annotations struct {
		ReadOnlyHint bool `json:"readOnlyHint"`
	}
	return json.Unmarshal(tool.Annotations, &annotations) == nil && annotations.ReadOnlyHint
}

func canonicalDiscoveredTool(tool Tool) (llm.MCPDiscoveredTool, error) {
	if tool.Name == "" || len(tool.InputSchema) == 0 || !json.Valid(tool.InputSchema) {
		return llm.MCPDiscoveredTool{}, errors.New("MCP tool requires a name and valid input schema")
	}
	for _, value := range []json.RawMessage{tool.OutputSchema, tool.Annotations, tool.Meta} {
		if len(value) > 0 && !json.Valid(value) {
			return llm.MCPDiscoveredTool{}, errors.New("MCP tool contains invalid JSON metadata")
		}
	}
	return llm.MCPDiscoveredTool{
		Name: tool.Name, Title: tool.Title, Description: tool.Description,
		InputSchema:  append(json.RawMessage(nil), tool.InputSchema...),
		OutputSchema: append(json.RawMessage(nil), tool.OutputSchema...),
		Annotations:  append(json.RawMessage(nil), tool.Annotations...),
		Meta:         append(json.RawMessage(nil), tool.Meta...),
	}, nil
}

func syntheticToolName(key []byte, serverLabel, toolName string) string {
	return syntheticName(key, "tool", serverLabel, toolName)
}

func syntheticItemID(key []byte, kind, identity string) string {
	return "mcp_" + opaqueToken(key, kind, identity)
}

func syntheticName(key []byte, values ...string) string {
	return syntheticNamePrefix + opaqueToken(key, values...)
}

func opaqueToken(key []byte, values ...string) string {
	mac := hmac.New(sha256.New, key)
	for _, value := range values {
		_, _ = mac.Write([]byte{0})
		_, _ = mac.Write([]byte(value))
	}
	digest := mac.Sum(nil)
	return base64.RawURLEncoding.EncodeToString(digest[:18])
}

func bindingIdentity(serverLabel, toolName string) string {
	return serverLabel + "\x00" + toolName
}

func cloneApprovalPolicy(policy *llm.MCPApprovalPolicy) *llm.MCPApprovalPolicy {
	if policy == nil {
		return nil
	}
	clone := *policy
	if policy.Always != nil {
		always := *policy.Always
		always.ToolNames = append([]string(nil), policy.Always.ToolNames...)
		clone.Always = &always
	}
	if policy.Never != nil {
		never := *policy.Never
		never.ToolNames = append([]string(nil), policy.Never.ToolNames...)
		clone.Never = &never
	}
	return &clone
}

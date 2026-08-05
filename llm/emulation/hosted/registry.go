package hosted

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/looplj/axonhub/llm"
)

var (
	ErrConfig        = errors.New("invalid hosted executor registry configuration")
	ErrNameCollision = errors.New("hosted executor synthetic name collision")
)

// Executor owns one provider-hosted semantic capability. Implementations use
// typed canonical inputs and outputs; protocol wire structs never cross this
// boundary.
type Executor interface {
	Kind() llm.ToolKind
	Function(llm.ToolDefinition) (llm.FunctionDefinition, error)
	Execute(context.Context, llm.ToolDefinition, llm.ToolInvocation) (*llm.ToolResult, error)
}

// LifecycleExecution carries the part of a hosted call lifecycle that cannot
// be represented by a plain function result. Computer executors use this to
// surface safety checks on the public invocation while the controller retains
// ownership of call identity, execution ownership, and source-protocol output.
type LifecycleExecution struct {
	Result              *llm.ToolResult
	PendingSafetyChecks []llm.ToolSafetyCheck
}

// LifecycleExecutor is optional. Most hosted capabilities only need Executor;
// implementations that receive lifecycle metadata from their deployment
// service can expose it without widening every executor's result contract.
type LifecycleExecutor interface {
	ExecuteLifecycle(context.Context, llm.ToolDefinition, llm.ToolInvocation) (*LifecycleExecution, error)
}

type Config struct {
	SyntheticNameKey []byte
	Executors        map[llm.ToolKind]Executor
}

type Binding struct {
	SyntheticName string
	Definition    llm.ToolDefinition
	Function      llm.FunctionDefinition
	Executor      Executor
}

type Registry struct {
	mu          sync.RWMutex
	bindings    map[string]Binding
	definitions []llm.ToolDefinition
	catalog     map[string]llm.ToolDefinition
	active      map[string]struct{}
}

// DiscoverRegistry selects only definitions that are not native on the target.
// The caller supplies that target decision so provider/channel capability
// overrides can evolve without coupling executors to a protocol package.
func DiscoverRegistry(request *llm.Request, config Config, emulate func(llm.ToolDefinition) bool) (*Registry, error) {
	registry := &Registry{
		bindings: make(map[string]Binding),
		catalog:  make(map[string]llm.ToolDefinition),
		active:   make(map[string]struct{}),
	}
	if request == nil {
		return registry, nil
	}
	requiresExecutor := false
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if definition.Execution == llm.ExecutionOwnerProvider && emulate != nil && emulate(definition) {
			requiresExecutor = true
			break
		}
	}
	if !requiresExecutor {
		return registry, nil
	}
	if len(config.SyntheticNameKey) < 32 {
		return nil, fmt.Errorf("%w: hosted executor synthetic name key is missing", ErrConfig)
	}
	reserved := make(map[string]struct{}, len(request.ToolDefinitions))
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		reserved[definition.LogicalName] = struct{}{}
		if definition.Execution == llm.ExecutionOwnerClient && definition.DeferLoading != nil && *definition.DeferLoading {
			cloned := cloneToolDefinition(definition)
			registry.catalog[catalogKey(cloned.Kind, cloned.LogicalName)] = cloned
		}
	}
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if definition.Execution != llm.ExecutionOwnerProvider || emulate == nil || !emulate(definition) {
			continue
		}
		executor := config.Executors[definition.Kind]
		var function llm.FunctionDefinition
		var err error
		if definition.Kind == llm.ToolKindToolSearch {
			function, err = toolSearchFunction(definition)
		} else {
			if executor == nil || executor.Kind() != definition.Kind {
				return nil, fmt.Errorf("%w: no executor for %s", ErrConfig, definition.Kind)
			}
			function, err = executor.Function(definition)
		}
		if err != nil {
			return nil, fmt.Errorf("prepare hosted %s executor: %w", definition.Kind, err)
		}
		if len(function.Parameters) == 0 {
			function.Parameters = []byte(`{"type":"object","properties":{}}`)
		}
		if !jsonValid(function.Parameters) {
			return nil, fmt.Errorf("%w: hosted %s executor returned invalid parameters", ErrConfig, definition.Kind)
		}
		name := syntheticName(config.SyntheticNameKey, definition.Kind, definition.LogicalName)
		if _, collision := reserved[name]; collision {
			return nil, fmt.Errorf("%w: %s", ErrNameCollision, name)
		}
		if _, collision := registry.bindings[name]; collision {
			return nil, fmt.Errorf("%w: %s", ErrNameCollision, name)
		}
		binding := Binding{SyntheticName: name, Definition: definition, Function: function, Executor: executor}
		registry.bindings[name] = binding
		registry.definitions = append(registry.definitions, llm.ToolDefinition{
			Kind: llm.ToolKindFunction, LogicalName: name, Description: hostedFunctionDescription(definition),
			Function: &llm.FunctionDefinition{
				Parameters: append([]byte(nil), function.Parameters...), Strict: function.Strict,
			},
			Execution: llm.ExecutionOwnerGateway,
		})
	}
	sort.SliceStable(registry.definitions, func(i, j int) bool {
		return registry.definitions[i].LogicalName < registry.definitions[j].LogicalName
	})
	registry.activateFromHistory(request.Input)
	return registry, nil
}

func hostedFunctionDescription(definition llm.ToolDefinition) string {
	if strings.TrimSpace(definition.Description) != "" {
		return definition.Description
	}
	switch definition.Kind {
	case llm.ToolKindWebSearch:
		return "Search the web for current information. After a successful result, use it to answer the user and do not repeat an identical query."
	case llm.ToolKindWebFetch:
		return "Fetch a web resource once and use the returned content to continue the answer."
	case llm.ToolKindFileSearch:
		return "Search the available files and use the returned evidence to answer the user."
	case llm.ToolKindCodeInterpreter, llm.ToolKindCodeExecution:
		return "Execute code only when needed and use the returned result to continue the answer."
	case llm.ToolKindShell, llm.ToolKindLocalShell:
		return "Run the requested shell operation and use its result to continue the answer."
	case llm.ToolKindComputer:
		return "Perform the requested computer action and continue from its recorded result."
	case llm.ToolKindImageGeneration:
		return "Generate the requested image and continue from the generated result."
	case llm.ToolKindToolSearch:
		return "Find the most relevant deferred tools, then call the discovered tool instead of repeating the same search."
	default:
		return "Execute the hosted capability once and continue from its result."
	}
}

func (registry *Registry) Binding(name string) (Binding, bool) {
	if registry == nil {
		return Binding{}, false
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	binding, ok := registry.bindings[name]
	return binding, ok
}

func (registry *Registry) BindingFor(kind llm.ToolKind, logicalName string) (Binding, bool) {
	if registry == nil {
		return Binding{}, false
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	for _, binding := range registry.bindings {
		if binding.Definition.Kind == kind && binding.Definition.LogicalName == logicalName {
			return binding, true
		}
	}
	return Binding{}, false
}

func (registry *Registry) FunctionDefinitions() []llm.ToolDefinition {
	if registry == nil {
		return nil
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	if len(registry.definitions) == 0 {
		return nil
	}
	return cloneToolDefinitions(registry.definitions)
}

func (registry *Registry) OwnsDefinition(definition llm.ToolDefinition) bool {
	if registry == nil {
		return false
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	for _, binding := range registry.bindings {
		if binding.Definition.Kind == definition.Kind && binding.Definition.LogicalName == definition.LogicalName {
			return true
		}
	}
	_, ok := registry.catalog[catalogKey(definition.Kind, definition.LogicalName)]
	return ok
}

func (registry *Registry) Empty() bool {
	if registry == nil {
		return true
	}
	registry.mu.RLock()
	defer registry.mu.RUnlock()
	return len(registry.bindings) == 0
}

func cloneToolDefinition(definition llm.ToolDefinition) llm.ToolDefinition {
	definitions := cloneToolDefinitions([]llm.ToolDefinition{definition})
	if len(definitions) == 0 {
		return llm.ToolDefinition{}
	}
	return definitions[0]
}

func cloneToolDefinitions(definitions []llm.ToolDefinition) []llm.ToolDefinition {
	return (&llm.Request{ToolDefinitions: definitions}).Clone().ToolDefinitions
}

func catalogKey(kind llm.ToolKind, logicalName string) string {
	return string(kind) + "\x00" + logicalName
}

func syntheticName(key []byte, kind llm.ToolKind, logicalName string) string {
	mac := hmac.New(sha256.New, key)
	_, _ = mac.Write([]byte(string(kind)))
	_, _ = mac.Write([]byte{0})
	_, _ = mac.Write([]byte(logicalName))
	return "axh_" + hex.EncodeToString(mac.Sum(nil)[:12])
}

func jsonValid(raw []byte) bool {
	trimmed := strings.TrimSpace(string(raw))
	return trimmed != "" && (trimmed[0] == '{' || trimmed[0] == '[') && json.Valid(raw)
}

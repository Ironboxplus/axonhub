package conversion

import (
	"fmt"
	"sort"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// ProjectAllowedTools converts a Responses allowed_tools selector into the
// equivalent protocol-neutral request shape: only selected definitions remain,
// and the target receives the selector's auto/required mode. The source request
// is never mutated.
func ProjectAllowedTools(request *llm.Request) (*llm.Request, error) {
	if request == nil || request.ToolChoice == nil || request.ToolChoice.AllowedTools == nil {
		return request, nil
	}
	allowed := request.ToolChoice.AllowedTools
	if allowed.Mode != "auto" && allowed.Mode != "required" {
		return nil, fmt.Errorf("%w: allowed_tools mode %q is not portable", ErrIncompletePlan, allowed.Mode)
	}
	if len(allowed.Tools) == 0 {
		return nil, fmt.Errorf("%w: allowed_tools has no tools", ErrIncompletePlan)
	}
	selected, mcpNames, unrestrictedMCP, err := resolveAllowedDefinitions(request.ToolDefinitions, allowed.Tools)
	if err != nil {
		return nil, err
	}
	projected := request.Clone()
	applyAllowedMCPRestrictions(projected.ToolDefinitions, mcpNames, unrestrictedMCP)
	projected.ToolDefinitions = filterAllowedDefinitions(projected.ToolDefinitions, selected)
	mode := allowed.Mode
	projected.ToolChoice = &llm.ToolChoice{ToolChoice: &mode}
	return projected, nil
}

func filterAllowedDefinitions(definitions []llm.ToolDefinition, selected map[int]struct{}) []llm.ToolDefinition {
	filtered := make([]llm.ToolDefinition, 0, len(selected))
	for index := range definitions {
		if _, ok := selected[index]; !ok {
			continue
		}
		filtered = append(filtered, definitions[index])
	}
	return filtered
}

func applyAllowedMCPRestrictions(definitions []llm.ToolDefinition, mcpNames map[int]map[string]struct{}, unrestrictedMCP map[int]bool) {
	for index := range definitions {
		definition := &definitions[index]
		if definition.Kind == llm.ToolKindMCP && definition.MCP != nil && !unrestrictedMCP[index] {
			names := make([]string, 0, len(mcpNames[index]))
			for name := range mcpNames[index] {
				names = append(names, name)
			}
			sort.Strings(names)
			if definition.MCP.AllowedTools == nil {
				definition.MCP.AllowedTools = &llm.MCPToolFilter{}
			}
			definition.MCP.AllowedTools.ToolNames = names
		}
	}
}

func resolveAllowedDefinitions(definitions []llm.ToolDefinition, refs []llm.AllowedToolRef) (
	map[int]struct{}, map[int]map[string]struct{}, map[int]bool, error,
) {
	selected := make(map[int]struct{})
	mcpNames := make(map[int]map[string]struct{})
	unrestrictedMCP := make(map[int]bool)
	for refIndex := range refs {
		ref := refs[refIndex]
		matched := false
		for definitionIndex := range definitions {
			definition := &definitions[definitionIndex]
			if !allowedRefMatchesDefinition(ref, definition) {
				continue
			}
			matched = true
			selected[definitionIndex] = struct{}{}
			if definition.Kind == llm.ToolKindMCP {
				if ref.Name == "" {
					unrestrictedMCP[definitionIndex] = true
					continue
				}
				if mcpNames[definitionIndex] == nil {
					mcpNames[definitionIndex] = make(map[string]struct{})
				}
				mcpNames[definitionIndex][ref.Name] = struct{}{}
			}
		}
		if !matched {
			return nil, nil, nil, fmt.Errorf("%w: allowed_tools selector %d (%s) has no matching definition", ErrIncompletePlan, refIndex, ref.Type)
		}
	}
	return selected, mcpNames, unrestrictedMCP, nil
}

func allowedRefMatchesDefinition(ref llm.AllowedToolRef, definition *llm.ToolDefinition) bool {
	if definition == nil {
		return false
	}
	kind := allowedToolKind(ref.Type)
	if definition.Kind != kind {
		return false
	}
	switch kind {
	case llm.ToolKindFunction, llm.ToolKindCustom:
		return ref.Name != "" && ref.Name == definition.LogicalName
	case llm.ToolKindMCP:
		if definition.MCP == nil || ref.ServerLabel == "" || ref.ServerLabel != definition.MCP.ServerLabel {
			return false
		}
		if ref.Name == "" || definition.MCP.AllowedTools == nil || len(definition.MCP.AllowedTools.ToolNames) == 0 {
			return true
		}
		for _, name := range definition.MCP.AllowedTools.ToolNames {
			if name == ref.Name {
				return true
			}
		}
		return false
	default:
		return true
	}
}

func allowedToolKind(value string) llm.ToolKind {
	switch strings.TrimSpace(value) {
	case "web_search_preview":
		return llm.ToolKindWebSearch
	case "computer_use_preview":
		return llm.ToolKindComputer
	default:
		return llm.ToolKind(strings.TrimSpace(value))
	}
}

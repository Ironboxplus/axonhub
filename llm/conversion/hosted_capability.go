package conversion

import (
	"strings"

	"github.com/looplj/axonhub/llm"
)

// HostedToolNativeEquivalent reports whether the target protocol can preserve
// the configured provider-owned tool behavior without gateway execution.
// Same-protocol identity is handled by the caller before this check.
func HostedToolNativeEquivalent(definition llm.ToolDefinition, target llm.APIFormat) bool {
	if definition.Execution != llm.ExecutionOwnerProvider || definition.Hosted == nil {
		return false
	}
	switch target {
	case llm.APIFormatOpenAIResponse:
		switch definition.Kind {
		case llm.ToolKindImageGeneration:
			return true
		case llm.ToolKindWebSearch:
			webSearch := definition.Hosted.WebSearch
			return webSearch != nil && webSearch.MaxUses == nil && webSearch.Strict == nil && len(webSearch.BlockedDomains) == 0
		}
	case llm.APIFormatAnthropicMessage:
		switch definition.Kind {
		case llm.ToolKindWebSearch:
			return definition.Hosted.WebSearch != nil
		case llm.ToolKindWebFetch:
			return definition.Hosted.WebFetch != nil
		case llm.ToolKindCodeExecution:
			return strings.HasPrefix(definition.Hosted.Type, "code_execution_")
		case llm.ToolKindToolSearch:
			return strings.HasPrefix(definition.Hosted.Type, "tool_search_tool_regex_") ||
				strings.HasPrefix(definition.Hosted.Type, "tool_search_tool_bm25_")
		}
	}
	return false
}

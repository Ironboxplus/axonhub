package hosted

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestHostedFunctionDescriptionCoversEveryCapabilityClass(t *testing.T) {
	t.Parallel()
	require.Equal(t, "custom description", hostedFunctionDescription(llm.ToolDefinition{Description: "custom description"}))
	for _, kind := range []llm.ToolKind{
		llm.ToolKindWebSearch,
		llm.ToolKindWebFetch,
		llm.ToolKindFileSearch,
		llm.ToolKindCodeInterpreter,
		llm.ToolKindCodeExecution,
		llm.ToolKindShell,
		llm.ToolKindLocalShell,
		llm.ToolKindComputer,
		llm.ToolKindImageGeneration,
		llm.ToolKindToolSearch,
		llm.ToolKind("future_hosted_capability"),
	} {
		kind := kind
		t.Run(string(kind), func(t *testing.T) {
			t.Parallel()
			require.NotEmpty(t, hostedFunctionDescription(llm.ToolDefinition{Kind: kind}))
		})
	}
}

package emulation

import (
	"testing"

	"github.com/looplj/axonhub/llm"
)

func BenchmarkCustomConstraintFastPathWithoutGrammar(b *testing.B) {
	request := &llm.Request{ToolDefinitions: []llm.ToolDefinition{
		{
			Kind: llm.ToolKindFunction, LogicalName: "lookup", Execution: llm.ExecutionOwnerClient,
			Function: &llm.FunctionDefinition{},
		},
		{
			Kind: llm.ToolKindCustom, LogicalName: "free_text", Execution: llm.ExecutionOwnerClient,
			Freeform: &llm.FreeformDefinition{Format: "text"},
		},
	}}
	b.ReportAllocs()
	for b.Loop() {
		if NeedsCustomConstraintEmulation(request, llm.APIFormatOpenAIChatCompletion) {
			b.Fatal("unconstrained request entered grammar emulation")
		}
	}
}

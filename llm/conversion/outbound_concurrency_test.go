package conversion

import (
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestResponseSummaryDoesNotMutateSharedDoneSentinel(t *testing.T) {
	beforeNil := llm.DoneResponse.TransformerMetadata == nil
	beforeLen := len(llm.DoneResponse.TransformerMetadata)
	setResponseSummary(llm.DoneResponse, llm.ConversionTraceSummary{
		SourceFormat: llm.APIFormatOpenAIResponse,
		TargetFormat: llm.APIFormatAnthropicMessage,
		Complete:     true,
	})
	if (llm.DoneResponse.TransformerMetadata == nil) != beforeNil ||
		len(llm.DoneResponse.TransformerMetadata) != beforeLen {
		t.Fatal("shared DoneResponse sentinel acquired request-local conversion metadata")
	}
}

package llm

import "testing"

func TestUsageNormalizeRepairsProviderTotalsAndIsIdempotent(t *testing.T) {
	usage := &Usage{
		PromptTokens:     2,
		CompletionTokens: 7,
		TotalTokens:      3,
		PromptTokensDetails: &PromptTokensDetails{
			CachedTokens:      5,
			WriteCachedTokens: 11,
		},
	}

	usage.Normalize()
	if usage.PromptTokens != 16 || usage.TotalTokens != 23 {
		t.Fatalf("normalized usage = prompt:%d total:%d, want 16/23", usage.PromptTokens, usage.TotalTokens)
	}
	if usage.CompletionTokensDetails == nil || !usage.Recovered {
		t.Fatalf("normalized details/recovery = %#v", usage)
	}

	usage.Normalize()
	if usage.PromptTokens != 16 || usage.TotalTokens != 23 {
		t.Fatalf("second normalization changed totals: %#v", usage)
	}
}

func TestUsageNormalizePreservesConsistentProviderTotals(t *testing.T) {
	usage := &Usage{PromptTokens: 20, CompletionTokens: 5, TotalTokens: 30}
	usage.Normalize()
	if usage.PromptTokens != 20 || usage.CompletionTokens != 5 || usage.TotalTokens != 30 {
		t.Fatalf("consistent usage changed: %#v", usage)
	}
	if usage.Recovered {
		t.Fatal("consistent usage incorrectly marked recovered")
	}
	if usage.PromptTokensDetails == nil || usage.CompletionTokensDetails == nil {
		t.Fatal("detail objects were not initialized")
	}
}

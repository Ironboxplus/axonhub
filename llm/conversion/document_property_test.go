package conversion

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/looplj/axonhub/llm"
	"pgregory.net/rapid"
)

func TestPortableDocumentsRemainPlannableAndCloneIsolated(t *testing.T) {
	rapid.Check(t, func(t *rapid.T) {
		source := rapid.SampledFrom([]llm.APIFormat{
			llm.APIFormatOpenAIChatCompletion,
			llm.APIFormatOpenAIResponse,
			llm.APIFormatAnthropicMessage,
		}).Draw(t, "source")
		documentCount := rapid.IntRange(1, 4).Draw(t, "document_count")
		content := []llm.ContentBlock{{Kind: llm.ContentKindText, Text: rapid.String().Draw(t, "text")}}
		for index := 0; index < documentCount; index++ {
			document := &llm.DocumentURL{
				Filename: rapid.StringMatching(`[A-Za-z0-9_-]{1,24}\.pdf`).Draw(t, "filename"),
				Title:    rapid.String().Draw(t, "title"),
				Context:  rapid.String().Draw(t, "context"),
			}
			if rapid.Bool().Draw(t, "use_file_id") {
				document.SourceType = llm.DocumentSourceFile
				document.FileID = rapid.StringMatching(`file_[A-Za-z0-9_-]{1,32}`).Draw(t, "file_id")
			} else {
				document.SourceType = llm.DocumentSourceBase64
				document.MIMEType = "application/pdf"
				document.Data = base64.StdEncoding.EncodeToString(rapid.SliceOfN(rapid.Byte(), 1, 256).Draw(t, "file_bytes"))
			}
			content = append(content, llm.ContentBlock{Kind: llm.ContentKindDocument, Document: document})
		}
		request := &llm.Request{
			APIFormat: source,
			Input:     []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: content}},
		}
		if err := llm.ValidateItemStructure(request.Input); err != nil {
			t.Fatalf("generated portable request is invalid: %v", err)
		}

		encoded, err := json.Marshal(request.Input)
		if err != nil {
			t.Fatalf("marshal generated canonical input: %v", err)
		}
		var decoded []llm.Item
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatalf("unmarshal generated canonical input: %v", err)
		}
		if err := llm.ValidateItemStructure(decoded); err != nil {
			t.Fatalf("JSON round-trip invalidated portable input: %v", err)
		}

		for _, target := range []llm.APIFormat{
			llm.APIFormatOpenAIChatCompletion,
			llm.APIFormatOpenAIResponse,
			llm.APIFormatAnthropicMessage,
		} {
			plan, err := NewPlanner().Plan(request, target)
			if err != nil || plan == nil || !plan.Complete() || plan.Summary.Unknown != 0 {
				t.Fatalf("portable %s -> %s document plan = %#v, err=%v", source, target, plan, err)
			}
		}

		clone := request.Clone()
		clonedDocument := clone.Input[0].Content[1].Document
		originalDocument := request.Input[0].Content[1].Document
		clonedDocument.Title = "changed"
		if clonedDocument.SourceType == llm.DocumentSourceBase64 {
			clonedDocument.Data = "changed"
		} else {
			clonedDocument.FileID = "changed"
		}
		if originalDocument.Title == "changed" || originalDocument.Data == "changed" || originalDocument.FileID == "changed" {
			t.Fatal("request clone shared a document payload with its source")
		}
	})
}

package llm

import (
	"encoding/json"
	"testing"
)

func TestValidateCanonicalOrderedToolLifecycle(t *testing.T) {
	items := []Item{
		{Kind: ItemKindMessage, Role: RoleUser, Content: []ContentBlock{{Kind: ContentKindText, Text: "apply the patch"}}},
		{Kind: ItemKindToolCall, ID: "item_1", ToolCall: &ToolInvocation{
			Kind: ToolKindCustom, CallID: "call_1", LogicalName: "apply_patch", InputText: "*** Begin Patch\n*** End Patch\n",
		}},
		{Kind: ItemKindToolResult, ToolResult: &ToolResult{
			Kind: ToolKindCustom, CallID: "call_1", LogicalName: "apply_patch",
			Content: []ContentBlock{{Kind: ContentKindText, Text: "Done!"}},
		}},
	}

	if err := ValidateItems(items); err != nil {
		t.Fatalf("valid ordered tool lifecycle rejected: %v", err)
	}
}

func TestValidateCanonicalRejectsAmbiguousUnionAndOrphanResult(t *testing.T) {
	ambiguous := []Item{{
		Kind:       ItemKindToolCall,
		ToolCall:   &ToolInvocation{Kind: ToolKindFunction, CallID: "call_1", LogicalName: "lookup"},
		ToolResult: &ToolResult{Kind: ToolKindFunction, CallID: "call_1"},
	}}
	if err := ValidateItems(ambiguous); err == nil {
		t.Fatal("ambiguous item union was accepted")
	}

	orphan := []Item{{Kind: ItemKindToolResult, ToolResult: &ToolResult{
		Kind: ToolKindFunction, CallID: "missing", Content: []ContentBlock{{Kind: ContentKindText, Text: "result"}},
	}}}
	if err := ValidateItems(orphan); err == nil {
		t.Fatal("orphan tool result was accepted")
	}
}

func TestValidateCanonicalHostedLifecycleRequiresCorrelatedTypedResult(t *testing.T) {
	valid := Item{
		Kind: ItemKindHostedCall, ID: "ws_1",
		HostedCall: &HostedToolCall{
			Invocation: ToolInvocation{
				Kind: ToolKindWebSearch, ID: "ws_1", CallID: "ws_1", LogicalName: "web_search",
				Execution: ExecutionOwnerProvider,
			},
			Result: &ToolResult{
				Kind: ToolKindWebSearch, CallID: "ws_1",
				Content: []ContentBlock{{Kind: ContentKindCitation, Citation: &URLCitation{URL: "https://example.invalid"}}},
			},
		},
	}
	if err := valid.Validate(); err != nil {
		t.Fatalf("valid hosted lifecycle rejected: %v", err)
	}

	wrongKind := CloneCanonicalItem(valid)
	wrongKind.HostedCall.Result.Kind = ToolKindImageGeneration
	if err := wrongKind.Validate(); err == nil {
		t.Fatal("hosted lifecycle with mismatched result kind was accepted")
	}
	wongCallID := CloneCanonicalItem(valid)
	wongCallID.HostedCall.Result.CallID = "other"
	if err := wongCallID.Validate(); err == nil {
		t.Fatal("hosted lifecycle with mismatched result call_id was accepted")
	}
}

func TestCanonicalCloneIsolatesOrderedItems(t *testing.T) {
	request := &Request{
		Input: []Item{{Kind: ItemKindMessage, Role: RoleUser, Content: []ContentBlock{{Kind: ContentKindText, Text: "original"}}}},
		ToolDefinitions: []ToolDefinition{{
			Kind: ToolKindFunction, LogicalName: "lookup", Execution: ExecutionOwnerClient,
			Function: &FunctionDefinition{Parameters: []byte(`{"type":"object"}`)},
		}},
	}

	cloned := request.Clone()
	cloned.Input[0].Content[0].Text = "changed"
	cloned.ToolDefinitions[0].Function.Parameters[0] = '['
	if request.Input[0].Content[0].Text != "original" || string(request.ToolDefinitions[0].Function.Parameters) != `{"type":"object"}` {
		t.Fatalf("canonical clone mutated source: %#v", request)
	}
}

func TestDocumentContentValidationRequiresOneTypedSource(t *testing.T) {
	valid := []*DocumentURL{
		{SourceType: DocumentSourceBase64, Data: "JVBERi0=", MIMEType: "application/pdf"},
		{SourceType: DocumentSourceURL, URL: "https://example.invalid/report.pdf"},
		{SourceType: DocumentSourceFile, FileID: "file_123"},
		{SourceType: DocumentSourceText, Data: "plain text", MIMEType: "text/plain"},
		{SourceType: DocumentSourceContent, Content: json.RawMessage(`[{"type":"text","text":"part"}]`)},
		{URL: "https://example.invalid/legacy.pdf"},
	}
	for index, document := range valid {
		block := ContentBlock{Kind: ContentKindDocument, Document: document}
		if err := block.Validate(); err != nil {
			t.Fatalf("valid document %d rejected: %v", index, err)
		}
	}

	invalid := []*DocumentURL{
		{},
		{SourceType: DocumentSourceBase64},
		{SourceType: DocumentSourceURL, URL: "https://example.invalid/a", Data: "also-set"},
		{SourceType: DocumentSourceFile, URL: "https://example.invalid/not-an-id"},
		{SourceType: DocumentSourceContent, Content: json.RawMessage(`not-json`)},
		{SourceType: "future", Data: "payload"},
	}
	for index, document := range invalid {
		block := ContentBlock{Kind: ContentKindDocument, Document: document}
		if err := block.Validate(); err == nil {
			t.Fatalf("invalid document %d was accepted: %#v", index, document)
		}
	}
}

func TestCanonicalCloneIsolatesDocumentPayload(t *testing.T) {
	enabled := true
	request := &Request{Input: []Item{{
		Kind: ItemKindMessage, Role: RoleUser,
		Content: []ContentBlock{{Kind: ContentKindDocument, Document: &DocumentURL{
			SourceType: DocumentSourceContent,
			Content:    json.RawMessage(`[{"type":"text","text":"original"}]`),
			Title:      "original", CitationsEnabled: &enabled,
		}}},
	}}}
	cloned := request.Clone()
	cloned.Input[0].Content[0].Document.Content[0] = '{'
	cloned.Input[0].Content[0].Document.Title = "changed"
	*cloned.Input[0].Content[0].Document.CitationsEnabled = false
	document := request.Input[0].Content[0].Document
	if string(document.Content) != `[{"type":"text","text":"original"}]` || document.Title != "original" || !*document.CitationsEnabled {
		t.Fatalf("canonical document clone mutated source: %#v", document)
	}
}

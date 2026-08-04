package responses

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolMarshalJSONFailureBranches(t *testing.T) {
	t.Parallel()
	_, err := (Tool{Type: "function", Parameters: map[string]any{"unsupported": make(chan int)}}).MarshalJSON()
	if err == nil || !strings.Contains(err.Error(), "unsupported type") {
		t.Fatalf("unsupported typed tool value error = %v", err)
	}
	_, err = (Tool{Type: "function", Residual: json.RawMessage(`{`)}).MarshalJSON()
	if err == nil {
		t.Fatal("invalid tool residual was accepted")
	}
}

func TestToolChoiceJSONBranches(t *testing.T) {
	t.Parallel()
	var invalid ToolChoice
	if err := invalid.UnmarshalJSON([]byte(`1`)); err == nil {
		t.Fatal("numeric tool choice was accepted")
	}

	mode := "auto"
	encoded, err := (&ToolChoice{Mode: &mode}).MarshalJSON()
	if err != nil || string(encoded) != `"auto"` {
		t.Fatalf("string tool choice = %s, %v", encoded, err)
	}

	typeName := "function"
	name := "lookup"
	encoded, err = (&ToolChoice{Type: &typeName, Name: &name, Residual: json.RawMessage(`{"future":1}`)}).MarshalJSON()
	if err != nil {
		t.Fatalf("object tool choice: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &object); err != nil {
		t.Fatalf("decode object tool choice: %v", err)
	}
	requireJSONField(t, object, "type", `"function"`)
	requireJSONField(t, object, "future", `1`)

	_, err = (&ToolChoice{Type: &typeName, Residual: json.RawMessage(`{`)}).MarshalJSON()
	if err == nil {
		t.Fatal("invalid tool choice residual was accepted")
	}
}

func TestResponseToolChoiceJSONBranches(t *testing.T) {
	t.Parallel()
	var choice ResponseToolChoice
	if err := choice.UnmarshalJSON([]byte(`1`)); err == nil {
		t.Fatal("numeric response tool choice was accepted")
	}

	encoded, err := (ResponseToolChoice{StringValue: "required"}).MarshalJSON()
	if err != nil || string(encoded) != `"required"` {
		t.Fatalf("response string tool choice = %s, %v", encoded, err)
	}

	mode := "auto"
	encoded, err = (ResponseToolChoice{ObjectValue: &ToolChoice{Mode: &mode}}).MarshalJSON()
	if err != nil || string(encoded) != `"auto"` {
		t.Fatalf("response object tool choice = %s, %v", encoded, err)
	}

	encoded, err = (ResponseToolChoice{}).MarshalJSON()
	if err != nil || string(encoded) != `null` {
		t.Fatalf("response empty tool choice = %s, %v", encoded, err)
	}
}

func TestItemActionJSONBranches(t *testing.T) {
	t.Parallel()
	var invalid ItemAction
	if err := invalid.UnmarshalJSON([]byte(`{"command":1}`)); err == nil {
		t.Fatal("invalid local shell command was accepted")
	}

	tests := []struct {
		name string
		item ItemAction
		want string
	}{
		{name: "image", item: *NewImageGenerationAction("generate"), want: `"generate"`},
		{name: "local shell", item: *NewLocalShellAction(&LocalShellAction{Type: "exec", Command: []string{"pwd"}}), want: `{"type":"exec","command":["pwd"]}`},
		{name: "web search", item: *NewWebSearchAction(&WebSearchAction{Type: "search", Query: "weather"}), want: `{"type":"search","query":"weather"}`},
		{name: "empty", item: ItemAction{}, want: `null`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			encoded, err := test.item.MarshalJSON()
			if err != nil {
				t.Fatalf("marshal action: %v", err)
			}
			var actual any
			var want any
			if err := json.Unmarshal(encoded, &actual); err != nil {
				t.Fatalf("decode action: %v", err)
			}
			if err := json.Unmarshal([]byte(test.want), &want); err != nil {
				t.Fatalf("decode expected action: %v", err)
			}
			actualJSON, _ := json.Marshal(actual)
			wantJSON, _ := json.Marshal(want)
			if string(actualJSON) != string(wantJSON) {
				t.Fatalf("action = %s, want %s", actualJSON, wantJSON)
			}
		})
	}
}

func TestItemActionMarshalPropagatesNestedResidualFailure(t *testing.T) {
	t.Parallel()
	action := ItemAction{WebSearch: &WebSearchAction{
		Type: "search",
		Sources: []WebSearchSource{{
			Type: "url", URL: "https://example.invalid", Residual: json.RawMessage(`{`),
		}},
	}}
	if _, err := action.MarshalJSON(); err == nil {
		t.Fatal("invalid nested web-search source residual was accepted")
	}
}

func TestNestedResidualCodecFailureBranches(t *testing.T) {
	t.Parallel()
	for name, decode := range map[string]func() error{
		"tool option":  func() error { var value ToolOption; return value.UnmarshalJSON([]byte(`{`)) },
		"web source":   func() error { var value WebSearchSource; return value.UnmarshalJSON([]byte(`{`)) },
		"safety check": func() error { var value ComputerSafetyCheck; return value.UnmarshalJSON([]byte(`{`)) },
		"screenshot":   func() error { var value ComputerScreenshot; return value.UnmarshalJSON([]byte(`{`)) },
		"MCP tool":     func() error { var value MCPListedTool; return value.UnmarshalJSON([]byte(`{`)) },
		"stream part":  func() error { var value StreamEventContentPart; return value.UnmarshalJSON([]byte(`{`)) },
	} {
		if err := decode(); err == nil {
			t.Fatalf("%s invalid JSON was accepted", name)
		}
	}

	for name, encode := range map[string]func() error{
		"tool option": func() error {
			_, err := (ToolOption{Type: "function", Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"web source": func() error {
			_, err := (WebSearchSource{Type: "url", Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"safety check": func() error {
			_, err := (ComputerSafetyCheck{ID: "safe", Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"screenshot": func() error {
			_, err := (ComputerScreenshot{Type: "computer_screenshot", Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"MCP tool typed": func() error {
			_, err := (MCPListedTool{Name: "lookup", InputSchema: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"MCP tool residual": func() error {
			_, err := (MCPListedTool{Name: "lookup", InputSchema: json.RawMessage(`{}`), Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
		"stream part typed": func() error {
			_, err := (StreamEventContentPart{Type: "output_text", Annotations: []Annotation{{Residual: json.RawMessage(`{`)}}}).MarshalJSON()
			return err
		},
		"stream part residual": func() error {
			_, err := (StreamEventContentPart{Type: "refusal", Residual: json.RawMessage(`{`)}).MarshalJSON()
			return err
		},
	} {
		if err := encode(); err == nil {
			t.Fatalf("%s invalid residual was accepted", name)
		}
	}
}

func TestStreamContentPartMarshalCoversNonTextSuccess(t *testing.T) {
	t.Parallel()
	refusal := "blocked"
	encoded, err := (StreamEventContentPart{Type: "refusal", Refusal: &refusal, Residual: json.RawMessage(`{"future":1}`)}).MarshalJSON()
	if err != nil {
		t.Fatalf("marshal refusal part: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &object); err != nil {
		t.Fatalf("decode refusal part: %v", err)
	}
	requireJSONField(t, object, "refusal", `"blocked"`)
	requireJSONField(t, object, "future", `1`)
	if _, err := MarshalStreamEvent(&StreamEvent{Type: StreamEventTypeResponseCreated}); err != nil {
		t.Fatalf("marshal stream event helper: %v", err)
	}
}

func TestResponseResidualCodecFailureBranches(t *testing.T) {
	t.Parallel()
	var response Response
	if err := response.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("direct Response.UnmarshalJSON accepted invalid JSON")
	}

	_, err := (Response{Output: []Item{{Type: "computer_call", ComputerAction: json.RawMessage(`{`)}}}).MarshalJSON()
	if err == nil || !strings.Contains(err.Error(), "action must be valid JSON") {
		t.Fatalf("invalid typed response output error = %v", err)
	}
	_, err = (Response{Residual: json.RawMessage(`{`)}).MarshalJSON()
	if err == nil {
		t.Fatal("invalid response residual was accepted")
	}
}

func TestNestedResponseObjectResidualCodecBranches(t *testing.T) {
	t.Parallel()

	var annotation Annotation
	if err := annotation.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("invalid annotation JSON was accepted")
	}
	if err := json.Unmarshal([]byte(`{"type":"url_citation","url_citation":{"url":"https://example.com","future":1}}`), &annotation); err != nil {
		t.Fatalf("unmarshal nested citation: %v", err)
	}
	if annotation.URLCitation == nil || annotation.URLCitation.URL != "https://example.com" || string(annotation.URLCitation.Residual) != `{"future":1}` {
		t.Fatalf("nested citation = %#v", annotation.URLCitation)
	}
	annotation.URLCitation.Residual = json.RawMessage(`{`)
	if _, err := annotation.MarshalJSON(); err == nil {
		t.Fatal("invalid nested citation residual was accepted by annotation codec")
	}
	annotation.URLCitation.Residual = json.RawMessage(`{"future":1}`)
	annotation.Residual = json.RawMessage(`{`)
	if _, err := annotation.MarshalJSON(); err == nil {
		t.Fatal("invalid annotation residual was accepted")
	}

	var citation URLCitation
	if err := citation.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("invalid citation JSON was accepted")
	}
	if err := json.Unmarshal([]byte(`{"url":"https://example.com","future":1}`), &citation); err != nil {
		t.Fatalf("unmarshal citation: %v", err)
	}
	citation.Residual = json.RawMessage(`{`)
	if _, err := citation.MarshalJSON(); err == nil {
		t.Fatal("invalid citation residual was accepted")
	}

	var summary ReasoningSummary
	if err := summary.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("invalid reasoning summary JSON was accepted")
	}
	if err := json.Unmarshal([]byte(`{"type":"summary_text","text":"summary","future":1}`), &summary); err != nil {
		t.Fatalf("unmarshal reasoning summary: %v", err)
	}
	summary.Residual = json.RawMessage(`{`)
	if _, err := summary.MarshalJSON(); err == nil {
		t.Fatal("invalid reasoning summary residual was accepted")
	}

	var content ReasoningContent
	if err := content.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("invalid reasoning content JSON was accepted")
	}
	if err := json.Unmarshal([]byte(`{"type":"reasoning_text","text":"detail","future":1}`), &content); err != nil {
		t.Fatalf("unmarshal reasoning content: %v", err)
	}
	content.Residual = json.RawMessage(`{`)
	if _, err := content.MarshalJSON(); err == nil {
		t.Fatal("invalid reasoning content residual was accepted")
	}
}

func TestStreamEventResidualCodecFailureBranches(t *testing.T) {
	t.Parallel()
	var event StreamEvent
	if err := event.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("invalid stream event JSON was accepted")
	}
	_, err := (StreamEvent{
		Type: StreamEventTypeOutputItemAdded,
		Item: &Item{Type: "computer_call", ComputerAction: json.RawMessage(`{`)},
	}).MarshalJSON()
	if err == nil || !strings.Contains(err.Error(), "action must be valid JSON") {
		t.Fatalf("invalid typed stream event error = %v", err)
	}
	_, err = (StreamEvent{Type: StreamEventTypeResponseCreated, Residual: json.RawMessage(`{`)}).MarshalJSON()
	if err == nil {
		t.Fatal("invalid stream event residual was accepted")
	}
}

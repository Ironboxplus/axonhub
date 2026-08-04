package responses

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestItemUnmarshalJSONUnionBranches(t *testing.T) {
	t.Parallel()
	var invalid Item
	if err := invalid.UnmarshalJSON([]byte(`{`)); err == nil {
		t.Fatal("direct Item.UnmarshalJSON accepted invalid JSON")
	}
	tests := []struct {
		name    string
		wire    string
		wantErr string
		check   func(*testing.T, Item)
	}{
		{name: "invalid item", wire: `{`, wantErr: "unexpected end"},
		{name: "null optional unions", wire: `{"type":"message","action":null,"output":null,"arguments":null}`},
		{name: "computer action", wire: `{"type":"computer_call","action":{"type":"click","x":1}}`, check: func(t *testing.T, item Item) {
			if string(item.ComputerAction) != `{"type":"click","x":1}` {
				t.Fatalf("computer action = %s", item.ComputerAction)
			}
		}},
		{name: "shell action", wire: `{"type":"shell_call","action":{"command":["go","test"]}}`, check: func(t *testing.T, item Item) {
			if string(item.ComputerAction) != `{"command":["go","test"]}` {
				t.Fatalf("shell action = %s", item.ComputerAction)
			}
		}},
		{name: "image action string", wire: `{"type":"image_generation_call","action":"generate"}`, check: func(t *testing.T, item Item) {
			if item.Action == nil || item.Action.ImageGenerationAction != "generate" {
				t.Fatalf("image action = %#v", item.Action)
			}
		}},
		{name: "web action object", wire: `{"type":"web_search_call","action":{"type":"search","query":"weather"}}`, check: func(t *testing.T, item Item) {
			if item.Action == nil || item.Action.WebSearch == nil || item.Action.WebSearch.Query != "weather" {
				t.Fatalf("web action = %#v", item.Action)
			}
		}},
		{name: "invalid action union", wire: `{"type":"web_search_call","action":1}`, wantErr: "action must be a string or object"},
		{name: "computer output", wire: `{"type":"computer_call_output","output":{"type":"computer_screenshot","file_id":"file_1"}}`, check: func(t *testing.T, item Item) {
			if item.ComputerOutput == nil || item.ComputerOutput.FileID != "file_1" {
				t.Fatalf("computer output = %#v", item.ComputerOutput)
			}
		}},
		{name: "invalid computer output", wire: `{"type":"computer_call_output","output":"bad"}`, wantErr: "cannot unmarshal"},
		{name: "shell output", wire: `{"type":"shell_call_output","output":[{"stdout":"ok"}]}`, check: func(t *testing.T, item Item) {
			if string(item.RawOutput) != `[{"stdout":"ok"}]` {
				t.Fatalf("shell output = %s", item.RawOutput)
			}
		}},
		{name: "text output", wire: `{"type":"function_call_output","output":"done"}`, check: func(t *testing.T, item Item) {
			if item.Output == nil || item.Output.Text == nil || *item.Output.Text != "done" {
				t.Fatalf("text output = %#v", item.Output)
			}
		}},
		{name: "invalid input output", wire: `{"type":"function_call_output","output":1}`, wantErr: "invalid input"},
		{name: "additional tools", wire: `{"type":"additional_tools","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`, check: func(t *testing.T, item Item) {
			if len(item.AdditionalTools) != 1 || item.AdditionalTools[0].Name != "lookup" || item.Tools != nil {
				t.Fatalf("additional tools = %#v / %#v", item.AdditionalTools, item.Tools)
			}
		}},
		{name: "tool search output", wire: `{"type":"tool_search_output","tools":[{"type":"custom","name":"exec"}]}`, check: func(t *testing.T, item Item) {
			if len(item.AdditionalTools) != 1 || item.AdditionalTools[0].Name != "exec" {
				t.Fatalf("tool search output = %#v", item.AdditionalTools)
			}
		}},
		{name: "invalid additional tool", wire: `{"type":"additional_tools","tools":[{"type":"function","parameters":1}]}`, wantErr: "cannot unmarshal"},
		{name: "string arguments", wire: `{"type":"function_call","arguments":"{\"q\":1}"}`, check: func(t *testing.T, item Item) {
			if item.Arguments != `{"q":1}` {
				t.Fatalf("string arguments = %q", item.Arguments)
			}
		}},
		{name: "object arguments", wire: `{"type":"tool_search_call","arguments":{"q":1}}`, check: func(t *testing.T, item Item) {
			if item.Arguments != `{"q":1}` {
				t.Fatalf("object arguments = %q", item.Arguments)
			}
		}},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var item Item
			err := json.Unmarshal([]byte(test.wire), &item)
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("unmarshal error = %v, want containing %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unmarshal item: %v", err)
			}
			if test.check != nil {
				test.check(t, item)
			}
		})
	}
}

func TestItemMarshalJSONUnionBranches(t *testing.T) {
	t.Parallel()
	text := "input"
	nonEmptyTools := []MCPListedTool{{Name: "lookup", InputSchema: json.RawMessage(`{"type":"object"}`)}}
	tests := []struct {
		name    string
		item    Item
		wantErr string
		check   func(*testing.T, map[string]json.RawMessage)
	}{
		{name: "computer default action", item: Item{Type: "computer_call"}, check: requireRawField("action", `{}`)},
		{name: "shell action", item: Item{Type: "shell_call", ComputerAction: json.RawMessage(`{"command":["pwd"]}`)}, check: requireRawField("action", `{"command":["pwd"]}`)},
		{name: "invalid computer action", item: Item{Type: "computer_call", ComputerAction: json.RawMessage(`{`)}, wantErr: "action must be valid JSON"},
		{name: "computer output", item: Item{Type: "computer_call_output", ComputerOutput: &ComputerScreenshot{Type: "computer_screenshot", FileID: "file_1"}}, check: requireRawField("output", `{"type":"computer_screenshot","file_id":"file_1"}`)},
		{name: "shell default output", item: Item{Type: "shell_call_output"}, check: requireRawField("output", `[]`)},
		{name: "shell output", item: Item{Type: "shell_call_output", RawOutput: json.RawMessage(`[{"stdout":"ok"}]`)}, check: requireRawField("output", `[{"stdout":"ok"}]`)},
		{name: "invalid shell output", item: Item{Type: "shell_call_output", RawOutput: json.RawMessage(`{`)}, wantErr: "output must be valid JSON"},
		{name: "additional tools", item: Item{Type: "additional_tools", AdditionalTools: []Tool{{Type: "custom", Name: "exec"}}}, check: requireArrayLength("tools", 1)},
		{name: "tool search output", item: Item{Type: "tool_search_output", AdditionalTools: []Tool{}}, check: requireArrayLength("tools", 0)},
		{name: "tool search default arguments", item: Item{Type: "tool_search_call"}, check: requireRawField("arguments", `{}`)},
		{name: "tool search arguments", item: Item{Type: "tool_search_call", Arguments: `{"q":1}`}, check: requireRawField("arguments", `{"q":1}`)},
		{name: "invalid tool search arguments", item: Item{Type: "tool_search_call", Arguments: `{`}, wantErr: "arguments must be valid JSON"},
		{name: "function arguments string", item: Item{Type: "function_call", Arguments: `{"q":1}`}, check: requireRawField("arguments", `"{\"q\":1}"`)},
		{name: "mcp call arguments string", item: Item{Type: "mcp_call", Arguments: `{"q":1}`}, check: requireRawField("arguments", `"{\"q\":1}"`)},
		{name: "mcp approval arguments string", item: Item{Type: "mcp_approval_request", Arguments: `{"q":1}`}, check: requireRawField("arguments", `"{\"q\":1}"`)},
		{name: "mcp list empty tools", item: Item{Type: "mcp_list_tools"}, check: requireArrayLength("tools", 0)},
		{name: "mcp list tools", item: Item{Type: "mcp_list_tools", Tools: nonEmptyTools}, check: requireArrayLength("tools", 1)},
		{name: "custom tool empty input", item: Item{Type: "custom_tool_call"}, check: requireRawField("input", `""`)},
		{name: "custom tool input", item: Item{Type: "custom_tool_call", Input: &text}, check: requireRawField("input", `"input"`)},
		{name: "compaction empty encrypted", item: Item{Type: "compaction"}, check: requireRawField("encrypted_content", `""`)},
		{name: "compaction encrypted", item: Item{Type: "compaction", EncryptedContent: &text}, check: requireRawField("encrypted_content", `"input"`)},
		{name: "non reasoning omits summary", item: Item{Type: "message", Summary: []ReasoningSummary{{Type: "summary_text", Text: "hidden"}}}, check: func(t *testing.T, object map[string]json.RawMessage) {
			if _, exists := object["summary"]; exists {
				t.Fatalf("non-reasoning item emitted summary: %#v", object)
			}
		}},
		{name: "reasoning empty summary", item: Item{Type: "reasoning"}, check: requireArrayLength("summary", 0)},
		{name: "reasoning summary", item: Item{Type: "reasoning", Summary: []ReasoningSummary{{Type: "summary_text", Text: "done"}}}, check: requireArrayLength("summary", 1)},
		{name: "typed field wins residual", item: Item{Type: "message", Role: "assistant", Residual: json.RawMessage(`{"role":"user","future":1}`)}, check: func(t *testing.T, object map[string]json.RawMessage) {
			requireJSONField(t, object, "role", `"assistant"`)
			requireJSONField(t, object, "future", `1`)
		}},
		{name: "invalid residual", item: Item{Type: "message", Residual: json.RawMessage(`{`)}, wantErr: "unexpected end"},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			encoded, err := test.item.MarshalJSON()
			if test.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), test.wantErr) {
					t.Fatalf("marshal error = %v, want containing %q", err, test.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("marshal item: %v", err)
			}
			var object map[string]json.RawMessage
			if err := json.Unmarshal(encoded, &object); err != nil {
				t.Fatalf("decode item: %v", err)
			}
			if test.check != nil {
				test.check(t, object)
			}
		})
	}
}

func requireRawField(field, want string) func(*testing.T, map[string]json.RawMessage) {
	return func(t *testing.T, object map[string]json.RawMessage) {
		t.Helper()
		requireJSONField(t, object, field, want)
	}
}

func requireArrayLength(field string, want int) func(*testing.T, map[string]json.RawMessage) {
	return func(t *testing.T, object map[string]json.RawMessage) {
		t.Helper()
		var values []json.RawMessage
		if err := json.Unmarshal(object[field], &values); err != nil {
			t.Fatalf("decode %s: %v", field, err)
		}
		if len(values) != want {
			t.Fatalf("%s length = %d, want %d", field, len(values), want)
		}
	}
}

func requireJSONField(t *testing.T, object map[string]json.RawMessage, field, want string) {
	t.Helper()
	actual, exists := object[field]
	if !exists {
		t.Fatalf("missing %s in %#v", field, object)
	}
	var wantValue any
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("decode expected %s: %v", field, err)
	}
	var actualValue any
	if err := json.Unmarshal(actual, &actualValue); err != nil {
		t.Fatalf("decode actual %s: %v", field, err)
	}
	wantJSON, _ := json.Marshal(wantValue)
	actualJSON, _ := json.Marshal(actualValue)
	if string(actualJSON) != string(wantJSON) {
		t.Fatalf("%s = %s, want %s", field, actualJSON, wantJSON)
	}
}

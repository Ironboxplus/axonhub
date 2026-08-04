package responses

import (
	"encoding/json"
	"testing"
)

func TestJSONResidualHelpersCoverAllInputShapes(t *testing.T) {
	t.Parallel()
	owned := map[string]struct{}{"known": {}}

	if residual := jsonObjectResidual(nil, owned); residual != nil {
		t.Fatalf("empty residual = %s", residual)
	}
	if residual := jsonObjectResidual(json.RawMessage(`{`), owned); residual != nil {
		t.Fatalf("invalid residual = %s", residual)
	}
	if residual := jsonObjectResidual(json.RawMessage(`{"known":1}`), owned); residual != nil {
		t.Fatalf("owned-only residual = %s", residual)
	}
	if residual := jsonObjectResidual(json.RawMessage(`{"known":1,"future":{"enabled":true}}`), owned); string(residual) != `{"future":{"enabled":true}}` {
		t.Fatalf("future residual = %s", residual)
	}
	for name, residual := range map[string]json.RawMessage{
		"request":     responseRequestResidual(json.RawMessage(`{"model":"grok-4.5","future":1}`)),
		"item":        responseItemResidual(json.RawMessage(`{"type":"message","future":1}`)),
		"tool":        responseToolResidual(json.RawMessage(`{"type":"function","future":1}`)),
		"tool_choice": responseToolChoiceResidual(json.RawMessage(`{"type":"function","future":1}`)),
		"response":    responseResponseResidual(json.RawMessage(`{"id":"resp_1","future":1}`)),
	} {
		if string(residual) != `{"future":1}` {
			t.Fatalf("%s wrapper residual = %s", name, residual)
		}
	}

	if items := rawItemArrayField(nil, "content"); items != nil {
		t.Fatalf("empty raw item array = %#v", items)
	}
	if items := rawItemArrayField(json.RawMessage(`{`), "content"); items != nil {
		t.Fatalf("invalid raw item array = %#v", items)
	}
	if items := rawItemArrayField(json.RawMessage(`{"content":{}}`), "content"); items != nil {
		t.Fatalf("non-array raw item field = %#v", items)
	}
	items := rawItemArrayField(json.RawMessage(`{"content":[{"type":"output_text"}]}`), "content")
	if len(items) != 1 || string(items[0]) != `{"type":"output_text"}` {
		t.Fatalf("raw item array = %#v", items)
	}
}

func TestMergeResidualObjectTypedFieldsWin(t *testing.T) {
	t.Parallel()
	current := json.RawMessage(`{"type":"message","role":"assistant"}`)
	merged, err := mergeResidualObject(current, nil)
	if err != nil || string(merged) != string(current) {
		t.Fatalf("empty residual merge = %s, %v", merged, err)
	}
	merged, err = mergeResidualObject(current, json.RawMessage(`{"role":"user","future":7}`))
	if err != nil {
		t.Fatalf("merge residual: %v", err)
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(merged, &object); err != nil {
		t.Fatalf("decode merged residual: %v", err)
	}
	if string(object["role"]) != `"assistant"` || string(object["future"]) != `7` {
		t.Fatalf("typed overlay did not win: %s", merged)
	}
}

func TestResponseItemActionResidualRejectsNonObjectShapes(t *testing.T) {
	t.Parallel()
	if residual := responseItemActionResidual(nil); residual != nil {
		t.Fatalf("nil action residual = %s", residual)
	}
	if residual := responseItemActionResidual(json.RawMessage(`"generate"`)); residual != nil {
		t.Fatalf("string action residual = %s", residual)
	}
}

package llm

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestEmbeddingInputJSONRepresentations(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		input   string
		want    EmbeddingInput
		encoded string
	}{
		{name: "string", input: `"hello"`, want: EmbeddingInput{String: "hello"}, encoded: `"hello"`},
		{name: "empty string", input: `""`, want: EmbeddingInput{String: ""}, encoded: `""`},
		{name: "strings", input: `["hello","world"]`, want: EmbeddingInput{StringArray: []string{"hello", "world"}}, encoded: `["hello","world"]`},
		{name: "empty array", input: `[]`, want: EmbeddingInput{StringArray: []string{}}, encoded: `[]`},
		{name: "tokens", input: `[1,2]`, want: EmbeddingInput{IntArray: []int64{1, 2}}, encoded: `[1,2]`},
		{name: "token batches", input: `[[1,2],[3]]`, want: EmbeddingInput{IntArrayArray: [][]int64{{1, 2}, {3}}}, encoded: `[[1,2],[3]]`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var got EmbeddingInput
			if err := json.Unmarshal([]byte(test.input), &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if !reflect.DeepEqual(got.String, test.want.String) ||
				!reflect.DeepEqual(got.StringArray, test.want.StringArray) ||
				!reflect.DeepEqual(got.IntArray, test.want.IntArray) ||
				!reflect.DeepEqual(got.IntArrayArray, test.want.IntArrayArray) {
				t.Fatalf("decoded = %#v, want %#v", got, test.want)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(encoded) != test.encoded {
				t.Fatalf("encoded = %s, want %s", encoded, test.encoded)
			}
		})
	}
}

func TestEmbeddingInputUnmarshalRejectsInvalidKinds(t *testing.T) {
	t.Parallel()
	for _, input := range []string{`null`, `123`, `true`, `{}`, `["valid",1]`} {
		t.Run(input, func(t *testing.T) {
			var got EmbeddingInput
			if err := json.Unmarshal([]byte(input), &got); err == nil || err.Error() != "invalid embedding input type" {
				t.Fatalf("unexpected error for %s: %v", input, err)
			}
		})
	}
}

func TestEmbeddingInputUnmarshalReplacesPreviousRepresentation(t *testing.T) {
	t.Parallel()
	got := EmbeddingInput{String: "stale"}
	if err := json.Unmarshal([]byte(`["fresh","values"]`), &got); err != nil {
		t.Fatalf("unmarshal strings: %v", err)
	}
	if got.String != "" || !reflect.DeepEqual(got.StringArray, []string{"fresh", "values"}) {
		t.Fatalf("stale string survived: %#v", got)
	}
	if err := json.Unmarshal([]byte(`[1,2]`), &got); err != nil {
		t.Fatalf("unmarshal tokens: %v", err)
	}
	if got.StringArray != nil || !reflect.DeepEqual(got.IntArray, []int64{1, 2}) {
		t.Fatalf("stale string array survived: %#v", got)
	}
}

func TestEmptyEmbeddingWireDefaults(t *testing.T) {
	t.Parallel()
	input, err := json.Marshal(EmbeddingInput{})
	if err != nil || string(input) != "null" {
		t.Fatalf("empty embedding input = %s, %v; want null", input, err)
	}
	value, err := json.Marshal(Embedding{})
	if err != nil || string(value) != "[]" {
		t.Fatalf("empty embedding value = %s, %v; want []", value, err)
	}
}

func TestEmbeddingJSONRepresentations(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		input   string
		want    Embedding
		encoded string
	}{
		{name: "float array", input: `[0.1,0.2]`, want: Embedding{Embedding: []float64{0.1, 0.2}}, encoded: `[0.1,0.2]`},
		{name: "empty array", input: `[]`, want: Embedding{Embedding: []float64{}}, encoded: `[]`},
		{name: "base64", input: `"YWJj"`, want: Embedding{Base64: "YWJj"}, encoded: `"YWJj"`},
		{name: "empty base64", input: `""`, want: Embedding{Base64: ""}, encoded: `""`},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var got Embedding
			if err := json.Unmarshal([]byte(test.input), &got); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if !reflect.DeepEqual(got.Embedding, test.want.Embedding) || got.Base64 != test.want.Base64 {
				t.Fatalf("decoded = %#v, want %#v", got, test.want)
			}
			encoded, err := json.Marshal(got)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if string(encoded) != test.encoded {
				t.Fatalf("encoded = %s, want %s", encoded, test.encoded)
			}
		})
	}
}

func TestEmbeddingUnmarshalRejectsInvalidKindsAndReplacesState(t *testing.T) {
	t.Parallel()
	for _, input := range []string{`null`, `123`, `true`, `{}`, `[0.1,"invalid"]`} {
		t.Run(input, func(t *testing.T) {
			var got Embedding
			if err := json.Unmarshal([]byte(input), &got); err == nil || err.Error() != "invalid embedding type" {
				t.Fatalf("unexpected error for %s: %v", input, err)
			}
		})
	}

	got := Embedding{Base64: "stale"}
	if err := json.Unmarshal([]byte(`[0.1,0.2]`), &got); err != nil {
		t.Fatalf("unmarshal float embedding: %v", err)
	}
	if got.Base64 != "" || !reflect.DeepEqual(got.Embedding, []float64{0.1, 0.2}) {
		t.Fatalf("stale base64 survived: %#v", got)
	}
	if err := json.Unmarshal([]byte(`"fresh"`), &got); err != nil {
		t.Fatalf("unmarshal base64 embedding: %v", err)
	}
	if got.Embedding != nil || got.Base64 != "fresh" {
		t.Fatalf("stale float array survived: %#v", got)
	}
}

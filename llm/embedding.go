package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
)

const (
	EmbeddingInputTypeString        = "string"
	EmbeddingInputTypeStringArray   = "string_array"
	EmbeddingInputTypeIntArray      = "int_array"
	EmbeddingInputTypeIntArrayArray = "int_array_array"
)

type EmbeddingInput struct {
	String        string    `json:"string,omitempty"`
	StringArray   []string  `json:"string_array,omitempty"`
	IntArray      []int64   `json:"int_array,omitempty"`
	IntArrayArray [][]int64 `json:"int_array_array,omitempty"`
	inputType     string
}

func (e EmbeddingInput) MarshalJSON() ([]byte, error) {
	if e.inputType == EmbeddingInputTypeString || e.String != "" {
		return json.Marshal(e.String)
	}

	if e.inputType == EmbeddingInputTypeStringArray || e.StringArray != nil {
		return json.Marshal(e.StringArray)
	}

	if e.inputType == EmbeddingInputTypeIntArray || e.IntArray != nil {
		return json.Marshal(e.IntArray)
	}

	if e.inputType == EmbeddingInputTypeIntArrayArray || e.IntArrayArray != nil {
		return json.Marshal(e.IntArrayArray)
	}

	return json.Marshal(nil)
}

func (e *EmbeddingInput) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("invalid embedding input type")
	}

	var str string

	err := json.Unmarshal(data, &str)
	if err == nil {
		*e = EmbeddingInput{String: str, inputType: EmbeddingInputTypeString}
		return nil
	}

	var strArray []string

	err = json.Unmarshal(data, &strArray)
	if err == nil {
		*e = EmbeddingInput{StringArray: strArray, inputType: EmbeddingInputTypeStringArray}
		return nil
	}

	var intArray []int64

	err = json.Unmarshal(data, &intArray)
	if err == nil {
		*e = EmbeddingInput{IntArray: intArray, inputType: EmbeddingInputTypeIntArray}
		return nil
	}

	var intArrayArray [][]int64

	err = json.Unmarshal(data, &intArrayArray)
	if err == nil {
		*e = EmbeddingInput{IntArrayArray: intArrayArray, inputType: EmbeddingInputTypeIntArrayArray}
		return nil
	}

	return fmt.Errorf("invalid embedding input type")
}

func (e EmbeddingInput) GetType() string {
	if e.inputType != "" {
		return e.inputType
	}

	if e.String != "" {
		return EmbeddingInputTypeString
	}

	if e.StringArray != nil {
		return EmbeddingInputTypeStringArray
	}

	if e.IntArray != nil {
		return EmbeddingInputTypeIntArray
	}

	if e.IntArrayArray != nil {
		return EmbeddingInputTypeIntArrayArray
	}

	return ""
}

// EmbeddingRequest represents the unified embedding request model.
// Based on OpenAI embedding request format for compatibility.
// Note: Common fields like Model are in the parent Request struct, not here.
type EmbeddingRequest struct {
	// Input is the text to embed. Can be string, []string, []int (tokens), or [][]int (multiple token arrays).
	Input EmbeddingInput `json:"input"`

	// Task is the task to embed.
	// For jina embedding, it can be:
	// text-matching
	// retrieval.query
	// retrieval.passag
	// separation
	// classification
	// none
	Task string `json:"task,omitempty"`

	// The format to return the embeddings in. Can be either `float` or
	// [`base64`](https://pypi.org/project/pybase64/).
	//
	// Any of "float", "base64".
	EncodingFormat string `json:"encoding_format,omitempty"`

	// Dimensions is the number of dimensions for the output embeddings.
	Dimensions *int `json:"dimensions,omitempty"`

	// User is a unique identifier for the end-user.
	User string `json:"user,omitempty"`
}

// EmbeddingResponse represents the unified embedding response model.
// Note: Common fields like Usage are in the parent Response struct, not here.
type EmbeddingResponse struct {
	// ID is the response identifier (some providers return this).
	ID string `json:"id,omitempty"`

	// Object is the object type, typically "list".
	Object string `json:"object"`

	// Data contains the embedding results.
	Data []EmbeddingData `json:"data"`
}

// EmbeddingData represents a single embedding result.
type EmbeddingData struct {
	// Object is the object type, typically "embedding".
	Object string `json:"object"`

	// Embedding is the embedding vector. Can be []float64 or base64 encoded string.
	Embedding Embedding `json:"embedding"`

	// Index is the index of the input this embedding corresponds to.
	Index int `json:"index"`
}

type Embedding struct {
	Embedding []float64 `json:"embedding,omitempty"`
	Base64    string    `json:"base64,omitempty"`
	valueType string
}

func (e Embedding) MarshalJSON() ([]byte, error) {
	if e.valueType == "float_array" || e.Embedding != nil {
		return json.Marshal(e.Embedding)
	}

	if e.valueType == "base64" || e.Base64 != "" {
		return json.Marshal(e.Base64)
	}

	return json.Marshal([]float64{})
}

func (e *Embedding) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("invalid embedding type")
	}

	var str string

	err := json.Unmarshal(data, &str)
	if err == nil {
		*e = Embedding{Base64: str, valueType: "base64"}
		return nil
	}

	var floatArray []float64

	err = json.Unmarshal(data, &floatArray)
	if err == nil {
		*e = Embedding{Embedding: floatArray, valueType: "float_array"}
		return nil
	}

	return fmt.Errorf("invalid embedding type")
}

package responses

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ValidateCompactionTriggerPlacement enforces the Responses remote-compaction
// wire invariant after all canonical encoding and raw sidecar merging. Requests
// without a compaction trigger are unaffected.
func ValidateCompactionTriggerPlacement(body []byte) error {
	var envelope struct {
		Input json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return fmt.Errorf("decode Responses request: %w", err)
	}
	input := bytes.TrimSpace(envelope.Input)
	if len(input) == 0 || bytes.Equal(input, []byte("null")) || input[0] != '[' {
		return nil
	}
	var items []struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(input, &items); err != nil {
		return fmt.Errorf("decode Responses input: %w", err)
	}
	triggerIndex := -1
	for index := range items {
		if items[index].Type != "compaction_trigger" {
			continue
		}
		if triggerIndex >= 0 {
			return fmt.Errorf("Responses input contains multiple compaction_trigger items")
		}
		triggerIndex = index
	}
	if triggerIndex >= 0 && triggerIndex != len(items)-1 {
		return fmt.Errorf("Responses compaction_trigger must be the final input item")
	}
	return nil
}

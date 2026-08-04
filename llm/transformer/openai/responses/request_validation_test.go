package responses

import (
	"strings"
	"testing"
)

func TestValidateCompactionTriggerPlacementBranches(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{name: "invalid envelope", body: `{`, wantErr: "decode Responses request"},
		{name: "input absent", body: `{}`},
		{name: "input null", body: `{"input":null}`},
		{name: "input text", body: `{"input":"hello"}`},
		{name: "input object", body: `{"input":{"type":"message"}}`},
		{name: "invalid array item", body: `{"input":[1]}`, wantErr: "decode Responses input"},
		{name: "no trigger", body: `{"input":[{"type":"message"}]}`},
		{name: "single final trigger", body: `{"input":[{"type":"message"},{"type":"compaction_trigger"}]}`},
		{name: "trigger not final", body: `{"input":[{"type":"compaction_trigger"},{"type":"message"}]}`, wantErr: "must be the final input item"},
		{name: "duplicate trigger", body: `{"input":[{"type":"compaction_trigger"},{"type":"compaction_trigger"}]}`, wantErr: "multiple compaction_trigger"},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := ValidateCompactionTriggerPlacement([]byte(test.body))
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("ValidateCompactionTriggerPlacement() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("ValidateCompactionTriggerPlacement() error = %v, want containing %q", err, test.wantErr)
			}
		})
	}
}

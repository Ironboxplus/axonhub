package responses

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestNormalizeAndValidateResponsesRequestBodyProfilesAndUnions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		profile    ResponsesWireProfile
		body       string
		wantCode   string
		wantPath   string
		wantRepair *WireRepair
	}{
		{
			name:    "standard top level function keeps optional description profile",
			profile: ResponsesWireProfileStandard,
			body:    `{"tools":[{"type":"function","name":"lookup","parameters":{}}]}`,
		},
		{
			name:       "lite additional web search gets deterministic description",
			profile:    ResponsesWireProfileLite,
			body:       `{"input":[{"type":"additional_tools","tools":[{"type":"web_search","external_web_access":true}]}]}`,
			wantRepair: &WireRepair{Code: "synthesized_tool_description", Path: "input[0].tools[0].description", ObjectType: "web_search"},
		},
		{
			name:       "lite additional namespace gets deterministic description",
			profile:    ResponsesWireProfileLite,
			body:       `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","tools":[{"type":"function","name":"exec","description":"Run.","parameters":{}}]}]}]}`,
			wantRepair: &WireRepair{Code: "synthesized_tool_description", Path: "input[0].tools[0].description", ObjectType: "namespace"},
		},
		{
			name:     "lite additional function missing description rejects",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"function","name":"exec","parameters":{}}]}]}`,
			wantCode: "missing_tool_description", wantPath: "input[0].tools[0].description",
		},
		{
			name:     "namespace child custom missing description rejects",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","description":"Apps.","tools":[{"type":"custom","name":"exec"}]}]}]}`,
			wantCode: "missing_tool_description", wantPath: "input[0].tools[0].tools[0].description",
		},
		{
			name:     "function parameters missing",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tools":[{"type":"function","name":"lookup","description":"Lookup."}]}`,
			wantCode: "missing_function_parameters", wantPath: "tools[0].parameters",
		},
		{
			name:     "function parameters null",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tools":[{"type":"function","name":"lookup","description":"Lookup.","parameters":null}]}`,
			wantCode: "missing_function_parameters", wantPath: "tools[0].parameters",
		},
		{
			name:    "function parameters empty object remains explicit",
			profile: ResponsesWireProfileStandard,
			body:    `{"tools":[{"type":"function","name":"lookup","description":"Lookup.","parameters":{}}]}`,
		},
		{
			name:     "function parameters non object",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tools":[{"type":"function","name":"lookup","description":"Lookup.","parameters":[]}]}`,
			wantCode: "invalid_function_parameters", wantPath: "tools[0].parameters",
		},
		{
			name:     "unknown tool without type",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tools":[{"name":"future","description":"Future."}]}`,
			wantCode: "missing_tool_type", wantPath: "tools[0].type",
		},
		{
			name:     "namespace child missing type",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","description":"Apps.","tools":[{"name":"exec","description":"Exec."}]}]}]}`,
			wantCode: "missing_tool_type", wantPath: "input[0].tools[0].tools[0].type",
		},
		{
			name:     "namespace duplicate child identity",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","description":"Apps.","tools":[{"type":"function","name":"exec","description":"Exec.","parameters":{}},{"type":"function","name":"exec","description":"Exec again.","parameters":{}}]}]}]}`,
			wantCode: "duplicate_tool_identity", wantPath: "input[0].tools[0].tools[1].name",
		},
		{
			name:     "empty additional declaration",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[]}]}`,
			wantCode: "empty_tool_declaration", wantPath: "input[0].tools",
		},
		{
			name:     "function choice requires name",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tool_choice":{"type":"function"}}`,
			wantCode: "invalid_tool_choice", wantPath: "tool_choice.name",
		},
		{
			name:    "built in choice needs no name",
			profile: ResponsesWireProfileStandard,
			body:    `{"tool_choice":{"type":"web_search"}}`,
		},
		{
			name:     "allowed tools option requires function name",
			profile:  ResponsesWireProfileStandard,
			body:     `{"tool_choice":{"type":"allowed_tools","mode":"auto","tools":[{"type":"function"}]}}`,
			wantCode: "invalid_tool_choice", wantPath: "tool_choice.tools[0].name",
		},
		{
			name:     "compaction trigger is final only",
			profile:  ResponsesWireProfileStandard,
			body:     `{"input":[{"type":"compaction_trigger"},{"type":"message","role":"user"}]}`,
			wantCode: "invalid_compaction_placement", wantPath: "input[0]",
		},
	}

	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			normalized, report, err := NormalizeAndValidateResponsesRequestBody([]byte(test.body), test.profile)
			if test.wantCode != "" {
				if err == nil {
					t.Fatalf("NormalizeAndValidateResponsesRequestBody() succeeded, want %s", test.wantCode)
				}
				var wireErr *ResponsesWireValidationError
				if !errors.As(err, &wireErr) {
					t.Fatalf("error type = %T (%v), want *ResponsesWireValidationError", err, err)
				}
				if wireErr.Code != test.wantCode || wireErr.Path != test.wantPath || wireErr.ObjectType == "" || wireErr.Repairable {
					t.Fatalf("wire error = %#v, want code=%q path=%q non-empty type and fail-closed", wireErr, test.wantCode, test.wantPath)
				}
				return
			}
			if err != nil {
				t.Fatalf("NormalizeAndValidateResponsesRequestBody() error = %v", err)
			}
			if !json.Valid(normalized) {
				t.Fatalf("normalized body is invalid JSON: %s", normalized)
			}
			if test.wantRepair == nil {
				if report == nil || len(report.Repairs) != 0 {
					t.Fatalf("repair report = %#v, want no repairs", report)
				}
				return
			}
			if report == nil || len(report.Repairs) != 1 || report.Repairs[0] != *test.wantRepair {
				t.Fatalf("repair report = %#v, want %#v", report, test.wantRepair)
			}
			var body map[string]any
			if err := json.Unmarshal(normalized, &body); err != nil {
				t.Fatalf("decode normalized body: %v", err)
			}
			if !containsDescription(body, test.wantRepair.Path) {
				t.Fatalf("repair was reported but description is absent: %s", normalized)
			}
		})
	}
}

func TestNormalizeAndValidateResponsesRequestBodyTopLevelBranches(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		profile ResponsesWireProfile
		body    string
		code    string
	}{
		{name: "unsupported profile", profile: "future", body: `{}`, code: "unsupported_wire_profile"},
		{name: "lite invalid JSON", profile: ResponsesWireProfileLite, body: `{`, code: "invalid_request_body"},
		{name: "standard invalid JSON", profile: ResponsesWireProfileStandard, body: `{`, code: "invalid_request_body"},
		{name: "lite top tools invalid", profile: ResponsesWireProfileLite, body: `{"tools":{}}`, code: "invalid_tool_array"},
		{name: "lite top tool normalizes", profile: ResponsesWireProfileLite, body: `{"tools":[{"type":"web_search"}]}`},
		{name: "lite input invalid", profile: ResponsesWireProfileLite, body: `{"input":[1]}`, code: "invalid_input_item"},
		{name: "lite choice invalid", profile: ResponsesWireProfileLite, body: `{"tool_choice":false}`, code: "invalid_tool_choice"},
		{name: "lite no union members", profile: ResponsesWireProfileLite, body: `{"model":"fixture"}`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			_, _, err := NormalizeAndValidateResponsesRequestBody([]byte(test.body), test.profile)
			if test.code == "" {
				if err != nil {
					t.Fatalf("NormalizeAndValidateResponsesRequestBody() error = %v", err)
				}
				return
			}
			var wireErr *ResponsesWireValidationError
			if !errors.As(err, &wireErr) || wireErr.Code != test.code {
				t.Fatalf("wire error = %#v, want code %q", wireErr, test.code)
			}
		})
	}
	if got := (&ResponsesWireValidationError{Code: "code", Path: "path", ObjectType: "tool"}).Error(); got == "" {
		t.Fatal("typed wire error formatted an empty message")
	}
	if got := (&ResponsesWireValidationError{Code: "code", Path: "path", ObjectType: "tool", message: "detail"}).Error(); got == "" {
		t.Fatal("typed wire error with detail formatted an empty message")
	}
	var nilWireErr *ResponsesWireValidationError
	if got := nilWireErr.Error(); got != "" {
		t.Fatalf("nil typed wire error = %q, want empty", got)
	}
}

func containsDescription(body map[string]any, path string) bool {
	input, _ := body["input"].([]any)
	if len(input) == 0 {
		return false
	}
	item, _ := input[0].(map[string]any)
	tools, _ := item["tools"].([]any)
	if len(tools) == 0 {
		return false
	}
	tool, _ := tools[0].(map[string]any)
	description, _ := tool["description"].(string)
	return description != "" && path != ""
}

package responses

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer"
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
			name:     "lite namespace child custom rejects before description inference",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","description":"Apps.","tools":[{"type":"custom","name":"exec"}]}]}]}`,
			wantCode: "responses_lite_namespace_child_must_be_function", wantPath: "input[0].tools[0].tools[0].type",
		},
		{
			name:     "lite namespace custom with description still rejects before provider dispatch",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"additional_tools","tools":[{"type":"namespace","name":"apps","description":"Apps.","tools":[{"type":"custom","name":"exec","description":"Execute."}]}]}]}`,
			wantCode: "responses_lite_namespace_child_must_be_function", wantPath: "input[0].tools[0].tools[0].type",
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
		{
			name:     "agent message requires routing boundaries",
			profile:  ResponsesWireProfileStandard,
			body:     `{"input":[{"type":"agent_message","author":"/root","content":[{"type":"input_text","text":"handoff"}]}]}`,
			wantCode: "invalid_agent_message", wantPath: "input[0].recipient",
		},
		{
			name:     "agent message rejects mixed secret and plaintext fields",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"visible","encrypted_content":"PRIVATE_AGENT_FRAGMENT"}]}]}`,
			wantCode: "invalid_agent_message_content", wantPath: "input[0].content[0]",
		},
		{
			name:     "agent message rejects status",
			profile:  ResponsesWireProfileStandard,
			body:     `{"input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","status":"completed","content":[{"type":"input_text","text":"handoff"}]}]}`,
			wantCode: "unsupported_item_status", wantPath: "input[0].status",
		},
		{
			name:    "agent message permits typed encrypted content on identity wire",
			profile: ResponsesWireProfileStandard,
			body:    `{"input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"encrypted_content","encrypted_content":"opaque"}]}]}`,
		},
		{
			name:    "standard preserves future agent content discriminator",
			profile: ResponsesWireProfileStandard,
			body:    `{"input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"future_agent_content","future":"opaque"}]}]}`,
		},
		{
			name:     "lite rejects future agent content discriminator",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"future_agent_content","future":"opaque"}]}]}`,
			wantCode: "invalid_agent_message_content", wantPath: "input[0].content[0].type",
		},
		{
			name:    "context compaction permits absent opaque checkpoint",
			profile: ResponsesWireProfileStandard,
			body:    `{"input":[{"id":"ctx_1","type":"context_compaction"}]}`,
		},
		{
			name:    "context compaction preserves null checkpoint",
			profile: ResponsesWireProfileLite,
			body:    `{"input":[{"id":"ctx_1","type":"context_compaction","encrypted_content":null}]}`,
		},
		{
			name:     "context compaction rejects non string checkpoint",
			profile:  ResponsesWireProfileStandard,
			body:     `{"input":[{"id":"ctx_1","type":"context_compaction","encrypted_content":{"private":"opaque"}}]}`,
			wantCode: "invalid_context_compaction", wantPath: "input[0].encrypted_content",
		},
		{
			name:     "context compaction rejects null status",
			profile:  ResponsesWireProfileLite,
			body:     `{"input":[{"type":"context_compaction","status":null}]}`,
			wantCode: "unsupported_item_status", wantPath: "input[0].status",
		},
		{
			name:     "compaction rejects status",
			profile:  ResponsesWireProfileStandard,
			body:     `{"input":[{"type":"compaction","encrypted_content":"opaque","status":"completed"}]}`,
			wantCode: "unsupported_item_status", wantPath: "input[0].status",
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

func TestValidateResponsesAgentMessageWireUnionBoundaries(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		author string
		to     string
		body   string
		code   string
		path   string
	}{
		{name: "blank author", to: "/root/worker", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message", path: "input[7].author"},
		{name: "newline recipient", author: "/root", to: "/root\nworker", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message", path: "input[7].recipient"},
		{name: "uppercase path", author: "/root/Worker", to: "/root/worker", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message_path", path: "input[7].author"},
		{name: "trailing slash", author: "/root", to: "/root/worker/", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message_path", path: "input[7].recipient"},
		{name: "root child forbidden", author: "/root/root", to: "/morpheus", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message_path", path: "input[7].author"},
		{name: "path traversal forbidden", author: "/root/..", to: "/root/worker", body: `[{"type":"input_text","text":"visible"}]`, code: "invalid_agent_message_path", path: "input[7].author"},
		{name: "morpheus and root accepted", author: "/morpheus", to: "/root", body: `[{"type":"input_text","text":"visible"}]`},
		{name: "missing content", author: "/root", to: "/root/worker", code: "invalid_agent_message", path: "input[7].content"},
		{name: "null content", author: "/root", to: "/root/worker", body: `null`, code: "invalid_agent_message", path: "input[7].content"},
		{name: "empty content", author: "/root", to: "/root/worker", body: `[]`, code: "invalid_agent_message", path: "input[7].content"},
		{name: "non-object part", author: "/root", to: "/root/worker", body: `[1]`, code: "invalid_agent_message_content", path: "input[7].content[0]"},
		{name: "missing part type", author: "/root", to: "/root/worker", body: `[{}]`, code: "invalid_agent_message_content", path: "input[7].content[0]"},
		{name: "missing input text", author: "/root", to: "/root/worker", body: `[{"type":"input_text"}]`, code: "invalid_agent_message_content", path: "input[7].content[0]"},
		{name: "encrypted blank", author: "/root", to: "/root/worker", body: `[{"type":"encrypted_content","encrypted_content":" "}]`, code: "invalid_agent_message_content", path: "input[7].content[0]"},
		{name: "future part is reserved for standard opaque identity", author: "/root", to: "/root/worker", body: `[{"type":"future_agent_content","payload":"opaque"}]`},
		{name: "future before malformed known member", author: "/root", to: "/root/worker", body: `[{"type":"future_agent_content"},{"type":"input_text","text":"visible","encrypted_content":"opaque"}]`, code: "invalid_agent_message_content", path: "input[7].content[1]"},
		{name: "malformed known member before future", author: "/root", to: "/root/worker", body: `[{"type":"input_text","text":"visible","encrypted_content":"opaque"},{"type":"future_agent_content"}]`, code: "invalid_agent_message_content", path: "input[7].content[0]"},
		{name: "typed residual survives", author: "/root", to: "/root/worker", body: `[{"type":"input_text","text":"visible","future":true},{"type":"encrypted_content","encrypted_content":"opaque","future":true}]`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			err := validateResponsesAgentMessageFields(test.author, test.to, json.RawMessage(test.body), "input[7]")
			if test.code == "" {
				if err != nil {
					t.Fatalf("validate typed agent message: %v", err)
				}
				return
			}
			var wireErr *ResponsesWireValidationError
			if !errors.As(err, &wireErr) || wireErr.Code != test.code || wireErr.Path != test.path || wireErr.Repairable {
				t.Fatalf("agent wire error = %#v err=%v", wireErr, err)
			}
		})
	}

	if err := validateResponsesAgentMessage(map[string]json.RawMessage{
		"author": json.RawMessage(`"/root"`), "recipient": json.RawMessage(`"/root/worker"`),
		"content": json.RawMessage(`[{"type":"input_text","text":"visible"}]`),
	}, "input[2]"); err != nil {
		t.Fatalf("lite agent-message validator: %v", err)
	}
	if err := validateResponsesAgentMessage(map[string]json.RawMessage{}, "input[2]"); err == nil {
		t.Fatal("missing lite agent-message fields were accepted")
	}
}

func TestValidateResponsesContextCompactionWireUnion(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name string
		raw  string
		code string
	}{
		{name: "absent"},
		{name: "null", raw: `null`},
		{name: "opaque string", raw: `"opaque"`},
		{name: "object", raw: `{}`, code: "invalid_context_compaction"},
		{name: "number", raw: `1`, code: "invalid_context_compaction"},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			err := validateResponsesContextCompaction(json.RawMessage(test.raw), "input[3]")
			if test.code == "" {
				if err != nil {
					t.Fatalf("context union error=%v", err)
				}
				return
			}
			var wireErr *ResponsesWireValidationError
			if !errors.As(err, &wireErr) || wireErr.Code != test.code || wireErr.Path != "input[3].encrypted_content" || wireErr.Repairable {
				t.Fatalf("context union error=%#v raw=%s", wireErr, test.raw)
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

func TestResponsesWireValidationErrorSurvivesOutboundInvalidRequestWrapping(t *testing.T) {
	t.Parallel()
	if got := (*ResponsesWireValidationError)(nil).SafeDiagnostic(); got != (llm.ErrorDiagnostic{}) {
		t.Fatalf("nil wire diagnostic = %#v", got)
	}

	wireErr := &ResponsesWireValidationError{
		Code: "missing_tool_description", Path: "input[0].tools[0].description", ObjectType: "function",
	}
	wrapped := fmt.Errorf("%w: %w", transformer.ErrInvalidRequest, wireErr)
	var typed *ResponsesWireValidationError
	if !errors.As(wrapped, &typed) || typed != wireErr {
		t.Fatalf("wrapped error did not preserve Responses wire type: %T %v", wrapped, wrapped)
	}
	diagnostic := llm.ErrorDiagnosticFrom(wrapped)
	if diagnostic == nil || diagnostic.Component != "outbound_wire_validation" ||
		diagnostic.Code != wireErr.Code || diagnostic.StatusCode != 400 ||
		diagnostic.Message != "Outbound Responses wire validation blocked before provider dispatch." {
		t.Fatalf("safe wire diagnostic = %#v", diagnostic)
	}
}

func TestResponsesIngressValidationUsesRequestParseDiagnostic(t *testing.T) {
	t.Parallel()
	err := ValidateResponsesIngressRequestBody(
		[]byte(`{"model":"fixture","input":[{"type":"agent_message","author":"/root/Worker","recipient":"/root/worker","content":[{"type":"input_text","text":"PRIVATE_INGRESS"}]}]}`),
		ResponsesWireProfileStandard,
	)
	if err == nil || strings.Contains(err.Error(), "PRIVATE_INGRESS") {
		t.Fatalf("ingress error=%v", err)
	}
	var wireErr *ResponsesWireValidationError
	if !errors.As(err, &wireErr) || wireErr.Code != "invalid_agent_message_path" || wireErr.Path != "input[0].author" {
		t.Fatalf("ingress wire error=%#v err=%v", wireErr, err)
	}
	diagnostic := llm.ErrorDiagnosticFrom(err)
	if diagnostic == nil || diagnostic.Component != "inbound_wire_validation" || diagnostic.Code != "invalid_agent_message_path" ||
		diagnostic.Message != "Inbound Responses request validation blocked before provider dispatch." || diagnostic.StatusCode != 400 {
		t.Fatalf("ingress safe diagnostic=%#v", diagnostic)
	}
}

func TestValidateParsedResponsesIngressRequestKeepsIngressThreeStatePolicy(t *testing.T) {
	t.Parallel()
	decode := func(t *testing.T, body string) *Request {
		t.Helper()
		var request Request
		if err := json.Unmarshal([]byte(body), &request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		return &request
	}

	validAgent := `{"model":"fixture","input":[{"type":"agent_message","author":"/root","recipient":"/morpheus","content":[{"type":"input_text","text":"visible"},{"type":"encrypted_content","encrypted_content":"opaque"}]}]}`
	futureInvalidOuter := `{"model":"fixture","input":[{"type":"agent_message","author":"/root/Worker","recipient":"/root/..","content":[{"type":"future_agent_content","opaque":true}]}]}`
	for _, test := range []struct {
		name    string
		request *Request
		profile ResponsesWireProfile
		code    string
		path    string
	}{
		{name: "nil is empty", request: nil, profile: ResponsesWireProfileStandard},
		{name: "scalar input needs no item scan", request: decode(t, `{"model":"fixture","input":"ordinary"}`), profile: ResponsesWireProfileStandard},
		{name: "ordinary item remains cheap", request: decode(t, `{"model":"fixture","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"ordinary"}]}]}`), profile: ResponsesWireProfileStandard},
		{name: "known valid typed content", request: decode(t, validAgent), profile: ResponsesWireProfileStandard},
		{name: "known malformed input text conflict", request: decode(t, `{"model":"fixture","input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"visible","encrypted_content":"opaque"}]}]}`), profile: ResponsesWireProfileStandard, code: "invalid_agent_message_content", path: "input[0].content[0]"},
		{name: "known malformed encrypted blank", request: decode(t, `{"model":"fixture","input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"encrypted_content","encrypted_content":" "}]}]}`), profile: ResponsesWireProfileStandard, code: "invalid_agent_message_content", path: "input[0].content[0]"},
		{name: "missing child discriminator is malformed", request: decode(t, `{"model":"fixture","input":[{"type":"agent_message","author":"/root","recipient":"/root/worker","content":[{"text":"visible"}]}]}`), profile: ResponsesWireProfileStandard, code: "invalid_agent_message_content", path: "input[0].content[0]"},
		{name: "future standard bypasses current outer path", request: decode(t, futureInvalidOuter), profile: ResponsesWireProfileStandard},
		{name: "future lite remains closed", request: decode(t, futureInvalidOuter), profile: ResponsesWireProfileLite, code: "invalid_agent_message_content", path: "input[0].content[0].type"},
		{name: "known invalid path still blocks", request: decode(t, strings.Replace(validAgent, `"/morpheus"`, `"/root/Worker"`, 1)), profile: ResponsesWireProfileStandard, code: "invalid_agent_message_path", path: "input[0].recipient"},
		{name: "known blank path blocks before grammar", request: decode(t, `{"model":"fixture","input":[{"type":"agent_message","author":" ","recipient":"/root/worker","content":[{"type":"input_text","text":"visible"}]}]}`), profile: ResponsesWireProfileStandard, code: "invalid_agent_message", path: "input[0].author"},
		{name: "typed status null still blocks", request: decode(t, `{"model":"fixture","input":[{"type":"agent_message","status":null,"author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"visible"}]}]}`), profile: ResponsesWireProfileStandard, code: "unsupported_item_status", path: "input[0].status"},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			err := validateParsedResponsesIngressRequest(test.request, test.profile)
			if test.code == "" {
				if err != nil {
					t.Fatalf("parsed ingress error=%v", err)
				}
				return
			}
			var wireErr *ResponsesWireValidationError
			if !errors.As(err, &wireErr) || wireErr.Code != test.code || wireErr.Path != test.path {
				t.Fatalf("parsed ingress wire error=%#v err=%v", wireErr, err)
			}
		})
	}
	if err := validateParsedResponsesAgentMessage(nil, "input[0]", agentMessageContentOpaqueFuture); err == nil {
		t.Fatal("nil parsed agent message was accepted")
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

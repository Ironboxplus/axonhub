package responses

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/looplj/axonhub/llm"
)

// ResponsesWireProfile selects the target Responses wire contract. The Lite
// profile applies the Codex additional_tools compatibility rules without
// leaking those provider-specific required fields into canonical models.
type ResponsesWireProfile string

const (
	ResponsesWireProfileStandard ResponsesWireProfile = "responses"
	ResponsesWireProfileLite     ResponsesWireProfile = "responses_lite"
)

// WireRepair records a deterministic, provider-profile compatibility repair.
// It contains structure only; it deliberately never records a sensitive value.
type WireRepair struct {
	Code       string `json:"code"`
	Path       string `json:"path"`
	ObjectType string `json:"object_type"`
}

// ResponsesWireRepairReport describes every deterministic repair applied to a
// final Responses request body.
type ResponsesWireRepairReport struct {
	Repairs []WireRepair `json:"repairs"`
}

// ResponsesWireValidationError is the stable, machine-readable failure
// returned when a final Responses wire body cannot be sent safely.
type ResponsesWireValidationError struct {
	Code       string `json:"code"`
	Path       string `json:"path"`
	ObjectType string `json:"object_type"`
	Repairable bool   `json:"repairable"`
	message    string
}

func (err *ResponsesWireValidationError) Error() string {
	if err == nil {
		return ""
	}
	if err.message != "" {
		return fmt.Sprintf("Responses wire validation %s at %s (%s): %s", err.Code, err.Path, err.ObjectType, err.message)
	}
	return fmt.Sprintf("Responses wire validation %s at %s (%s)", err.Code, err.Path, err.ObjectType)
}

// SafeDiagnostic exposes a fixed, payload-free classification for a final
// wire rejection. The code and JSON path stay in WireEvidence; no body value,
// header, credential, or provider message crosses the processing-error
// boundary.
func (err *ResponsesWireValidationError) SafeDiagnostic() llm.ErrorDiagnostic {
	if err == nil || err.Code == "" {
		return llm.ErrorDiagnostic{}
	}
	return llm.ErrorDiagnostic{
		Component:  "outbound_wire_validation",
		Code:       err.Code,
		Message:    "Outbound Responses wire validation blocked before provider dispatch.",
		StatusCode: 400,
	}
}

// NormalizeAndValidateResponsesRequestBody validates the exact final Responses
// body after typed encoding and residual/raw merging. It may only perform the
// narrow deterministic repairs allowed by profile and returns the normalized
// body that must be passed to transport. Callers that mutate raw JSON after an
// Axon transformer can invoke this exported API again immediately before HTTP.
func NormalizeAndValidateResponsesRequestBody(
	body []byte,
	profile ResponsesWireProfile,
) ([]byte, *ResponsesWireRepairReport, error) {
	if profile != ResponsesWireProfileStandard && profile != ResponsesWireProfileLite {
		return nil, nil, wireValidationError("unsupported_wire_profile", "", "request", false, "unknown profile %q", profile)
	}
	report := &ResponsesWireRepairReport{}
	if profile == ResponsesWireProfileStandard {
		if err := validateStandardResponsesWireBody(body); err != nil {
			return nil, report, err
		}
		return body, report, nil
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(body, &envelope); err != nil || envelope == nil {
		return nil, nil, wireValidationError("invalid_request_body", "", "request", false, "must be a JSON object")
	}

	if rawTools, present := envelope["tools"]; present {
		normalized, err := normalizeResponsesToolArray(rawTools, "tools", profile, false, report)
		if err != nil {
			return nil, report, err
		}
		envelope["tools"] = normalized
	}

	if rawInput, present := envelope["input"]; present {
		normalized, err := normalizeResponsesInput(rawInput, profile, report)
		if err != nil {
			return nil, report, err
		}
		envelope["input"] = normalized
	}

	if rawChoice, present := envelope["tool_choice"]; present {
		if err := validateResponsesToolChoice(rawChoice, "tool_choice"); err != nil {
			return nil, report, err
		}
	}

	// envelope values originate from successful JSON parsing or json.Marshal,
	// so this object cannot contain an unsupported Go value or invalid raw JSON.
	normalized, _ := json.Marshal(envelope)
	return normalized, report, nil
}

type standardResponsesWireEnvelope struct {
	Tools      json.RawMessage `json:"tools"`
	Input      json.RawMessage `json:"input"`
	ToolChoice json.RawMessage `json:"tool_choice"`
}

type standardResponsesWireTool struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	ServerLabel string            `json:"server_label"`
	Parameters  json.RawMessage   `json:"parameters"`
	Tools       []json.RawMessage `json:"tools"`
}

func validateStandardResponsesWireBody(body []byte) error {
	var envelope standardResponsesWireEnvelope
	if err := json.Unmarshal(body, &envelope); err != nil {
		return wireValidationError("invalid_request_body", "", "request", false, "must be a JSON object")
	}
	if len(envelope.Tools) > 0 {
		if err := validateStandardResponsesToolArray(envelope.Tools, "tools", false); err != nil {
			return err
		}
	}
	if err := validateStandardResponsesInput(envelope.Input); err != nil {
		return err
	}
	if len(envelope.ToolChoice) > 0 {
		return validateResponsesToolChoice(envelope.ToolChoice, "tool_choice")
	}
	return nil
}

func validateStandardResponsesInput(raw json.RawMessage) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] != '[' {
		return nil
	}
	var items []struct {
		Type  string            `json:"type"`
		Tools []json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return wireValidationError("invalid_input", "input", "input", false, "must be an array")
	}
	triggerIndex := -1
	for index := range items {
		if items[index].Type == "compaction_trigger" {
			if triggerIndex >= 0 {
				return wireValidationError("invalid_compaction_placement", fmt.Sprintf("input[%d]", index), "compaction_trigger", false, "multiple compaction_trigger items")
			}
			triggerIndex = index
		}
		if items[index].Type == "additional_tools" {
			if len(items[index].Tools) == 0 {
				return wireValidationError("empty_tool_declaration", fmt.Sprintf("input[%d].tools", index), "additional_tools", false, "tools must be a non-empty array")
			}
			encoded, err := json.Marshal(items[index].Tools)
			if err != nil {
				return wireValidationError("invalid_tool_array", fmt.Sprintf("input[%d].tools", index), "additional_tools", false, "marshal tools: %v", err)
			}
			if err := validateStandardResponsesToolArray(encoded, fmt.Sprintf("input[%d].tools", index), true); err != nil {
				return err
			}
		}
	}
	if triggerIndex >= 0 && triggerIndex != len(items)-1 {
		return wireValidationError("invalid_compaction_placement", fmt.Sprintf("input[%d]", triggerIndex), "compaction_trigger", false, "must be the final input item")
	}
	return nil
}

func validateStandardResponsesToolArray(raw json.RawMessage, path string, additional bool) error {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		if additional {
			return wireValidationError("empty_tool_declaration", path, "additional_tools", false, "tools must be a non-empty array")
		}
		return wireValidationError("invalid_tool_array", path, "tools", false, "tools must be an array")
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(trimmed, &tools); err != nil {
		return wireValidationError("invalid_tool_array", path, "tools", false, "tools must be an array")
	}
	if additional && len(tools) == 0 {
		return wireValidationError("empty_tool_declaration", path, "additional_tools", false, "tools must be a non-empty array")
	}
	for index := range tools {
		if err := validateStandardResponsesTool(tools[index], fmt.Sprintf("%s[%d]", path, index)); err != nil {
			return err
		}
	}
	return nil
}

func validateStandardResponsesTool(raw json.RawMessage, path string) error {
	var tool standardResponsesWireTool
	if err := json.Unmarshal(raw, &tool); err != nil {
		return wireValidationError("invalid_tool", path, "tool", false, "must be an object")
	}
	if strings.TrimSpace(tool.Type) == "" {
		return wireValidationError("missing_tool_type", path+".type", "tool", false, "type is required")
	}
	switch tool.Type {
	case "function":
		if strings.TrimSpace(tool.Name) == "" {
			return wireValidationError("missing_tool_name", path+".name", tool.Type, false, "name is required")
		}
		if err := validateRawResponsesFunctionParameters(tool.Parameters, len(tool.Parameters) > 0, path); err != nil {
			return err
		}
	case "custom":
		if strings.TrimSpace(tool.Name) == "" {
			return wireValidationError("missing_tool_name", path+".name", tool.Type, false, "name is required")
		}
	case "namespace":
		if strings.TrimSpace(tool.Name) == "" {
			return wireValidationError("missing_tool_name", path+".name", tool.Type, false, "name is required")
		}
		if len(tool.Tools) == 0 {
			return wireValidationError("empty_tool_declaration", path+".tools", tool.Type, false, "namespace tools must be a non-empty array")
		}
		seen := make(map[string]struct{}, len(tool.Tools))
		for index := range tool.Tools {
			if err := validateStandardResponsesTool(tool.Tools[index], fmt.Sprintf("%s.tools[%d]", path, index)); err != nil {
				return err
			}
			var child standardResponsesWireTool
			_ = json.Unmarshal(tool.Tools[index], &child)
			if child.Name == "" {
				continue
			}
			if _, duplicate := seen[child.Name]; duplicate {
				return wireValidationError("duplicate_tool_identity", fmt.Sprintf("%s.tools[%d].name", path, index), tool.Type, false, "duplicate namespace child name")
			}
			seen[child.Name] = struct{}{}
		}
	case "mcp":
		if strings.TrimSpace(tool.ServerLabel) == "" {
			return wireValidationError("missing_mcp_server_label", path+".server_label", tool.Type, false, "server_label is required")
		}
	}
	return nil
}

func normalizeResponsesInput(
	raw json.RawMessage,
	profile ResponsesWireProfile,
	report *ResponsesWireRepairReport,
) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] != '[' {
		return raw, nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(trimmed, &items); err != nil {
		return nil, wireValidationError("invalid_input", "input", "input", false, "must be an array: %v", err)
	}
	triggerIndex := -1
	for index := range items {
		path := fmt.Sprintf("input[%d]", index)
		var item map[string]json.RawMessage
		if err := json.Unmarshal(items[index], &item); err != nil || item == nil {
			return nil, wireValidationError("invalid_input_item", path, "input", false, "must be an object")
		}
		itemType, _ := rawJSONString(item["type"])
		if itemType == "compaction_trigger" {
			if triggerIndex >= 0 {
				return nil, wireValidationError("invalid_compaction_placement", path, "compaction_trigger", false, "multiple compaction_trigger items")
			}
			triggerIndex = index
		}
		if itemType != "additional_tools" {
			continue
		}
		rawTools, present := item["tools"]
		if !present {
			return nil, wireValidationError("empty_tool_declaration", path+".tools", "additional_tools", false, "tools is required")
		}
		normalized, err := normalizeResponsesToolArray(rawTools, path+".tools", profile, true, report)
		if err != nil {
			return nil, err
		}
		item["tools"] = normalized
		encoded, err := json.Marshal(item)
		if err != nil {
			return nil, wireValidationError("invalid_input_item", path, "additional_tools", false, "marshal item: %v", err)
		}
		items[index] = encoded
	}
	if triggerIndex >= 0 && triggerIndex != len(items)-1 {
		return nil, wireValidationError("invalid_compaction_placement", fmt.Sprintf("input[%d]", triggerIndex), "compaction_trigger", false, "must be the final input item")
	}
	normalized, err := json.Marshal(items)
	if err != nil {
		return nil, wireValidationError("invalid_input", "input", "input", false, "marshal input: %v", err)
	}
	return normalized, nil
}

func normalizeResponsesToolArray(
	raw json.RawMessage,
	path string,
	profile ResponsesWireProfile,
	additional bool,
	report *ResponsesWireRepairReport,
) (json.RawMessage, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		if additional {
			return nil, wireValidationError("empty_tool_declaration", path, "additional_tools", false, "tools must be a non-empty array")
		}
		return nil, wireValidationError("invalid_tool_array", path, "tools", false, "tools must be an array")
	}
	var tools []json.RawMessage
	if err := json.Unmarshal(trimmed, &tools); err != nil {
		return nil, wireValidationError("invalid_tool_array", path, "tools", false, "tools must be an array")
	}
	if additional && len(tools) == 0 {
		return nil, wireValidationError("empty_tool_declaration", path, "additional_tools", false, "tools must be a non-empty array")
	}
	for index := range tools {
		normalized, err := normalizeResponsesTool(tools[index], fmt.Sprintf("%s[%d]", path, index), profile, additional, report, false)
		if err != nil {
			return nil, err
		}
		tools[index] = normalized
	}
	normalized, err := json.Marshal(tools)
	if err != nil {
		return nil, wireValidationError("invalid_tool_array", path, "tools", false, "marshal tools: %v", err)
	}
	return normalized, nil
}

func normalizeResponsesTool(
	raw json.RawMessage,
	path string,
	profile ResponsesWireProfile,
	additional bool,
	report *ResponsesWireRepairReport,
	namespaceChild bool,
) (json.RawMessage, error) {
	var tool map[string]json.RawMessage
	if err := json.Unmarshal(raw, &tool); err != nil || tool == nil {
		return nil, wireValidationError("invalid_tool", path, "tool", false, "must be an object")
	}
	toolType, ok := rawJSONString(tool["type"])
	if !ok || strings.TrimSpace(toolType) == "" {
		return nil, wireValidationError("missing_tool_type", path+".type", "tool", false, "type is required")
	}
	if profile == ResponsesWireProfileLite && namespaceChild && toolType != "function" {
		return nil, wireValidationError(
			"responses_lite_namespace_child_must_be_function", path+".type", toolType, false,
			"Responses Lite namespace tools can only contain function tools",
		)
	}

	switch toolType {
	case "function":
		if err := requireResponsesToolText(tool, "name", path, toolType); err != nil {
			return nil, err
		}
		if err := requireResponsesToolDescription(tool, path, toolType, profile, additional, report); err != nil {
			return nil, err
		}
		if err := validateResponsesFunctionParameters(tool, path); err != nil {
			return nil, err
		}
	case "custom":
		if err := requireResponsesToolText(tool, "name", path, toolType); err != nil {
			return nil, err
		}
		if err := requireResponsesToolDescription(tool, path, toolType, profile, additional, report); err != nil {
			return nil, err
		}
	case "namespace":
		if _, err := requiredResponsesToolText(tool, "name", path, toolType); err != nil {
			return nil, err
		}
		if err := requireResponsesToolDescription(tool, path, toolType, profile, additional, report); err != nil {
			return nil, err
		}
		rawChildren, present := tool["tools"]
		if !present {
			return nil, wireValidationError("empty_tool_declaration", path+".tools", toolType, false, "namespace tools is required")
		}
		var childItems []json.RawMessage
		if err := json.Unmarshal(rawChildren, &childItems); err != nil || len(childItems) == 0 {
			return nil, wireValidationError("empty_tool_declaration", path+".tools", toolType, false, "namespace tools must be a non-empty array")
		}
		seenNames := make(map[string]struct{}, len(childItems))
		for index := range childItems {
			normalized, childErr := normalizeResponsesTool(childItems[index], fmt.Sprintf("%s.tools[%d]", path, index), profile, additional, report, true)
			if childErr != nil {
				return nil, childErr
			}
			childItems[index] = normalized
			var child map[string]json.RawMessage
			_ = json.Unmarshal(normalized, &child)
			childName, _ := rawJSONString(child["name"])
			if _, duplicate := seenNames[childName]; duplicate {
				return nil, wireValidationError("duplicate_tool_identity", fmt.Sprintf("%s.tools[%d].name", path, index), toolType, false, "duplicate namespace child name")
			}
			seenNames[childName] = struct{}{}
		}
		encodedChildren, err := json.Marshal(childItems)
		if err != nil {
			return nil, wireValidationError("invalid_tool", path+".tools", toolType, false, "marshal namespace children: %v", err)
		}
		tool["tools"] = encodedChildren
	case "mcp":
		if err := requireResponsesToolText(tool, "server_label", path, toolType); err != nil {
			return nil, err
		}
	default:
		if profile == ResponsesWireProfileLite && additional && toolType == "web_search" {
			if err := requireResponsesToolDescription(tool, path, toolType, profile, additional, report); err != nil {
				return nil, err
			}
		}
	}
	encoded, err := json.Marshal(tool)
	if err != nil {
		return nil, wireValidationError("invalid_tool", path, toolType, false, "marshal tool: %v", err)
	}
	return encoded, nil
}

func requireResponsesToolText(object map[string]json.RawMessage, field, path, objectType string) error {
	_, err := requiredResponsesToolText(object, field, path, objectType)
	return err
}

func requiredResponsesToolText(object map[string]json.RawMessage, field, path, objectType string) (string, error) {
	value, ok := rawJSONString(object[field])
	if !ok || strings.TrimSpace(value) == "" {
		code := "missing_tool_" + field
		if field == "server_label" {
			code = "missing_mcp_server_label"
		}
		return "", wireValidationError(code, path+"."+field, objectType, false, "%s is required", field)
	}
	return value, nil
}

func requireResponsesToolDescription(
	object map[string]json.RawMessage,
	path, objectType string,
	profile ResponsesWireProfile,
	additional bool,
	report *ResponsesWireRepairReport,
) error {
	description, ok := rawJSONString(object["description"])
	if ok && strings.TrimSpace(description) != "" {
		return nil
	}
	if profile != ResponsesWireProfileLite || !additional {
		// Standard Responses tools retain their profile's optional-description
		// behavior. Lite additional_tools is the strict provider contract that
		// requires a non-empty description for every recursive declaration.
		return nil
	}
	if profile == ResponsesWireProfileLite && additional {
		switch objectType {
		case "web_search":
			object["description"] = json.RawMessage(`"Search the web for up-to-date information."`)
		case "namespace":
			name, nameOK := rawJSONString(object["name"])
			if !nameOK || strings.TrimSpace(name) == "" {
				return wireValidationError("missing_tool_name", path+".name", objectType, false, "name is required before description can be synthesized")
			}
			object["description"] = json.RawMessage(fmt.Sprintf("%q", "Tools in the "+name+" namespace."))
		default:
			return wireValidationError("missing_tool_description", path+".description", objectType, false, "description is required and cannot be safely inferred")
		}
		report.Repairs = append(report.Repairs, WireRepair{
			Code: "synthesized_tool_description", Path: path + ".description", ObjectType: objectType,
		})
		return nil
	}
	return wireValidationError("missing_tool_description", path+".description", objectType, false, "description is required")
}

func validateResponsesFunctionParameters(object map[string]json.RawMessage, path string) error {
	raw, present := object["parameters"]
	return validateRawResponsesFunctionParameters(raw, present, path)
}

func validateRawResponsesFunctionParameters(raw json.RawMessage, present bool, path string) error {
	trimmed := bytes.TrimSpace(raw)
	if !present || len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return wireValidationError("missing_function_parameters", path+".parameters", "function", false, "parameters is required and must be an object")
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &schema); err != nil || schema == nil {
		return wireValidationError("invalid_function_parameters", path+".parameters", "function", false, "parameters must be a JSON object")
	}
	return nil
}

func validateResponsesToolChoice(raw json.RawMessage, path string) error {
	trimmed := bytes.TrimSpace(raw)
	var mode string
	if err := json.Unmarshal(trimmed, &mode); err == nil {
		if mode == "auto" || mode == "none" || mode == "required" {
			return nil
		}
		return wireValidationError("invalid_tool_choice", path, "tool_choice", false, "unsupported mode")
	}
	var choice map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &choice); err != nil || choice == nil {
		return wireValidationError("invalid_tool_choice", path, "tool_choice", false, "must be a string or object")
	}
	choiceType, ok := rawJSONString(choice["type"])
	if !ok || strings.TrimSpace(choiceType) == "" {
		return wireValidationError("invalid_tool_choice", path+".type", "tool_choice", false, "type is required")
	}
	if choiceType == "allowed_tools" {
		mode, modeOK := rawJSONString(choice["mode"])
		if !modeOK || (mode != "auto" && mode != "required") {
			return wireValidationError("invalid_tool_choice", path+".mode", "allowed_tools", false, "mode must be auto or required")
		}
		var options []map[string]json.RawMessage
		if err := json.Unmarshal(choice["tools"], &options); err != nil || len(options) == 0 {
			return wireValidationError("invalid_tool_choice", path+".tools", "allowed_tools", false, "tools must be a non-empty array")
		}
		for index := range options {
			optionType, optionOK := rawJSONString(options[index]["type"])
			if !optionOK || strings.TrimSpace(optionType) == "" {
				return wireValidationError("invalid_tool_choice", fmt.Sprintf("%s.tools[%d].type", path, index), "allowed_tools", false, "type is required")
			}
			if optionType == "function" || optionType == "custom" {
				if _, err := requiredResponsesToolText(options[index], "name", fmt.Sprintf("%s.tools[%d]", path, index), "allowed_tools"); err != nil {
					return wireValidationError("invalid_tool_choice", fmt.Sprintf("%s.tools[%d].name", path, index), "allowed_tools", false, "name is required")
				}
			}
			if optionType == "mcp" {
				if _, err := requiredResponsesToolText(options[index], "server_label", fmt.Sprintf("%s.tools[%d]", path, index), "allowed_tools"); err != nil {
					return wireValidationError("invalid_tool_choice", fmt.Sprintf("%s.tools[%d].server_label", path, index), "allowed_tools", false, "server_label is required")
				}
			}
		}
		return nil
	}
	if choiceType == "function" || choiceType == "custom" {
		if _, err := requiredResponsesToolText(choice, "name", path, "tool_choice"); err != nil {
			return wireValidationError("invalid_tool_choice", path+".name", "tool_choice", false, "name is required")
		}
	}
	if _, hasMode := choice["mode"]; hasMode {
		return wireValidationError("invalid_tool_choice", path+".mode", "tool_choice", false, "mode cannot be combined with a named tool choice")
	}
	return nil
}

func rawJSONString(raw json.RawMessage) (string, bool) {
	var value string
	if len(raw) == 0 || json.Unmarshal(raw, &value) != nil {
		return "", false
	}
	return value, true
}

func wireValidationError(code, path, objectType string, repairable bool, format string, args ...any) *ResponsesWireValidationError {
	return &ResponsesWireValidationError{
		Code: code, Path: path, ObjectType: objectType, Repairable: repairable, message: fmt.Sprintf(format, args...),
	}
}

package conversion

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/looplj/axonhub/llm"
)

func restoreInvocationSchemaArguments(call *llm.ToolInvocation, session *Session, direction llm.ConversionDirection, ref ObjectRef) {
	if call == nil || session == nil {
		return
	}
	// Always retain provider-raw bytes before any client-facing cleanup so a
	// later Responses continuation can replay unmodified arguments.
	session.recordProviderArgumentBytes(call.CallID, call.LogicalName, call.ArgumentsJSON, call.ArgumentsText)
	paths := session.schemaRestoration(call.LogicalName, call.Namespace)
	if len(call.ArgumentsJSON) > 0 {
		if restored, schemaChanged, numberChanged := normalizeToolArgumentJSON(call.ArgumentsJSON, paths); schemaChanged || numberChanged {
			call.ArgumentsJSON = restored
			if schemaChanged {
				session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
			}
			if numberChanged {
				session.recordToolArgumentCanonicalization(direction, ref, call.CallID)
			}
		}
		return
	}
	if call.ArgumentsText == "" {
		return
	}
	if restored, schemaChanged, numberChanged := normalizeToolArgumentJSON([]byte(call.ArgumentsText), paths); schemaChanged || numberChanged {
		call.ArgumentsText = string(restored)
		if schemaChanged {
			session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
		}
		if numberChanged {
			session.recordToolArgumentCanonicalization(direction, ref, call.CallID)
		}
	}
}

func restoreLegacySchemaArguments(call *llm.ToolCall, session *Session, direction llm.ConversionDirection, ref ObjectRef) {
	if call == nil || session == nil || call.Function.Arguments == "" {
		return
	}
	session.recordProviderArgumentBytes(call.ID, call.Function.Name, nil, call.Function.Arguments)
	paths := session.schemaRestoration(call.Function.Name, call.Function.Namespace)
	if restored, schemaChanged, numberChanged := normalizeToolArgumentJSON([]byte(call.Function.Arguments), paths); schemaChanged || numberChanged {
		call.Function.Arguments = string(restored)
		if schemaChanged {
			session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
		}
		if numberChanged {
			session.recordToolArgumentCanonicalization(direction, ref, call.ID)
		}
	}
}

// normalizeToolArgumentJSON composes semantic schema restoration with the
// protocol-independent exact-number spelling required by typed tool clients.
// Provider raw bytes are recorded before this function is reached.
func normalizeToolArgumentJSON(raw []byte, paths []schemaOptionalPath) (json.RawMessage, bool, bool) {
	restored := json.RawMessage(raw)
	schemaChanged := false
	if len(paths) > 0 {
		if candidate, changed := stripOptionalNullArguments(restored, paths); changed {
			restored, schemaChanged = candidate, true
		}
	}
	canonical, numberChanged := canonicalizeIntegralJSONNumbers(restored)
	return canonical, schemaChanged, numberChanged
}

func stripOptionalNullArguments(raw []byte, paths []schemaOptionalPath) (json.RawMessage, bool) {
	if len(raw) == 0 || len(paths) == 0 {
		return raw, false
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return raw, false
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return raw, false
	}
	changed := false
	for _, path := range paths {
		if stripOptionalNullAtPath(value, path) {
			changed = true
		}
	}
	if !changed {
		return raw, false
	}
	restored, err := json.Marshal(value)
	if err != nil {
		return raw, false
	}
	return restored, true
}

func stripOptionalNullAtPath(value any, path schemaOptionalPath) bool {
	if len(path) == 0 {
		return false
	}
	step := path[0]
	if step.array {
		items, ok := value.([]any)
		if !ok {
			return false
		}
		changed := false
		for _, item := range items {
			if stripOptionalNullAtPath(item, path[1:]) {
				changed = true
			}
		}
		return changed
	}
	object, ok := value.(map[string]any)
	if !ok {
		return false
	}
	child, exists := object[step.property]
	if !exists {
		return false
	}
	if len(path) == 1 {
		if child != nil {
			return false
		}
		delete(object, step.property)
		return true
	}
	return stripOptionalNullAtPath(child, path[1:])
}

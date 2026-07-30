package conversion

import (
	"encoding/json"

	"github.com/looplj/axonhub/llm"
)

func restoreInvocationSchemaArguments(call *llm.ToolInvocation, session *Session, direction llm.ConversionDirection, ref ObjectRef) {
	if call == nil || session == nil {
		return
	}
	paths := session.schemaRestoration(call.LogicalName)
	if len(paths) == 0 {
		return
	}
	if len(call.ArgumentsJSON) > 0 {
		if restored, changed := stripOptionalNullArguments(call.ArgumentsJSON, paths); changed {
			call.ArgumentsJSON = restored
			session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
		}
		return
	}
	if call.ArgumentsText == "" {
		return
	}
	if restored, changed := stripOptionalNullArguments([]byte(call.ArgumentsText), paths); changed {
		call.ArgumentsText = string(restored)
		session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
	}
}

func restoreLegacySchemaArguments(call *llm.ToolCall, session *Session, direction llm.ConversionDirection, ref ObjectRef) {
	if call == nil || session == nil || call.Function.Arguments == "" {
		return
	}
	paths := session.schemaRestoration(call.Function.Name)
	if len(paths) == 0 {
		return
	}
	if restored, changed := stripOptionalNullArguments([]byte(call.Function.Arguments), paths); changed {
		call.Function.Arguments = string(restored)
		session.recordDebug(direction, ref, "restore", StrategySchemaNormalize, ReasonSemanticProjection, true)
	}
}

func stripOptionalNullArguments(raw []byte, paths []schemaOptionalPath) (json.RawMessage, bool) {
	if len(raw) == 0 || len(paths) == 0 {
		return raw, false
	}
	var value any
	if json.Unmarshal(raw, &value) != nil {
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

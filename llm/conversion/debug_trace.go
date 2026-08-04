package conversion

import (
	"context"
	"encoding/binary"
	"strconv"

	"github.com/looplj/axonhub/llm"
)

func buildConversionDebugTrace(ctx context.Context, actions []Action) *llm.ConversionDebugTrace {
	trace := llm.NewConversionDebugTrace(ctx, len(actions))
	if trace == nil {
		attentionActions := 0
		for index := range actions {
			if requiredPlanEvidence(actions[index]) {
				attentionActions++
			}
		}
		if attentionActions == 0 {
			return nil
		}
		trace = llm.NewRequiredConversionDebugTrace(attentionActions)
	}
	for index := range actions {
		action := &actions[index]
		if trace.Mode == llm.ConversionEvidenceRequired && !requiredPlanEvidence(*action) {
			continue
		}
		var logicalRef []byte
		if trace.Mode == llm.ConversionEvidenceSampled {
			logicalRef = objectRefBytes(action.Ref)
		}
		trace.Append(planActionEvidence(*action), logicalRef)
	}
	return trace
}

func requiredPlanEvidence(action Action) bool {
	return action.Kind != ActionNative
}

func planActionEvidence(action Action) llm.ConversionActionTrace {
	objectID, fieldPath := objectEvidenceLocation(llm.ConversionDirectionRequest, action.Ref, action.Strategy)
	result, severity := planEvidenceResult(action.Kind)
	return llm.ConversionActionTrace{
		Direction:       llm.ConversionDirectionRequest,
		ObjectKind:      string(action.Ref.Kind),
		ObjectID:        objectID,
		FieldPath:       fieldPath,
		DestinationPath: action.DestinationPath,
		Stage:           llm.ConversionStageRequestPlanning,
		Action:          string(action.Kind),
		Strategy:        string(action.Strategy),
		Reason:          string(action.Reason),
		Result:          result,
		Severity:        severity,
		Reversible:      action.Reversible,
	}
}

func runtimeActionEvidence(
	direction llm.ConversionDirection,
	ref ObjectRef,
	action string,
	strategy StrategyID,
	reason ReasonCode,
	reversible bool,
) llm.ConversionActionTrace {
	objectID, fieldPath := objectEvidenceLocation(direction, ref, strategy)
	result, severity := runtimeEvidenceResult(action, reversible)
	return llm.ConversionActionTrace{
		Direction:  direction,
		ObjectKind: string(ref.Kind),
		ObjectID:   objectID,
		FieldPath:  fieldPath,
		Stage:      runtimeEvidenceStage(direction),
		Action:     action,
		Strategy:   string(strategy),
		Reason:     string(reason),
		Result:     result,
		Severity:   severity,
		Reversible: reversible,
	}
}

func requiredRuntimeEvidence(action llm.ConversionActionTrace) bool {
	switch action.Result {
	case llm.ConversionResultRestoreMiss, llm.ConversionResultNormalized, llm.ConversionResultRepaired:
		return true
	default:
		return action.Severity != llm.ConversionSeverityInfo
	}
}

func planEvidenceResult(kind ActionKind) (llm.ConversionEvidenceResult, llm.ConversionEvidenceSeverity) {
	switch kind {
	case ActionNative:
		return llm.ConversionResultNative, llm.ConversionSeverityInfo
	case ActionLower:
		return llm.ConversionResultLowered, llm.ConversionSeverityWarning
	case ActionEmulate:
		return llm.ConversionResultEmulated, llm.ConversionSeverityWarning
	case ActionOpaque:
		return llm.ConversionResultOpaque, llm.ConversionSeverityWarning
	default:
		return llm.ConversionResultUnknown, llm.ConversionSeverityCritical
	}
}

func runtimeEvidenceResult(action string, reversible bool) (llm.ConversionEvidenceResult, llm.ConversionEvidenceSeverity) {
	switch action {
	case "restore":
		return llm.ConversionResultRestored, llm.ConversionSeverityInfo
	case "restore_miss":
		return llm.ConversionResultRestoreMiss, llm.ConversionSeverityCritical
	case "normalize":
		if reversible {
			return llm.ConversionResultNormalized, llm.ConversionSeverityInfo
		}
		return llm.ConversionResultNormalized, llm.ConversionSeverityWarning
	case "repair":
		return llm.ConversionResultRepaired, llm.ConversionSeverityWarning
	default:
		if reversible {
			return llm.ConversionResultNative, llm.ConversionSeverityInfo
		}
		return llm.ConversionResultUnknown, llm.ConversionSeverityCritical
	}
}

func runtimeEvidenceStage(direction llm.ConversionDirection) llm.ConversionEvidenceStage {
	switch direction {
	case llm.ConversionDirectionResponse:
		return llm.ConversionStageResponseRestore
	case llm.ConversionDirectionStream:
		return llm.ConversionStageStreamRestore
	default:
		return llm.ConversionStageRequestTransform
	}
}

func objectEvidenceLocation(direction llm.ConversionDirection, ref ObjectRef, strategy StrategyID) (string, string) {
	sequence := "input"
	if direction != llm.ConversionDirectionRequest {
		sequence = "output"
	}
	indexed := func(name string, index int) string {
		if index < 0 {
			return name
		}
		return name + "[" + strconv.Itoa(index) + "]"
	}

	var objectID string
	switch ref.Kind {
	case ObjectToolDefinition:
		objectID = indexed("tools", ref.ToolIndex)
		if ref.ItemIndex >= 0 {
			objectID = indexed(sequence, ref.ItemIndex) + "." + indexed("tools", ref.ToolIndex)
		}
	case ObjectToolChoice:
		objectID = "tool_choice"
	case ObjectToolCall:
		switch {
		case ref.ItemIndex >= 0:
			objectID = indexed(sequence, ref.ItemIndex) + ".tool_call"
		case ref.MessageIndex >= 0:
			objectID = indexed("messages", ref.MessageIndex) + "." + indexed("tool_calls", ref.ToolCallIndex)
		default:
			objectID = "tool_call"
		}
	case ObjectToolResult:
		switch {
		case ref.ItemIndex >= 0:
			objectID = indexed(sequence, ref.ItemIndex) + ".tool_result"
		case ref.MessageIndex >= 0 && ref.ToolCallIndex >= 0:
			objectID = indexed("messages", ref.MessageIndex) + "." + indexed("tool_results", ref.ToolCallIndex)
		case ref.MessageIndex >= 0:
			objectID = indexed("messages", ref.MessageIndex) + ".tool_result"
		default:
			objectID = "tool_result"
		}
	case ObjectContentBlock:
		objectID = indexed(sequence, ref.ItemIndex) + "." + indexed("content", ref.ContentIndex)
	case ObjectProviderData:
		objectID = "provider_data"
		if ref.ItemIndex >= 0 {
			objectID = indexed(sequence, ref.ItemIndex) + ".provider_data"
		}
	case ObjectCompaction, ObjectRequestControl, ObjectInputItem:
		objectID = indexed(sequence, ref.ItemIndex)
	default:
		objectID = string(ref.Kind)
	}

	fieldPath := objectID
	switch strategy {
	case StrategySchemaNormalize:
		fieldPath += ".parameters"
	case StrategyRequestControl:
		fieldPath += ".position"
	case StrategyRequestControlMultiplicity:
		fieldPath += ".type"
	}
	return objectID, fieldPath
}

func objectRefBytes(ref ObjectRef) []byte {
	// The object kind and structural indexes locate one logical object without
	// touching tool names, call IDs, model payloads, or provider data.
	buffer := make([]byte, len(ref.Kind)+1+5*8)
	copy(buffer, ref.Kind)
	offset := len(ref.Kind) + 1
	indexes := [...]int{ref.ToolIndex, ref.ItemIndex, ref.ContentIndex, ref.MessageIndex, ref.ToolCallIndex}
	for _, index := range indexes {
		binary.BigEndian.PutUint64(buffer[offset:], uint64(int64(index)))
		offset += 8
	}
	return buffer
}

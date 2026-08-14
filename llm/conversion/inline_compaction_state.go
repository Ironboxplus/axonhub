package conversion

import (
	"encoding/json"
	"errors"
)

func inlineCompactionStateBytes(state CompactionState) (uint32, error) {
	encoded, err := json.Marshal(state)
	if err != nil {
		return 0, &InlineCompactionError{Code: InlineCompactionInvalidState, Err: errors.New("checkpoint state cannot be encoded")}
	}
	if len(encoded) > maxInlineCompactionStateBytes {
		return 0, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("checkpoint state exceeds the gateway limit")}
	}
	return uint32(len(encoded)), nil
}

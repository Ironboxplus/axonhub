package conversion

import (
	"context"
	"encoding/binary"

	"github.com/looplj/axonhub/llm"
)

func buildConversionDebugTrace(ctx context.Context, actions []Action) *llm.ConversionDebugTrace {
	trace := llm.NewConversionDebugTrace(ctx, len(actions))
	if trace == nil {
		return nil
	}
	for index := range actions {
		action := &actions[index]
		trace.Append(
			llm.ConversionDirectionRequest,
			string(action.Ref.Kind),
			string(action.Kind),
			string(action.Strategy),
			string(action.Reason),
			action.Reversible,
			objectRefBytes(action.Ref),
		)
	}
	return trace
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

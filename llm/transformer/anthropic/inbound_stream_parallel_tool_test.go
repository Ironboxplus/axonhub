package anthropic

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/streams"
)

func TestInboundStreamPreservesInterleavedParallelToolArguments(t *testing.T) {
	chunk := func(toolCalls ...llm.ToolCall) *llm.Response {
		return &llm.Response{
			ID: "parallel-tools", Object: "chat.completion.chunk", Model: "test-model",
			Choices: []llm.Choice{{Index: 0, Delta: &llm.Message{Role: "assistant", ToolCalls: toolCalls}}},
		}
	}
	source := streams.SliceStream([]*llm.Response{
		chunk(
			llm.ToolCall{Index: 0, ID: "call_lookup", Type: "function", Function: llm.FunctionCall{Name: "lookup", Arguments: `{"city":"`}},
			llm.ToolCall{Index: 1, ID: "call_weather", Type: "function", Function: llm.FunctionCall{Name: "weather", Arguments: `{"unit":"`}},
		),
		chunk(
			llm.ToolCall{Index: 0, Function: llm.FunctionCall{Arguments: `Tokyo"}`}},
			llm.ToolCall{Index: 1, Function: llm.FunctionCall{Arguments: `C"}`}},
		),
		{
			ID: "parallel-tools", Object: "chat.completion.chunk", Model: "test-model",
			Choices: []llm.Choice{{Index: 0, Delta: &llm.Message{}, FinishReason: lo.ToPtr("tool_calls")}},
		},
		llm.DoneResponse,
	})

	transformed, err := NewInboundTransformer().TransformStream(t.Context(), source)
	require.NoError(t, err)
	arguments := map[int]*strings.Builder{0: {}, 1: {}}
	starts := map[int]string{}
	stops := map[int]int{}
	for transformed.Next() {
		var event StreamEvent
		require.NoError(t, json.Unmarshal(transformed.Current().Data, &event))
		if event.Index == nil {
			continue
		}
		switch event.Type {
		case "content_block_start":
			if event.ContentBlock != nil && event.ContentBlock.Type == "tool_use" {
				starts[int(*event.Index)] = event.ContentBlock.ID
			}
		case "content_block_delta":
			if event.Delta != nil && event.Delta.Type != nil && *event.Delta.Type == "input_json_delta" && event.Delta.PartialJSON != nil {
				arguments[int(*event.Index)].WriteString(*event.Delta.PartialJSON)
			}
		case "content_block_stop":
			stops[int(*event.Index)]++
		}
	}
	require.NoError(t, transformed.Err())
	require.Equal(t, map[int]string{0: "call_lookup", 1: "call_weather"}, starts)
	require.Equal(t, `{"city":"Tokyo"}`, arguments[0].String())
	require.Equal(t, `{"unit":"C"}`, arguments[1].String())
	require.Equal(t, map[int]int{0: 1, 1: 1}, stops)
}

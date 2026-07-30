package conversion

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func BenchmarkPlannerObserverOff(b *testing.B) {
	request := benchmarkCanonicalRequest(100, 32)
	planner := NewPlanner()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		plan, err := planner.Plan(request, llm.APIFormatAnthropicMessage)
		if err != nil || !plan.Complete() {
			b.Fatalf("plan canonical request: plan=%#v err=%v", plan, err)
		}
	}
}

func BenchmarkCanonicalProjection100Messages32Tools(b *testing.B) {
	request := benchmarkCanonicalRequest(100, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		projected, err := projectCanonical(request, llm.APIFormatAnthropicMessage)
		if err != nil || len(projected.Messages) != 100 || len(projected.Tools) != 32 {
			b.Fatalf("project canonical request: messages=%d tools=%d err=%v", len(projected.Messages), len(projected.Tools), err)
		}
	}
}

func BenchmarkStreamJSONAccumulator64KiB(b *testing.B) {
	payload := make([]byte, 64<<10)
	for index := range payload {
		payload[index] = 'x'
	}
	fragments := []string{`{"input":"`, string(payload[:21<<10]), string(payload[21<<10 : 43<<10]), string(payload[43<<10:]), `"}`}
	b.SetBytes(int64(len(payload)))
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		var accumulator jsonValueAccumulator
		for _, fragment := range fragments {
			accumulator.Add(fragment)
		}
		if !accumulator.complete || accumulator.Len() < len(payload) {
			b.Fatal("stream JSON accumulator did not complete")
		}
	}
}

func benchmarkCanonicalRequest(messageCount, toolCount int) *llm.Request {
	request := &llm.Request{APIFormat: llm.APIFormatOpenAIChatCompletion}
	request.Input = make([]llm.Item, 0, messageCount)
	for index := 0; index < messageCount; index++ {
		role := llm.RoleUser
		if index%2 == 1 {
			role = llm.RoleAssistant
		}
		request.Input = append(request.Input, llm.Item{
			Kind: llm.ItemKindMessage, Role: role,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: fmt.Sprintf("message-%03d", index)}},
		})
	}
	request.ToolDefinitions = make([]llm.ToolDefinition, 0, toolCount)
	for index := 0; index < toolCount; index++ {
		request.ToolDefinitions = append(request.ToolDefinitions, llm.ToolDefinition{
			Kind: llm.ToolKindFunction, LogicalName: fmt.Sprintf("tool_%02d", index),
			Function:  &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object"}`)},
			Execution: llm.ExecutionOwnerClient,
		})
	}
	return request
}

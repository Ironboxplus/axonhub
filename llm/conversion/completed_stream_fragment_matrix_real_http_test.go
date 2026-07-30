package conversion_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type completedFragmentProtocol struct {
	name        string
	path        string
	requestBody string
	newInbound  func() transformer.Inbound
	newOutbound func(string) (transformer.Outbound, error)
	write       func(*testing.T, http.ResponseWriter)
}

func TestCompletedTextReasoningToolUsageLifecycleSurvivesRandomTCPSplits3x3(t *testing.T) {
	protocols := completedFragmentProtocols()
	for _, providerCase := range protocols {
		for _, clientCase := range protocols {
			providerCase, clientCase := providerCase, clientCase
			t.Run(providerCase.name+"_to_"+clientCase.name, func(t *testing.T) {
				t.Parallel()
				provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
					if request.URL.Path != providerCase.path {
						http.Error(writer, "unexpected path", http.StatusNotFound)
						return
					}
					writer.Header().Set("Content-Type", "text/event-stream")
					providerCase.write(t, &fragmentingResponseWriter{
						ResponseWriter: writer,
						pattern:        []int{1, 7, 31, 127, 1024, 3, 19},
					})
				}))
				t.Cleanup(provider.Close)

				providerOutbound, err := providerCase.newOutbound(provider.URL)
				if err != nil {
					t.Fatalf("create %s provider outbound: %v", providerCase.name, err)
				}
				executor := httpclient.NewHttpClientWithClient(provider.Client())
				t.Cleanup(executor.CloseIdleConnections)
				clientInbound := clientCase.newInbound()
				result, err := pipeline.NewFactory(executor).
					Pipeline(clientInbound, conversion.NewOutbound(providerOutbound)).
					Process(context.Background(), &httpclient.Request{
						Method: http.MethodPost, URL: clientCase.path,
						Headers: http.Header{"Content-Type": []string{"application/json"}},
						Body:    []byte(clientCase.requestBody),
					})
				if err != nil {
					t.Fatalf("start %s-to-%s fragmented stream: %v", providerCase.name, clientCase.name, err)
				}
				defer result.EventStream.Close()
				var clientEvents []*httpclient.StreamEvent
				for result.EventStream.Next() {
					event := result.EventStream.Current()
					clientEvents = append(clientEvents, &httpclient.StreamEvent{
						LastEventID: event.LastEventID,
						Type:        event.Type,
						Data:        append([]byte(nil), event.Data...),
						Size:        event.Size,
					})
				}
				if err := result.EventStream.Err(); err != nil {
					t.Fatalf("consume %s-to-%s fragmented stream: %v", providerCase.name, clientCase.name, err)
				}

				aggregated, _, err := clientInbound.AggregateStreamChunks(context.Background(), clientEvents)
				if err != nil {
					t.Fatalf("aggregate %s client stream: %v", clientCase.name, err)
				}
				clientDecoder, err := clientCase.newOutbound("http://127.0.0.1:1")
				if err != nil {
					t.Fatalf("create %s verification decoder: %v", clientCase.name, err)
				}
				canonical, err := clientDecoder.TransformResponse(context.Background(), &httpclient.Response{
					StatusCode: http.StatusOK,
					Headers:    http.Header{"Content-Type": []string{"application/json"}},
					Body:       aggregated,
				})
				if err != nil {
					t.Fatalf("decode aggregated %s client response: %v\n%s", clientCase.name, err, aggregated)
				}
				assertCompletedFragmentCanonical(t, providerCase.name, clientCase.name, canonical, aggregated)
			})
		}
	}
}

func completedFragmentProtocols() []completedFragmentProtocol {
	return []completedFragmentProtocol{
		{
			name: "chat", path: "/v1/chat/completions",
			requestBody: `{"model":"matrix-model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"run"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`,
			newInbound:  func() transformer.Inbound { return openai.NewInboundTransformer() },
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: writeCompletedChatFragmentStream,
		},
		{
			name: "responses", path: "/v1/responses",
			requestBody: `{"model":"matrix-model","stream":true,"input":"run","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`,
			newInbound:  func() transformer.Inbound { return responses.NewInboundTransformer() },
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: writeCompletedResponsesFragmentStream,
		},
		{
			name: "anthropic", path: "/v1/messages",
			requestBody: `{"model":"matrix-model","stream":true,"max_tokens":128,"messages":[{"role":"user","content":"run"}],"tools":[{"name":"lookup","input_schema":{"type":"object"}}]}`,
			newInbound:  func() transformer.Inbound { return anthropic.NewInboundTransformer() },
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			write: writeCompletedAnthropicFragmentStream,
		},
	}
}

func writeCompletedChatFragmentStream(t *testing.T, writer http.ResponseWriter) {
	t.Helper()
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"role": "assistant", "reasoning_content": "think"}, "finish_reason": nil}},
	})
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": "hello"}, "finish_reason": nil}},
	})
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "id": "call_fragment", "type": "function",
			"function": map[string]any{"name": "lookup", "arguments": `{"q":"hel`},
		}}}, "finish_reason": nil}},
	})
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{map[string]any{
			"index": 0, "function": map[string]any{"arguments": `lo"}`},
		}}}, "finish_reason": nil}},
	})
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": "tool_calls"}},
	})
	writeSSEJSON(t, writer, map[string]any{
		"id": "chat_fragment", "object": "chat.completion.chunk", "model": "matrix-model", "choices": []any{},
		"usage": map[string]any{"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
	})
	_, _ = fmt.Fprint(writer, "data: [DONE]\n\n")
}

func writeCompletedResponsesFragmentStream(t *testing.T, writer http.ResponseWriter) {
	t.Helper()
	events := []map[string]any{
		{"type": "response.created", "sequence_number": 0, "response": map[string]any{"id": "resp_fragment", "object": "response", "model": "matrix-model", "status": "in_progress", "output": []any{}}},
		{"type": "response.output_item.added", "sequence_number": 1, "output_index": 0, "item": map[string]any{"id": "reason_fragment", "type": "reasoning", "status": "in_progress", "summary": []any{}}},
		{"type": "response.reasoning_summary_text.delta", "sequence_number": 2, "output_index": 0, "item_id": "reason_fragment", "summary_index": 0, "delta": "think"},
		{"type": "response.output_item.done", "sequence_number": 3, "output_index": 0, "item": map[string]any{"id": "reason_fragment", "type": "reasoning", "status": "completed", "summary": []any{map[string]any{"type": "summary_text", "text": "think"}}, "encrypted_content": "sig_fragment"}},
		{"type": "response.output_item.added", "sequence_number": 4, "output_index": 1, "item": map[string]any{"id": "msg_fragment", "type": "message", "role": "assistant", "status": "in_progress", "content": []any{}}},
		{"type": "response.output_text.delta", "sequence_number": 5, "output_index": 1, "item_id": "msg_fragment", "content_index": 0, "delta": "hello"},
		{"type": "response.output_item.done", "sequence_number": 6, "output_index": 1, "item": map[string]any{"id": "msg_fragment", "type": "message", "role": "assistant", "status": "completed", "content": []any{map[string]any{"type": "output_text", "text": "hello", "annotations": []any{}}}}},
		{"type": "response.output_item.added", "sequence_number": 7, "output_index": 2, "item": map[string]any{"id": "fc_fragment", "type": "function_call", "status": "in_progress", "call_id": "call_fragment", "name": "lookup", "arguments": ""}},
		{"type": "response.function_call_arguments.delta", "sequence_number": 8, "output_index": 2, "item_id": "fc_fragment", "delta": `{"q":"hel`},
		{"type": "response.function_call_arguments.delta", "sequence_number": 9, "output_index": 2, "item_id": "fc_fragment", "delta": `lo"}`},
		{"type": "response.function_call_arguments.done", "sequence_number": 10, "output_index": 2, "item_id": "fc_fragment", "call_id": "call_fragment", "name": "lookup", "arguments": `{"q":"hello"}`},
		{"type": "response.output_item.done", "sequence_number": 11, "output_index": 2, "item": map[string]any{"id": "fc_fragment", "type": "function_call", "status": "completed", "call_id": "call_fragment", "name": "lookup", "arguments": `{"q":"hello"}`}},
		{"type": "response.completed", "sequence_number": 12, "response": map[string]any{"id": "resp_fragment", "object": "response", "model": "matrix-model", "status": "completed", "output": []any{}, "usage": map[string]any{"input_tokens": 7, "output_tokens": 5, "total_tokens": 12}}},
	}
	for _, event := range events {
		writeNamedSSEJSON(t, writer, event["type"].(string), event)
	}
}

func writeCompletedAnthropicFragmentStream(t *testing.T, writer http.ResponseWriter) {
	t.Helper()
	writeNamedSSEJSON(t, writer, "message_start", map[string]any{
		"type": "message_start", "message": map[string]any{
			"id": "msg_fragment", "type": "message", "role": "assistant", "model": "matrix-model", "content": []any{},
			"usage": map[string]any{"input_tokens": 7, "output_tokens": 0},
		},
	})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 0,
		"content_block": map[string]any{"type": "thinking", "thinking": "", "signature": ""},
	})
	writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "thinking_delta", "thinking": "think"},
	})
	writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 0, "delta": map[string]any{"type": "signature_delta", "signature": "sig_fragment"},
	})
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 0})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 1, "content_block": map[string]any{"type": "text", "text": ""},
	})
	writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 1, "delta": map[string]any{"type": "text_delta", "text": "hello"},
	})
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 1})
	writeNamedSSEJSON(t, writer, "content_block_start", map[string]any{
		"type": "content_block_start", "index": 2,
		"content_block": map[string]any{"type": "tool_use", "id": "call_fragment", "name": "lookup", "input": map[string]any{}},
	})
	writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 2, "delta": map[string]any{"type": "input_json_delta", "partial_json": `{"q":"hel`},
	})
	writeNamedSSEJSON(t, writer, "content_block_delta", map[string]any{
		"type": "content_block_delta", "index": 2, "delta": map[string]any{"type": "input_json_delta", "partial_json": `lo"}`},
	})
	writeNamedSSEJSON(t, writer, "content_block_stop", map[string]any{"type": "content_block_stop", "index": 2})
	writeNamedSSEJSON(t, writer, "message_delta", map[string]any{
		"type": "message_delta", "delta": map[string]any{"stop_reason": "tool_use", "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": 5},
	})
	writeNamedSSEJSON(t, writer, "message_stop", map[string]any{"type": "message_stop"})
}

func assertCompletedFragmentCanonical(t *testing.T, provider, client string, response *llm.Response, wire []byte) {
	t.Helper()
	if response == nil {
		t.Fatalf("%s-to-%s canonical response is nil", provider, client)
	}
	var reasoning, text, arguments strings.Builder
	callID, logicalName := "", ""
	for index := range response.Output {
		item := &response.Output[index]
		switch item.Kind {
		case llm.ItemKindReasoning:
			if item.Reasoning != nil {
				reasoning.WriteString(item.Reasoning.Content)
			}
		case llm.ItemKindMessage:
			for contentIndex := range item.Content {
				if item.Content[contentIndex].Kind == llm.ContentKindText {
					text.WriteString(item.Content[contentIndex].Text)
				}
			}
		case llm.ItemKindToolCall:
			if item.ToolCall != nil {
				callID = item.ToolCall.CallID
				logicalName = item.ToolCall.LogicalName
				arguments.Write(item.ToolCall.ArgumentsJSON)
				arguments.WriteString(item.ToolCall.ArgumentsText)
			}
		}
	}
	if reasoning.String() != "think" || text.String() != "hello" || callID != "call_fragment" ||
		logicalName != "lookup" || arguments.String() != `{"q":"hello"}` {
		t.Fatalf("%s-to-%s fragmented lifecycle degraded: reasoning=%q text=%q call=%q name=%q arguments=%q\nwire=%s\ncanonical=%#v",
			provider, client, reasoning.String(), text.String(), callID, logicalName, arguments.String(), wire, response.Output)
	}
	if response.Usage == nil || response.Usage.PromptTokens != 7 || response.Usage.CompletionTokens != 5 || response.Usage.TotalTokens != 12 {
		t.Fatalf("%s-to-%s fragmented usage degraded: %#v\nwire=%s", provider, client, response.Usage, wire)
	}
	if response.Status != "" && response.Status != llm.ResponseStatusCompleted {
		t.Fatalf("%s-to-%s fragmented terminal status = %q", provider, client, response.Status)
	}
	if !json.Valid(wire) {
		t.Fatalf("%s aggregated %s response is invalid JSON: %s", client, provider, wire)
	}
}

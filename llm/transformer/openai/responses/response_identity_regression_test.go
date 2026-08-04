package responses

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesResponseIdentityPreservesTopLevelAndObjectResidual(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_identity","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"future_response_control":{"trace":"keep"},
		"output":[{
			"id":"msg_1","type":"message","status":"completed","role":"assistant",
			"future_output_control":{"retain":true},
			"content":[{"type":"output_text","text":"ok","annotations":[],"future_text_control":{"retain":true}}]
		}],
		"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
	}`)

	body := encodeClientResponsesBody(t, canonical)
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.JSONEq(t, `{"trace":"keep"}`, string(envelope["future_response_control"]))
	var output []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["output"], &output))
	require.Len(t, output, 1)
	require.JSONEq(t, `{"retain":true}`, string(output[0]["future_output_control"]))
	var content []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(output[0]["content"], &content))
	require.JSONEq(t, `{"retain":true}`, string(content[0]["future_text_control"]))
}

func TestResponsesStreamingAndNonStreamingIdentityMaterializeSameResponse(t *testing.T) {
	t.Parallel()

	responseBody := `{
		"id":"resp_parity","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"future_response":{"trace":"same"},
		"output":[{
			"id":"msg_parity","type":"message","status":"completed","role":"assistant","future_item":{"route":"same"},
			"content":[{"type":"output_text","text":"hello","annotations":[],"future_content":{"confidence":1}}]
		}],
		"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3,"future_usage":{"tier":"same"}}
	}`
	nonStreaming := decodeProviderResponsesBody(t, responseBody)

	decoder := newCanonicalStreamDecoder()
	accumulator := llm.NewCanonicalResponseAccumulator()
	for _, raw := range []string{
		`{"type":"response.created","response":{"id":"resp_parity","object":"response","created_at":1,"model":"fixture-model","status":"in_progress","output":[],"future_response":{"trace":"same"}}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_parity","type":"message","status":"in_progress","role":"assistant","content":[],"future_item":{"route":"same"}}}`,
		`{"type":"response.content_part.added","item_id":"msg_parity","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[],"future_content":{"confidence":1}}}`,
		`{"type":"response.output_text.delta","item_id":"msg_parity","output_index":0,"content_index":0,"delta":"hello"}`,
		`{"type":"response.output_text.done","item_id":"msg_parity","output_index":0,"content_index":0,"text":"hello"}`,
		`{"type":"response.content_part.done","item_id":"msg_parity","output_index":0,"content_index":0,"part":{"type":"output_text","text":"hello","annotations":[],"future_content":{"confidence":1}}}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_parity","type":"message","status":"completed","role":"assistant","future_item":{"route":"same"},"content":[{"type":"output_text","text":"hello","annotations":[],"future_content":{"confidence":1}}]}}`,
		`{"type":"response.completed","response":{"id":"resp_parity","object":"response","created_at":1,"model":"fixture-model","status":"completed","output":[{"id":"msg_parity","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hello","annotations":[]}]}],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3,"future_usage":{"tier":"same"}},"future_response":{"trace":"same"}}}`,
	} {
		var wire StreamEvent
		require.NoError(t, json.Unmarshal([]byte(raw), &wire))
		events, err := decoder.decode(&wire)
		require.NoError(t, err, "decode %s", wire.Type)
		chunk := &llm.Response{Events: events}
		if wire.Response != nil {
			chunk.ID = wire.Response.ID
			chunk.Model = wire.Response.Model
			chunk.Created = wire.Response.CreatedAt
		}
		require.NoError(t, accumulator.Observe(chunk), "observe %s", wire.Type)
	}
	streaming := accumulator.Snapshot()
	require.NotNil(t, streaming)
	require.Equal(t, llm.ResponseStatusCompleted, streaming.Status)

	nonStreamingBody := encodeClientResponsesBody(t, nonStreaming)
	streamingBody := encodeClientResponsesBody(t, streaming)
	require.JSONEq(t, string(nonStreamingBody), string(streamingBody))
}

func TestResponsesResponseIdentityUsesCurrentCanonicalOutputOwnership(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_mutation","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"output":[
			{"id":"future","type":"future_output","future_mode":"must-not-reappear"},
			{"id":"first","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"first","annotations":[]}]},
			{"id":"second","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"second","annotations":[]}]}
		],
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
	}`)
	require.Len(t, canonical.Output, 3)
	canonical.Output = []llm.Item{canonical.Output[2], canonical.Output[1]}

	body := encodeClientResponsesBody(t, canonical)
	var envelope struct {
		Output []json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Output, 2)
	require.Equal(t, []string{"second", "first"}, responseInputIDs(t, envelope.Output))
	for _, raw := range envelope.Output {
		require.NotContains(t, string(raw), "must-not-reappear")
	}
}

func TestResponsesResponseIdentityPreservesCitationAndReasoningPartResiduals(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_nested_identity","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"output":[
			{
				"id":"msg_1","type":"message","status":"completed","role":"assistant",
				"content":[{
					"type":"output_text","text":"source","future_text":true,
					"annotations":[{
						"type":"url_citation","url":"https://example.com","title":"Example",
						"start_index":0,"end_index":6,"future_annotation":{"rank":1}
					}]
				}]
			},
			{
				"id":"reason_1","type":"reasoning","status":"completed",
				"summary":[{"type":"summary_text","text":"summary","future_summary":1}],
				"reasoning_content":[{"type":"reasoning_text","text":"detail","future_reasoning":2}]
			}
		],
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}
	}`)

	body := encodeClientResponsesBody(t, canonical)
	var envelope struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Output, 2)

	var content []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[0]["content"], &content))
	require.Len(t, content, 1)
	require.JSONEq(t, `true`, string(content[0]["future_text"]))
	var annotations []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(content[0]["annotations"], &annotations))
	require.Len(t, annotations, 1)
	require.JSONEq(t, `"https://example.com"`, string(annotations[0]["url"]))
	require.JSONEq(t, `{"rank":1}`, string(annotations[0]["future_annotation"]))

	var summary []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[1]["summary"], &summary))
	require.Len(t, summary, 1)
	require.JSONEq(t, `1`, string(summary[0]["future_summary"]))
	var reasoning []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[1]["reasoning_content"], &reasoning))
	require.Len(t, reasoning, 1)
	require.JSONEq(t, `2`, string(reasoning[0]["future_reasoning"]))
}

func TestResponsesResponseIdentityPreservesNestedResponseAndHostedActionResiduals(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_nested_owner","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"conversation":{"id":"conv_1","future_conversation":{"shard":"sg"}},
		"text":{"format":{"type":"text","future_format":{"renderer":"v2"}},"future_text":{"verbosity_source":"provider"}},
		"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3,"input_tokens_details":{"cached_tokens":1,"future_cache":{"tier":"warm"}},"future_usage":{"billing":"reserved"}},
		"output":[{
			"id":"ws_1","type":"web_search_call","status":"completed",
			"action":{"type":"search","query":"old-query","future_search_constraint":{"region":"sg"},"sources":[{"type":"url","url":"https://example.com","title":"Example","future_source":{"rank":1}}]}
		}]
	}`)
	require.Len(t, canonical.Output, 1)
	require.NotNil(t, canonical.Output[0].HostedCall)
	canonical.Output[0].HostedCall.Invocation.ArgumentsJSON = json.RawMessage(`{"type":"search","query":"changed-query"}`)

	body := encodeClientResponsesBody(t, canonical)
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &envelope))
	var conversation map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["conversation"], &conversation))
	require.JSONEq(t, `{"shard":"sg"}`, string(conversation["future_conversation"]))
	var textOptions map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["text"], &textOptions))
	require.JSONEq(t, `{"verbosity_source":"provider"}`, string(textOptions["future_text"]))
	var format map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(textOptions["format"], &format))
	require.JSONEq(t, `{"renderer":"v2"}`, string(format["future_format"]))
	var usage map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["usage"], &usage))
	require.JSONEq(t, `{"billing":"reserved"}`, string(usage["future_usage"]))
	var inputDetails map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(usage["input_tokens_details"], &inputDetails))
	require.JSONEq(t, `{"tier":"warm"}`, string(inputDetails["future_cache"]))

	var output []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["output"], &output))
	require.Len(t, output, 1)
	var action map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(output[0]["action"], &action))
	require.JSONEq(t, `"changed-query"`, string(action["query"]))
	require.JSONEq(t, `{"region":"sg"}`, string(action["future_search_constraint"]))
	var sources []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(action["sources"], &sources))
	require.Len(t, sources, 1)
	require.JSONEq(t, `{"rank":1}`, string(sources[0]["future_source"]))
}

func TestResponsesResponseIdentityDoesNotReviveClearedLocalShellActionFields(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_shell_owner","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"output":[{
			"id":"shell_1","type":"shell_call","call_id":"call_1","status":"completed",
			"action":{"type":"exec","command":["old"],"working_directory":"/old","future_shell":{"sandbox":"strict"}}
		}]
	}`)
	require.Len(t, canonical.Output, 1)
	require.NotNil(t, canonical.Output[0].HostedCall)
	canonical.Output[0].HostedCall.Invocation.ArgumentsJSON = json.RawMessage(`{"type":"exec","command":["new"]}`)

	body := encodeClientResponsesBody(t, canonical)
	var envelope struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Output, 1)
	var action map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[0]["action"], &action))
	require.JSONEq(t, `["new"]`, string(action["command"]))
	require.NotContains(t, action, "working_directory", "cleared canonical action field was revived from residual")
	require.JSONEq(t, `{"sandbox":"strict"}`, string(action["future_shell"]))
}

func TestResponsesResponseIdentityPreservesNestedComputerAndMCPObjectOwners(t *testing.T) {
	t.Parallel()

	canonical := decodeProviderResponsesBody(t, `{
		"id":"resp_nested_arrays","object":"response","created_at":1,"model":"fixture-model","status":"completed",
		"output":[
			{
				"id":"computer_1","type":"computer_call","call_id":"call_computer","status":"completed",
				"action":{"type":"click","x":10,"y":20},
				"pending_safety_checks":[{"id":"safe_1","code":"confirm","message":"old","future_check":{"policy":"strict"}}]
			},
			{
				"id":"mcp_1","type":"mcp_list_tools","server_label":"docs","status":"completed",
				"tools":[{"name":"search","description":"old","input_schema":{"type":"object"},"future_tool":{"version":2}}]
			},
			{
				"id":"computer_output_1","type":"computer_call_output","call_id":"call_computer","status":"completed",
				"output":{"type":"computer_screenshot","image_url":"https://old.example/image.png","future_output":{"frame":7}}
			}
		]
	}`)
	require.Len(t, canonical.Output, 3)
	canonical.Output[0].ToolCall.PendingSafetyChecks[0].Message = "changed"
	canonical.Output[1].MCPListTools.Tools[0].Description = "changed"
	require.NotNil(t, canonical.Output[2].ToolResult)
	var screenshot map[string]any
	require.NoError(t, json.Unmarshal(canonical.Output[2].ToolResult.StructuredContent, &screenshot))
	screenshot["image_url"] = "https://new.example/image.png"
	canonical.Output[2].ToolResult.StructuredContent, _ = json.Marshal(screenshot)

	body := encodeClientResponsesBody(t, canonical)
	var envelope struct {
		Output []map[string]json.RawMessage `json:"output"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Output, 3)
	var checks []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[0]["pending_safety_checks"], &checks))
	require.JSONEq(t, `"changed"`, string(checks[0]["message"]))
	require.JSONEq(t, `{"policy":"strict"}`, string(checks[0]["future_check"]))
	var tools []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[1]["tools"], &tools))
	require.JSONEq(t, `"changed"`, string(tools[0]["description"]))
	require.JSONEq(t, `{"version":2}`, string(tools[0]["future_tool"]))
	var output map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Output[2]["output"], &output))
	require.JSONEq(t, `"https://new.example/image.png"`, string(output["image_url"]))
	require.JSONEq(t, `{"frame":7}`, string(output["future_output"]))
}

func decodeProviderResponsesBody(t *testing.T, body string) *llm.Response {
	t.Helper()
	outbound, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
	require.NoError(t, err)
	canonical, err := outbound.TransformResponse(context.Background(), &httpclient.Response{
		StatusCode: http.StatusOK,
		Body:       []byte(body),
		Request:    &httpclient.Request{},
	})
	require.NoError(t, err)
	return canonical
}

func encodeClientResponsesBody(t *testing.T, response *llm.Response) []byte {
	t.Helper()
	wire, err := NewInboundTransformer().TransformResponse(context.Background(), response)
	require.NoError(t, err)
	return wire.Body
}

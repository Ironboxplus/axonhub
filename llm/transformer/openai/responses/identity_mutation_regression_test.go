package responses

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesIdentityUsesCurrentCanonicalInputOrder(t *testing.T) {
	t.Parallel()

	t.Run("reorders known items", func(t *testing.T) {
		request := decodeResponsesIdentityRequest(t, `{
			"model":"fixture-model",
			"input":[
				{"type":"message","id":"first","role":"user","content":[{"type":"input_text","text":"first"}]},
				{"type":"message","id":"second","role":"user","content":[{"type":"input_text","text":"second"}]}
			]
		}`)
		require.Len(t, request.Input, 2)
		request.Input[0], request.Input[1] = request.Input[1], request.Input[0]

		input := encodeResponsesIdentityInput(t, request)
		require.Equal(t, []string{"second", "first"}, responseInputIDs(t, input))
	})

	t.Run("moves opaque item with its canonical node", func(t *testing.T) {
		request := decodeResponsesIdentityRequest(t, `{
			"model":"fixture-model",
			"input":[
				{"type":"future_control","id":"future","mode":"preserve"},
				{"type":"message","id":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}
			]
		}`)
		require.Len(t, request.Input, 2)
		request.Input[0], request.Input[1] = request.Input[1], request.Input[0]

		input := encodeResponsesIdentityInput(t, request)
		require.Equal(t, []string{"message", "future"}, responseInputIDs(t, input))
	})
}

func TestResponsesIdentityDoesNotReplayDeletedOpaqueCanonicalItem(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":[
			{"type":"future_control","id":"delete-me","mode":"must_not_reappear"},
			{"type":"message","id":"keep-me","role":"user","content":[{"type":"input_text","text":"hello"}]}
		]
	}`)
	require.Len(t, request.Input, 2)
	request.Input = append([]llm.Item(nil), request.Input[1:]...)

	input := encodeResponsesIdentityInput(t, request)
	require.Equal(t, []string{"keep-me"}, responseInputIDs(t, input))
}

func TestResponsesIdentityPreservesTopLevelAndKnownItemResidualFields(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"client_metadata":{"trace_mode":"bounded"},
		"internal_chat_message_metadata_passthrough":{"request_scope":"identity"},
		"input":[{
			"type":"message",
			"id":"message-1",
			"role":"user",
			"phase":"commentary",
			"future_message_control":{"retain":true},
			"content":[
				{"type":"input_text","text":"hello","future_text_control":{"retain":true}},
				{"type":"future_input_part","mode":"opaque"}
			]
		}]
	}`)

	body := encodeResponsesIdentityBody(t, request)
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.JSONEq(t, `{"trace_mode":"bounded"}`, string(envelope["client_metadata"]))
	require.JSONEq(t, `{"request_scope":"identity"}`, string(envelope["internal_chat_message_metadata_passthrough"]))

	var input []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["input"], &input))
	require.Len(t, input, 1)
	require.JSONEq(t, `"commentary"`, string(input[0]["phase"]))
	require.JSONEq(t, `{"retain":true}`, string(input[0]["future_message_control"]))

	var content []map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(input[0]["content"], &content))
	require.Len(t, content, 2)
	require.JSONEq(t, `{"retain":true}`, string(content[0]["future_text_control"]))
	require.JSONEq(t, `"future_input_part"`, string(content[1]["type"]))
	require.JSONEq(t, `"opaque"`, string(content[1]["mode"]))
}

func TestResponsesIdentityPreservesReasoningContentAndSummarySeparately(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":[{
			"type":"reasoning",
			"id":"reasoning-1",
			"content":[{"type":"reasoning_text","text":"reasoning-content"}],
			"summary":[{"type":"summary_text","text":"reasoning-summary"}],
			"encrypted_content":"opaque-signature"
		}]
	}`)

	input := encodeResponsesIdentityInput(t, request)
	require.Len(t, input, 1)
	var item map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(input[0], &item))
	require.JSONEq(t, `[{"type":"reasoning_text","text":"reasoning-content"}]`, string(item["content"]))
	require.JSONEq(t, `[{"type":"summary_text","text":"reasoning-summary"}]`, string(item["summary"]))
}

func TestResponsesIdentityPreservesExplicitParallelFalseWithAdditionalTools(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"parallel_tool_calls":false,
		"input":[
			{"type":"additional_tools","role":"developer","tools":[{"type":"custom","name":"exec","description":"run code"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}
		]
	}`)

	body := encodeResponsesIdentityBody(t, request)
	var envelope struct {
		ParallelToolCalls *bool `json:"parallel_tool_calls"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.NotNil(t, envelope.ParallelToolCalls, "explicit false is protocol state, not an omitted default")
	require.False(t, *envelope.ParallelToolCalls)
}

func TestResponsesIdentityToolResidualFollowsCanonicalDefinition(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"tools":[
			{"type":"function","name":"keep","description":"keep tool","parameters":{"type":"object"},"x_tool_marker":"keep-with-owner"},
			{"type":"future_provider_tool","name":"delete","x_behavior":"must-not-reappear"}
		],
		"input":"hello"
	}`)
	require.Len(t, request.ToolDefinitions, 2, "every behavioral tool needs a canonical owner")
	request.ToolDefinitions = append([]llm.ToolDefinition(nil), request.ToolDefinitions[:1]...)

	body := encodeResponsesIdentityBody(t, request)
	var envelope struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Tools, 1, "deleted opaque tool was replayed from an absolute-index sidecar")
	require.JSONEq(t, `"keep"`, string(envelope.Tools[0]["name"]))
	require.JSONEq(t, `"keep-with-owner"`, string(envelope.Tools[0]["x_tool_marker"]))
}

func TestResponsesIdentityPreservesNestedToolResidualsWhileCanonicalFieldsChange(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"tools":[
			{
				"type":"custom","name":"exec","description":"run code",
				"format":{"type":"grammar","syntax":"lark","definition":"start: WORD","future_format":{"dialect":"v2"}}
			},
			{
				"type":"web_search",
				"filters":{"allowed_domains":["old.example"],"future_filter":{"mode":"strict"}},
				"user_location":{"type":"approximate","city":"old-city","country":"SG","future_location":{"precision":"region"}}
			}
		],
		"input":"hello"
	}`)
	require.Len(t, request.ToolDefinitions, 2)
	require.NotNil(t, request.ToolDefinitions[0].Freeform)
	request.ToolDefinitions[0].Freeform.Definition = "start: CHANGED"
	require.NotNil(t, request.ToolDefinitions[1].Hosted)
	require.NotNil(t, request.ToolDefinitions[1].Hosted.WebSearch)
	request.ToolDefinitions[1].Hosted.WebSearch.AllowedDomains = []string{"new.example"}
	request.ToolDefinitions[1].Hosted.WebSearch.UserLocation.City = "new-city"

	body := encodeResponsesIdentityBody(t, request)
	var envelope struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Tools, 2)

	var format map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Tools[0]["format"], &format))
	require.JSONEq(t, `"start: CHANGED"`, string(format["definition"]))
	require.JSONEq(t, `{"dialect":"v2"}`, string(format["future_format"]))

	var filters map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Tools[1]["filters"], &filters))
	require.JSONEq(t, `["new.example"]`, string(filters["allowed_domains"]))
	require.JSONEq(t, `{"mode":"strict"}`, string(filters["future_filter"]))
	var location map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope.Tools[1]["user_location"], &location))
	require.JSONEq(t, `"new-city"`, string(location["city"]))
	require.JSONEq(t, `{"precision":"region"}`, string(location["future_location"]))
}

func TestResponsesIdentityPreservesNestedRequestResidualsWhileCanonicalFieldsChange(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":"hello",
		"reasoning":{"effort":"medium","future_reasoning":{"budget_class":"interactive"}},
		"stream":true,
		"stream_options":{"include_obfuscation":true,"future_stream":{"heartbeat_ms":250}},
		"conversation":{"id":"conv_old","future_conversation":{"shard":"sg"}},
		"text":{"verbosity":"low","format":{"type":"json_object","future_format":{"renderer":"v2"}},"future_text":{"source":"client"}}
	}`)
	request.ReasoningEffort = "high"
	require.NotNil(t, request.Conversation)
	request.Conversation.ID = "conv_new"
	require.NotNil(t, request.ResponseFormat)
	request.ResponseFormat.Type = "text"

	body := encodeResponsesIdentityBody(t, request)
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &envelope))
	var reasoning map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["reasoning"], &reasoning))
	require.JSONEq(t, `"high"`, string(reasoning["effort"]))
	require.JSONEq(t, `{"budget_class":"interactive"}`, string(reasoning["future_reasoning"]))
	var streamOptions map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["stream_options"], &streamOptions))
	require.JSONEq(t, `{"heartbeat_ms":250}`, string(streamOptions["future_stream"]))
	var conversation map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["conversation"], &conversation))
	require.JSONEq(t, `"conv_new"`, string(conversation["id"]))
	require.JSONEq(t, `{"shard":"sg"}`, string(conversation["future_conversation"]))
	var textOptions map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(envelope["text"], &textOptions))
	require.JSONEq(t, `{"source":"client"}`, string(textOptions["future_text"]))
	var format map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(textOptions["format"], &format))
	require.JSONEq(t, `"text"`, string(format["type"]))
	require.JSONEq(t, `{"renderer":"v2"}`, string(format["future_format"]))
}

func TestResponsesIdentityDoesNotAttachResidualAfterToolTypeChanges(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"tools":[{
			"type":"custom","name":"exec","description":"old custom",
			"format":{"type":"grammar","syntax":"lark","definition":"start: WORD","future_custom":{"dialect":"v2"}},
			"future_tool":{"owner":"custom"}
		}],
		"input":"hello"
	}`)
	require.Len(t, request.ToolDefinitions, 1)
	definition := &request.ToolDefinitions[0]
	definition.Kind = llm.ToolKindFunction
	definition.Description = "new function"
	definition.Freeform = nil
	definition.Function = &llm.FunctionDefinition{Parameters: json.RawMessage(`{"type":"object","properties":{}}`)}

	body := encodeResponsesIdentityBody(t, request)
	var envelope struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	require.Len(t, envelope.Tools, 1)
	require.JSONEq(t, `"function"`, string(envelope.Tools[0]["type"]))
	require.NotContains(t, envelope.Tools[0], "format")
	require.NotContains(t, envelope.Tools[0], "future_tool")
}

func TestResponsesIdentityDoesNotAttachResidualAfterItemTypeChanges(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":[{
			"type":"message","id":"message-1","role":"user","future_message":{"owner":"message"},
			"content":[{"type":"input_text","text":"hello"}]
		}]
	}`)
	require.Len(t, request.Input, 1)
	request.Input[0].Kind = llm.ItemKindReasoning
	request.Input[0].Role = llm.RoleAssistant
	request.Input[0].Content = nil
	request.Input[0].Reasoning = &llm.ReasoningItem{
		SummaryParts: []llm.ReasoningPart{{Type: "summary_text", Text: "changed"}},
	}

	input := encodeResponsesIdentityInput(t, request)
	require.Len(t, input, 1)
	var item map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(input[0], &item))
	require.JSONEq(t, `"reasoning"`, string(item["type"]))
	require.NotContains(t, item, "future_message")
}

func TestResponsesIdentityDoesNotAttachResidualAfterContentTypeChanges(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":[{
			"type":"message","id":"message-1","role":"user",
			"content":[{"type":"input_text","text":"old","future_text":{"owner":"input_text"}}]
		}]
	}`)
	require.Len(t, request.Input, 1)
	require.Len(t, request.Input[0].Content, 1)
	block := &request.Input[0].Content[0]
	block.Kind = llm.ContentKindImage
	block.Text = ""
	block.Image = &llm.ImageURL{URL: "https://example.invalid/changed.png"}

	input := encodeResponsesIdentityInput(t, request)
	require.Len(t, input, 1)
	var message struct {
		Content []map[string]json.RawMessage `json:"content"`
	}
	require.NoError(t, json.Unmarshal(input[0], &message))
	require.Len(t, message.Content, 1)
	require.JSONEq(t, `"input_image"`, string(message.Content[0]["type"]))
	require.NotContains(t, message.Content[0], "future_text")
}

func TestResponsesIdentityDoesNotAttachResidualAfterReasoningPartTypeChanges(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"input":[{
			"type":"reasoning","id":"reasoning-1",
			"summary":[{"type":"summary_text","text":"old","future_summary":{"owner":"summary_text"}}]
		}]
	}`)
	require.Len(t, request.Input, 1)
	require.NotNil(t, request.Input[0].Reasoning)
	require.Len(t, request.Input[0].Reasoning.SummaryParts, 1)
	request.Input[0].Reasoning.SummaryParts[0].Type = "future_summary_text"
	request.Input[0].Reasoning.SummaryParts[0].Text = "changed"

	input := encodeResponsesIdentityInput(t, request)
	require.Len(t, input, 1)
	var reasoning struct {
		Summary []map[string]json.RawMessage `json:"summary"`
	}
	require.NoError(t, json.Unmarshal(input[0], &reasoning))
	require.Len(t, reasoning.Summary, 1)
	require.JSONEq(t, `"future_summary_text"`, string(reasoning.Summary[0]["type"]))
	require.NotContains(t, reasoning.Summary[0], "future_summary")
}

func TestResponsesIdentityToolChoiceResidualDoesNotReviveDeletedChoice(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"tools":[{"type":"tool_search","execution":"client","description":"search"}],
		"tool_choice":{
			"type":"tool_search",
			"tools":[{"type":"tool_search","name":"search"}],
			"x_choice_marker":"owned-by-choice"
		},
		"input":"hello"
	}`)
	require.NotNil(t, request.ToolChoice)

	body := encodeResponsesIdentityBody(t, request)
	var preserved map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &preserved))
	require.Contains(t, preserved, "tool_choice")
	var choice map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(preserved["tool_choice"], &choice))
	require.Contains(t, choice, "tools")
	require.JSONEq(t, `"owned-by-choice"`, string(choice["x_choice_marker"]))

	request.ToolChoice = nil
	body = encodeResponsesIdentityBody(t, request)
	var cleared map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &cleared))
	require.NotContains(t, cleared, "tool_choice", "cleared canonical tool choice was revived from raw state")
}

func TestResponsesIdentityToolChoiceOptionResidualFollowsCanonicalOption(t *testing.T) {
	t.Parallel()

	request := decodeResponsesIdentityRequest(t, `{
		"model":"fixture-model",
		"tools":[{"type":"function","name":"lookup","description":"lookup","parameters":{"type":"object"}}],
		"tool_choice":{
			"type":"allowed_tools","mode":"required",
			"tools":[{"type":"function","name":"lookup","future_option":{"routing":"primary"}}]
		},
		"input":"hello"
	}`)
	require.NotNil(t, request.ToolChoice)
	require.NotNil(t, request.ToolChoice.AllowedTools)
	require.Len(t, request.ToolChoice.AllowedTools.Tools, 1)
	request.ToolChoice.AllowedTools.Tools[0].Name = "lookup_changed"

	body := encodeResponsesIdentityBody(t, request)
	var envelope map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &envelope))
	var choice struct {
		Tools []map[string]json.RawMessage `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(envelope["tool_choice"], &choice))
	require.Len(t, choice.Tools, 1)
	require.JSONEq(t, `"lookup_changed"`, string(choice.Tools[0]["name"]))
	require.JSONEq(t, `{"routing":"primary"}`, string(choice.Tools[0]["future_option"]))
}

func decodeResponsesIdentityRequest(t *testing.T, body string) *llm.Request {
	t.Helper()
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Body: []byte(body),
	})
	require.NoError(t, err)
	return request
}

func encodeResponsesIdentityBody(t *testing.T, request *llm.Request) []byte {
	t.Helper()
	outbound, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
	require.NoError(t, err)
	wire, err := outbound.TransformRequest(context.Background(), request)
	require.NoError(t, err)
	return wire.Body
}

func encodeResponsesIdentityInput(t *testing.T, request *llm.Request) []json.RawMessage {
	t.Helper()
	body := encodeResponsesIdentityBody(t, request)
	var envelope struct {
		Input []json.RawMessage `json:"input"`
	}
	require.NoError(t, json.Unmarshal(body, &envelope))
	return envelope.Input
}

func responseInputIDs(t *testing.T, input []json.RawMessage) []string {
	t.Helper()
	ids := make([]string, 0, len(input))
	for _, raw := range input {
		var identity struct {
			ID string `json:"id"`
		}
		require.NoError(t, json.Unmarshal(raw, &identity))
		ids = append(ids, identity.ID)
	}
	return ids
}

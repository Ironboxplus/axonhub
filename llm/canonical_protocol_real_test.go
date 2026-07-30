package llm_test

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestThreeProtocolDecodersProduceEquivalentOrderedCanonicalToolLifecycle(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		path    string
		inbound transformer.Inbound
		body    string
	}{
		{
			name: "chat_completions", path: "/v1/chat/completions", inbound: openai.NewInboundTransformer(),
			body: `{"model":"fixture-model","messages":[{"role":"user","content":"lookup weather"},{"role":"assistant","tool_calls":[{"id":"call_weather","type":"function","function":{"name":"lookup_weather","arguments":"{\"city\":\"Singapore\"}"}}]},{"role":"tool","tool_call_id":"call_weather","content":"sunny"}],"tools":[{"type":"function","function":{"name":"lookup_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]}`,
		},
		{
			name: "responses", path: "/v1/responses", inbound: responses.NewInboundTransformer(),
			body: `{"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"lookup weather"}]},{"type":"function_call","id":"item_weather","call_id":"call_weather","name":"lookup_weather","arguments":"{\"city\":\"Singapore\"}"},{"type":"function_call_output","call_id":"call_weather","output":"sunny"}],"tools":[{"type":"function","name":"lookup_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}`,
		},
		{
			name: "anthropic_messages", path: "/v1/messages", inbound: anthropic.NewInboundTransformer(),
			body: `{"model":"fixture-model","max_tokens":128,"messages":[{"role":"user","content":"lookup weather"},{"role":"assistant","content":[{"type":"tool_use","id":"call_weather","name":"lookup_weather","input":{"city":"Singapore"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_weather","content":"sunny"}]}],"tools":[{"name":"lookup_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			request, err := tt.inbound.TransformRequest(context.Background(), &httpclient.Request{
				Method: http.MethodPost, URL: tt.path, Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(tt.body),
			})
			if err != nil {
				t.Fatalf("decode real protocol request: %v", err)
			}
			if err := llm.ValidateItems(request.Input); err != nil {
				t.Fatalf("validate decoder canonical items: %v\n%#v", err, request.Input)
			}
			if len(request.Input) != 3 || request.Input[0].Kind != llm.ItemKindMessage ||
				request.Input[1].Kind != llm.ItemKindToolCall || request.Input[2].Kind != llm.ItemKindToolResult {
				t.Fatalf("ordered canonical kinds = %#v", request.Input)
			}
			call := request.Input[1].ToolCall
			result := request.Input[2].ToolResult
			if call == nil || result == nil || call.Kind != llm.ToolKindFunction || call.CallID != "call_weather" ||
				call.LogicalName != "lookup_weather" || string(call.ArgumentsJSON) != `{"city":"Singapore"}` ||
				result.Kind != llm.ToolKindFunction || result.CallID != call.CallID || len(result.Content) != 1 || result.Content[0].Text != "sunny" {
				t.Fatalf("canonical tool lifecycle degraded: call=%#v result=%#v", call, result)
			}
			if tt.name == "responses" && (request.Input[1].ID != "item_weather" || call.ID != "item_weather") {
				t.Fatalf("Responses item identity collapsed into call_id: item=%q invocation=%q call_id=%q", request.Input[1].ID, call.ID, call.CallID)
			}
			if len(request.ToolDefinitions) != 1 || request.ToolDefinitions[0].Kind != llm.ToolKindFunction ||
				request.ToolDefinitions[0].LogicalName != "lookup_weather" {
				t.Fatalf("canonical tool definitions = %#v", request.ToolDefinitions)
			}
		})
	}
}

func TestThreeProtocolDecodersPreserveExplicitEmptyTextItems(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name, path, body string
		inbound          transformer.Inbound
	}{
		{name: "chat", path: "/v1/chat/completions", inbound: openai.NewInboundTransformer(), body: `{"model":"fixture-model","messages":[{"role":"assistant","content":""}]}`},
		{name: "responses", path: "/v1/responses", inbound: responses.NewInboundTransformer(), body: `{"model":"fixture-model","input":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":""}]}]}`},
		{name: "anthropic", path: "/v1/messages", inbound: anthropic.NewInboundTransformer(), body: `{"model":"fixture-model","max_tokens":16,"messages":[{"role":"assistant","content":""}]}`},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			request, err := test.inbound.TransformRequest(context.Background(), jsonRequest(test.path, test.body))
			if err != nil {
				t.Fatalf("decode explicit empty text: %v", err)
			}
			if len(request.Input) != 1 || request.Input[0].Kind != llm.ItemKindMessage || len(request.Input[0].Content) != 1 ||
				request.Input[0].Content[0].Kind != llm.ContentKindText || request.Input[0].Content[0].Text != "" {
				t.Fatalf("explicit empty text was dropped: %#v", request.Input)
			}
		})
	}
}

func TestThreeProtocolFunctionLifecycleRequestMatrix(t *testing.T) {
	t.Parallel()
	type protocol struct {
		name        string
		path        string
		body        string
		inbound     transformer.Inbound
		newOutbound func() transformer.Outbound
	}
	protocols := []protocol{
		{
			name: "chat", path: "/v1/chat/completions", inbound: openai.NewInboundTransformer(),
			body: `{"model":"fixture-model","messages":[{"role":"user","content":"lookup weather"},{"role":"assistant","tool_calls":[{"id":"call_weather","type":"function","function":{"name":"lookup_weather","arguments":"{\"city\":\"Singapore\"}"}}]},{"role":"tool","tool_call_id":"call_weather","content":"sunny"}],"tools":[{"type":"function","function":{"name":"lookup_weather","parameters":{"type":"object"}}}]}`,
			newOutbound: func() transformer.Outbound {
				outbound, err := openai.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
				if err != nil {
					t.Fatalf("create Chat outbound: %v", err)
				}
				return outbound
			},
		},
		{
			name: "responses", path: "/v1/responses", inbound: responses.NewInboundTransformer(),
			body: `{"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"lookup weather"}]},{"type":"function_call","id":"item_weather","call_id":"call_weather","name":"lookup_weather","arguments":{"city":"Singapore"}},{"type":"function_call_output","call_id":"call_weather","output":"sunny"}],"tools":[{"type":"function","name":"lookup_weather","parameters":{"type":"object"}}]}`,
			newOutbound: func() transformer.Outbound {
				outbound, err := responses.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
				if err != nil {
					t.Fatalf("create Responses outbound: %v", err)
				}
				return outbound
			},
		},
		{
			name: "anthropic", path: "/v1/messages", inbound: anthropic.NewInboundTransformer(),
			body: `{"model":"fixture-model","max_tokens":128,"messages":[{"role":"user","content":"lookup weather"},{"role":"assistant","content":[{"type":"tool_use","id":"call_weather","name":"lookup_weather","input":{"city":"Singapore"}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call_weather","content":"sunny"}]}],"tools":[{"name":"lookup_weather","input_schema":{"type":"object"}}]}`,
			newOutbound: func() transformer.Outbound {
				outbound, err := anthropic.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
				if err != nil {
					t.Fatalf("create Anthropic outbound: %v", err)
				}
				return outbound
			},
		},
	}

	for _, source := range protocols {
		for _, target := range protocols {
			source, target := source, target
			t.Run(source.name+"_to_"+target.name, func(t *testing.T) {
				t.Parallel()
				canonical, err := source.inbound.TransformRequest(context.Background(), jsonRequest(source.path, source.body))
				if err != nil {
					t.Fatalf("decode %s source: %v", source.name, err)
				}
				encoded, err := conversion.NewOutbound(target.newOutbound()).TransformRequest(context.Background(), canonical)
				if err != nil {
					t.Fatalf("convert %s to %s: %v", source.name, target.name, err)
				}
				decoded, err := target.inbound.TransformRequest(context.Background(), &httpclient.Request{
					Method: http.MethodPost, URL: target.path,
					Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: encoded.Body,
				})
				if err != nil {
					t.Fatalf("decode converted %s wire request: %v\n%s", target.name, err, encoded.Body)
				}
				want, got := canonicalLifecycle(canonical.Input), canonicalLifecycle(decoded.Input)
				if got != want {
					t.Fatalf("%s -> %s lifecycle mismatch\nwant: %s\n got: %s\nwire: %s", source.name, target.name, want, got, encoded.Body)
				}
				if source.name == "responses" && target.name == "responses" &&
					(decoded.Input[1].ID != "item_weather" || decoded.Input[1].ToolCall.ID != "item_weather") {
					t.Fatalf("Responses identity edge lost item_id: %#v\nwire: %s", decoded.Input[1], encoded.Body)
				}
			})
		}
	}
}

func TestWebSearchCapabilityConversionPreservesSupportedConfiguration(t *testing.T) {
	t.Parallel()
	responsesInbound := responses.NewInboundTransformer()
	anthropicInbound := anthropic.NewInboundTransformer()

	responsesRequest, err := responsesInbound.TransformRequest(context.Background(), jsonRequest("/v1/responses",
		`{"model":"fixture-model","input":"search","tools":[{"type":"web_search","filters":{"allowed_domains":["example.com"]},"user_location":{"type":"approximate","city":"Singapore","country":"SG","timezone":"Asia/Singapore"}}]}`))
	if err != nil {
		t.Fatalf("decode Responses web search: %v", err)
	}
	anthropicOutbound, err := anthropic.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
	if err != nil {
		t.Fatalf("create Anthropic outbound: %v", err)
	}
	toAnthropic, err := conversion.NewOutbound(anthropicOutbound).TransformRequest(context.Background(), responsesRequest)
	if err != nil {
		t.Fatalf("convert Responses web search to Anthropic: %v", err)
	}
	decodedAnthropic, err := anthropicInbound.TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/messages", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: toAnthropic.Body,
	})
	if err != nil {
		t.Fatalf("decode converted Anthropic web search: %v\n%s", err, toAnthropic.Body)
	}
	assertCanonicalWebSearch(t, decodedAnthropic, []string{"example.com"}, nil, "Singapore", "SG", "Asia/Singapore")

	anthropicRequest, err := anthropicInbound.TransformRequest(context.Background(), jsonRequest("/v1/messages",
		`{"model":"fixture-model","max_tokens":128,"messages":[{"role":"user","content":"search"}],"tools":[{"type":"web_search_20250305","name":"web_search","allowed_domains":["example.org"],"user_location":{"type":"approximate","city":"London","country":"GB","timezone":"Europe/London"}}]}`))
	if err != nil {
		t.Fatalf("decode Anthropic web search: %v", err)
	}
	responsesOutbound, err := responses.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	toResponses, err := conversion.NewOutbound(responsesOutbound).TransformRequest(context.Background(), anthropicRequest)
	if err != nil {
		t.Fatalf("convert Anthropic web search to Responses: %v", err)
	}
	decodedResponses, err := responsesInbound.TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: toResponses.Body,
	})
	if err != nil {
		t.Fatalf("decode converted Responses web search: %v\n%s", err, toResponses.Body)
	}
	assertCanonicalWebSearch(t, decodedResponses, []string{"example.org"}, nil, "London", "GB", "Europe/London")
}

func TestWebSearchCapabilityConversionRejectsUnsupportedControls(t *testing.T) {
	t.Parallel()
	request, err := anthropic.NewInboundTransformer().TransformRequest(context.Background(), jsonRequest("/v1/messages",
		`{"model":"fixture-model","max_tokens":128,"messages":[{"role":"user","content":"search"}],"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":2,"blocked_domains":["blocked.example"]}]}`))
	if err != nil {
		t.Fatalf("decode Anthropic web search controls: %v", err)
	}
	plan, err := conversion.NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err == nil || plan == nil || plan.Complete() || plan.Summary.Unknown != 1 {
		t.Fatalf("unsupported Responses web search plan = %#v, err=%v", plan, err)
	}
}

func TestResponsesHostedHistoryIdentityRouteReplaysNativeItem(t *testing.T) {
	t.Parallel()
	inbound := responses.NewInboundTransformer()
	body := `{"model":"fixture-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"draw"}]},{"type":"image_generation_call","id":"ig_identity","status":"completed","action":"generate","background":"opaque","output_format":"webp","quality":"low","result":"aW1hZ2U=","revised_prompt":"draw a fox"}]}`
	canonical, err := inbound.TransformRequest(context.Background(), jsonRequest("/v1/responses", body))
	if err != nil {
		t.Fatalf("decode hosted identity source: %v", err)
	}
	outbound, err := responses.NewOutboundTransformer("http://127.0.0.1:1", "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	wire, err := conversion.NewOutbound(outbound).TransformRequest(context.Background(), canonical)
	if err != nil {
		t.Fatalf("encode hosted identity route: %v", err)
	}
	for _, fragment := range []string{`"id":"ig_identity"`, `"type":"image_generation_call"`, `"result":"aW1hZ2U="`, `"revised_prompt":"draw a fox"`} {
		if !strings.Contains(string(wire.Body), fragment) {
			t.Fatalf("Responses hosted identity lost %s: %s", fragment, wire.Body)
		}
	}
	decoded, err := inbound.TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: wire.Body,
	})
	if err != nil || len(decoded.Input) != 2 || decoded.Input[1].Kind != llm.ItemKindHostedCall {
		t.Fatalf("decode hosted identity wire: request=%#v err=%v", decoded, err)
	}
}

func assertCanonicalWebSearch(t *testing.T, request *llm.Request, allowed, blocked []string, city, country, timezone string) {
	t.Helper()
	if request == nil || len(request.ToolDefinitions) != 1 || request.ToolDefinitions[0].Hosted == nil ||
		request.ToolDefinitions[0].Hosted.WebSearch == nil {
		t.Fatalf("canonical web search definition = %#v", request)
	}
	webSearch := request.ToolDefinitions[0].Hosted.WebSearch
	if strings.Join(webSearch.AllowedDomains, ",") != strings.Join(allowed, ",") ||
		strings.Join(webSearch.BlockedDomains, ",") != strings.Join(blocked, ",") ||
		webSearch.UserLocation.City != city || webSearch.UserLocation.Country != country || webSearch.UserLocation.Timezone != timezone {
		t.Fatalf("canonical web search configuration = %#v", webSearch)
	}
}

func jsonRequest(path, body string) *httpclient.Request {
	return &httpclient.Request{
		Method: http.MethodPost, URL: path,
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(body),
	}
}

func canonicalLifecycle(items []llm.Item) string {
	parts := make([]string, 0, len(items))
	for index := range items {
		item := &items[index]
		switch item.Kind {
		case llm.ItemKindMessage:
			var text strings.Builder
			for contentIndex := range item.Content {
				if item.Content[contentIndex].Kind == llm.ContentKindText {
					text.WriteString(item.Content[contentIndex].Text)
				}
			}
			parts = append(parts, fmt.Sprintf("message:%s:%s", item.Role, text.String()))
		case llm.ItemKindToolCall:
			call := item.ToolCall
			parts = append(parts, fmt.Sprintf("call:%s:%s:%s:%s", call.Kind, call.CallID, call.LogicalName, string(call.ArgumentsJSON)+call.ArgumentsText))
		case llm.ItemKindToolResult:
			result := item.ToolResult
			text := ""
			if len(result.Content) > 0 {
				text = result.Content[0].Text
			}
			parts = append(parts, fmt.Sprintf("result:%s:%s:%s", result.Kind, result.CallID, text))
		}
	}
	return strings.Join(parts, "|")
}

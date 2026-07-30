package conversion_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/transformer"
	"github.com/looplj/axonhub/llm/transformer/anthropic"
	"github.com/looplj/axonhub/llm/transformer/openai"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestClientToolSearchDiscoversThenCallsToolOverRealHTTP3Targets(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name        string
		path        string
		format      string
		newOutbound func(string) (transformer.Outbound, error)
	}{
		{
			name: "chat", path: "/v1/chat/completions", format: "chat",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
		},
		{
			name: "responses", path: "/v1/responses", format: "responses",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
		},
		{
			name: "anthropic", path: "/v1/messages", format: "anthropic",
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-provider-key")
			},
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			fixture := &toolSearchLifecycleProvider{format: tt.format}
			provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != tt.path {
					http.Error(w, "unexpected provider path", http.StatusNotFound)
					return
				}
				fixture.ServeHTTP(w, r)
			}))
			t.Cleanup(provider.Close)

			target, err := tt.newOutbound(provider.URL)
			if err != nil {
				t.Fatalf("create %s outbound: %v", tt.name, err)
			}
			executor := httpclient.NewHttpClientWithClient(provider.Client())
			t.Cleanup(executor.CloseIdleConnections)
			protocolPipeline := pipeline.NewFactory(executor).Pipeline(
				responses.NewInboundTransformer(), conversion.NewOutbound(target),
			)
			ctx := context.Background()

			first, err := protocolPipeline.Process(ctx, toolSearchClientRequest(toolSearchTurnInitial))
			if err != nil {
				t.Fatalf("%s search turn: %v", tt.name, err)
			}
			assertResponsesOutputItem(t, first.Response.Body, "tool_search_call", "search_call_1", "")

			second, err := protocolPipeline.Process(ctx, toolSearchClientRequest(toolSearchTurnDiscovered))
			if err != nil {
				t.Fatalf("%s discovered turn: %v", tt.name, err)
			}
			assertResponsesOutputItem(t, second.Response.Body, "function_call", "create_call_1", "create_event")

			third, err := protocolPipeline.Process(ctx, toolSearchClientRequest(toolSearchTurnCompleted))
			if err != nil {
				t.Fatalf("%s completion turn: %v", tt.name, err)
			}
			assertResponsesFinalText(t, third.Response.Body, "event created")
			if got := fixture.Rounds(); got != 3 {
				t.Fatalf("%s provider rounds = %d, want 3", tt.name, got)
			}
		})
	}
}

type toolSearchTurn int

const (
	toolSearchTurnInitial toolSearchTurn = iota
	toolSearchTurnDiscovered
	toolSearchTurnCompleted
)

func toolSearchClientRequest(turn toolSearchTurn) *httpclient.Request {
	input := `[{"type":"message","role":"user","content":[{"type":"input_text","text":"create a calendar event"}]}]`
	if turn >= toolSearchTurnDiscovered {
		input = `[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"create a calendar event"}]},
			{"type":"tool_search_call","id":"tsc_1","call_id":"search_call_1","status":"completed","execution":"client","arguments":{"query":"calendar event","limit":1}},
			{"type":"tool_search_output","id":"tso_1","call_id":"search_call_1","status":"completed","execution":"client","tools":[
				{"type":"function","name":"create_event","description":"Create a calendar event","defer_loading":true,"parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}
			]}
		]`
	}
	if turn >= toolSearchTurnCompleted {
		input = `[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"create a calendar event"}]},
			{"type":"tool_search_call","id":"tsc_1","call_id":"search_call_1","status":"completed","execution":"client","arguments":{"query":"calendar event","limit":1}},
			{"type":"tool_search_output","id":"tso_1","call_id":"search_call_1","status":"completed","execution":"client","tools":[
				{"type":"function","name":"create_event","description":"Create a calendar event","defer_loading":true,"parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}}
			]},
			{"type":"function_call","id":"fc_1","call_id":"create_call_1","name":"create_event","status":"completed","arguments":{"title":"Architecture review"}},
			{"type":"function_call_output","id":"fco_1","call_id":"create_call_1","status":"completed","output":"created"}
		]`
	}
	body := fmt.Sprintf(`{
		"model":"fixture-model","input":%s,
		"tools":[
			{"type":"tool_search","execution":"client","description":"Search deferred tools","parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"}},"required":["query"],"additionalProperties":false}},
			{"type":"function","name":"always_available","description":"Always available","parameters":{"type":"object","properties":{}}},
			{"type":"function","name":"create_event","description":"Create a calendar event","defer_loading":true,"parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"],"additionalProperties":false}},
			{"type":"function","name":"delete_event","description":"Delete a calendar event","defer_loading":true,"parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}}
		]
	}`, input)
	return &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: []byte(body),
	}
}

type toolSearchLifecycleProvider struct {
	mu         sync.Mutex
	format     string
	rounds     int
	searchName string
}

func (provider *toolSearchLifecycleProvider) Rounds() int {
	provider.mu.Lock()
	defer provider.mu.Unlock()
	return provider.rounds
}

func (provider *toolSearchLifecycleProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read provider request", http.StatusBadRequest)
		return
	}
	provider.mu.Lock()
	defer provider.mu.Unlock()
	provider.rounds++
	round := provider.rounds

	names, deferred, history, err := parseToolSearchProviderRequest(provider.format, body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := provider.validateRound(round, names, deferred, history); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, provider.response(round))
}

func (provider *toolSearchLifecycleProvider) validateRound(round int, names []string, deferred map[string]bool, history string) error {
	contains := func(wanted string) bool {
		for _, name := range names {
			if name == wanted {
				return true
			}
		}
		return false
	}
	if provider.format == "responses" {
		if len(names) != 4 || !contains("tool_search") || !contains("always_available") || !contains("create_event") || !contains("delete_event") {
			return fmt.Errorf("native Responses catalog degraded in round %d: %v", round, names)
		}
		if !deferred["create_event"] || !deferred["delete_event"] {
			return fmt.Errorf("native Responses deferred flags degraded in round %d: %#v", round, deferred)
		}
	} else {
		if provider.searchName == "" {
			for _, name := range names {
				if strings.HasPrefix(name, "axc_") {
					provider.searchName = name
					break
				}
			}
		}
		if provider.searchName == "" || !contains(provider.searchName) || !contains("always_available") {
			return fmt.Errorf("lowered search/eager tools missing in round %d: %v", round, names)
		}
		if round == 1 && (contains("create_event") || contains("delete_event")) {
			return fmt.Errorf("deferred tools leaked before discovery: %v", names)
		}
		if round >= 2 && (!contains("create_event") || contains("delete_event")) {
			return fmt.Errorf("discovered catalog is not exact in round %d: %v", round, names)
		}
		if round >= 2 && deferred["create_event"] {
			return fmt.Errorf("unsupported defer_loading leaked to %s target", provider.format)
		}
	}
	if round >= 2 && !strings.Contains(history, "search_call_1") {
		return fmt.Errorf("tool search history missing in round %d", round)
	}
	if round >= 3 && (!strings.Contains(history, "create_call_1") || !strings.Contains(history, "created")) {
		return fmt.Errorf("discovered tool result history missing")
	}
	return nil
}

func (provider *toolSearchLifecycleProvider) response(round int) string {
	searchName := provider.searchName
	if provider.format == "responses" {
		searchName = "tool_search"
	}
	switch provider.format {
	case "chat":
		switch round {
		case 1:
			return fmt.Sprintf(`{"id":"chat_search_1","object":"chat.completion","created":1785429000,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"search_call_1","type":"function","function":{"name":%q,"arguments":"{\"query\":\"calendar event\",\"limit\":1}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6}}`, searchName)
		case 2:
			return `{"id":"chat_search_2","object":"chat.completion","created":1785429001,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[{"id":"create_call_1","type":"function","function":{"name":"create_event","arguments":"{\"title\":\"Architecture review\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":6,"completion_tokens":2,"total_tokens":8}}`
		default:
			return `{"id":"chat_search_3","object":"chat.completion","created":1785429002,"model":"fixture-model","choices":[{"index":0,"message":{"role":"assistant","content":"event created"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`
		}
	case "anthropic":
		switch round {
		case 1:
			return fmt.Sprintf(`{"id":"msg_search_1","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"search_call_1","name":%q,"input":{"query":"calendar event","limit":1}}],"stop_reason":"tool_use","usage":{"input_tokens":4,"output_tokens":2}}`, searchName)
		case 2:
			return `{"id":"msg_search_2","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"tool_use","id":"create_call_1","name":"create_event","input":{"title":"Architecture review"}}],"stop_reason":"tool_use","usage":{"input_tokens":6,"output_tokens":2}}`
		default:
			return `{"id":"msg_search_3","type":"message","role":"assistant","model":"fixture-model","content":[{"type":"text","text":"event created"}],"stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":2}}`
		}
	default:
		switch round {
		case 1:
			return `{"id":"resp_search_1","object":"response","created_at":1785429000,"model":"fixture-model","status":"completed","output":[{"type":"tool_search_call","id":"tsc_provider_1","call_id":"search_call_1","status":"completed","execution":"client","arguments":{"query":"calendar event","limit":1}}],"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}`
		case 2:
			return `{"id":"resp_search_2","object":"response","created_at":1785429001,"model":"fixture-model","status":"completed","output":[{"type":"function_call","id":"fc_provider_1","call_id":"create_call_1","name":"create_event","status":"completed","arguments":{"title":"Architecture review"}}],"usage":{"input_tokens":6,"output_tokens":2,"total_tokens":8}}`
		default:
			return `{"id":"resp_search_3","object":"response","created_at":1785429002,"model":"fixture-model","status":"completed","output":[{"type":"message","id":"msg_provider_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"event created","annotations":[]}]}],"usage":{"input_tokens":7,"output_tokens":2,"total_tokens":9}}`
		}
	}
}

func parseToolSearchProviderRequest(format string, body []byte) ([]string, map[string]bool, string, error) {
	deferred := make(map[string]bool)
	names := make([]string, 0)
	switch format {
	case "chat":
		var payload struct {
			Tools []struct {
				Type     string `json:"type"`
				Function struct {
					Name         string `json:"name"`
					DeferLoading *bool  `json:"defer_loading"`
				} `json:"function"`
			} `json:"tools"`
			Messages json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, nil, "", fmt.Errorf("decode Chat provider request: %w", err)
		}
		for _, tool := range payload.Tools {
			names = append(names, tool.Function.Name)
			deferred[tool.Function.Name] = tool.Function.DeferLoading != nil && *tool.Function.DeferLoading
		}
		return names, deferred, string(payload.Messages), nil
	case "anthropic":
		var payload struct {
			Tools []struct {
				Name         string `json:"name"`
				DeferLoading *bool  `json:"defer_loading"`
			} `json:"tools"`
			Messages json.RawMessage `json:"messages"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, nil, "", fmt.Errorf("decode Anthropic provider request: %w", err)
		}
		for _, tool := range payload.Tools {
			names = append(names, tool.Name)
			deferred[tool.Name] = tool.DeferLoading != nil && *tool.DeferLoading
		}
		return names, deferred, string(payload.Messages), nil
	default:
		var payload struct {
			Tools []struct {
				Type         string `json:"type"`
				Name         string `json:"name"`
				DeferLoading *bool  `json:"defer_loading"`
			} `json:"tools"`
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, nil, "", fmt.Errorf("decode Responses provider request: %w", err)
		}
		for _, tool := range payload.Tools {
			name := tool.Name
			if name == "" {
				name = tool.Type
			}
			names = append(names, name)
			deferred[name] = tool.DeferLoading != nil && *tool.DeferLoading
		}
		return names, deferred, string(payload.Input), nil
	}
}

func assertResponsesOutputItem(t *testing.T, body []byte, itemType, callID, name string) {
	t.Helper()
	var payload struct {
		Output []struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Responses client body: %v body=%s", err, body)
	}
	if len(payload.Output) != 1 || payload.Output[0].Type != itemType || payload.Output[0].CallID != callID || payload.Output[0].Name != name {
		t.Fatalf("Responses output = %#v, want type=%s call_id=%s name=%s; body=%s", payload.Output, itemType, callID, name, body)
	}
}

func assertResponsesFinalText(t *testing.T, body []byte, wanted string) {
	t.Helper()
	var payload struct {
		Output []struct {
			Type    string `json:"type"`
			Content []struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"output"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode final Responses body: %v body=%s", err, body)
	}
	if len(payload.Output) != 1 || payload.Output[0].Type != "message" || len(payload.Output[0].Content) != 1 || payload.Output[0].Content[0].Text != wanted {
		t.Fatalf("final Responses output = %#v, want %q; body=%s", payload.Output, wanted, body)
	}
}

package conversion_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sort"
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

// TestCodexCollaborationToolSchemasSurviveAllProviderWiresOverRealHTTP freezes
// the schema-bearing part of Codex 0.147.0's Responses Lite declaration.
//
// Source construction, deliberately read rather than inferred from an Axon
// failure payload:
//   - codex-rs/core/src/client.rs: build_responses_request() puts every ToolSpec
//     into input[0].additional_tools when use_responses_lite is true;
//   - codex-rs/core/src/tools/spec_plan.rs:add_collaboration_tools() selects
//     exactly one V1 or V2 collaboration family;
//   - codex-rs/core/src/tools/handlers/multi_agents_spec.rs defines both
//     families and request_user_input_spec.rs / shell_spec.rs define the
//     standalone request_user_input and exec_command schemas.
//
// The wire digest intentionally excludes descriptions, encrypted markers,
// argument payloads, provider credentials, and every other non-schema field.
// It is therefore safe to emit in failures while still catching type, required,
// property, bounds, pattern, enum, default, and additionalProperties loss.
func TestCodexCollaborationToolSchemasSurviveAllProviderWiresOverRealHTTP(t *testing.T) {
	t.Parallel()

	for _, source := range []struct {
		name string
		body []byte
	}{
		{name: "multi_agent_v1", body: codexResponsesLiteCollaborationV1Fixture()},
		{name: "multi_agent_v2", body: codexResponsesLiteCollaborationV2Fixture()},
	} {
		source := source
		t.Run(source.name, func(t *testing.T) {
			t.Parallel()

			want := codexSourceToolDigest(t, source.body)
			canonical, err := responses.NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
				Method: http.MethodPost,
				URL:    "/v1/responses",
				Headers: http.Header{
					"Content-Type":                []string{"application/json"},
					responses.ResponsesLiteHeader: []string{"true"},
				},
				Body: source.body,
			})
			if err != nil {
				t.Fatalf("decode frozen Codex %s source declaration: %v", source.name, err)
			}
			if got := canonicalRegistryToolDigest(t, canonical); !slices.Equal(got, want) {
				t.Fatalf("Codex %s source -> canonical registry schema digest changed\nwant: %s\n got: %s", source.name, renderCodexToolDigest(want), renderCodexToolDigest(got))
			}

			for _, target := range codexSchemaHTTPProviderTargets(t) {
				target := target
				t.Run(target.name, func(t *testing.T) {
					t.Parallel()

					provider := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
						if request.URL.Path != target.path {
							http.Error(writer, "unexpected provider path", http.StatusNotFound)
							return
						}
						body, readErr := io.ReadAll(request.Body)
						if readErr != nil {
							http.Error(writer, "read provider request", http.StatusBadRequest)
							return
						}
						if got := target.digest(t, body); !slices.Equal(got, want) {
							t.Logf("%s -> %s schema digest mismatch\nwant: %s\n got: %s", source.name, target.name, renderCodexToolDigest(want), renderCodexToolDigest(got))
							http.Error(writer, "Codex tool schema changed before provider dispatch: want="+renderCodexToolDigest(want)+" got="+renderCodexToolDigest(got), http.StatusBadRequest)
							return
						}
						writer.Header().Set("Content-Type", "application/json")
						_, _ = io.WriteString(writer, target.response)
					}))
					t.Cleanup(provider.Close)

					outbound, err := target.newOutbound(provider.URL)
					if err != nil {
						t.Fatalf("create %s outbound: %v", target.name, err)
					}
					executor := httpclient.NewHttpClientWithClient(provider.Client())
					t.Cleanup(executor.CloseIdleConnections)

					result, err := pipeline.NewFactory(executor).
						Pipeline(responses.NewInboundTransformer(), conversion.NewOutbound(outbound)).
						Process(context.Background(), &httpclient.Request{
							Method: http.MethodPost,
							URL:    "/v1/responses",
							Headers: http.Header{
								"Content-Type":                []string{"application/json"},
								responses.ResponsesLiteHeader: []string{"true"},
							},
							Body: source.body,
						})
					if err != nil {
						t.Fatalf("convert Codex %s declaration to %s over real HTTP: %v", source.name, target.name, err)
					}
					if result == nil || result.Response == nil {
						t.Fatalf("Codex %s -> %s missing provider response: %#v", source.name, target.name, result)
					}
				})
			}
		})
	}
}

type codexSchemaHTTPProviderTarget struct {
	name        string
	path        string
	response    string
	newOutbound func(string) (transformer.Outbound, error)
	digest      func(*testing.T, []byte) []codexToolDigest
}

func codexSchemaHTTPProviderTargets(t *testing.T) []codexSchemaHTTPProviderTarget {
	t.Helper()
	return []codexSchemaHTTPProviderTarget{
		{
			name:     "openai_responses",
			path:     "/v1/responses",
			response: `{"id":"resp_schema","object":"response","model":"fixture","status":"completed","output":[{"id":"msg_schema","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"ok","annotations":[]}]}]}`,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return responses.NewOutboundTransformer(baseURL, "fixture-key")
			},
			digest: codexResponsesProviderToolDigest,
		},
		{
			name:     "openai_chat_completions",
			path:     "/v1/chat/completions",
			response: `{"id":"chat_schema","object":"chat.completion","model":"fixture","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return openai.NewOutboundTransformer(baseURL, "fixture-key")
			},
			digest: codexChatProviderToolDigest,
		},
		{
			name:     "anthropic_messages",
			path:     "/v1/messages",
			response: `{"id":"msg_schema","type":"message","role":"assistant","model":"fixture","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`,
			newOutbound: func(baseURL string) (transformer.Outbound, error) {
				return anthropic.NewOutboundTransformer(baseURL, "fixture-key")
			},
			digest: codexAnthropicProviderToolDigest,
		},
	}
}

// codexToolDigest is a semantic, source-independent comparison unit. Namespaces
// are first-class here even though Chat and Anthropic require Axon to encode
// them into a safe flattened function name on the wire.
type codexToolDigest struct {
	Namespace string
	Name      string
	Schema    string
}

func renderCodexToolDigest(digest []codexToolDigest) string {
	encoded, err := json.Marshal(digest)
	if err != nil {
		return fmt.Sprintf("marshal digest: %v", err)
	}
	return string(encoded)
}

func codexSourceToolDigest(t *testing.T, body []byte) []codexToolDigest {
	t.Helper()
	var request struct {
		Input []struct {
			Type  string            `json:"type"`
			Tools []json.RawMessage `json:"tools"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &request); err != nil {
		t.Fatalf("decode frozen Codex source: %v", err)
	}
	if len(request.Input) == 0 || request.Input[0].Type != "additional_tools" {
		t.Fatalf("frozen Codex source must start with additional_tools")
	}
	return codexResponsesToolsDigest(t, request.Input[0].Tools)
}

func canonicalRegistryToolDigest(t *testing.T, request *llm.Request) []codexToolDigest {
	t.Helper()
	if request == nil {
		t.Fatal("missing canonical request")
	}
	digest := make([]codexToolDigest, 0, len(request.ToolDefinitions))
	for index := range request.ToolDefinitions {
		definition := request.ToolDefinitions[index]
		if definition.Kind != llm.ToolKindFunction || definition.Function == nil {
			t.Fatalf("canonical definition %d is not a typed Codex function: %#v", index, definition)
		}
		requireCodexFunctionContract(t, definition.Description, definition.Function.Strict, "canonical definition", definition.LogicalName)
		name := definition.LogicalName
		namespace := definition.Function.Namespace
		if namespace != "" {
			name = strings.TrimPrefix(name, namespace+"__")
		}
		digest = append(digest, codexToolDigest{
			Namespace: namespace,
			Name:      name,
			Schema:    codexStructuralSchemaDigest(t, definition.Function.Parameters),
		})
	}
	return sortCodexToolDigest(digest)
}

func codexResponsesProviderToolDigest(t *testing.T, body []byte) []codexToolDigest {
	t.Helper()
	var payload struct {
		Input []struct {
			Type  string            `json:"type"`
			Tools []json.RawMessage `json:"tools"`
		} `json:"input"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Responses provider wire: %v", err)
	}
	for _, item := range payload.Input {
		if item.Type == "additional_tools" {
			return codexResponsesToolsDigest(t, item.Tools)
		}
	}
	t.Fatalf("Responses provider wire lacks additional_tools")
	return nil
}

func codexResponsesToolsDigest(t *testing.T, rawTools []json.RawMessage) []codexToolDigest {
	t.Helper()
	digest := make([]codexToolDigest, 0, len(rawTools))
	for index, raw := range rawTools {
		var tool struct {
			Type        string            `json:"type"`
			Name        string            `json:"name"`
			Description string            `json:"description"`
			Strict      *bool             `json:"strict"`
			Parameters  json.RawMessage   `json:"parameters"`
			Tools       []json.RawMessage `json:"tools"`
		}
		if err := json.Unmarshal(raw, &tool); err != nil {
			t.Fatalf("decode Responses tool %d: %v", index, err)
		}
		switch tool.Type {
		case "function":
			requireCodexFunctionContract(t, tool.Description, tool.Strict, "Responses provider function", tool.Name)
			digest = append(digest, codexToolDigest{Name: tool.Name, Schema: codexStructuralSchemaDigest(t, tool.Parameters)})
		case "namespace":
			if strings.TrimSpace(tool.Description) == "" {
				t.Fatalf("Responses provider namespace %q has no description", tool.Name)
			}
			for childIndex, childRaw := range tool.Tools {
				var child struct {
					Type        string          `json:"type"`
					Name        string          `json:"name"`
					Description string          `json:"description"`
					Strict      *bool           `json:"strict"`
					Parameters  json.RawMessage `json:"parameters"`
				}
				if err := json.Unmarshal(childRaw, &child); err != nil {
					t.Fatalf("decode Responses namespace %q child %d: %v", tool.Name, childIndex, err)
				}
				if child.Type != "function" {
					t.Fatalf("Codex namespace %q child %q type = %q, want function", tool.Name, child.Name, child.Type)
				}
				requireCodexFunctionContract(t, child.Description, child.Strict, "Responses provider namespace function", tool.Name+"."+child.Name)
				digest = append(digest, codexToolDigest{Namespace: tool.Name, Name: child.Name, Schema: codexStructuralSchemaDigest(t, child.Parameters)})
			}
		default:
			t.Fatalf("Codex source fixture contains unsupported behavioral tool type %q at %d", tool.Type, index)
		}
	}
	return sortCodexToolDigest(digest)
}

func codexChatProviderToolDigest(t *testing.T, body []byte) []codexToolDigest {
	t.Helper()
	var payload struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Strict      *bool           `json:"strict"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Chat provider wire: %v", err)
	}
	digest := make([]codexToolDigest, 0, len(payload.Tools))
	for index, tool := range payload.Tools {
		if tool.Type != "function" {
			t.Fatalf("Chat provider tool %d type = %q, want function", index, tool.Type)
		}
		requireCodexFunctionContract(t, tool.Function.Description, tool.Function.Strict, "Chat provider function", tool.Function.Name)
		namespace, name := splitCodexFlattenedToolName(tool.Function.Name)
		digest = append(digest, codexToolDigest{Namespace: namespace, Name: name, Schema: codexStructuralSchemaDigest(t, tool.Function.Parameters)})
	}
	return sortCodexToolDigest(digest)
}

func codexAnthropicProviderToolDigest(t *testing.T, body []byte) []codexToolDigest {
	t.Helper()
	var payload struct {
		Tools []struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Strict      *bool           `json:"strict"`
			InputSchema json.RawMessage `json:"input_schema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("decode Anthropic provider wire: %v", err)
	}
	digest := make([]codexToolDigest, 0, len(payload.Tools))
	for index, tool := range payload.Tools {
		if tool.Name == "" {
			t.Fatalf("Anthropic provider tool %d missing name", index)
		}
		requireCodexFunctionContract(t, tool.Description, tool.Strict, "Anthropic provider function", tool.Name)
		namespace, name := splitCodexFlattenedToolName(tool.Name)
		digest = append(digest, codexToolDigest{Namespace: namespace, Name: name, Schema: codexStructuralSchemaDigest(t, tool.InputSchema)})
	}
	return sortCodexToolDigest(digest)
}

func splitCodexFlattenedToolName(name string) (string, string) {
	if namespace, child, ok := strings.Cut(name, "__"); ok {
		return namespace, child
	}
	return "", name
}

func requireCodexFunctionContract(t *testing.T, description string, strict *bool, scope, name string) {
	t.Helper()
	if strings.TrimSpace(description) == "" {
		t.Fatalf("%s %q has no description", scope, name)
	}
	if strict == nil || *strict {
		t.Fatalf("%s %q strict=%v, want explicit false from Codex source", scope, name, strict)
	}
}

func sortCodexToolDigest(digest []codexToolDigest) []codexToolDigest {
	sort.Slice(digest, func(left, right int) bool {
		if digest[left].Namespace != digest[right].Namespace {
			return digest[left].Namespace < digest[right].Namespace
		}
		return digest[left].Name < digest[right].Name
	})
	return digest
}

func codexStructuralSchemaDigest(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var schema any
	if len(raw) == 0 || json.Unmarshal(raw, &schema) != nil {
		t.Fatalf("decode tool JSON schema: %s", raw)
	}
	projected := codexProjectSchema(schema)
	encoded, err := json.Marshal(projected)
	if err != nil {
		t.Fatalf("encode safe structural schema digest: %v", err)
	}
	return string(encoded)
}

func codexProjectSchema(value any) any {
	object, ok := value.(map[string]any)
	if !ok {
		return nil
	}
	projected := make(map[string]any)
	for _, key := range []string{
		"type", "required", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
		"minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties",
		"pattern", "enum", "default", "additionalProperties",
	} {
		if current, exists := object[key]; exists {
			projected[key] = current
		}
	}
	if properties, ok := object["properties"].(map[string]any); ok {
		projectedProperties := make(map[string]any, len(properties))
		for name, property := range properties {
			projectedProperties[name] = codexProjectSchema(property)
		}
		projected["properties"] = projectedProperties
	}
	if items, exists := object["items"]; exists {
		projected["items"] = codexProjectSchema(items)
	}
	return projected
}

func codexResponsesLiteCollaborationV1Fixture() []byte {
	// Frozen from Codex c775eb7e tool constructors. V1 is intentionally kept
	// separate from V2 because spec_plan selects one collaboration family per
	// turn; merging them would cease to be a real client declaration.
	return []byte(`{
  "model":"gpt-5.6-sol",
  "input":[
    {"type":"additional_tools","role":"developer","tools":[
      {"type":"function","name":"exec_command","description":"Run command.","strict":false,"parameters":{"type":"object","properties":{"cmd":{"type":"string"},"workdir":{"type":"string"},"tty":{"type":"boolean"},"yield_time_ms":{"type":"number"},"max_output_tokens":{"type":"number"},"shell":{"type":"string"},"login":{"type":"boolean"}},"required":["cmd"],"additionalProperties":false}},
      {"type":"function","name":"write_stdin","description":"Write stdin.","strict":false,"parameters":{"type":"object","properties":{"session_id":{"type":"number"},"chars":{"type":"string"},"yield_time_ms":{"type":"number"},"max_output_tokens":{"type":"number"}},"required":["session_id"],"additionalProperties":false}},
      {"type":"function","name":"request_user_input","description":"Ask user.","strict":false,"parameters":{"type":"object","properties":{"questions":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"header":{"type":"string"},"question":{"type":"string"},"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}},"required":["label","description"],"additionalProperties":false}}},"required":["id","header","question","options"],"additionalProperties":false}},"autoResolutionMs":{"type":"number"}},"required":["questions"],"additionalProperties":false}},
      {"type":"namespace","name":"multi_agent_v1","description":"Tools for spawning and managing sub-agents.","tools":[
        {"type":"function","name":"spawn_agent","description":"Spawn agent.","strict":false,"parameters":{"type":"object","properties":{"message":{"type":"string"},"items":{"type":"array","items":{"type":"object","properties":{"type":{"type":"string"},"text":{"type":"string"},"image_url":{"type":"string"},"path":{"type":"string"},"name":{"type":"string"}},"additionalProperties":false}},"agent_type":{"type":"string"},"fork_context":{"type":"boolean"},"model":{"type":"string"},"reasoning_effort":{"type":"string"},"service_tier":{"type":"string"}},"additionalProperties":false}},
        {"type":"function","name":"send_input","description":"Send input.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"},"message":{"type":"string"},"items":{"type":"array","items":{"type":"object","properties":{"type":{"type":"string"},"text":{"type":"string"},"image_url":{"type":"string"},"path":{"type":"string"},"name":{"type":"string"}},"additionalProperties":false}},"interrupt":{"type":"boolean"}},"required":["target"],"additionalProperties":false}},
        {"type":"function","name":"resume_agent","description":"Resume agent.","strict":false,"parameters":{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}},
        {"type":"function","name":"wait_agent","description":"Wait agent.","strict":false,"parameters":{"type":"object","properties":{"targets":{"type":"array","items":{"type":"string"}},"timeout_ms":{"type":"number"}},"required":["targets"],"additionalProperties":false}},
        {"type":"function","name":"close_agent","description":"Close agent.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"],"additionalProperties":false}}
      ]}
    ]},
    {"type":"message","role":"user","content":[{"type":"input_text","text":"schema check"}]}
  ]
}`)
}

func codexResponsesLiteCollaborationV2Fixture() []byte {
	return []byte(`{
  "model":"gpt-5.6-sol",
  "input":[
    {"type":"additional_tools","role":"developer","tools":[
      {"type":"function","name":"exec_command","description":"Run command.","strict":false,"parameters":{"type":"object","properties":{"cmd":{"type":"string"},"workdir":{"type":"string"},"tty":{"type":"boolean"},"yield_time_ms":{"type":"number"},"max_output_tokens":{"type":"number"},"shell":{"type":"string"},"login":{"type":"boolean"}},"required":["cmd"],"additionalProperties":false}},
      {"type":"function","name":"write_stdin","description":"Write stdin.","strict":false,"parameters":{"type":"object","properties":{"session_id":{"type":"number"},"chars":{"type":"string"},"yield_time_ms":{"type":"number"},"max_output_tokens":{"type":"number"}},"required":["session_id"],"additionalProperties":false}},
      {"type":"function","name":"request_user_input","description":"Ask user.","strict":false,"parameters":{"type":"object","properties":{"questions":{"type":"array","items":{"type":"object","properties":{"id":{"type":"string"},"header":{"type":"string"},"question":{"type":"string"},"options":{"type":"array","items":{"type":"object","properties":{"label":{"type":"string"},"description":{"type":"string"}},"required":["label","description"],"additionalProperties":false}}},"required":["id","header","question","options"],"additionalProperties":false}},"autoResolutionMs":{"type":"number"}},"required":["questions"],"additionalProperties":false}},
      {"type":"namespace","name":"collaboration","description":"Tools for spawning and managing sub-agents.","tools":[
        {"type":"function","name":"spawn_agent","description":"Spawn agent.","strict":false,"parameters":{"type":"object","properties":{"message":{"type":"string","encrypted":true},"fork_turns":{"type":"string"},"model":{"type":"string"},"reasoning_effort":{"type":"string"},"service_tier":{"type":"string"},"task_name":{"type":"string"}},"required":["task_name","message"],"additionalProperties":false}},
        {"type":"function","name":"send_message","description":"Send message.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"},"message":{"type":"string","encrypted":true}},"required":["target","message"],"additionalProperties":false}},
        {"type":"function","name":"followup_task","description":"Follow up.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"},"message":{"type":"string","encrypted":true}},"required":["target","message"],"additionalProperties":false}},
        {"type":"function","name":"wait_agent","description":"Wait agent.","strict":false,"parameters":{"type":"object","properties":{"timeout_ms":{"type":"number"}},"additionalProperties":false}},
        {"type":"function","name":"interrupt_agent","description":"Interrupt agent.","strict":false,"parameters":{"type":"object","properties":{"target":{"type":"string"}},"required":["target"],"additionalProperties":false}},
        {"type":"function","name":"list_agents","description":"List agents.","strict":false,"parameters":{"type":"object","properties":{"path_prefix":{"type":"string"}},"additionalProperties":false}}
      ]}
    ]},
    {"type":"message","role":"user","content":[{"type":"input_text","text":"schema check"}]}
  ]
}`)
}

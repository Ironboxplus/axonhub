package hosted

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestHTTPExecutorCarriesHostedFamiliesOverRealHTTP(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name   string
		kind   llm.ToolKind
		args   json.RawMessage
		result json.RawMessage
		text   string
	}{
		{name: "file_search", kind: llm.ToolKindFileSearch, args: json.RawMessage(`{"queries":["canonical conversion"]}`), result: json.RawMessage(`[{"file_id":"file_1","filename":"design.md","score":0.9,"text":"typed"}]`), text: `[{"file_id":"file_1","filename":"design.md","score":0.9,"text":"typed"}]`},
		{name: "code_interpreter", kind: llm.ToolKindCodeInterpreter, args: json.RawMessage(`{"code":"print(42)","container_id":"ctr_1"}`), result: json.RawMessage(`[{"type":"logs","logs":"42\\n"}]`), text: `42\n`},
		{name: "shell", kind: llm.ToolKindShell, args: json.RawMessage(`{"commands":["go test ./conversion"],"timeout_ms":30000}`), result: json.RawMessage(`[{"stdout":"ok\\n","stderr":"","outcome":{"type":"exit","exit_code":0}}]`), text: `[{"stdout":"ok\\n","stderr":"","outcome":{"type":"exit","exit_code":0}}]`},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
				defer request.Body.Close()
				if request.Method != http.MethodPost || request.Header.Get("X-Hosted-Test") != "enabled" {
					http.Error(writer, "unexpected transport contract", http.StatusBadRequest)
					return
				}
				var body HTTPExecutorRequest
				if err := json.NewDecoder(request.Body).Decode(&body); err != nil || body.Kind != test.kind ||
					body.LogicalName != test.name || !json.Valid(body.Arguments) || !json.Valid(body.Configuration) {
					http.Error(writer, "unexpected executor request", http.StatusBadRequest)
					return
				}
				if string(body.Arguments) != string(test.args) {
					http.Error(writer, "arguments changed", http.StatusBadRequest)
					return
				}
				writer.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(writer).Encode(HTTPExecutorResponse{Result: test.result})
			}))
			t.Cleanup(server.Close)

			executor, err := NewHTTPExecutor(HTTPExecutorConfig{
				Kind: test.kind, Endpoint: server.URL, Client: server.Client(),
				Headers: http.Header{"X-Hosted-Test": []string{"enabled"}},
				EndpointPolicy: func(candidate *url.URL) error {
					if candidate.String() != server.URL {
						return errors.New("unexpected endpoint")
					}
					return nil
				},
			})
			if err != nil {
				t.Fatalf("create %s executor: %v", test.kind, err)
			}
			definition := llm.ToolDefinition{
				Kind: test.kind, LogicalName: test.name, Execution: llm.ExecutionOwnerProvider,
				Hosted: &llm.HostedToolDefinition{Type: test.name, Configuration: json.RawMessage(`{"type":"` + test.name + `","future_option":true}`)},
			}
			function, err := executor.Function(definition)
			if err != nil || len(function.Parameters) == 0 || !json.Valid(function.Parameters) {
				t.Fatalf("function schema for %s: schema=%s err=%v", test.kind, function.Parameters, err)
			}
			result, err := executor.Execute(context.Background(), definition, llm.ToolInvocation{
				Kind: test.kind, CallID: "call_1", LogicalName: test.name, ArgumentsJSON: test.args,
			})
			if err != nil {
				t.Fatalf("execute %s: %v", test.kind, err)
			}
			if result.Kind != test.kind || result.CallID != "call_1" || result.Execution != llm.ExecutionOwnerGateway ||
				result.Status != llm.ToolResultStatusCompleted || len(result.Content) != 1 ||
				result.Content[0].Kind != llm.ContentKindText || result.Content[0].Text != test.text ||
				string(result.StructuredContent) != string(test.result) {
				t.Fatalf("%s result degraded: %#v", test.kind, result)
			}
		})
	}
}

func TestHTTPExecutorRejectsInvalidTypedArgumentsBeforeNetwork(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name string
		kind llm.ToolKind
		args json.RawMessage
	}{
		{name: "file_search", kind: llm.ToolKindFileSearch, args: json.RawMessage(`{"queries":[]}`)},
		{name: "code", kind: llm.ToolKindCodeExecution, args: json.RawMessage(`{"code":""}`)},
		{name: "shell", kind: llm.ToolKindShell, args: json.RawMessage(`{"commands":[]}`)},
		{name: "image", kind: llm.ToolKindImageGeneration, args: json.RawMessage(`{"prompt":""}`)},
		{name: "computer", kind: llm.ToolKindComputer, args: json.RawMessage(`{"x":1,"y":2}`)},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var calls atomic.Int64
			server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, _ *http.Request) {
				calls.Add(1)
				http.Error(writer, "must not be called", http.StatusInternalServerError)
			}))
			t.Cleanup(server.Close)
			executor, err := NewHTTPExecutor(HTTPExecutorConfig{
				Kind: test.kind, Endpoint: server.URL, Client: server.Client(),
				EndpointPolicy: func(*url.URL) error { return nil },
			})
			if err != nil {
				t.Fatalf("create executor: %v", err)
			}
			_, err = executor.Execute(context.Background(), llm.ToolDefinition{
				Kind: test.kind, LogicalName: test.name, Execution: llm.ExecutionOwnerProvider,
				Hosted: &llm.HostedToolDefinition{Type: test.name, Configuration: json.RawMessage(`{"type":"` + test.name + `","future_option":true}`)},
			}, llm.ToolInvocation{Kind: test.kind, CallID: "call_1", LogicalName: test.name, ArgumentsJSON: test.args})
			if err == nil || calls.Load() != 0 {
				t.Fatalf("invalid typed arguments: err=%v calls=%d", err, calls.Load())
			}
		})
	}
}

func TestHTTPExecutorMapsTypedCodeOutputsOverRealHTTP(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body HTTPExecutorRequest
		if json.NewDecoder(request.Body).Decode(&body) != nil {
			http.Error(writer, "decode", http.StatusBadRequest)
			return
		}
		var arguments CodeArguments
		if json.Unmarshal(body.Arguments, &arguments) != nil || arguments.Code != "render()" || arguments.ContainerID != "ctr_1" {
			http.Error(writer, "typed arguments", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"result":[{"type":"logs","logs":"ready"},{"type":"image","data_base64":"SU1BR0U=","output_format":"png"},{"type":"file","file_id":"file_1","filename":"report.txt"},{"type":"future_output","value":7}]}`))
	}))
	t.Cleanup(server.Close)
	executor, err := NewHTTPExecutor(HTTPExecutorConfig{
		Kind: llm.ToolKindCodeExecution, Endpoint: server.URL, Client: server.Client(),
		EndpointPolicy: func(*url.URL) error { return nil },
	})
	if err != nil {
		t.Fatalf("create code executor: %v", err)
	}
	result, err := executor.Execute(context.Background(), llm.ToolDefinition{
		Kind: llm.ToolKindCodeExecution, LogicalName: "code_execution", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "code_execution", Configuration: json.RawMessage(`{"type":"code_execution","container":{"type":"auto"},"future_option":true}`)},
	}, llm.ToolInvocation{
		Kind: llm.ToolKindCodeExecution, CallID: "code_1", LogicalName: "code_execution",
		ArgumentsJSON: json.RawMessage(`{"code":"render()","container_id":"ctr_1"}`),
	})
	if err != nil {
		t.Fatalf("execute code: %v", err)
	}
	if len(result.Content) != 4 || result.Content[0].Kind != llm.ContentKindText || result.Content[0].Text != "ready" ||
		result.Content[1].Kind != llm.ContentKindImage || result.Content[1].Image == nil ||
		result.Content[2].Kind != llm.ContentKindDocument || result.Content[2].Document == nil || result.Content[2].Document.FileID != "file_1" ||
		result.Content[3].Kind != llm.ContentKindUnknown || !json.Valid(result.Content[3].UnknownRaw) || !json.Valid(result.StructuredContent) {
		t.Fatalf("typed code result = %#v", result)
	}
}

func TestHTTPExecutorCarriesImageGenerationOverRealHTTP(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body HTTPExecutorRequest
		if request.Method != http.MethodPost || json.NewDecoder(request.Body).Decode(&body) != nil ||
			body.Kind != llm.ToolKindImageGeneration || string(body.Arguments) != `{"prompt":"draw axon","action":"generate"}` {
			http.Error(writer, "unexpected image request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{"result":{"data_base64":"SU1BR0VfTUFSS0VS","output_format":"png"}}`))
	}))
	t.Cleanup(server.Close)
	executor, err := NewHTTPExecutor(HTTPExecutorConfig{
		Kind: llm.ToolKindImageGeneration, Endpoint: server.URL, Client: server.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.String() != server.URL {
				return errors.New("unexpected endpoint")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("create image executor: %v", err)
	}
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindImageGeneration, LogicalName: "image_generation", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "image_generation", Configuration: json.RawMessage(`{"type":"image_generation"}`)},
	}
	result, err := executor.Execute(context.Background(), definition, llm.ToolInvocation{
		Kind: llm.ToolKindImageGeneration, CallID: "image_1", LogicalName: "image_generation",
		ArgumentsJSON: json.RawMessage(`{"prompt":"draw axon","action":"generate"}`),
	})
	if err != nil {
		t.Fatalf("execute image generation: %v", err)
	}
	if result == nil || len(result.Content) != 1 || result.Content[0].Kind != llm.ContentKindImage || result.Content[0].Image == nil ||
		result.Content[0].Image.URL != "data:image/png;base64,SU1BR0VfTUFSS0VS" ||
		!json.Valid(result.StructuredContent) {
		t.Fatalf("image executor result = %#v", result)
	}
}

func TestHTTPExecutorCarriesComputerLifecycleOverRealHTTP(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		var body HTTPExecutorRequest
		if request.Method != http.MethodPost || json.NewDecoder(request.Body).Decode(&body) != nil ||
			body.Kind != llm.ToolKindComputer || body.LogicalName != "computer" ||
			string(body.Arguments) != `{"type":"click","x":12,"y":34,"button":"left"}` {
			http.Error(writer, "unexpected computer request", http.StatusBadRequest)
			return
		}
		writer.Header().Set("Content-Type", "application/json")
		_, _ = writer.Write([]byte(`{
			"status":"completed",
			"result":{"type":"computer_screenshot","image_url":"data:image/png;base64,Q09NUFVURVJfTUFSS0VS"},
			"pending_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}],
			"acknowledged_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}]
		}`))
	}))
	t.Cleanup(server.Close)
	executor, err := NewHTTPExecutor(HTTPExecutorConfig{
		Kind: llm.ToolKindComputer, Endpoint: server.URL, Client: server.Client(),
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.String() != server.URL {
				return errors.New("unexpected endpoint")
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("create computer executor: %v", err)
	}
	lifecycle, ok := any(executor).(LifecycleExecutor)
	if !ok {
		t.Fatal("computer HTTP executor does not expose hosted lifecycle execution")
	}
	definition := llm.ToolDefinition{
		Kind: llm.ToolKindComputer, LogicalName: "computer", Execution: llm.ExecutionOwnerProvider,
		Hosted: &llm.HostedToolDefinition{Type: "computer", Configuration: json.RawMessage(`{"type":"computer","display_width":1024,"display_height":768,"environment":"browser"}`)},
	}
	execution, err := lifecycle.ExecuteLifecycle(context.Background(), definition, llm.ToolInvocation{
		Kind: llm.ToolKindComputer, CallID: "computer_1", LogicalName: "computer",
		ArgumentsJSON: json.RawMessage(`{"type":"click","x":12,"y":34,"button":"left"}`),
	})
	if err != nil {
		t.Fatalf("execute computer lifecycle: %v", err)
	}
	if execution == nil || execution.Result == nil || len(execution.PendingSafetyChecks) != 1 ||
		execution.PendingSafetyChecks[0].ID != "safe_1" || len(execution.Result.AcknowledgedSafetyChecks) != 1 ||
		execution.Result.AcknowledgedSafetyChecks[0].ID != "safe_1" ||
		execution.Result.Status != llm.ToolResultStatusCompleted || len(execution.Result.Content) != 1 ||
		execution.Result.Content[0].Kind != llm.ContentKindImage || execution.Result.Content[0].Image == nil ||
		execution.Result.Content[0].Image.URL != "data:image/png;base64,Q09NUFVURVJfTUFSS0VS" ||
		!json.Valid(execution.Result.StructuredContent) {
		t.Fatalf("computer lifecycle degraded: %#v", execution)
	}
}

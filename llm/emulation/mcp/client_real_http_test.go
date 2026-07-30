package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestClientRunsMCPStreamableHTTPLifecycleOverRealHTTP(t *testing.T) {
	t.Parallel()
	const sessionID = "session-real-1"
	const authorization = "Bearer private-mcp-token"
	const privateHeader = "private-header-value"

	serverRequestAnswered := make(chan struct{})
	issues := make(chan string, 16)
	var answerOnce sync.Once
	var methodsMu sync.Mutex
	var methods []string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != authorization || r.Header.Get("X-MCP-Secret") != privateHeader {
			issues <- "private execution headers missing"
			http.Error(w, "missing headers", http.StatusUnauthorized)
			return
		}
		if got := r.Header.Get("Accept"); got != "application/json, text/event-stream" {
			issues <- "invalid Accept header: " + got
		}
		if r.Method == http.MethodDelete {
			if r.Header.Get("Mcp-Session-Id") != sessionID {
				issues <- "DELETE missing session"
			}
			methodsMu.Lock()
			methods = append(methods, "session/delete")
			methodsMu.Unlock()
			w.WriteHeader(http.StatusNoContent)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			issues <- "read request body"
			http.Error(w, "read", http.StatusBadRequest)
			return
		}
		var envelope rpcEnvelope
		if err := json.Unmarshal(body, &envelope); err != nil {
			issues <- "decode JSON-RPC request"
			http.Error(w, "decode", http.StatusBadRequest)
			return
		}
		if envelope.Method == "" {
			if string(envelope.ID) != `"server-request-1"` || envelope.Error == nil || envelope.Error.Code != -32601 {
				issues <- "invalid response to server request: " + string(body)
			}
			answerOnce.Do(func() { close(serverRequestAnswered) })
			w.WriteHeader(http.StatusAccepted)
			return
		}

		methodsMu.Lock()
		methods = append(methods, envelope.Method)
		methodsMu.Unlock()
		if envelope.Method != "initialize" {
			if r.Header.Get("Mcp-Session-Id") != sessionID {
				issues <- envelope.Method + " missing session"
			}
			if r.Header.Get("MCP-Protocol-Version") != ProtocolVersion {
				issues <- envelope.Method + " missing protocol version"
			}
		}

		switch envelope.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Mcp-Session-Id", sessionID)
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"real-fixture","version":"1.0.0"}}}`, envelope.ID, ProtocolVersion)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"lookup","description":"Lookup inventory","inputSchema":{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]}}]}}`, envelope.ID)
		case "tools/call":
			w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
			flusher, ok := w.(http.Flusher)
			if !ok {
				issues <- "fixture server has no flusher"
				http.Error(w, "no flusher", http.StatusInternalServerError)
				return
			}
			writeSSEFragmented(w, flusher, `{"jsonrpc":"2.0","method":"notifications/progress","params":{"progress":1}}`)
			writeSSEFragmented(w, flusher, `{"jsonrpc":"2.0","id":"server-request-1","method":"sampling/createMessage","params":{}}`)
			select {
			case <-serverRequestAnswered:
			case <-r.Context().Done():
				return
			}
			response := fmt.Sprintf(`{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"in stock"},{"type":"resource_link","uri":"inventory://sku/42","name":"sku-42"}],"structuredContent":{"available":true}}}`, envelope.ID)
			writeSSEFragmented(w, flusher, response)
		default:
			issues <- "unexpected method: " + envelope.Method
			http.Error(w, "unexpected method", http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)

	allowedEndpoint, err := url.Parse(server.URL)
	require.NoError(t, err)
	client, err := NewClient(Config{
		Endpoint: server.URL,
		Secrets: llm.MCPConnectionSecrets{
			Authorization: authorization,
			Headers:       map[string]string{"X-MCP-Secret": privateHeader},
		},
		EndpointPolicy: func(candidate *url.URL) error {
			if candidate.Scheme != allowedEndpoint.Scheme || candidate.Host != allowedEndpoint.Host {
				return fmt.Errorf("host is not allowlisted")
			}
			return nil
		},
		ClientName: "axon-real-test", ClientVersion: "1.0.0",
	})
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	initialized, err := client.Initialize(ctx)
	require.NoError(t, err)
	require.Equal(t, ProtocolVersion, initialized.ProtocolVersion)
	require.Equal(t, "real-fixture", initialized.ServerInfo.Name)

	listed, err := client.ListTools(ctx, "")
	require.NoError(t, err)
	require.Len(t, listed.Tools, 1)
	require.Equal(t, "lookup", listed.Tools[0].Name)
	require.JSONEq(t, `{"type":"object","properties":{"sku":{"type":"string"}},"required":["sku"]}`, string(listed.Tools[0].InputSchema))

	result, err := client.CallTool(ctx, "lookup", json.RawMessage(`{"sku":"42"}`))
	require.NoError(t, err)
	require.False(t, result.IsError)
	require.Len(t, result.Content, 2)
	require.Equal(t, "in stock", result.Content[0].Text)
	require.Equal(t, "inventory://sku/42", result.Content[1].URI)
	require.JSONEq(t, `{"available":true}`, string(result.StructuredContent))
	require.JSONEq(t, `{"type":"resource_link","uri":"inventory://sku/42","name":"sku-42"}`, string(result.Content[1].Raw))

	require.NoError(t, client.Close(ctx))
	close(issues)
	for issue := range issues {
		t.Error(issue)
	}
	methodsMu.Lock()
	defer methodsMu.Unlock()
	require.Equal(t, []string{
		"initialize", "notifications/initialized", "tools/list", "tools/call", "session/delete",
	}, methods)
}

func TestClientRejectsUnboundedOrUnapprovedEndpoints(t *testing.T) {
	t.Parallel()
	_, err := NewClient(Config{Endpoint: "https://example.invalid/mcp"})
	require.ErrorContains(t, err, "endpoint policy is required")

	_, err = NewClient(Config{
		Endpoint:       "http://127.0.0.1/internal",
		EndpointPolicy: func(*url.URL) error { return fmt.Errorf("private network denied") },
	})
	require.ErrorContains(t, err, "endpoint rejected")
}

func TestClientRedactsRemoteMessageOverRealHTTP(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var envelope rpcEnvelope
		_ = json.Unmarshal(body, &envelope)
		w.Header().Set("Content-Type", "application/json")
		if envelope.Method == "initialize" {
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"error":{"code":-32001,"message":"private-token-must-not-leak"}}`, envelope.ID)
			return
		}
	}))
	t.Cleanup(server.Close)
	client, err := NewClient(Config{
		Endpoint: server.URL,
		EndpointPolicy: func(candidate *url.URL) error {
			if strings.HasPrefix(candidate.String(), server.URL) {
				return nil
			}
			return fmt.Errorf("denied")
		},
	})
	require.NoError(t, err)
	_, err = client.Initialize(context.Background())
	require.Error(t, err)
	require.NotContains(t, err.Error(), "private-token-must-not-leak")
	var mcpErr *Error
	require.ErrorAs(t, err, &mcpErr)
	require.Equal(t, ErrorRemote, mcpErr.Kind)
	require.Equal(t, -32001, mcpErr.RPCCode)
}

func TestClientSendsExplicitCancellationOverRealHTTP(t *testing.T) {
	t.Parallel()
	const sessionID = "session-cancel-1"
	callStarted := make(chan struct{})
	cancelledRequest := make(chan uint64, 1)
	callCtx, cancelCall := context.WithCancel(context.Background())
	defer cancelCall()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var envelope struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
			Params struct {
				RequestID uint64 `json:"requestId"`
			} `json:"params"`
		}
		_ = json.Unmarshal(body, &envelope)
		switch envelope.Method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Mcp-Session-Id", sessionID)
			_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{"tools":{}},"serverInfo":{"name":"cancel-fixture","version":"1"}}}`, envelope.ID, ProtocolVersion)
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/call":
			close(callStarted)
			cancelCall()
			<-r.Context().Done()
		case "notifications/cancelled":
			cancelledRequest <- envelope.Params.RequestID
			w.WriteHeader(http.StatusAccepted)
		default:
			http.Error(w, "unexpected", http.StatusBadRequest)
		}
	}))
	t.Cleanup(server.Close)
	client, err := NewClient(Config{
		Endpoint: server.URL,
		EndpointPolicy: func(candidate *url.URL) error {
			if strings.HasPrefix(candidate.String(), server.URL) {
				return nil
			}
			return fmt.Errorf("denied")
		},
	})
	require.NoError(t, err)
	require.NotNil(t, client)
	_, err = client.Initialize(context.Background())
	require.NoError(t, err)

	_, callErr := client.CallTool(callCtx, "long_running", json.RawMessage(`{"value":1}`))
	require.ErrorIs(t, callErr, context.Canceled)
	select {
	case <-callStarted:
	default:
		t.Fatal("real MCP call did not start")
	}
	select {
	case requestID := <-cancelledRequest:
		require.NotZero(t, requestID)
	case <-time.After(3 * time.Second):
		t.Fatal("MCP cancellation notification was not received")
	}
}

func TestClientBoundsRealHTTPJSONResponse(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var envelope rpcEnvelope
		_ = json.Unmarshal(body, &envelope)
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"%s","capabilities":{},"serverInfo":{"name":"%s","version":"1"}}}`, envelope.ID, ProtocolVersion, strings.Repeat("x", 1024))
	}))
	t.Cleanup(server.Close)
	client, err := NewClient(Config{
		Endpoint: server.URL, MaxResponseBytes: 256,
		EndpointPolicy: func(candidate *url.URL) error {
			if strings.HasPrefix(candidate.String(), server.URL) {
				return nil
			}
			return fmt.Errorf("denied")
		},
	})
	require.NoError(t, err)
	_, err = client.Initialize(context.Background())
	require.Error(t, err)
	var mcpErr *Error
	require.ErrorAs(t, err, &mcpErr)
	require.Equal(t, ErrorLimit, mcpErr.Kind)
}

func writeSSEFragmented(writer io.Writer, flusher http.Flusher, data string) {
	payload := "event: message\ndata: " + data + "\n\n"
	for index := range payload {
		_, _ = io.WriteString(writer, payload[index:index+1])
		flusher.Flush()
	}
}

package mcp

import (
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
)

func TestErrorSafeDiagnosticContainsOnlyBoundedMCPMetadata(t *testing.T) {
	err := &Error{Kind: ErrorRemote, Method: "initialize", StatusCode: 401}
	diagnostic := llm.ErrorDiagnosticFrom(err)
	if diagnostic == nil || diagnostic.Component != "mcp_gateway" || diagnostic.Code != "mcp_remote" ||
		diagnostic.Message != "MCP initialize failed: remote (HTTP 401)" || diagnostic.StatusCode != 401 {
		t.Fatalf("MCP diagnostic = %#v", diagnostic)
	}
	if diagnostic := llm.ErrorDiagnosticFrom((*Error)(nil)); diagnostic != nil {
		t.Fatalf("nil MCP error produced diagnostic %#v", diagnostic)
	}
}

func TestErrorSafeDiagnosticRejectsUnrecognizedFreeFormMetadata(t *testing.T) {
	t.Parallel()
	secret := "https://private.example.test/mcp?token=secret-value"
	for name, err := range map[string]*Error{
		"kind":   {Kind: ErrorKind("transport " + secret), Method: "tools/call", StatusCode: 401},
		"method": {Kind: ErrorTransport, Method: "tools/call " + secret, StatusCode: 401},
	} {
		t.Run(name, func(t *testing.T) {
			diagnostic := llm.ErrorDiagnosticFrom(err)
			if diagnostic != nil {
				encoded := diagnostic.Component + diagnostic.Code + diagnostic.Message
				if strings.Contains(encoded, secret) {
					t.Fatalf("safe diagnostic leaked free-form metadata: %#v", diagnostic)
				}
				t.Fatalf("unrecognized metadata produced safe diagnostic %#v", diagnostic)
			}
		})
	}
}

func TestMCPSafeDiagnosticAllowlistAndMessageShapes(t *testing.T) {
	t.Parallel()
	for _, kind := range []ErrorKind{ErrorTransport, ErrorProtocol, ErrorRemote, ErrorSessionExpired, ErrorLimit} {
		code, label, ok := safeDiagnosticKind(kind)
		if !ok || code == "" || label == "" {
			t.Fatalf("safeDiagnosticKind(%q) = %q, %q, %v", kind, code, label, ok)
		}
	}
	if code, label, ok := safeDiagnosticKind(ErrorKind("future")); ok || code != "" || label != "" {
		t.Fatalf("unknown kind accepted: %q, %q, %v", code, label, ok)
	}
	for _, method := range []string{
		"initialize", "notifications/initialized", "tools/list", "tools/call",
		"notifications/cancelled", "session/delete", "server/request",
	} {
		if got, ok := safeDiagnosticMethod(method); !ok || got != method {
			t.Fatalf("safeDiagnosticMethod(%q) = %q, %v", method, got, ok)
		}
	}
	if got, ok := safeDiagnosticMethod("future/method"); ok || got != "" {
		t.Fatalf("unknown method accepted: %q, %v", got, ok)
	}
	for _, test := range []struct {
		status, rpc int
		want        string
	}{
		{status: 401, want: "MCP tools/call failed: remote (HTTP 401)"},
		{rpc: -32601, want: "MCP tools/call failed: remote (JSON-RPC -32601)"},
		{want: "MCP tools/call failed: remote"},
	} {
		if got := safeDiagnosticMessage("tools/call", "remote", test.status, test.rpc); got != test.want {
			t.Fatalf("safeDiagnosticMessage() = %q, want %q", got, test.want)
		}
	}
}

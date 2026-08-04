package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/looplj/axonhub/llm"
)

const ProtocolVersion = "2025-06-18"

type ErrorKind string

const (
	ErrorTransport      ErrorKind = "transport"
	ErrorProtocol       ErrorKind = "protocol"
	ErrorRemote         ErrorKind = "remote"
	ErrorSessionExpired ErrorKind = "session_expired"
	ErrorLimit          ErrorKind = "limit"
)

// Error intentionally excludes endpoint URLs, response bodies, JSON-RPC
// messages, tool arguments, and credentials so it is safe for bounded traces.
type Error struct {
	Kind       ErrorKind
	Method     string
	StatusCode int
	RPCCode    int
	cause      error
}

func (err *Error) Error() string {
	if err == nil {
		return "MCP error"
	}
	switch {
	case err.StatusCode != 0:
		return fmt.Sprintf("MCP %s failed: %s (HTTP %d)", err.Method, err.Kind, err.StatusCode)
	case err.RPCCode != 0:
		return fmt.Sprintf("MCP %s failed: %s (JSON-RPC %d)", err.Method, err.Kind, err.RPCCode)
	default:
		return fmt.Sprintf("MCP %s failed: %s", err.Method, err.Kind)
	}
}

// SafeDiagnostic exposes only the fixed-cardinality MCP method, error kind,
// and numeric status already carried by Error. Remote messages, endpoint URLs,
// request bodies, headers, credentials, and tool arguments are intentionally
// absent from Error and therefore cannot cross this boundary.
func (err *Error) SafeDiagnostic() llm.ErrorDiagnostic {
	if err == nil {
		return llm.ErrorDiagnostic{}
	}
	code, kind, kindOK := safeDiagnosticKind(err.Kind)
	method, methodOK := safeDiagnosticMethod(err.Method)
	if !kindOK || !methodOK {
		return llm.ErrorDiagnostic{}
	}
	return llm.ErrorDiagnostic{
		Component:  "mcp_gateway",
		Code:       code,
		Message:    safeDiagnosticMessage(method, kind, err.StatusCode, err.RPCCode),
		StatusCode: err.StatusCode,
	}
}

func safeDiagnosticKind(kind ErrorKind) (code, label string, ok bool) {
	switch kind {
	case ErrorTransport:
		return "mcp_transport", "transport", true
	case ErrorProtocol:
		return "mcp_protocol", "protocol", true
	case ErrorRemote:
		return "mcp_remote", "remote", true
	case ErrorSessionExpired:
		return "mcp_session_expired", "session expired", true
	case ErrorLimit:
		return "mcp_limit", "limit", true
	default:
		return "", "", false
	}
}

func safeDiagnosticMethod(method string) (string, bool) {
	switch method {
	case "initialize", "notifications/initialized", "tools/list", "tools/call",
		"notifications/cancelled", "session/delete", "server/request":
		return method, true
	default:
		return "", false
	}
}

func safeDiagnosticMessage(method, kind string, statusCode, rpcCode int) string {
	switch {
	case statusCode != 0:
		return fmt.Sprintf("MCP %s failed: %s (HTTP %d)", method, kind, statusCode)
	case rpcCode != 0:
		return fmt.Sprintf("MCP %s failed: %s (JSON-RPC %d)", method, kind, rpcCode)
	default:
		return fmt.Sprintf("MCP %s failed: %s", method, kind)
	}
}

func (err *Error) Unwrap() error {
	if err == nil {
		return nil
	}
	return err.cause
}

type ServerInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title,omitempty"`
	Version string `json:"version"`
}

type InitializeResult struct {
	ProtocolVersion string          `json:"protocolVersion"`
	Capabilities    json.RawMessage `json:"capabilities"`
	ServerInfo      ServerInfo      `json:"serverInfo"`
	Instructions    string          `json:"instructions,omitempty"`
}

type Tool struct {
	Name         string          `json:"name"`
	Title        string          `json:"title,omitempty"`
	Description  string          `json:"description,omitempty"`
	InputSchema  json.RawMessage `json:"inputSchema"`
	OutputSchema json.RawMessage `json:"outputSchema,omitempty"`
	Annotations  json.RawMessage `json:"annotations,omitempty"`
	Meta         json.RawMessage `json:"_meta,omitempty"`
}

type ListToolsResult struct {
	Tools      []Tool `json:"tools"`
	NextCursor string `json:"nextCursor,omitempty"`
}

type Content struct {
	Type        string          `json:"type"`
	Text        string          `json:"text,omitempty"`
	Data        string          `json:"data,omitempty"`
	MimeType    string          `json:"mimeType,omitempty"`
	URI         string          `json:"uri,omitempty"`
	Name        string          `json:"name,omitempty"`
	Title       string          `json:"title,omitempty"`
	Description string          `json:"description,omitempty"`
	Resource    json.RawMessage `json:"resource,omitempty"`
	Annotations json.RawMessage `json:"annotations,omitempty"`
	Meta        json.RawMessage `json:"_meta,omitempty"`
	Raw         json.RawMessage `json:"-"`
}

type CallToolResult struct {
	Content           []Content       `json:"content"`
	StructuredContent json.RawMessage `json:"structuredContent,omitempty"`
	IsError           bool            `json:"isError,omitempty"`
	Meta              json.RawMessage `json:"_meta,omitempty"`
}

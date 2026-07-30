package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
)

const (
	defaultMaxRequestBytes  = 8 << 20
	defaultMaxResponseBytes = 16 << 20
	defaultMaxSSEEventBytes = 16 << 20
	maxSessionIDBytes       = 1024
)

type EndpointPolicy func(*url.URL) error

type Config struct {
	Endpoint         string
	HTTPClient       *http.Client
	Secrets          llm.MCPConnectionSecrets
	EndpointPolicy   EndpointPolicy
	ProtocolVersion  string
	ClientName       string
	ClientVersion    string
	MaxRequestBytes  int64
	MaxResponseBytes int64
	MaxSSEEventBytes int
}

type Client struct {
	endpoint         *url.URL
	httpClient       *http.Client
	secrets          llm.MCPConnectionSecrets
	protocolVersion  string
	clientName       string
	clientVersion    string
	maxRequestBytes  int64
	maxResponseBytes int64
	maxSSEEventBytes int
	nextID           atomic.Uint64

	mu           sync.Mutex
	initialized  bool
	initializing chan struct{}
	sessionID    string
	server       InitializeResult
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      uint64 `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code int `json:"code"`
}

type exchangeResult struct {
	result    json.RawMessage
	sessionID string
}

func NewClient(config Config) (*Client, error) {
	if config.EndpointPolicy == nil {
		return nil, errors.New("MCP endpoint policy is required")
	}
	endpoint, err := url.Parse(config.Endpoint)
	if err != nil || endpoint.Scheme != "http" && endpoint.Scheme != "https" || endpoint.Host == "" || endpoint.User != nil || endpoint.Fragment != "" {
		return nil, errors.New("invalid MCP endpoint")
	}
	if err := config.EndpointPolicy(endpoint); err != nil {
		return nil, fmt.Errorf("MCP endpoint rejected: %w", err)
	}

	baseClient := config.HTTPClient
	if baseClient == nil {
		baseClient = http.DefaultClient
	}
	clientCopy := *baseClient
	originalRedirect := baseClient.CheckRedirect
	clientCopy.CheckRedirect = func(request *http.Request, via []*http.Request) error {
		if err := config.EndpointPolicy(request.URL); err != nil {
			return fmt.Errorf("MCP redirect rejected: %w", err)
		}
		if originalRedirect != nil {
			return originalRedirect(request, via)
		}
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}

	protocolVersion := config.ProtocolVersion
	if protocolVersion == "" {
		protocolVersion = ProtocolVersion
	}
	clientName := config.ClientName
	if clientName == "" {
		clientName = "axonhub"
	}
	clientVersion := config.ClientVersion
	if clientVersion == "" {
		clientVersion = "dev"
	}
	maxRequestBytes := config.MaxRequestBytes
	if maxRequestBytes <= 0 {
		maxRequestBytes = defaultMaxRequestBytes
	}
	maxResponseBytes := config.MaxResponseBytes
	if maxResponseBytes <= 0 {
		maxResponseBytes = defaultMaxResponseBytes
	}
	maxSSEEventBytes := config.MaxSSEEventBytes
	if maxSSEEventBytes <= 0 {
		maxSSEEventBytes = defaultMaxSSEEventBytes
	}

	return &Client{
		endpoint: endpoint, httpClient: &clientCopy,
		secrets:         llm.MCPConnectionSecrets{Authorization: config.Secrets.Authorization, Headers: cloneHeaders(config.Secrets.Headers)},
		protocolVersion: protocolVersion, clientName: clientName, clientVersion: clientVersion,
		maxRequestBytes: maxRequestBytes, maxResponseBytes: maxResponseBytes, maxSSEEventBytes: maxSSEEventBytes,
	}, nil
}

func (client *Client) Initialize(ctx context.Context) (InitializeResult, error) {
	for {
		client.mu.Lock()
		if client.initialized {
			result := client.server
			client.mu.Unlock()
			return result, nil
		}
		if waiting := client.initializing; waiting != nil {
			client.mu.Unlock()
			select {
			case <-ctx.Done():
				return InitializeResult{}, ctx.Err()
			case <-waiting:
				continue
			}
		}
		client.initializing = make(chan struct{})
		client.mu.Unlock()
		break
	}

	result, err := client.initialize(ctx)
	client.mu.Lock()
	if err == nil {
		client.initialized = true
		client.server = result
	}
	close(client.initializing)
	client.initializing = nil
	client.mu.Unlock()
	return result, err
}

func (client *Client) initialize(ctx context.Context) (InitializeResult, error) {
	response, err := client.exchange(ctx, "initialize", map[string]any{
		"protocolVersion": client.protocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name": client.clientName, "version": client.clientVersion,
		},
	})
	if err != nil {
		return InitializeResult{}, err
	}
	var result InitializeResult
	if err := json.Unmarshal(response.result, &result); err != nil || result.ProtocolVersion == "" || len(result.ProtocolVersion) > 64 {
		return InitializeResult{}, &Error{Kind: ErrorProtocol, Method: "initialize", cause: err}
	}
	if len(response.sessionID) > maxSessionIDBytes || strings.ContainsAny(response.sessionID, "\r\n") {
		return InitializeResult{}, &Error{Kind: ErrorProtocol, Method: "initialize"}
	}
	client.mu.Lock()
	client.sessionID = response.sessionID
	client.protocolVersion = result.ProtocolVersion
	client.mu.Unlock()

	if err := client.sendOneWay(ctx, "notifications/initialized", rpcNotification{
		JSONRPC: "2.0", Method: "notifications/initialized",
	}); err != nil {
		client.clearSession()
		return InitializeResult{}, err
	}
	return result, nil
}

func (client *Client) ListTools(ctx context.Context, cursor string) (ListToolsResult, error) {
	if _, err := client.Initialize(ctx); err != nil {
		return ListToolsResult{}, err
	}
	params := make(map[string]string)
	if cursor != "" {
		params["cursor"] = cursor
	}
	response, err := client.exchange(ctx, "tools/list", params)
	if err != nil {
		return ListToolsResult{}, err
	}
	var result ListToolsResult
	if err := json.Unmarshal(response.result, &result); err != nil {
		return ListToolsResult{}, &Error{Kind: ErrorProtocol, Method: "tools/list", cause: err}
	}
	for index := range result.Tools {
		tool := &result.Tools[index]
		if tool.Name == "" || len(tool.InputSchema) == 0 || !json.Valid(tool.InputSchema) {
			return ListToolsResult{}, &Error{Kind: ErrorProtocol, Method: "tools/list"}
		}
	}
	return result, nil
}

func (client *Client) CallTool(ctx context.Context, name string, arguments json.RawMessage) (CallToolResult, error) {
	if _, err := client.Initialize(ctx); err != nil {
		return CallToolResult{}, err
	}
	if name == "" {
		return CallToolResult{}, errors.New("MCP tool name is required")
	}
	if len(arguments) == 0 {
		arguments = json.RawMessage(`{}`)
	}
	var argumentObject map[string]json.RawMessage
	if err := json.Unmarshal(arguments, &argumentObject); err != nil || argumentObject == nil {
		return CallToolResult{}, errors.New("MCP tool arguments must be a JSON object")
	}
	response, err := client.exchange(ctx, "tools/call", map[string]any{
		"name": name, "arguments": argumentObject,
	})
	if err != nil {
		return CallToolResult{}, err
	}
	var raw struct {
		Content           []json.RawMessage `json:"content"`
		StructuredContent json.RawMessage   `json:"structuredContent,omitempty"`
		IsError           bool              `json:"isError,omitempty"`
		Meta              json.RawMessage   `json:"_meta,omitempty"`
	}
	if err := json.Unmarshal(response.result, &raw); err != nil {
		return CallToolResult{}, &Error{Kind: ErrorProtocol, Method: "tools/call", cause: err}
	}
	result := CallToolResult{
		Content: make([]Content, 0, len(raw.Content)), StructuredContent: raw.StructuredContent,
		IsError: raw.IsError, Meta: raw.Meta,
	}
	for index := range raw.Content {
		var content Content
		if err := json.Unmarshal(raw.Content[index], &content); err != nil || content.Type == "" {
			return CallToolResult{}, &Error{Kind: ErrorProtocol, Method: "tools/call", cause: err}
		}
		content.Raw = append(json.RawMessage(nil), raw.Content[index]...)
		result.Content = append(result.Content, content)
	}
	return result, nil
}

func (client *Client) Close(ctx context.Context) error {
	client.mu.Lock()
	sessionID := client.sessionID
	client.mu.Unlock()
	if sessionID == "" {
		client.clearSession()
		return nil
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodDelete, client.endpoint.String(), nil)
	if err != nil {
		return &Error{Kind: ErrorTransport, Method: "session/delete", cause: err}
	}
	client.applyHeaders(request, sessionID)
	response, err := client.httpClient.Do(request)
	if err != nil {
		return &Error{Kind: ErrorTransport, Method: "session/delete", cause: safeContextCause(ctx, err)}
	}
	_ = response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		if response.StatusCode != http.StatusNotFound {
			return &Error{Kind: ErrorTransport, Method: "session/delete", StatusCode: response.StatusCode}
		}
	}
	client.clearSession()
	return nil
}

func (client *Client) exchange(ctx context.Context, method string, params any) (exchangeResult, error) {
	id := client.nextID.Add(1)
	payload, err := json.Marshal(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method, cause: err}
	}
	if int64(len(payload)) > client.maxRequestBytes {
		return exchangeResult{}, &Error{Kind: ErrorLimit, Method: method}
	}
	response, err := client.post(ctx, method, payload)
	if err != nil {
		if ctx.Err() != nil {
			cancelCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 2*time.Second)
			_ = client.sendOneWay(cancelCtx, "notifications/cancelled", rpcNotification{
				JSONRPC: "2.0", Method: "notifications/cancelled",
				Params: map[string]any{"requestId": id, "reason": "request context ended"},
			})
			cancel()
		}
		return exchangeResult{}, err
	}
	defer response.Body.Close()

	mediaType, _, err := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if err != nil {
		return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method, cause: err}
	}
	var envelope rpcEnvelope
	switch mediaType {
	case "application/json":
		body, err := readBounded(response.Body, client.maxResponseBytes)
		if err != nil {
			return exchangeResult{}, &Error{Kind: ErrorLimit, Method: method, cause: err}
		}
		if err := json.Unmarshal(body, &envelope); err != nil || !matchesID(envelope.ID, id) {
			return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method, cause: err}
		}
	case "text/event-stream":
		decoder := httpclient.NewSSEDecoderWithMaxEventSize(ctx, response.Body, client.maxSSEEventBytes)
		defer decoder.Close()
		matched := false
		for decoder.Next() {
			event := decoder.Current()
			if event == nil || len(bytes.TrimSpace(event.Data)) == 0 {
				continue
			}
			var candidate rpcEnvelope
			if err := json.Unmarshal(event.Data, &candidate); err != nil || candidate.JSONRPC != "2.0" {
				return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method, cause: err}
			}
			if candidate.Method != "" && len(candidate.ID) > 0 {
				if err := client.respondMethodNotFound(ctx, candidate.ID); err != nil {
					return exchangeResult{}, err
				}
				continue
			}
			if matchesID(candidate.ID, id) {
				envelope = candidate
				matched = true
				break
			}
		}
		if err := decoder.Err(); err != nil {
			return exchangeResult{}, &Error{Kind: ErrorTransport, Method: method, cause: safeContextCause(ctx, err)}
		}
		if !matched {
			return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method}
		}
	default:
		return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method}
	}
	if envelope.JSONRPC != "2.0" || envelope.Error == nil && len(envelope.Result) == 0 {
		return exchangeResult{}, &Error{Kind: ErrorProtocol, Method: method}
	}
	if envelope.Error != nil {
		return exchangeResult{}, &Error{Kind: ErrorRemote, Method: method, RPCCode: envelope.Error.Code}
	}
	return exchangeResult{result: envelope.Result, sessionID: strings.TrimSpace(response.Header.Get("Mcp-Session-Id"))}, nil
}

func (client *Client) post(ctx context.Context, method string, payload []byte) (*http.Response, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, client.endpoint.String(), bytes.NewReader(payload))
	if err != nil {
		return nil, &Error{Kind: ErrorTransport, Method: method, cause: err}
	}
	client.mu.Lock()
	sessionID := client.sessionID
	client.mu.Unlock()
	client.applyHeaders(request, sessionID)
	response, err := client.httpClient.Do(request)
	if err != nil {
		return nil, &Error{Kind: ErrorTransport, Method: method, cause: safeContextCause(ctx, err)}
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		_ = response.Body.Close()
		if response.StatusCode == http.StatusNotFound && sessionID != "" {
			client.clearSession()
			return nil, &Error{Kind: ErrorSessionExpired, Method: method, StatusCode: response.StatusCode}
		}
		return nil, &Error{Kind: ErrorTransport, Method: method, StatusCode: response.StatusCode}
	}
	return response, nil
}

func (client *Client) sendOneWay(ctx context.Context, method string, payload any) error {
	encoded, err := json.Marshal(payload)
	if err != nil {
		return &Error{Kind: ErrorProtocol, Method: method, cause: err}
	}
	if int64(len(encoded)) > client.maxRequestBytes {
		return &Error{Kind: ErrorLimit, Method: method}
	}
	response, err := client.post(ctx, method, encoded)
	if err != nil {
		return err
	}
	_ = response.Body.Close()
	return nil
}

func (client *Client) respondMethodNotFound(ctx context.Context, id json.RawMessage) error {
	payload := struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Error   struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}{JSONRPC: "2.0", ID: append(json.RawMessage(nil), id...)}
	payload.Error.Code = -32601
	payload.Error.Message = "Method not supported"
	return client.sendOneWay(ctx, "server/request", payload)
}

func (client *Client) applyHeaders(request *http.Request, sessionID string) {
	for key, value := range client.secrets.Headers {
		switch http.CanonicalHeaderKey(key) {
		case "Content-Type", "Accept", "Mcp-Protocol-Version", "Mcp-Session-Id":
			continue
		}
		request.Header.Set(key, value)
	}
	if client.secrets.Authorization != "" {
		request.Header.Set("Authorization", client.secrets.Authorization)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json, text/event-stream")
	client.mu.Lock()
	protocolVersion := client.protocolVersion
	client.mu.Unlock()
	if protocolVersion != "" {
		request.Header.Set("MCP-Protocol-Version", protocolVersion)
	}
	if sessionID != "" {
		request.Header.Set("Mcp-Session-Id", sessionID)
	}
}

func (client *Client) clearSession() {
	client.mu.Lock()
	client.initialized = false
	client.sessionID = ""
	client.server = InitializeResult{}
	client.mu.Unlock()
}

func readBounded(reader io.Reader, limit int64) ([]byte, error) {
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, errors.New("MCP response exceeds limit")
	}
	return data, nil
}

func matchesID(raw json.RawMessage, expected uint64) bool {
	if len(raw) == 0 {
		return false
	}
	value, err := strconv.ParseUint(string(bytes.TrimSpace(raw)), 10, 64)
	return err == nil && value == expected
}

func cloneHeaders(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	clone := make(map[string]string, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func safeContextCause(ctx context.Context, err error) error {
	if errors.Is(err, context.Canceled) || ctx != nil && errors.Is(ctx.Err(), context.Canceled) {
		return context.Canceled
	}
	if errors.Is(err, context.DeadlineExceeded) || ctx != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return context.DeadlineExceeded
	}
	return nil
}

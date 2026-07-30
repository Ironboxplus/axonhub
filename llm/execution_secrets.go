package llm

import "maps"

// ToolExecutionSecrets carries request-lifetime executor credentials outside
// canonical tool definitions, conversion ledgers, persistence, and JSON
// serialization. Keys are request-unique MCP server labels.
type ToolExecutionSecrets struct {
	MCP map[string]MCPConnectionSecrets `json:"-"`
}

type MCPConnectionSecrets struct {
	Authorization string            `json:"-"`
	Headers       map[string]string `json:"-"`
}

func (secrets *ToolExecutionSecrets) Clone() *ToolExecutionSecrets {
	if secrets == nil {
		return nil
	}
	clone := &ToolExecutionSecrets{MCP: make(map[string]MCPConnectionSecrets, len(secrets.MCP))}
	for label, connection := range secrets.MCP {
		connection.Headers = maps.Clone(connection.Headers)
		clone.MCP[label] = connection
	}
	return clone
}

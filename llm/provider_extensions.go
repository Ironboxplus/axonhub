package llm

import "encoding/json"

// ProviderExtensions carries provider/API-format private data that should not
// be serialized through the common llm request/response JSON model.
type ProviderExtensions struct {
	OpenAIChat      *OpenAIChatProviderExtensions      `json:"-"`
	OpenAIResponses *OpenAIResponsesProviderExtensions `json:"-"`
}

type OpenAIChatProviderExtensions struct {
	Request *OpenAIChatRequestExtensions `json:"-"`
}

type OpenAIChatRequestExtensions struct {
	// RawThinking preserves the complete provider-specific thinking object. The
	// common request still exposes ReasoningEffort for cross-provider routing,
	// while this sidecar makes OpenAI-compatible round trips lossless.
	RawThinking json.RawMessage `json:"-"`
}

type OpenAIResponsesProviderExtensions struct {
	Request *OpenAIResponsesRequestExtensions `json:"-"`
}

type ResponseProviderExtensions struct {
	OpenAIResponses *OpenAIResponsesResponseExtensions `json:"-"`
}

type OpenAIResponsesResponseExtensions struct {
	ResidualFields json.RawMessage `json:"-"`
}

func CloneResponseProviderExtensions(src *ResponseProviderExtensions) *ResponseProviderExtensions {
	if src == nil {
		return nil
	}
	clone := &ResponseProviderExtensions{}
	if src.OpenAIResponses != nil {
		clone.OpenAIResponses = &OpenAIResponsesResponseExtensions{
			ResidualFields: cloneRawMessage(src.OpenAIResponses.ResidualFields),
		}
	}
	return clone
}

type OpenAIResponsesRequestExtensions struct {
	ReasoningContext string                     `json:"-"`
	ResidualFields   json.RawMessage            `json:"-"`
	ToolNamespaces   []ToolNamespaceDeclaration `json:"-"`
	// Deprecated raw-index fields are retained only for source compatibility
	// while callers migrate. The Responses encoder and planner never consume
	// them; canonical objects own tool, choice, and input identity.
	RawTools       []OpenAIResponsesRawFragment `json:"-"`
	ToolSignatures []string                     `json:"-"`
	RawToolChoice  json.RawMessage              `json:"-"`
	RawInputItems  []OpenAIResponsesRawFragment `json:"-"`
}

type OpenAIResponsesRawFragment struct {
	Type          string `json:"-"`
	Name          string `json:"-"`
	CallID        string `json:"-"`
	OriginalIndex int    `json:"-"`
	// RepresentedToolCount is the number of structured tools replaced when Raw is replayed.
	RepresentedToolCount int `json:"-"`
	// RepresentedInputItemCount is the number of structured input items replaced
	// when Raw is replayed. A value of one is used for Responses Lite
	// additional_tools: canonical state owns the behavioral tool definitions,
	// while Raw retains provider-private envelope and namespace metadata for an
	// identity Responses route.
	RepresentedInputItemCount int             `json:"-"`
	BehaviorFullyRepresented  bool            `json:"-"`
	Raw                       json.RawMessage `json:"-"`
}

func EnsureOpenAIChatProviderExtensions(req *Request) *OpenAIChatProviderExtensions {
	if req == nil {
		return nil
	}

	if req.ProviderExtensions == nil {
		req.ProviderExtensions = &ProviderExtensions{}
	}

	if req.ProviderExtensions.OpenAIChat == nil {
		req.ProviderExtensions.OpenAIChat = &OpenAIChatProviderExtensions{}
	}

	return req.ProviderExtensions.OpenAIChat
}

func EnsureOpenAIResponsesProviderExtensions(req *Request) *OpenAIResponsesProviderExtensions {
	if req == nil {
		return nil
	}

	if req.ProviderExtensions == nil {
		req.ProviderExtensions = &ProviderExtensions{}
	}

	if req.ProviderExtensions.OpenAIResponses == nil {
		req.ProviderExtensions.OpenAIResponses = &OpenAIResponsesProviderExtensions{}
	}

	return req.ProviderExtensions.OpenAIResponses
}

func CloneProviderExtensions(src *ProviderExtensions) *ProviderExtensions {
	if src == nil {
		return nil
	}

	cloned := &ProviderExtensions{}
	if src.OpenAIChat != nil {
		cloned.OpenAIChat = &OpenAIChatProviderExtensions{}
		if src.OpenAIChat.Request != nil {
			cloned.OpenAIChat.Request = &OpenAIChatRequestExtensions{
				RawThinking: cloneRawMessage(src.OpenAIChat.Request.RawThinking),
			}
		}
	}
	if src.OpenAIResponses != nil {
		cloned.OpenAIResponses = &OpenAIResponsesProviderExtensions{}
		if src.OpenAIResponses.Request != nil {
			cloned.OpenAIResponses.Request = &OpenAIResponsesRequestExtensions{
				ReasoningContext: src.OpenAIResponses.Request.ReasoningContext,
				ResidualFields:   cloneRawMessage(src.OpenAIResponses.Request.ResidualFields),
				ToolNamespaces:   cloneToolNamespaceDeclarations(src.OpenAIResponses.Request.ToolNamespaces),
				RawTools:         cloneOpenAIResponsesRawFragments(src.OpenAIResponses.Request.RawTools),
				ToolSignatures:   append([]string(nil), src.OpenAIResponses.Request.ToolSignatures...),
				RawToolChoice:    cloneRawMessage(src.OpenAIResponses.Request.RawToolChoice),
				RawInputItems:    cloneOpenAIResponsesRawFragments(src.OpenAIResponses.Request.RawInputItems),
			}
		}
	}

	return cloned
}

func cloneToolNamespaceDeclarations(src []ToolNamespaceDeclaration) []ToolNamespaceDeclaration {
	if len(src) == 0 {
		return nil
	}
	out := make([]ToolNamespaceDeclaration, len(src))
	copy(out, src)
	for index := range out {
		out[index].SourceResidual = cloneRawMessage(src[index].SourceResidual)
	}
	return out
}

func cloneOpenAIResponsesRawFragments(src []OpenAIResponsesRawFragment) []OpenAIResponsesRawFragment {
	if len(src) == 0 {
		return nil
	}

	out := make([]OpenAIResponsesRawFragment, len(src))
	for i := range src {
		out[i] = src[i]
		out[i].Raw = cloneRawMessage(src[i].Raw)
	}

	return out
}

func cloneRawMessage(src json.RawMessage) json.RawMessage {
	if len(src) == 0 {
		return nil
	}

	return append(json.RawMessage(nil), src...)
}

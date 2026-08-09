package llm

import (
	"strings"

	"github.com/looplj/axonhub/llm/httpclient"
)

// ResponsesWireProfileMetadataKey is the canonical request metadata key used
// by Responses adapters after recognizing an effective wire profile.
const ResponsesWireProfileMetadataKey = "responses_wire_profile"

const (
	// ResponsesWireProfileLiteValue selects Codex's Responses Lite wire
	// contract. It is a final-wire profile, not a different canonical API.
	ResponsesWireProfileLiteValue = "responses_lite"
	// OpenAIResponsesLiteHeader is the client signal which selects the Lite
	// contract before the inbound adapter has materialized metadata.
	OpenAIResponsesLiteHeader = "X-OpenAI-Internal-Codex-Responses-Lite"
)

// HasResponsesLiteWireHeader reports whether a raw client request explicitly
// selected the Responses Lite wire contract.
func HasResponsesLiteWireHeader(request *httpclient.Request) bool {
	if request == nil {
		return false
	}
	for name, values := range request.Headers {
		if !strings.EqualFold(name, OpenAIResponsesLiteHeader) {
			continue
		}
		for _, value := range values {
			if strings.EqualFold(strings.TrimSpace(value), "true") {
				return true
			}
		}
	}
	return false
}

// UsesResponsesLiteWireRequest is the raw-request form of the shared
// effective-profile decision. Outbound raw middleware uses it after arbitrary
// request mutations, where only the HTTP representation is available.
func UsesResponsesLiteWireRequest(request *httpclient.Request) bool {
	if request == nil {
		return false
	}
	if request.TransformerMetadata != nil && request.TransformerMetadata[ResponsesWireProfileMetadataKey] == ResponsesWireProfileLiteValue {
		return true
	}
	return HasResponsesLiteWireHeader(request)
}

// UsesResponsesLiteWireProfile is the single effective-profile decision used
// by response planning, encoding, and relay validation. Inbound adapters set
// metadata eagerly, while the raw-header fallback keeps direct Request users
// and pre-inbound planning consistent with that final wire contract.
func UsesResponsesLiteWireProfile(request *Request) bool {
	if request == nil {
		return false
	}
	if request.TransformerMetadata != nil && request.TransformerMetadata[ResponsesWireProfileMetadataKey] == ResponsesWireProfileLiteValue {
		return true
	}
	return UsesResponsesLiteWireRequest(request.RawRequest)
}

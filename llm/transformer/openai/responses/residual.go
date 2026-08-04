package responses

import (
	"encoding/json"
	"reflect"
	"strings"

	"github.com/looplj/axonhub/llm"
)

var (
	responsesRequestOwnedFields      = ownedJSONFields(reflect.TypeFor[Request]())
	responsesItemOwnedFields         = ownedJSONFields(reflect.TypeFor[Item]())
	responsesToolOwnedFields         = ownedJSONFields(reflect.TypeFor[Tool]())
	responsesChoiceOwnedFields       = ownedJSONFields(reflect.TypeFor[ToolChoice]())
	responsesResponseOwnedFields     = ownedJSONFields(reflect.TypeFor[Response]())
	responsesAnnotationFields        = ownedJSONFields(reflect.TypeFor[Annotation]())
	responsesCitationFields          = ownedJSONFields(reflect.TypeFor[URLCitation]())
	responsesSummaryFields           = ownedJSONFields(reflect.TypeFor[ReasoningSummary]())
	responsesReasoningFields         = ownedJSONFields(reflect.TypeFor[ReasoningContent]())
	responsesStreamEventFields       = ownedJSONFields(reflect.TypeFor[StreamEvent]())
	responsesCustomFormatFields      = ownedJSONFields(reflect.TypeFor[CustomToolFormat]())
	responsesWebFilterFields         = ownedJSONFields(reflect.TypeFor[WebSearchFilters]())
	responsesWebLocationFields       = ownedJSONFields(reflect.TypeFor[WebSearchUserLocation]())
	responsesReasoningOptionFields   = ownedJSONFields(reflect.TypeFor[Reasoning]())
	responsesStreamOptionFields      = ownedJSONFields(reflect.TypeFor[StreamOptions]())
	responsesConversationFields      = ownedJSONFields(reflect.TypeFor[Conversation]())
	responsesTextOptionFields        = ownedJSONFields(reflect.TypeFor[TextOptions]())
	responsesTextFormatFields        = ownedJSONFields(reflect.TypeFor[TextFormat]())
	responsesWebActionFields         = ownedJSONFields(reflect.TypeFor[WebSearchAction]())
	responsesWebSourceFields         = ownedJSONFields(reflect.TypeFor[WebSearchSource]())
	responsesLocalShellActionFields  = ownedJSONFields(reflect.TypeFor[LocalShellAction]())
	responsesKnownItemActionFields   = unionStringSets(responsesWebActionFields, responsesLocalShellActionFields)
	responsesToolOptionFields        = ownedJSONFields(reflect.TypeFor[ToolOption]())
	responsesSafetyCheckFields       = ownedJSONFields(reflect.TypeFor[ComputerSafetyCheck]())
	responsesScreenshotFields        = ownedJSONFields(reflect.TypeFor[ComputerScreenshot]())
	responsesMCPListedToolFields     = ownedJSONFields(reflect.TypeFor[MCPListedTool]())
	responsesStreamPartFields        = ownedJSONFields(reflect.TypeFor[StreamEventContentPart]())
	responsesCanonicalResponseFields = stringSet(
		"object", "id", "created_at", "model", "output", "status",
		"previous_response_id", "background", "usage",
	)
	responsesUsageFields        = stringSet("input_tokens", "input_tokens_details", "output_tokens", "output_tokens_details", "total_tokens")
	responsesInputDetailFields  = stringSet("cache_write_tokens", "cached_tokens")
	responsesOutputDetailFields = stringSet("reasoning_tokens")

	responsesRequestNestedResiduals = map[string]residualExtractor{
		"reasoning":      responseReasoningOptionsResidual,
		"stream_options": responseStreamOptionsResidual,
		"conversation":   responseConversationResidual,
		"text":           responseTextOptionsResidual,
	}
	responsesWebItemNestedResiduals = map[string]residualExtractor{
		"action": responseWebActionResidual,
	}
	responsesShellItemNestedResiduals = map[string]residualExtractor{
		"action": responseLocalShellActionResidual,
	}
	responsesToolNestedResiduals = map[string]residualExtractor{
		"format":        responseCustomFormatResidual,
		"filters":       responseWebFiltersResidual,
		"user_location": responseWebLocationResidual,
	}
	responsesResponseNestedResiduals = map[string]residualExtractor{
		"usage": responseUsageResidual,
	}
	responsesTextNestedResiduals = map[string]residualExtractor{
		"format": responseTextFormatResidual,
	}
	responsesUsageNestedResiduals = map[string]residualExtractor{
		"input_tokens_details":  responseInputDetailsResidual,
		"output_tokens_details": responseOutputDetailsResidual,
	}
)

type residualExtractor func(json.RawMessage) json.RawMessage

func stringSet(values ...string) map[string]struct{} {
	result := make(map[string]struct{}, len(values))
	for _, value := range values {
		result[value] = struct{}{}
	}
	return result
}

func unionStringSets(sets ...map[string]struct{}) map[string]struct{} {
	result := make(map[string]struct{})
	for _, set := range sets {
		for value := range set {
			result[value] = struct{}{}
		}
	}
	return result
}

func responseResidualOwnedByWire(hints llm.ProtocolHints, wireType string) bool {
	if hints.SourceFormat != llm.APIFormatOpenAIResponse {
		return false
	}
	ownerType := hints.ResidualOwnerType
	if ownerType == "" {
		ownerType = hints.SourceType
	}
	if ownerType == wireType {
		return true
	}
	return ownerType == "reasoning_text" && wireType == "reasoning"
}

func ownedJSONFields(typ reflect.Type) map[string]struct{} {
	fields := make(map[string]struct{}, typ.NumField())
	for index := 0; index < typ.NumField(); index++ {
		field := typ.Field(index)
		name := field.Tag.Get("json")
		if comma := strings.IndexByte(name, ','); comma >= 0 {
			name = name[:comma]
		}
		if name == "" || name == "-" {
			continue
		}
		fields[name] = struct{}{}
	}
	return fields
}

func jsonObjectResidual(raw json.RawMessage, owned map[string]struct{}) json.RawMessage {
	return jsonObjectResidualWithNested(raw, owned, nil)
}

func jsonObjectResidualWithNested(
	raw json.RawMessage,
	owned map[string]struct{},
	nested map[string]residualExtractor,
) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return nil
	}
	for key := range owned {
		extractor := nested[key]
		if extractor == nil {
			delete(object, key)
			continue
		}
		residual := extractor(object[key])
		if len(residual) == 0 {
			delete(object, key)
			continue
		}
		object[key] = residual
	}
	if len(object) == 0 {
		return nil
	}
	// Every RawMessage value came from a successful object decode, so the map
	// cannot contain an unsupported Go value or invalid raw JSON here.
	residual, _ := json.Marshal(object)
	return residual
}

func responseRequestResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidualWithNested(raw, responsesRequestOwnedFields, responsesRequestNestedResiduals)
}

func responseItemResidual(raw json.RawMessage) json.RawMessage {
	var identity struct {
		Type string `json:"type"`
	}
	_ = json.Unmarshal(raw, &identity)
	var nested map[string]residualExtractor
	switch identity.Type {
	case "web_search_call":
		nested = responsesWebItemNestedResiduals
	case "local_shell_call", "shell_call":
		nested = responsesShellItemNestedResiduals
	}
	return jsonObjectResidualWithNested(raw, responsesItemOwnedFields, nested)
}

func responseToolResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidualWithNested(raw, responsesToolOwnedFields, responsesToolNestedResiduals)
}

func responseToolChoiceResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesChoiceOwnedFields)
}

func responseResponseResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidualWithNested(raw, responsesCanonicalResponseFields, responsesResponseNestedResiduals)
}

func responseTextOptionsResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidualWithNested(raw, responsesTextOptionFields, responsesTextNestedResiduals)
}

func responseReasoningOptionsResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesReasoningOptionFields)
}

func responseStreamOptionsResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesStreamOptionFields)
}

func responseConversationResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesConversationFields)
}

func responseTextFormatResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesTextFormatFields)
}

func responseCustomFormatResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesCustomFormatFields)
}

func responseWebFiltersResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesWebFilterFields)
}

func responseWebLocationResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesWebLocationFields)
}

func responseWebActionResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesWebActionFields)
}

func responseLocalShellActionResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesLocalShellActionFields)
}

func responseItemActionResidual(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || raw[0] != '{' {
		return nil
	}
	return jsonObjectResidual(raw, responsesKnownItemActionFields)
}

func responseUsageResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidualWithNested(raw, responsesUsageFields, responsesUsageNestedResiduals)
}

func responseInputDetailsResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesInputDetailFields)
}

func responseOutputDetailsResidual(raw json.RawMessage) json.RawMessage {
	return jsonObjectResidual(raw, responsesOutputDetailFields)
}

func rawItemArrayField(raw json.RawMessage, field string) []json.RawMessage {
	if len(raw) == 0 {
		return nil
	}
	var object map[string]json.RawMessage
	if json.Unmarshal(raw, &object) != nil {
		return nil
	}
	var items []json.RawMessage
	if json.Unmarshal(object[field], &items) != nil {
		return nil
	}
	return items
}

func mergeResidualObject(current, residual json.RawMessage) (json.RawMessage, error) {
	if len(residual) == 0 {
		return current, nil
	}
	return mergeRawObject(current, residual)
}

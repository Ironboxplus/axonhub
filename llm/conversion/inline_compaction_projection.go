package conversion

import (
	"errors"
	"fmt"
	"strconv"
	"strings"

	"github.com/looplj/axonhub/llm"
)

type inlineCompactionDocumentProjector func(*llm.DocumentURL, llm.APIFormat) (inlineCompactionDocumentProjectionResult, error)
type inlineCompactionAgentEnvelopeEncoder func(*llm.AgentMessage) (string, error)

// checkedInlineCompactionDocumentProjection is the checked projection boundary
// used after a history item has been admitted. It has an injectable pure
// projector solely so its typed failure contract is tested without a provider.
func checkedInlineCompactionDocumentProjection(document *llm.DocumentURL, target llm.APIFormat, project inlineCompactionDocumentProjector) (inlineCompactionDocumentProjectionResult, error) {
	projection, err := project(document, target)
	if err != nil {
		return inlineCompactionDocumentProjectionResult{}, err
	}
	return projection, nil
}

func checkedInlineCompactionAgentEnvelope(message *llm.AgentMessage, encode inlineCompactionAgentEnvelopeEncoder) (string, error) {
	payload, err := encode(message)
	if err != nil {
		return "", &InlineCompactionError{Code: InlineCompactionInvalidState, Err: err}
	}
	return payload, nil
}

// inlineCompactionSummaryMessageBudget counts the safe model-visible payload
// used by both summary selection and truncation.
func inlineCompactionSummaryMessageBudget(message llm.Message) int {
	if message.Content.Content != nil {
		return len(*message.Content.Content)
	}
	budget := 0
	for index := range message.Content.MultipleContent {
		part := message.Content.MultipleContent[index]
		if part.Text != nil {
			budget += len(*part.Text)
		}
		if part.ImageURL != nil {
			budget += len(part.ImageURL.URL)
		}
		if part.Document != nil {
			budget += inlineCompactionDocumentProjectionBudget(part.Document)
		}
	}
	if budget == 0 {
		return 1
	}
	return budget
}

func inlineCompactionSummaryMessage(item llm.Item, target llm.APIFormat, drops *inlineCompactionDrops) (*llm.Message, error) {
	return inlineCompactionSummaryMessageWithProjectors(item, target, drops, inlineCompactionDocumentProjection, func(message *llm.AgentMessage) (string, error) {
		return message.LegacyInterAgentMessageJSON()
	})
}

func inlineCompactionSummaryMessageWithProjectors(item llm.Item, target llm.APIFormat, drops *inlineCompactionDrops, documentProjector inlineCompactionDocumentProjector, agentEncoder inlineCompactionAgentEnvelopeEncoder) (*llm.Message, error) {
	decision := classifyInlineCompactionItemForTarget(item, target)
	reject := func() (*llm.Message, error) {
		return nil, &InlineCompactionError{Code: inlineCompactionPlanError(decision).Code, Err: errors.New(decision.semantic)}
	}
	switch item.Kind {
	case llm.ItemKindMessage:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		content := make([]llm.MessageContentPart, 0, len(item.Content))
		for index := range item.Content {
			block := item.Content[index]
			switch block.Kind {
			case llm.ContentKindText, llm.ContentKindRefusal:
				text := block.Text
				content = append(content, llm.MessageContentPart{Type: "text", Text: &text})
			case llm.ContentKindImage:
				image := *block.Image
				content = append(content, llm.MessageContentPart{Type: "image_url", ImageURL: &image})
			case llm.ContentKindDocument:
				projection, err := checkedInlineCompactionDocumentProjection(block.Document, target, documentProjector)
				if err != nil {
					return nil, err
				}
				if drops != nil {
					drops.DocumentProjected++
					drops.DocumentTruncated += projection.truncated
					drops.DocumentTruncatedBytes += projection.truncatedBytes
					drops.DocumentSidecars += projection.sidecars
				}
				content = append(content, projection.part)
			}
		}
		if len(content) == 0 {
			return nil, nil
		}
		role := string(item.Role)
		if item.Role == llm.RoleAssistant {
			role = string(llm.RoleUser)
		}
		return &llm.Message{Role: role, Content: llm.MessageContent{MultipleContent: content}}, nil
	case llm.ItemKindToolCall:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		if len(item.ToolCall.ProviderData) > 0 && drops != nil {
			drops.Private++
		}
		return inlineCompactionMarker(inlineCompactionToolCallHistory(item.ToolCall)), nil
	case llm.ItemKindToolResult:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		return inlineCompactionToolResultMessage(item.ToolResult, target, drops)
	case llm.ItemKindMCPListTools:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		return inlineCompactionMarker("Historical MCP tool discovery was recorded."), nil
	case llm.ItemKindMCPApprovalRequest, llm.ItemKindMCPApprovalResponse:
		return inlineCompactionMarker("Historical MCP approval state was recorded."), nil
	case llm.ItemKindMCPCall:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		return inlineCompactionMarker(inlineCompactionMCPCallHistory(item.MCPCall)), nil
	case llm.ItemKindAgentMessage:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		payload, err := checkedInlineCompactionAgentEnvelope(item.AgentMessage, agentEncoder)
		if err != nil {
			return nil, err
		}
		return inlineCompactionMarker("Historical agent routing message:\n" + boundedInlineText(payload)), nil
	case llm.ItemKindReasoning:
		if drops != nil {
			drops.Reasoning++
		}
		return nil, nil
	case llm.ItemKindHostedCall:
		if decision.kind != inlineCompactionProjectionSummarize {
			return reject()
		}
		return inlineHostedCompactionSummary(item, drops)
	case llm.ItemKindToolDeclaration:
		return nil, nil
	default:
		return reject()
	}
}

func inlineHostedCompactionSummary(item llm.Item, drops *inlineCompactionDrops) (*llm.Message, error) {
	if item.HostedCall == nil {
		return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("hosted call is incomplete")}
	}
	call := item.HostedCall
	if call.Invocation.Kind != llm.ToolKindWebSearch && call.Invocation.Kind != llm.ToolKindWebFetch {
		return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: fmt.Errorf("hosted tool kind %q has no safe compact projection", call.Invocation.Kind)}
	}
	if drops != nil {
		drops.HostedProjected++
		if len(call.Invocation.ProviderData) > 0 {
			drops.Private++
		}
	}
	if call.Result == nil {
		return inlineCompactionMarker("Historical hosted web operation was recorded."), nil
	}
	if len(call.Result.ProviderData) > 0 && drops != nil {
		drops.Private++
	}
	if len(call.Result.StructuredContent) > 0 && drops != nil {
		drops.Structured++
	}
	text, err := inlineHostedCompactionText(call.Result.Content, drops)
	if err != nil {
		return nil, err
	}
	return inlineCompactionMarker("Historical hosted web outcome:\n" + text), nil
}
func inlineCompactionMarker(text string) *llm.Message {
	return &llm.Message{Role: "user", Content: llm.MessageContent{Content: stringPointer(boundedInlineText("[AXON_COMPACTION_HISTORY_DATA]\n" + text))}}
}

type inlineCompactionDocumentProjectionResult struct {
	part           llm.MessageContentPart
	truncated      uint32
	truncatedBytes uint32
	sidecars       uint32
}

func inlineCompactionDocumentProjection(document *llm.DocumentURL, target llm.APIFormat) (inlineCompactionDocumentProjectionResult, error) {
	if document == nil {
		return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("document is not safely projectable")}
	}
	sourceType := document.SourceType
	if sourceType == "" && document.URL != "" && document.Data == "" && document.FileID == "" && len(document.Content) == 0 {
		sourceType = llm.DocumentSourceURL
	}
	result := inlineCompactionDocumentProjectionResult{}
	if document.CacheControl != nil || document.Context != "" || document.CitationsEnabled != nil || document.Title != "" {
		result.sidecars++
	}
	clone := func() *llm.DocumentURL {
		return &llm.DocumentURL{SourceType: sourceType, Filename: document.Filename, MIMEType: document.MIMEType}
	}
	switch sourceType {
	case llm.DocumentSourceBase64:
		projected := clone()
		projected.Data = document.Data
		if inlineCompactionDocumentProjectionBudget(projected) > maxInlineCompactionSummaryVisibleBytes {
			return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("document data exceeds summary input budget")}
		}
		result.part = llm.MessageContentPart{Type: "document", Document: projected}
	case llm.DocumentSourceURL:
		if target == llm.APIFormatOpenAIChatCompletion {
			return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("target cannot encode URL document in safe summary")}
		}
		projected := clone()
		projected.URL = document.URL
		if inlineCompactionDocumentProjectionBudget(projected) > maxInlineCompactionSummaryVisibleBytes {
			return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("document URL exceeds summary input budget")}
		}
		result.part = llm.MessageContentPart{Type: "document", Document: projected}
	case llm.DocumentSourceFile:
		projected := clone()
		projected.FileID = document.FileID
		if inlineCompactionDocumentProjectionBudget(projected) > maxInlineCompactionSummaryVisibleBytes {
			return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("document file ID exceeds summary input budget")}
		}
		result.part = llm.MessageContentPart{Type: "document", Document: projected}
	case llm.DocumentSourceText:
		prefix := "[AXON_COMPACTION_DOCUMENT_TEXT"
		if document.Filename != "" {
			prefix += " " + document.Filename
		}
		prefix += "]\n"
		text := boundedInlineTextToLimit(document.Data, maxInlineCompactionProjectedText-len(prefix))
		if text == "" {
			return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("document text has no safe summary content")}
		}
		if len(document.Data) > len(text) {
			result.truncated = 1
			result.truncatedBytes = uint32(len(document.Data) - len(text))
		}
		result.part = safeInlineTextPart(prefix + text)
	case llm.DocumentSourceContent:
		return inlineCompactionDocumentProjectionResult{}, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("document content JSON has no closed summary projection")}
	}
	return result, nil
}
func inlineCompactionDocumentProjectionBudget(document *llm.DocumentURL) int {
	if document == nil {
		return 0
	}
	budget := 64 + len(document.Filename) + len(document.MIMEType)
	switch document.SourceType {
	case llm.DocumentSourceBase64, llm.DocumentSourceText:
		budget += len(document.Data)
	case llm.DocumentSourceURL:
		budget += len(document.URL)
	case llm.DocumentSourceFile:
		budget += len(document.FileID)
	}
	return budget
}

func inlineCompactionToolResultMessage(result *llm.ToolResult, target llm.APIFormat, drops *inlineCompactionDrops) (*llm.Message, error) {
	if result == nil {
		return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("tool result is not safely projectable")}
	}
	if len(result.Content) == 0 && (len(result.ProviderData) > 0 || len(result.StructuredContent) > 0) {
		return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("tool result has no model-visible content")}
	}
	if drops != nil {
		if len(result.ProviderData) > 0 {
			drops.Private++
		}
		if len(result.StructuredContent) > 0 {
			drops.Structured++
		}
	}
	parts := []llm.MessageContentPart{safeInlineTextPart(inlineCompactionToolResultHistory(result))}
	for index := range result.Content {
		block := result.Content[index]
		provenance, err := inlineCompactionToolResultBlockProvenance(block)
		if err != nil {
			return nil, err
		}
		if provenance && drops != nil {
			drops.SourceSidecars++
		}
		switch block.Kind {
		case llm.ContentKindText, llm.ContentKindRefusal:
			text := block.Text
			parts = append(parts, llm.MessageContentPart{Type: "text", Text: &text})
		case llm.ContentKindImage:
			if block.Image == nil || block.Image.URL == "" {
				return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("tool result has no typed image")}
			}
			if len(block.Image.URL) > maxInlineCompactionRetainedItemBytes {
				return nil, &InlineCompactionError{Code: InlineCompactionOversize, Err: errors.New("tool result image exceeds the gateway limit")}
			}
			image := *block.Image
			parts = append(parts, llm.MessageContentPart{Type: "image_url", ImageURL: &image})
		case llm.ContentKindDocument:
			projection, err := inlineCompactionDocumentProjection(block.Document, target)
			if err != nil {
				return nil, err
			}
			if drops != nil {
				drops.DocumentProjected++
				drops.DocumentTruncated += projection.truncated
				drops.DocumentTruncatedBytes += projection.truncatedBytes
				drops.DocumentSidecars += projection.sidecars
			}
			parts = append(parts, projection.part)
		default:
			return nil, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: fmt.Errorf("tool result contains unsupported content kind %q", block.Kind)}
		}
	}
	return &llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: parts}}, nil
}

func inlineCompactionToolCallHistory(call *llm.ToolInvocation) string {
	if call == nil {
		return "Historical client tool call."
	}
	return inlineCompactionHistoryFields("Historical client tool call", []inlineCompactionHistoryField{{"kind", string(call.Kind)}, {"name", call.LogicalName}, {"namespace", call.Namespace}, {"status", string(call.Status)}})
}
func inlineCompactionToolResultHistory(result *llm.ToolResult) string {
	if result == nil {
		return "[AXON_COMPACTION_HISTORY_DATA]\nHistorical client tool outcome:"
	}
	return inlineCompactionHistoryFields("[AXON_COMPACTION_HISTORY_DATA]\nHistorical client tool outcome", []inlineCompactionHistoryField{{"kind", string(result.Kind)}, {"name", result.LogicalName}, {"status", string(result.Status)}, {"error", strconv.FormatBool(result.IsError)}})
}
func inlineCompactionMCPCallHistory(call *llm.MCPCall) string {
	if call == nil {
		return "Historical MCP outcome:"
	}
	return inlineCompactionHistoryFields("Historical MCP outcome", []inlineCompactionHistoryField{{"server", call.ServerLabel}, {"name", call.LogicalName}, {"status", string(call.Status)}, {"error", call.Error}, {"output", call.Output}})
}

type inlineCompactionHistoryField struct {
	label string
	value string
}

func inlineCompactionHistoryFields(prefix string, fields []inlineCompactionHistoryField) string {
	var builder strings.Builder
	builder.WriteString(prefix)
	for _, field := range fields {
		if field.value == "" {
			continue
		}
		builder.WriteByte('\n')
		builder.WriteString(field.label)
		builder.WriteString(": ")
		builder.WriteString(boundedInlineText(field.value))
	}
	return boundedInlineText(builder.String())
}
func inlineCompactionToolResultContentProvenance(content []llm.ContentBlock) (uint32, error) {
	var stripped uint32
	for index := range content {
		provenance, err := inlineCompactionToolResultBlockProvenance(content[index])
		if err != nil {
			return 0, err
		}
		if provenance {
			stripped++
		}
	}
	return stripped, nil
}
func inlineCompactionToolResultBlockProvenance(block llm.ContentBlock) (bool, error) {
	if len(block.UnknownRaw) > 0 || len(block.SourceResidual) > 0 {
		return false, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("tool result content has a private residual")}
	}
	if block.ResidualOwnerType == "" {
		return false, nil
	}
	allowed := false
	switch block.Kind {
	case llm.ContentKindText, llm.ContentKindRefusal:
		allowed = block.ResidualOwnerType == "text" || block.ResidualOwnerType == "input_text" || block.ResidualOwnerType == "output_text"
	case llm.ContentKindImage:
		allowed = block.ResidualOwnerType == "input_image"
	case llm.ContentKindDocument:
		allowed = block.ResidualOwnerType == "input_file"
	}
	if !allowed {
		return false, &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("tool result content has an unrecognized source union")}
	}
	return true, nil
}
func inlineHostedCompactionText(content []llm.ContentBlock, drops *inlineCompactionDrops) (string, error) {
	parts := make([]string, 0, len(content))
	for index := range content {
		block := content[index]
		switch block.Kind {
		case llm.ContentKindText, llm.ContentKindRefusal:
			parts = append(parts, block.Text)
		case llm.ContentKindCitation:
			if block.Citation == nil {
				return "", &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: errors.New("hosted citation is provider-private")}
			}
			if block.Citation.EncryptedIndex != nil && drops != nil {
				drops.Private++
			}
			citation := "citation"
			if block.Citation.Title != "" {
				citation += ": " + block.Citation.Title
			}
			if block.Citation.URL != "" {
				citation += " (" + block.Citation.URL + ")"
			}
			parts = append(parts, citation)
		default:
			return "", &InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: fmt.Errorf("hosted result contains unsupported content kind %q", block.Kind)}
		}
	}
	return boundedInlineText(strings.Join(parts, "\n")), nil
}

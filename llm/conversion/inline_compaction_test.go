package conversion

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/looplj/axonhub/llm"
)

func TestInlineCompactionTextBoundsPreserveUTF8AndContinuationPrefix(t *testing.T) {
	if MaxInlineCompactionCheckpointTokenBytes != maxInlineCompactionTokenBytes {
		t.Fatalf("checkpoint token contract=%d internal=%d", MaxInlineCompactionCheckpointTokenBytes, maxInlineCompactionTokenBytes)
	}
	text := strings.Repeat("中文🙂", maxInlineCompactionProjectedText/len("中文🙂")+2)
	bounded := boundedInlineText(text)
	if len(bounded) > maxInlineCompactionProjectedText || !utf8.ValidString(bounded) {
		t.Fatalf("bounded text len=%d utf8=%v", len(bounded), utf8.ValidString(bounded))
	}
	continuation := inlineCompactionContinuationText(text)
	if !strings.HasPrefix(continuation, inlineCompactionSummaryPrefix) || len(continuation) > maxInlineCompactionVisibleTextBytes || !utf8.ValidString(continuation) {
		t.Fatalf("continuation prefix/len/utf8 invalid: len=%d valid=%v", len(continuation), utf8.ValidString(continuation))
	}
	state := CompactionState{
		Version:    inlineCompactionStateVersion,
		Generation: 1,
		Retained: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "current"}},
		}},
		Continuation: llm.Item{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content:       []llm.ContentBlock{{Kind: llm.ContentKindText, Text: continuation}},
			ProtocolHints: llm.ProtocolHints{SourceGroup: inlineCompactionContinuationSourceGroup},
		},
	}
	if err := validateInlineCompactionState(state); err != nil {
		t.Fatalf("valid bounded continuation rejected: %v", err)
	}
}

func validInlineCompactionStateForTest() CompactionState {
	return CompactionState{
		Version:    inlineCompactionStateVersion,
		Generation: 1,
		Retained: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained user context"}},
		}},
		Continuation: llm.Item{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content:       []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionSummaryPrefix + "safe continuation"}},
			ProtocolHints: llm.ProtocolHints{SourceGroup: inlineCompactionContinuationSourceGroup},
		},
	}
}

func TestInlineCompactionClosedStateAndDiagnosticBoundaries(t *testing.T) {
	var nilInlineError *InlineCompactionError
	if diagnostic := nilInlineError.SafeDiagnostic(); diagnostic.Component != "" || diagnostic.Code != "" || diagnostic.StatusCode != 0 {
		t.Fatalf("nil inline error leaked a diagnostic: %#v", diagnostic)
	}
	if err := inlineCompactionCheckpointStructuralError(nil); err != nil {
		t.Fatalf("nil request has no checkpoint invariant: %v", err)
	}

	tests := []struct {
		name  string
		state func() CompactionState
		code  InlineCompactionErrorCode
	}{
		{name: "state version", code: InlineCompactionInvalidState, state: func() CompactionState {
			state := validInlineCompactionStateForTest()
			state.Version++
			return state
		}},
		{name: "generation", code: InlineCompactionInvalidState, state: func() CompactionState {
			state := validInlineCompactionStateForTest()
			state.Generation = 0
			return state
		}},
		{name: "empty retained", code: InlineCompactionInvalidState, state: func() CompactionState {
			state := validInlineCompactionStateForTest()
			state.Retained = nil
			return state
		}},
		{name: "visible retained budget", code: InlineCompactionOversize, state: func() CompactionState {
			state := validInlineCompactionStateForTest()
			state.Retained[0].Content[0].Text = strings.Repeat("x", maxInlineCompactionVisibleTextBytes+1)
			return state
		}},
		{name: "serialized durable state budget", code: InlineCompactionOversize, state: func() CompactionState {
			state := validInlineCompactionStateForTest()
			state.Retained = nil
			for index := 0; index < 5; index++ {
				state.Retained = append(state.Retained, llm.Item{
					Kind: llm.ItemKindMessage, Role: llm.RoleUser,
					Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", 900<<10)}}},
				})
			}
			return state
		}},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			_, err := sanitizeInlineCompactionState(test.state())
			var inlineErr *InlineCompactionError
			if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != test.code {
				t.Fatalf("sanitize error=%T %v inline=%#v", err, err, inlineErr)
			}
		})
	}
	if safe, err := sanitizeInlineCompactionState(validInlineCompactionStateForTest()); err != nil || safe.Continuation.Role != llm.RoleUser || len(safe.Retained) != 1 {
		t.Fatalf("valid state was not reconstructed safely: state=%#v err=%v", safe, err)
	}
}

func TestInlineCompactionForcesNonStreamingOnlyForStreamingTrigger(t *testing.T) {
	outbound := &Outbound{}
	stream := true
	notStream := false
	tests := []struct {
		name    string
		request *llm.Request
		want    bool
	}{
		{name: "nil", want: false},
		{name: "non stream", request: &llm.Request{Stream: &notStream, Input: []llm.Item{{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}}}}, want: false},
		{name: "ordinary stream", request: &llm.Request{Stream: &stream, Input: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser}}}, want: false},
		{name: "inline checkpoint stream", request: &llm.Request{Stream: &stream, Input: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser}, {Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}}}}, want: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if got := outbound.ForceNonStreaming(nil, test.request); got != test.want {
				t.Fatalf("ForceNonStreaming()=%v want=%v", got, test.want)
			}
		})
	}
}

func TestInlineCompactionPlannerOwnershipPredicatesCoverBothLifecycleHalves(t *testing.T) {
	trigger := Action{Ref: ObjectRef{Kind: ObjectRequestControl}, Kind: ActionEmulate, Strategy: StrategyInlineCompactionGateway}
	hydrate := Action{Ref: ObjectRef{Kind: ObjectCompaction}, Kind: ActionEmulate, Strategy: StrategyInlineCompactionHydrate}
	notOwned := Action{Ref: ObjectRef{Kind: ObjectCompaction}, Kind: ActionUnknown, Strategy: StrategyInlineCompactionHydrate}
	tests := []struct {
		name     string
		plan     *Plan
		requires bool
		executes bool
	}{
		{name: "nil"},
		{name: "unrelated", plan: &Plan{Actions: []Action{notOwned}}},
		{name: "trigger", plan: &Plan{Actions: []Action{trigger}}, requires: true, executes: true},
		{name: "checkpoint only", plan: &Plan{Actions: []Action{hydrate}}, requires: true},
		{name: "checkpoint then trigger", plan: &Plan{Actions: []Action{hydrate, trigger}}, requires: true, executes: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			if got := planRequiresGatewayInlineCompactionCodec(test.plan); got != test.requires {
				t.Fatalf("requires codec=%v want=%v", got, test.requires)
			}
			if got := planExecutesGatewayInlineCompactionSummary(test.plan); got != test.executes {
				t.Fatalf("executes summary=%v want=%v", got, test.executes)
			}
		})
	}
}

func TestClassifyInlineCompactionItemForTargetUsesClosedProjectionMatrix(t *testing.T) {
	validMessage := func(text string) llm.Item {
		return llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: text}}}
	}
	validAgent := &llm.AgentMessage{Author: "/root/worker", Recipient: "/root", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "route"}}}
	validHosted := &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}}
	tests := []struct {
		name     string
		item     llm.Item
		target   llm.APIFormat
		kind     inlineCompactionProjectionKind
		semantic string
	}{
		{name: "message text", item: validMessage("safe"), target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize, semantic: "inline_history_message"},
		{name: "assistant history", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "assistant history"}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "unsupported message role", item: llm.Item{Kind: llm.ItemKindMessage, Role: "future", Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "future"}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_message_role"},
		{name: "message refusal", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindRefusal, Text: "no"}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "message residual sidecar is stripped", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, ProtocolHints: llm.ProtocolHints{SourceResidual: []byte(`{"private":true}`)}, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "visible"}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize, semantic: "inline_history_message"},
		{name: "message empty image", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_oversize_image"},
		{name: "message valid image", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,SAFE"}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "message image beyond summary cap", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", maxInlineCompactionSummaryVisibleBytes+1)}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_oversize_image"},
		{name: "message URL document unsupported by chat", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceURL, URL: "https://example.test/file.pdf"}}}}, target: llm.APIFormatOpenAIChatCompletion, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_document"},
		{name: "message oversized document", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: strings.Repeat("d", maxInlineCompactionSummaryVisibleBytes)}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_oversize_document"},
		{name: "message unknown content", item: llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindUnknown}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_content"},
		{name: "tool call missing", item: llm.Item{Kind: llm.ItemKindToolCall}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_invalid_tool_call"},
		{name: "tool call", item: llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "tool result missing", item: llm.Item{Kind: llm.ItemKindToolResult}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_invalid_tool_result"},
		{name: "tool result unsupported content", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindImage}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_tool_result"},
		{name: "tool result private only", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{ProviderData: []byte(`{"private":true}`)}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_private_only_tool_result"},
		{name: "tool result typed content", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "result"}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "tool result oversized document", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: strings.Repeat("d", maxInlineCompactionSummaryVisibleBytes)}}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_oversize_tool_result"},
		{name: "tool result raw residual is blocked", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "result", SourceResidual: []byte(`{"future":true}`)}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_private_tool_result_content"},
		{name: "tool result unknown raw is blocked", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "result", UnknownRaw: []byte(`{"future":true}`)}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_private_tool_result_content"},
		{name: "tool result unknown provenance is blocked", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "result", ResidualOwnerType: "future_output"}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_private_tool_result_content"},
		{name: "tool result known Responses provenance is visibly stripped", item: llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "result", ResidualOwnerType: "input_text"}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize, semantic: "inline_history_tool_result_provenance_stripped"},
		{name: "MCP discovery missing", item: llm.Item{Kind: llm.ItemKindMCPListTools}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_invalid_mcp_discovery"},
		{name: "MCP discovery", item: llm.Item{Kind: llm.ItemKindMCPListTools, MCPListTools: &llm.MCPListTools{}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "MCP approval request", item: llm.Item{Kind: llm.ItemKindMCPApprovalRequest}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "MCP approval response", item: llm.Item{Kind: llm.ItemKindMCPApprovalResponse}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "MCP call missing", item: llm.Item{Kind: llm.ItemKindMCPCall}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_invalid_mcp_call"},
		{name: "MCP call", item: llm.Item{Kind: llm.ItemKindMCPCall, MCPCall: &llm.MCPCall{}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "agent missing", item: llm.Item{Kind: llm.ItemKindAgentMessage}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_invalid_agent_message"},
		{name: "agent private", item: llm.Item{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_private_agent_message"},
		{name: "agent plaintext", item: llm.Item{Kind: llm.ItemKindAgentMessage, AgentMessage: validAgent}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "reasoning", item: llm.Item{Kind: llm.ItemKindReasoning}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionDrop, semantic: "inline_history_reasoning_dropped"},
		{name: "hosted missing", item: llm.Item{Kind: llm.ItemKindHostedCall}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_hosted_call"},
		{name: "hosted wrong tool", item: llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindFunction}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_hosted_call"},
		{name: "hosted unsupported result", item: llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}, Result: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindImage}}}}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported_hosted_result"},
		{name: "hosted web", item: llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: validHosted}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize},
		{name: "tool declaration", item: llm.Item{Kind: llm.ItemKindToolDeclaration}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionStrip, semantic: "inline_history_tool_declaration_stripped"},
		{name: "checkpoint missing", item: llm.Item{Kind: llm.ItemKindCompaction}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_checkpoint_missing_opaque_token"},
		{name: "checkpoint oversized", item: llm.Item{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: strings.Repeat("t", maxInlineCompactionTokenBytes+1)}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_checkpoint_token_oversize"},
		{name: "checkpoint", item: llm.Item{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: "gateway-token"}}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionSummarize, semantic: "compaction_checkpoint"},
		{name: "trigger", item: llm.Item{Kind: llm.ItemKindCompactionTrigger}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionStrip, semantic: "compaction_trigger_request_control"},
		{name: "context compaction", item: llm.Item{Kind: llm.ItemKindContextCompaction}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_context_compaction"},
		{name: "unknown", item: llm.Item{Kind: llm.ItemKindUnknown}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unknown"},
		{name: "future kind", item: llm.Item{Kind: "future"}, target: llm.APIFormatOpenAIResponse, kind: inlineCompactionProjectionBlock, semantic: "inline_history_unsupported"},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			decision := classifyInlineCompactionItemForTarget(test.item, test.target)
			if decision.kind != test.kind || test.semantic != "" && decision.semantic != test.semantic {
				t.Fatalf("decision=%#v want kind=%v semantic=%q", decision, test.kind, test.semantic)
			}
		})
	}
}

type recordingInlineCompactionCodec struct {
	sealed    CompactionState
	token     string
	err       error
	openState CompactionState
	owned     bool
}

func (codec *recordingInlineCompactionCodec) Seal(_ context.Context, state CompactionState) (string, error) {
	codec.sealed = state
	return codec.token, codec.err
}

func (codec *recordingInlineCompactionCodec) Open(context.Context, string) (CompactionState, bool, error) {
	return codec.openState, codec.owned, codec.err
}

func TestInlineCompactionRestoreSealsOnlySanitizedState(t *testing.T) {
	codec := &recordingInlineCompactionCodec{token: "gateway-owned-token"}
	plan := &Plan{}
	session := newSession(plan, &llm.Request{}, true)
	session.inlineCompaction = &inlineCompactionState{
		codec: codec, generation: 1,
		retained: []llm.Item{{
			Kind: llm.ItemKindMessage, ID: "source-id-must-not-persist", Role: llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained"}},
		}},
	}
	restored, err := restoreInlineCompaction(context.Background(), &llm.Response{
		ID: "provider-summary-id", Model: "fixture", Status: llm.ResponseStatusCompleted,
		Usage:  &llm.Usage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
		Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "summary plaintext"}}}},
	}, session)
	if err != nil {
		t.Fatal(err)
	}
	if len(codec.sealed.Retained) != 1 || codec.sealed.Retained[0].ID != "" || codec.sealed.Continuation.Role != llm.RoleUser || codec.sealed.Continuation.ProtocolHints.SourceGroup != inlineCompactionContinuationSourceGroup {
		t.Fatalf("codec received unsanitized durable state: %#v", codec.sealed)
	}
	if restored == nil || restored.ID == "provider-summary-id" || restored.Usage == nil || restored.Usage.TotalTokens != 3 || len(restored.Output) != 1 || restored.Output[0].Compaction == nil || restored.Output[0].Compaction.EncryptedContent != codec.token {
		t.Fatalf("restore did not return a fresh public checkpoint response: %#v", restored)
	}
}

func TestInlineCompactionGenerationOverflowBlocksBeforeSummaryProjection(t *testing.T) {
	stream := true
	request := &llm.Request{
		Stream:              &stream,
		TransformerMetadata: map[string]any{inlineCompactionGenerationMetadataKey: ^uint32(0)},
		Input: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "current"}}},
			{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
		},
	}
	plan := &Plan{Actions: []Action{{Ref: ObjectRef{Kind: ObjectRequestControl}, Kind: ActionEmulate, Strategy: StrategyInlineCompactionGateway}}}
	_, state, err := projectInlineCompactionExecution(request, plan, nil)
	var inlineErr *InlineCompactionError
	if state != nil || !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState {
		t.Fatalf("overflow must block before codec/provider projection: state=%#v err=%v inline=%#v", state, err, inlineErr)
	}
}

func TestInlineCompactionGenerationAcceptsOnlyBoundedPositiveMetadata(t *testing.T) {
	tests := []struct {
		name  string
		value any
		want  uint32
		fail  bool
	}{
		{name: "nil request defaults", want: 1},
		{name: "missing metadata defaults", value: nil, want: 1},
		{name: "uint32 generation advances", value: uint32(7), want: 8},
		{name: "int generation advances", value: 7, want: 8},
		{name: "zero uint32 defaults", value: uint32(0), want: 1},
		{name: "negative int defaults", value: -1, want: 1},
		{name: "max int fails closed", value: int(^uint32(0)), fail: true},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			var request *llm.Request
			if test.name != "nil request defaults" {
				metadata := map[string]any{}
				if test.value != nil {
					metadata[inlineCompactionGenerationMetadataKey] = test.value
				}
				request = &llm.Request{TransformerMetadata: metadata}
			}
			got, err := inlineCompactionGeneration(request)
			if test.fail {
				var inlineErr *InlineCompactionError
				if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState {
					t.Fatalf("generation err=%v inline=%#v", err, inlineErr)
				}
				return
			}
			if err != nil || got != test.want {
				t.Fatalf("generation=%d err=%v want=%d", got, err, test.want)
			}
		})
	}
}

func TestInlineCompactionHydrationRejectsInvalidOwnershipBeforeInjection(t *testing.T) {
	validState := validInlineCompactionStateForTest()
	checkpoint := func(token string) *llm.Request {
		return &llm.Request{TransformerMetadata: map[string]any{"preserve": "metadata"}, Input: []llm.Item{
			{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: token}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "current"}}},
		}}
	}
	plan := &Plan{}
	if hydrated, used, err := hydrateGatewayInlineCompaction(context.Background(), nil, plan, nil); err != nil || hydrated != nil || used {
		t.Fatalf("nil request hydration=%#v used=%v err=%v", hydrated, used, err)
	}
	ordinary := &llm.Request{Input: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser}}}
	if hydrated, used, err := hydrateGatewayInlineCompaction(context.Background(), ordinary, plan, nil); err != nil || hydrated != ordinary || used {
		t.Fatalf("ordinary hydration=%#v used=%v err=%v", hydrated, used, err)
	}
	tests := []struct {
		name  string
		input *llm.Request
		codec CompactionStateCodec
		code  InlineCompactionErrorCode
	}{
		{name: "multiple checkpoints", input: &llm.Request{Input: []llm.Item{{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: "one"}}, {Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: "two"}}}}, code: InlineCompactionInvalidState},
		{name: "codec missing", input: checkpoint("owned"), code: InlineCompactionCodecMissing},
		{name: "empty token", input: checkpoint(""), codec: &recordingInlineCompactionCodec{}, code: InlineCompactionInvalidToken},
		{name: "oversized token", input: checkpoint(strings.Repeat("x", maxInlineCompactionTokenBytes+1)), codec: &recordingInlineCompactionCodec{}, code: InlineCompactionOversize},
		{name: "codec failure", input: checkpoint("owned"), codec: &recordingInlineCompactionCodec{err: errors.New("store down")}, code: InlineCompactionCodecUnavailable},
		{name: "foreign token", input: checkpoint("foreign"), codec: &recordingInlineCompactionCodec{owned: false}, code: InlineCompactionForeignToken},
		{name: "unsafe stored state", input: checkpoint("owned"), codec: &recordingInlineCompactionCodec{owned: true, openState: CompactionState{Version: 1}}, code: InlineCompactionInvalidState},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			_, used, err := hydrateGatewayInlineCompaction(context.Background(), test.input, plan, test.codec)
			var inlineErr *InlineCompactionError
			if used || !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != test.code {
				t.Fatalf("used=%v err=%v inline=%#v", used, err, inlineErr)
			}
		})
	}
	hydrated, used, err := hydrateGatewayInlineCompaction(context.Background(), checkpoint("owned"), plan, &recordingInlineCompactionCodec{owned: true, openState: validState})
	if err != nil || !used || len(hydrated.Input) != 3 || hydrated.Input[0].Content[0].Text != "retained user context" || hydrated.Input[1].ProtocolHints.SourceGroup != inlineCompactionContinuationSourceGroup || hydrated.TransformerMetadata["preserve"] != "metadata" || hydrated.TransformerMetadata[inlineCompactionGenerationMetadataKey] != uint32(1) {
		t.Fatalf("safe hydration=%#v used=%v err=%v", hydrated, used, err)
	}
}

func TestInlineCompactionProjectionBlocksEveryPreProviderInvalidState(t *testing.T) {
	triggerPlan := &Plan{Target: CapabilityProfile{APIFormat: llm.APIFormatOpenAIResponse}, Actions: []Action{{Ref: ObjectRef{Kind: ObjectRequestControl}, Kind: ActionEmulate, Strategy: StrategyInlineCompactionGateway}}}
	triggerRequest := func(history []llm.Item) *llm.Request {
		return &llm.Request{Model: "fixture", APIFormat: llm.APIFormatOpenAIResponse, Input: append(history, llm.Item{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}})}
	}
	if projected, state, err := projectInlineCompactionExecution(&llm.Request{}, &Plan{}, nil); err != nil || projected == nil || state != nil {
		t.Fatalf("non-inline plan must pass through: projected=%#v state=%#v err=%v", projected, state, err)
	}
	tests := []struct {
		name    string
		request *llm.Request
		codec   CompactionStateCodec
		code    InlineCompactionErrorCode
	}{
		{name: "missing trigger", request: &llm.Request{}, codec: &recordingInlineCompactionCodec{}, code: InlineCompactionInvalidState},
		{name: "codec missing", request: triggerRequest([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retain"}}}}), code: InlineCompactionCodecMissing},
		{name: "no history", request: triggerRequest(nil), codec: &recordingInlineCompactionCodec{}, code: InlineCompactionInvalidState},
		{name: "unretainable history", request: triggerRequest([]llm.Item{{Kind: llm.ItemKindToolDeclaration, ToolDeclaration: &llm.ToolDeclarationItem{}}}), codec: &recordingInlineCompactionCodec{}, code: InlineCompactionUnsafeInput},
		{name: "unsafe retained image", request: triggerRequest([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", maxInlineCompactionRetainedItemBytes+1)}}}}}), codec: &recordingInlineCompactionCodec{}, code: InlineCompactionOversize},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			_, state, err := projectInlineCompactionExecution(test.request, triggerPlan, test.codec)
			var inlineErr *InlineCompactionError
			if state != nil || !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != test.code {
				t.Fatalf("state=%#v err=%v inline=%#v", state, err, inlineErr)
			}
		})
	}
	projected, state, err := projectInlineCompactionExecution(triggerRequest([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retain"}}}}), triggerPlan, &recordingInlineCompactionCodec{})
	if err != nil || state == nil || projected == nil || projected.Stream == nil || *projected.Stream || len(projected.Input) < 2 || projected.Input[0].Role != llm.RoleDeveloper {
		t.Fatalf("safe summary projection=%#v state=%#v err=%v", projected, state, err)
	}
}

func TestInlineCompactionSanitizerSubcontractsRejectEveryNonClosedShape(t *testing.T) {
	validItem := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "safe"}}}
	for _, item := range []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser},
		{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: validItem.Content},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Status: llm.ItemStatusCompleted, Content: validItem.Content},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, ProtocolHints: llm.ProtocolHints{SourceGroup: "private"}, Content: validItem.Content},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{}},
	} {
		if _, err := sanitizeInlineCompactionRetainedItem(item); err == nil {
			t.Fatalf("unsafe retained item accepted: %#v", item)
		}
	}
	if item, err := sanitizeInlineCompactionRetainedItem(validItem); err != nil || item.ID != "" || item.Content[0].Text != "safe" {
		t.Fatalf("valid retained item=%#v err=%v", item, err)
	}
	for _, block := range []llm.ContentBlock{
		{Kind: llm.ContentKindText, ID: "private"},
		{Kind: llm.ContentKindText, Text: "text", Image: &llm.ImageURL{URL: "extra"}},
		{Kind: llm.ContentKindImage},
		{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: ""}},
		{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "file"}},
	} {
		if _, err := sanitizeInlineCompactionRetainedContent(block); err == nil {
			t.Fatalf("unsafe retained block accepted: %#v", block)
		}
	}
	if image, err := sanitizeInlineCompactionRetainedContent(llm.ContentBlock{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,SAFE"}}); err != nil || image.Image == nil {
		t.Fatalf("valid image block=%#v err=%v", image, err)
	}
	validContinuation := validInlineCompactionStateForTest().Continuation
	for _, item := range []llm.Item{
		{},
		{Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper, Content: validContinuation.Content, ProtocolHints: validContinuation.ProtocolHints},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "not a prefix"}}, ProtocolHints: validContinuation.ProtocolHints},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionSummaryPrefix + "safe", SourceResidual: []byte(`{"private":true}`)}}, ProtocolHints: validContinuation.ProtocolHints},
	} {
		if _, err := sanitizeInlineContinuation(item); err == nil {
			t.Fatalf("unsafe continuation accepted: %#v", item)
		}
	}
	if continuation, err := sanitizeInlineContinuation(validContinuation); err != nil || continuation.Role != llm.RoleUser {
		t.Fatalf("valid continuation=%#v err=%v", continuation, err)
	}
}

func TestInlineHostedCompactionProjectionStripsPrivateSidecars(t *testing.T) {
	encrypted := "opaque-index"
	text, err := inlineHostedCompactionText([]llm.ContentBlock{
		{Kind: llm.ContentKindText, Text: "WEB_MARKER"},
		{Kind: llm.ContentKindRefusal, Text: "REFUSAL_MARKER"},
		{Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{Title: "Citation", URL: "https://example.test", EncryptedIndex: &encrypted}},
	}, &inlineCompactionDrops{})
	if err != nil || !strings.Contains(text, "WEB_MARKER") || !strings.Contains(text, "Citation") || strings.Contains(text, encrypted) {
		t.Fatalf("hosted text=%q err=%v", text, err)
	}
	if _, err := inlineHostedCompactionText([]llm.ContentBlock{{Kind: llm.ContentKindCitation}}, nil); err == nil {
		t.Fatal("citation without typed payload was accepted")
	}
	if _, err := inlineHostedCompactionText([]llm.ContentBlock{{Kind: llm.ContentKindImage}}, nil); err == nil {
		t.Fatal("unsupported hosted content was accepted")
	}

	base := llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}}}
	if message, err := inlineHostedCompactionSummary(base, nil); err != nil || message == nil || message.Role != "user" {
		t.Fatalf("hosted operation summary=%#v err=%v", message, err)
	}
	drops := inlineCompactionDrops{}
	base.HostedCall.Invocation.ProviderData = []byte(`{"PRIVATE":"invocation"}`)
	base.HostedCall.Result = &llm.ToolResult{Kind: llm.ToolKindWebSearch, ProviderData: []byte(`{"PRIVATE":"result"}`), StructuredContent: []byte(`{"PRIVATE":"structured"}`), Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "RESULT_MARKER"}, {Kind: llm.ContentKindCitation, Citation: &llm.URLCitation{Title: "Safe", URL: "https://example.test", EncryptedIndex: &encrypted}}}}
	message, err := inlineHostedCompactionSummary(base, &drops)
	if err != nil || message == nil || !inlineSummaryMessageContains(*message, "RESULT_MARKER") || drops.HostedProjected != 1 || drops.Private < 3 || drops.Structured != 1 {
		t.Fatalf("hosted summary=%#v drops=%#v err=%v", message, drops, err)
	}
	if _, err := inlineHostedCompactionSummary(llm.Item{Kind: llm.ItemKindHostedCall}, nil); err == nil {
		t.Fatal("missing hosted call was accepted")
	}
	if _, err := inlineHostedCompactionSummary(llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindFunction}}}, nil); err == nil {
		t.Fatal("non-web hosted call was accepted")
	}
	if _, err := inlineHostedCompactionSummary(llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}, Result: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindImage}}}}}, nil); err == nil {
		t.Fatal("hosted result with unsupported typed content was accepted")
	}
}

func TestInlineCompactionSummaryMessageProjectsTypedHistoryWithoutPrivateSidecars(t *testing.T) {
	drops := inlineCompactionDrops{}
	message, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{
		{Kind: llm.ContentKindText, Text: "TEXT"}, {Kind: llm.ContentKindRefusal, Text: "REFUSAL"}, {Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,SAFE"}},
		{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "FILE_MARKER", Filename: "readme.txt"}},
	}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || message == nil || len(message.Content.MultipleContent) != 4 || drops.DocumentProjected != 1 {
		t.Fatalf("message projection=%#v drops=%#v err=%v", message, drops, err)
	}
	assistant, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "ASSISTANT_HISTORY"}}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || assistant == nil || assistant.Role != string(llm.RoleUser) {
		t.Fatalf("assistant history projection=%#v err=%v", assistant, err)
	}
	for _, item := range []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage}}},
		{Kind: llm.ItemKindToolCall},
		{Kind: llm.ItemKindToolResult},
		{Kind: llm.ItemKindMCPListTools},
		{Kind: llm.ItemKindMCPCall},
		{Kind: llm.ItemKindAgentMessage},
		{Kind: llm.ItemKindHostedCall},
		{Kind: llm.ItemKindUnknown},
	} {
		if _, err := inlineCompactionSummaryMessage(item, llm.APIFormatOpenAIResponse, &drops); err == nil {
			t.Fatalf("unsafe summary history was projected: %#v", item)
		}
	}
	if message, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser}, llm.APIFormatOpenAIResponse, &drops); err == nil || message != nil {
		t.Fatalf("empty typed message must fail before provider context: message=%#v err=%v", message, err)
	}
	if message, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindReasoning}, llm.APIFormatOpenAIResponse, &drops); err != nil || message != nil || drops.Reasoning == 0 {
		t.Fatalf("reasoning was not explicitly dropped: message=%#v drops=%#v err=%v", message, drops, err)
	}
	if message, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindToolDeclaration}, llm.APIFormatOpenAIResponse, &drops); err != nil || message != nil {
		t.Fatalf("tool declaration was not stripped: message=%#v err=%v", message, err)
	}
	toolCall, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction, LogicalName: "read_file", Namespace: "workspace", Status: llm.ToolCallStatusCompleted, ProviderData: []byte(`{"PRIVATE":true}`)}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || toolCall == nil || !inlineSummaryMessageContains(*toolCall, "name: read_file") || !inlineSummaryMessageContains(*toolCall, "namespace: workspace") || !inlineSummaryMessageContains(*toolCall, "status: completed") {
		t.Fatalf("tool call marker=%#v err=%v", toolCall, err)
	}
	toolResult, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{ProviderData: []byte(`{"PRIVATE":true}`), StructuredContent: []byte(`{"PRIVATE":true}`), Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "RESULT_MARKER"}}}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || toolResult == nil || !inlineSummaryMessageContains(*toolResult, "RESULT_MARKER") || drops.Private == 0 || drops.Structured == 0 {
		t.Fatalf("tool result marker=%#v drops=%#v err=%v", toolResult, drops, err)
	}
	toolResultWithProvenance, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "PROVENANCE_MARKER", ResidualOwnerType: "input_text"}}}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || toolResultWithProvenance == nil || !inlineSummaryMessageContains(*toolResultWithProvenance, "PROVENANCE_MARKER") || drops.Private == 0 || drops.SourceSidecars != 1 {
		t.Fatalf("tool-result provenance was not safely stripped: message=%#v drops=%#v err=%v", toolResultWithProvenance, drops, err)
	}
	for _, item := range []llm.Item{
		{Kind: llm.ItemKindMCPListTools, MCPListTools: &llm.MCPListTools{}},
		{Kind: llm.ItemKindMCPApprovalRequest},
		{Kind: llm.ItemKindMCPApprovalResponse},
		{Kind: llm.ItemKindMCPCall, MCPCall: &llm.MCPCall{ServerLabel: "audit", LogicalName: "lookup", Status: llm.MCPCallStatusCompleted, Output: "MCP_MARKER"}},
		{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "AGENT_MARKER"}}}},
	} {
		if message, err := inlineCompactionSummaryMessage(item, llm.APIFormatOpenAIResponse, &drops); err != nil || message == nil {
			t.Fatalf("typed history marker=%#v item=%#v err=%v", message, item, err)
		}
	}
	failedMCP, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindMCPCall, MCPCall: &llm.MCPCall{ServerLabel: "audit-failed", LogicalName: "lookup_fallback", Status: llm.MCPCallStatusFailed, Error: "MCP_ERROR_MARKER"}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || failedMCP == nil || !inlineSummaryMessageContains(*failedMCP, "server: audit-failed") || !inlineSummaryMessageContains(*failedMCP, "name: lookup_fallback") || !inlineSummaryMessageContains(*failedMCP, "status: failed") || !inlineSummaryMessageContains(*failedMCP, "error: MCP_ERROR_MARKER") {
		t.Fatalf("failed MCP history=%#v err=%v", failedMCP, err)
	}
	hosted, err := inlineCompactionSummaryMessage(llm.Item{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}}}, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || hosted == nil || !inlineSummaryMessageContains(*hosted, "hosted web operation") {
		t.Fatalf("hosted history=%#v err=%v", hosted, err)
	}
}

func TestInlineCompactionKnownMessageProjectionStripsResidualSidecarsAndRejectsSecondArms(t *testing.T) {
	item := llm.Item{
		Kind: llm.ItemKindMessage, Role: llm.RoleUser,
		ProtocolHints: llm.ProtocolHints{ResidualOwnerType: "message", SourceResidual: json.RawMessage(`{"future_item":"ITEM_SIDECAR_SECRET"}`)},
		Content: []llm.ContentBlock{{
			Kind: llm.ContentKindText, ID: "text_1", Text: "VISIBLE_HISTORY", ResidualOwnerType: "input_text",
			SourceResidual: json.RawMessage(`{"future_content":"CONTENT_SIDECAR_SECRET"}`), UnknownRaw: json.RawMessage(`{"unknown":"UNKNOWN_SIDECAR_SECRET"}`),
		}},
	}
	drops := inlineCompactionDrops{}
	message, err := inlineCompactionSummaryMessage(item, llm.APIFormatOpenAIResponse, &drops)
	if err != nil || message == nil || !inlineSummaryMessageContains(*message, "VISIBLE_HISTORY") || inlineSummaryMessageContains(*message, "SIDECAR_SECRET") || drops.SourceSidecars != 2 {
		t.Fatalf("known message projection=%#v drops=%#v err=%v", message, drops, err)
	}
	retained, err := retainedInlineCompactionItem(item)
	if err != nil || !inlineCompactionEmptyProtocolHints(retained.ProtocolHints) || retained.ID != "" || len(retained.Content) != 1 || retained.Content[0].ID != "" || len(retained.Content[0].SourceResidual) != 0 || len(retained.Content[0].UnknownRaw) != 0 || retained.Content[0].ResidualOwnerType != "" || retained.Content[0].Text != "VISIBLE_HISTORY" {
		t.Fatalf("known message retention=%#v err=%v", retained, err)
	}

	secondArm := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "visible", Image: &llm.ImageURL{URL: "data:image/png;base64,QUJD"}}}}
	decision := classifyInlineCompactionItemForTarget(secondArm, llm.APIFormatOpenAIResponse)
	if decision.kind != inlineCompactionProjectionBlock || decision.semantic != "inline_history_message_second_content_arm" {
		t.Fatalf("second content arm decision=%#v", decision)
	}
}

func TestInlineCompactionSummaryCanonicalItemsAdmitsAssistantHistoryOnlyAsData(t *testing.T) {
	text := "assistant result"
	items, err := inlineCompactionSummaryCanonicalItems([]llm.Message{{Role: string(llm.RoleAssistant), Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text", Text: &text}}}}})
	if err != nil || len(items) != 1 || items[0].Role != llm.RoleAssistant || items[0].Content[0].Text != text {
		t.Fatalf("assistant canonical summary items=%#v err=%v", items, err)
	}
}

func TestInlineCompactionContinuationAcceptsOnlyOneSafePlaintextOutput(t *testing.T) {
	text := "choice continuation"
	tests := []struct {
		name string
		resp *llm.Response
		ok   bool
	}{
		{name: "nil"},
		{name: "single assistant output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "output continuation"}}}}}, ok: true},
		{name: "reasoning then assistant output", resp: &llm.Response{Output: []llm.Item{
			{Kind: llm.ItemKindReasoning, Reasoning: &llm.ReasoningItem{Content: "PRIVATE_REASONING", Signature: "PRIVATE_SIGNATURE"}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "output continuation"}}},
		}}, ok: true},
		{name: "legacy stopped choice remains valid", resp: &llm.Response{Choices: []llm.Choice{{FinishReason: stringPointer("stop"), Message: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: stringPointer("legacy summary")}}}}}, ok: true},
		{name: "incomplete status", resp: &llm.Response{Status: llm.ResponseStatusIncomplete, Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "partial"}}}}}},
		{name: "failed status", resp: &llm.Response{Status: llm.ResponseStatusFailed, Error: &llm.ResponseError{Detail: llm.ErrorDetail{Code: "failed", Message: "provider failed"}}, Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "partial"}}}}}},
		{name: "cancelled status", resp: &llm.Response{Status: llm.ResponseStatusCancelled, Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "partial"}}}}}},
		{name: "failed lifecycle event", resp: &llm.Response{Events: []llm.Event{{Kind: llm.EventKindResponseFailed, Error: &llm.ResponseError{Detail: llm.ErrorDetail{Code: "failed", Message: "provider failed"}}}}, Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "partial"}}}}}},
		{name: "Chat length finish", resp: &llm.Response{Choices: []llm.Choice{{FinishReason: stringPointer("length"), Message: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: stringPointer("partial")}}}}}},
		{name: "Anthropic max tokens finish", resp: &llm.Response{Choices: []llm.Choice{{FinishReason: stringPointer("max_tokens"), Message: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: stringPointer("partial")}}}}}},
		{name: "output sidecar is stripped", resp: &llm.Response{Output: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, ProtocolHints: llm.ProtocolHints{SourceResidual: []byte(`{"private":true}`)},
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "safe"}},
		}}}, ok: true},
		{name: "multiple assistant outputs", resp: &llm.Response{Output: []llm.Item{
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "first"}}},
			{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "second"}}},
		}}},
		{name: "nontext assistant output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,QUJD"}}}}}}},
		{name: "tool output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindToolCall, ToolCall: &llm.ToolInvocation{Kind: llm.ToolKindFunction}}}}},
		{name: "hosted output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindHostedCall, HostedCall: &llm.HostedToolCall{Invocation: llm.ToolInvocation{Kind: llm.ToolKindWebSearch}}}}}},
		{name: "agent output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/child", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "route"}}}}}}},
		{name: "compaction output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: "provider-private"}}}}},
		{name: "context compaction output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindContextCompaction, ContextCompaction: &llm.ContextCompactionItem{EncryptedContent: stringPointer("provider-private")}}}}},
		{name: "unknown output", resp: &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindUnknown, Unknown: &llm.UnknownItem{Type: "future", Raw: json.RawMessage(`{"private":true}`), Behavioral: true}}}}},
		{name: "legacy choice", resp: &llm.Response{Choices: []llm.Choice{{Message: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: &text}}}}}, ok: true},
		{name: "legacy choice with tool", resp: &llm.Response{Choices: []llm.Choice{{Message: &llm.Message{Role: "assistant", Content: llm.MessageContent{Content: &text}, ToolCalls: []llm.ToolCall{{}}}}}}},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			item, err := inlineCompactionContinuation(test.resp)
			if test.ok {
				if err != nil || item.Role != llm.RoleUser || len(item.Content) != 1 || !strings.HasPrefix(item.Content[0].Text, inlineCompactionSummaryPrefix) {
					t.Fatalf("continuation=%#v err=%v", item, err)
				}
				return
			}
			var inlineErr *InlineCompactionError
			if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionUnsafeOutput {
				t.Fatalf("unsafe output continuation=%#v err=%v", item, err)
			}
		})
	}
}

func TestInlineCompactionRestoreAndRuntimeEvidenceFailClosed(t *testing.T) {
	newSessionWith := func(codec CompactionStateCodec, retained []llm.Item) *Session {
		session := newSession(&Plan{}, &llm.Request{}, true)
		session.inlineCompaction = &inlineCompactionState{codec: codec, generation: 1, retained: retained}
		return session
	}
	validRetained := []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained"}}}}
	validSummary := &llm.Response{Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "summary"}}}}}
	if response, err := restoreInlineCompaction(context.Background(), validSummary, nil); err != nil || response != validSummary {
		t.Fatalf("unarmed restore changed response=%#v err=%v", response, err)
	}
	if _, err := restoreInlineCompaction(context.Background(), &llm.Response{}, newSessionWith(&recordingInlineCompactionCodec{}, validRetained)); err == nil {
		t.Fatal("unsafe summary output was sealed")
	}
	if _, err := restoreInlineCompaction(context.Background(), validSummary, newSessionWith(&recordingInlineCompactionCodec{}, []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "x", Document: &llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "sidecar"}}}}})); err == nil {
		t.Fatal("unsafe retained state was sealed")
	}
	for _, test := range []struct {
		name  string
		codec *recordingInlineCompactionCodec
		code  InlineCompactionErrorCode
	}{
		{name: "codec failure", codec: &recordingInlineCompactionCodec{err: errors.New("private store")}, code: InlineCompactionCodecUnavailable},
		{name: "empty token", codec: &recordingInlineCompactionCodec{}, code: InlineCompactionInvalidToken},
		{name: "oversized token", codec: &recordingInlineCompactionCodec{token: strings.Repeat("x", maxInlineCompactionTokenBytes+1)}, code: InlineCompactionOversize},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			session := newSessionWith(test.codec, validRetained)
			_, err := restoreInlineCompaction(context.Background(), validSummary, session)
			var inlineErr *InlineCompactionError
			if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != test.code || session.plan.Debug == nil {
				t.Fatalf("restore err=%v inline=%#v trace=%#v", err, inlineErr, session.plan.Debug)
			}
		})
	}
	var nilSession *Session
	nilSession.recordInlineCompactionEvidence(llm.ConversionDirectionRequest, "summary", "emulate", StrategyInlineCompactionGateway, ReasonGatewayCompaction, 0, "", 0, 0, 0, inlineCompactionDrops{}, false)
	recordInlineCompactionFailure(nil, "plan", &InlineCompactionError{Code: InlineCompactionUnsafeInput})
	recordInlineCompactionFailure(&Plan{}, "plan", errors.New("not an inline compaction error"))
}

func TestRetainedInlineCompactionUsesNewestFirstOriginalContentOrder(t *testing.T) {
	largeChinese := strings.Repeat("中", maxInlineCompactionVisibleTextBytes/len("中")+10)
	items := []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "OLDER_MUST_BE_DROPPED"}}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{
			{Kind: llm.ContentKindText, Text: largeChinese},
			{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: "data:image/png;base64,QUJD"}},
			{Kind: llm.ContentKindText, Text: "TRAILING_TEXT"},
		}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "NEWEST_MUST_REMAIN"}}},
	}
	retained, stats, err := retainedInlineCompactionItemsWithBudget(items)
	if err != nil {
		t.Fatal(err)
	}
	if len(retained) != 2 || stats.Truncated < 2 || stats.TruncatedBytes == 0 {
		t.Fatalf("retained=%#v stats=%#v", retained, stats)
	}
	if retained[0].Content[0].Kind != llm.ContentKindText || !utf8.ValidString(retained[0].Content[0].Text) || len(retained[0].Content) < 2 || retained[0].Content[1].Kind != llm.ContentKindImage {
		t.Fatalf("boundary message lost original order or UTF-8/image: %#v", retained[0])
	}
	if retained[1].Content[0].Text != "NEWEST_MUST_REMAIN" {
		t.Fatalf("newest retained message=%#v", retained[1])
	}
	for _, item := range retained {
		for _, content := range item.Content {
			if strings.Contains(content.Text, "OLDER_MUST_BE_DROPPED") {
				t.Fatalf("older item survived newest-first retention: %#v", retained)
			}
		}
	}
}

func TestRetainedInlineCompactionStateBytesExcludeSourceIdentifiers(t *testing.T) {
	item := llm.Item{
		Kind: llm.ItemKindMessage, ID: strings.Repeat("source-id-", 4096), Role: llm.RoleUser,
		Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "RETAINED_CONTEXT"}},
	}
	retained, err := retainedInlineCompactionItem(item)
	if err != nil || retained.ID != "" || retained.Content[0].Text != "RETAINED_CONTEXT" {
		t.Fatalf("retained closed state=%#v err=%v", retained, err)
	}
	stateBytes, err := inlineCompactionStateBytes(CompactionState{Version: inlineCompactionStateVersion, Generation: 1, Retained: []llm.Item{retained}, Continuation: pendingInlineContinuation()})
	if err != nil || stateBytes >= uint32(len(item.ID)) {
		t.Fatalf("state bytes must exclude source identifier: bytes=%d sourceID=%d err=%v", stateBytes, len(item.ID), err)
	}
}

func TestInlineCompactionSummaryHasAggregateBoundAndTrustedPriority(t *testing.T) {
	history := []llm.Item{{
		Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper,
		Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "TRUSTED_DEVELOPER_MUST_REMAIN " + strings.Repeat("d", maxInlineCompactionSummaryVisibleBytes)}},
	}}
	for index := 0; index < 16; index++ {
		history = append(history, llm.Item{Kind: llm.ItemKindToolResult, ToolResult: &llm.ToolResult{
			Kind: llm.ToolKindFunction, CallID: "call_" + string(rune('a'+index)),
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "TOOL_" + string(rune('a'+index)) + strings.Repeat("x", maxInlineCompactionProjectedText)}},
		}})
	}
	messages, drops, stats, err := inlineCompactionSummaryMessages(history, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatal(err)
	}
	bytes := 0
	trusted := false
	latest := false
	for index := range messages {
		bytes += inlineCompactionSummaryMessageBudget(messages[index])
		if messages[index].Role == "developer" && inlineSummaryMessageContains(messages[index], "TRUSTED_DEVELOPER_MUST_REMAIN") {
			trusted = true
		}
		if inlineSummaryMessageContains(messages[index], "TOOL_p") {
			latest = true
		}
	}
	if bytes > maxInlineCompactionSummaryVisibleBytes || !trusted || !latest || stats.Truncated == 0 || drops.SummaryTruncated != 0 {
		t.Fatalf("summary bytes=%d trusted=%v latest=%v stats=%#v drops=%#v", bytes, trusted, latest, stats, drops)
	}
}

func TestInlineCompactionSummaryRejectsAnAtomicNewestExternalItemAboveAggregateBudget(t *testing.T) {
	content := make([]llm.ContentBlock, 0, 9)
	for index := 0; index < 9; index++ {
		content = append(content, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", maxInlineCompactionProjectedText)}})
	}
	_, _, _, err := inlineCompactionSummaryMessages([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: content}}, llm.APIFormatOpenAIResponse)
	var inlineErr *InlineCompactionError
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionOversize {
		t.Fatalf("oversized newest external item err=%v inline=%#v", err, inlineErr)
	}
}

func TestInlineCompactionEvidenceUsesActualDirectionAndPayloadFreeFailure(t *testing.T) {
	plan := &Plan{}
	session := newSession(plan, &llm.Request{}, true)
	drops := inlineCompactionDrops{SourceSidecars: 2}
	session.recordInlineCompactionEvidence(llm.ConversionDirectionRequest, "summary", "emulate", StrategyInlineCompactionGateway, ReasonGatewayCompaction, 12, "digest", 1, 24, 2, drops, true)
	session.recordInlineCompactionEvidence(llm.ConversionDirectionResponse, "restore", "restore", StrategyInlineCompactionGateway, ReasonGatewayCheckpoint, 0, "digest", 1, 24, 2, inlineCompactionDrops{}, true)
	recordInlineCompactionFailure(plan, "seal", &InlineCompactionError{Code: InlineCompactionCodecUnavailable, Err: errors.New("PRIVATE_DATABASE_DSN")})
	trace := session.DebugTrace()
	if trace == nil || len(trace.Actions) != 3 {
		t.Fatalf("trace=%#v", trace)
	}
	request := trace.Actions[0]
	response := trace.Actions[1]
	failure := trace.Actions[2]
	if request.Direction != llm.ConversionDirectionRequest || request.Stage != llm.ConversionStageRequestTransform || request.CompactionStage != "summary" || request.CompactionSourceSidecars != 2 || request.CompactionDroppedPrivate != 0 {
		t.Fatalf("summary evidence=%#v", request)
	}
	if response.Direction != llm.ConversionDirectionResponse || response.Stage != llm.ConversionStageResponseRestore || response.CompactionStage != "restore" {
		t.Fatalf("restore evidence=%#v", response)
	}
	if failure.Direction != llm.ConversionDirectionResponse || failure.Stage != llm.ConversionStageResponseRestore || failure.CompactionStage != "seal" || failure.CompactionErrorCode != string(InlineCompactionCodecUnavailable) || failure.Severity != llm.ConversionSeverityCritical {
		t.Fatalf("failure evidence=%#v", failure)
	}
	encoded, err := json.Marshal(trace)
	if err != nil || strings.Contains(string(encoded), "PRIVATE_DATABASE_DSN") || strings.Contains(string(encoded), "digest") {
		t.Fatalf("evidence leaked protected data: err=%v trace=%s", err, encoded)
	}
}

func TestInlineCompactionDocumentProjectionTargetMatrixAndBudget(t *testing.T) {
	t.Parallel()
	targets := []llm.APIFormat{llm.APIFormatOpenAIResponse, llm.APIFormatOpenAIChatCompletion, llm.APIFormatAnthropicMessage}
	for _, target := range targets {
		target := target
		t.Run(string(target), func(t *testing.T) {
			base64, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: "UERGX01BUk", Filename: "report.pdf", MIMEType: "application/pdf"}, target)
			if err != nil || base64.part.Type != "document" || base64.part.Document == nil || base64.part.Document.Data != "UERGX01BUk" {
				t.Fatalf("base64 projection=%#v err=%v", base64, err)
			}
			file, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceFile, FileID: "file_123", Filename: "report.pdf", MIMEType: "application/pdf"}, target)
			if err != nil || file.part.Type != "document" || file.part.Document == nil || file.part.Document.FileID != "file_123" {
				t.Fatalf("file projection=%#v err=%v", file, err)
			}
			text, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "DOCUMENT_TEXT_MARKER", Filename: "report.txt"}, target)
			if err != nil || text.part.Type != "text" || text.part.Text == nil || !strings.Contains(*text.part.Text, "DOCUMENT_TEXT_MARKER") {
				t.Fatalf("text projection=%#v err=%v", text, err)
			}
			url, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceURL, URL: "https://example.test/report.pdf", Filename: "report.pdf"}, target)
			if target == llm.APIFormatOpenAIChatCompletion {
				if err == nil {
					t.Fatalf("Chat URL document must block, projection=%#v", url)
				}
			} else if err != nil || url.part.Type != "document" || url.part.Document == nil || url.part.Document.URL == "" {
				t.Fatalf("URL projection=%#v err=%v", url, err)
			}
			if _, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceContent, Content: []byte(`{"private":"opaque"}`)}, target); err == nil {
				t.Fatal("structured document content must not be generically forwarded")
			}
		})
	}
	oversize := &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: strings.Repeat("x", maxInlineCompactionSummaryVisibleBytes), Filename: strings.Repeat("n", 128), MIMEType: "application/pdf"}
	if _, err := inlineCompactionDocumentProjection(oversize, llm.APIFormatOpenAIResponse); err == nil {
		t.Fatal("document source plus filename/MIME framing must respect aggregate budget")
	}
	if _, err := inlineCompactionDocumentProjection(nil, llm.APIFormatOpenAIResponse); err == nil {
		t.Fatal("nil document must not acquire a generic projection")
	}
	legacy, err := inlineCompactionDocumentProjection(&llm.DocumentURL{URL: "https://example.test/legacy.pdf", Filename: "legacy.pdf"}, llm.APIFormatOpenAIResponse)
	if err != nil || legacy.part.Document == nil || legacy.part.Document.SourceType != llm.DocumentSourceURL || legacy.part.Document.URL == "" {
		t.Fatalf("legacy URL projection=%#v err=%v", legacy, err)
	}
	cache := &llm.CacheControl{Type: "ephemeral"}
	flag := true
	sidecars, err := inlineCompactionDocumentProjection(&llm.DocumentURL{
		SourceType: llm.DocumentSourceBase64, Data: "SAFE", Filename: "sidecar.pdf", MIMEType: "application/pdf",
		CacheControl: cache, Context: "PRIVATE_CONTEXT", CitationsEnabled: &flag, Title: "PRIVATE_TITLE",
	}, llm.APIFormatOpenAIResponse)
	if err != nil || sidecars.sidecars != 1 || sidecars.part.Document == nil || sidecars.part.Document.CacheControl != nil || sidecars.part.Document.Context != "" || sidecars.part.Document.Title != "" {
		t.Fatalf("document sidecars were not stripped: projection=%#v err=%v", sidecars, err)
	}
	for _, document := range []*llm.DocumentURL{
		{SourceType: llm.DocumentSourceURL, URL: "https://example.test/" + strings.Repeat("u", maxInlineCompactionSummaryVisibleBytes)},
		{SourceType: llm.DocumentSourceFile, FileID: strings.Repeat("f", maxInlineCompactionSummaryVisibleBytes)},
	} {
		if _, err := inlineCompactionDocumentProjection(document, llm.APIFormatOpenAIResponse); err == nil {
			t.Fatalf("oversize atomic document was accepted: %#v", document)
		}
	}
	truncated, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: strings.Repeat("文", maxInlineCompactionProjectedText), Filename: "long.txt"}, llm.APIFormatOpenAIResponse)
	if err != nil || truncated.truncated != 1 || truncated.truncatedBytes == 0 || truncated.part.Text == nil || !utf8.ValidString(*truncated.part.Text) {
		t.Fatalf("long text document did not use bounded plaintext projection: %#v err=%v", truncated, err)
	}
	if _, err := inlineCompactionDocumentProjection(&llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "x", Filename: strings.Repeat("n", maxInlineCompactionProjectedText)}, llm.APIFormatOpenAIResponse); err == nil {
		t.Fatal("a text document whose framing exhausts its safe plaintext budget must fail closed")
	}
}

func TestInlineCompactionClosedHelperFailurePaths(t *testing.T) {
	t.Parallel()

	var nilInlineErr *InlineCompactionError
	if nilInlineErr.Error() != "inline compaction error" || nilInlineErr.Unwrap() != nil {
		t.Fatalf("nil inline error behavior changed: error=%q unwrap=%v", nilInlineErr.Error(), nilInlineErr.Unwrap())
	}
	if err := (&InlineCompactionError{}); err.Error() != "inline compaction error" || err.Unwrap() != nil {
		t.Fatalf("empty inline error behavior changed: %#v", err)
	}
	wrapped := errors.New("inner")
	if got := (&InlineCompactionError{Code: InlineCompactionUnsafeInput, Err: wrapped}); got.Error() != "inline compaction checkpoint_input_unsafe" || !errors.Is(got, wrapped) {
		t.Fatalf("wrapped inline error behavior changed: %v", got)
	}

	if _, _, found := inlineCompactionTrigger(nil); found {
		t.Fatal("nil request has a trigger")
	}
	if index, trigger, found := inlineCompactionTrigger(&llm.Request{Input: []llm.Item{{Kind: llm.ItemKindCompactionTrigger}, {Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "after"}}}}}); !found || index != 0 || trigger == nil {
		t.Fatalf("trigger lookup=%d %#v found=%v", index, trigger, found)
	}

	if item, kept := truncateRetainedInlineItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "text"}}}, -1); kept != 0 || item.Kind != "" {
		t.Fatalf("negative retained budget=%#v kept=%d", item, kept)
	}
	if item, kept := truncateRetainedInlineItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: ""}}}, 8); kept != 0 || item.Kind != "" {
		t.Fatalf("empty retained text=%#v kept=%d", item, kept)
	}
	if _, err := retainedInlineCompactionItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindAudio, Audio: &llm.InputAudio{Format: "wav", Data: "YXVkaW8="}}}}); err == nil {
		t.Fatal("unsupported retained audio was accepted")
	}
	if item, err := retainedInlineCompactionItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "document only"}}}}); !errors.Is(err, errInlineCompactionNotRetained) || item.Kind != "" {
		t.Fatalf("document-only retained item=%#v err=%v", item, err)
	}
	if item, err := retainedInlineCompactionItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "safe", SourceResidual: json.RawMessage(`{"private":true}`)}}}); err != nil || len(item.Content) != 1 || item.Content[0].Text != "safe" || len(item.Content[0].SourceResidual) != 0 {
		t.Fatalf("retained source residual was not safely stripped: item=%#v err=%v", item, err)
	}
	large := make([]llm.ContentBlock, 0, 70)
	for index := 0; index < 70; index++ {
		large = append(large, llm.ContentBlock{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", maxInlineCompactionProjectedText)}})
	}
	if _, err := retainedInlineCompactionItem(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: large}); err == nil {
		t.Fatal("oversized retained item was accepted")
	}

	for _, message := range []llm.Message{
		{Role: "future", Content: llm.MessageContent{Content: stringPointer("x")}},
		{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text"}}}},
		{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "image_url"}}}},
		{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "document"}}}},
		{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "audio"}}}},
		{Role: "user"},
	} {
		if _, err := inlineCompactionSummaryCanonicalItems([]llm.Message{message}); err == nil {
			t.Fatalf("unsafe summary message accepted: %#v", message)
		}
	}
	if inlineCompactionSummaryMessageBudget(llm.Message{}) != 1 {
		t.Fatal("empty summary message lost minimum budget")
	}

	text := "abcdef"
	emptyText := ""
	image := llm.ImageURL{URL: "image"}
	document := llm.DocumentURL{SourceType: llm.DocumentSourceFile, FileID: "file"}
	for _, test := range []struct {
		name    string
		message llm.Message
		budget  int
		ok      bool
	}{
		{name: "single content truncated", message: llm.Message{Role: "user", Content: llm.MessageContent{Content: &text}}, budget: 3, ok: true},
		{name: "single content no room", message: llm.Message{Role: "user", Content: llm.MessageContent{Content: &text}}, budget: 0},
		{name: "text nil", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text"}}}}, budget: 4},
		{name: "text empty", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text", Text: &emptyText}}}}, budget: 4},
		{name: "image whole", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "image_url", ImageURL: &image}}}}, budget: 5, ok: true},
		{name: "image too large", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "image_url", ImageURL: &image}}}}, budget: 1},
		{name: "image nil", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "image_url"}}}}, budget: 5},
		{name: "document whole", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "document", Document: &document}}}}, budget: 68, ok: true},
		{name: "document too large", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "document", Document: &document}}}}, budget: 1},
		{name: "document nil", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "document"}}}}, budget: 5},
		{name: "unknown part", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "audio"}}}}, budget: 5},
		{name: "remaining exhausted", message: llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text", Text: stringPointer("x")}, {Type: "text", Text: stringPointer("y")}}}}, budget: 1, ok: true},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			got, _, ok := truncateInlineCompactionSummaryMessage(test.message, test.budget)
			if ok != test.ok || test.ok && got == nil {
				t.Fatalf("truncated=%#v ok=%v want=%v", got, ok, test.ok)
			}
		})
	}
	if message, _, ok := truncateInlineCompactionSummaryMessage(llm.Message{Role: "user", Content: llm.MessageContent{Content: &emptyText}}, 4); ok || message != nil {
		t.Fatalf("empty legacy summary content=%#v ok=%v", message, ok)
	}
	if message, _, ok := truncateInlineCompactionSummaryMessage(llm.Message{Role: "user", Content: llm.MessageContent{MultipleContent: []llm.MessageContentPart{{Type: "text", Text: &emptyText}}}}, 4); ok || message != nil {
		t.Fatalf("empty multipart summary content=%#v ok=%v", message, ok)
	}
	if inlineCompactionDocumentProjectionBudget(nil) != 0 {
		t.Fatal("nil document has nonzero projection budget")
	}

	if _, err := inlineCompactionToolResultMessage(nil, llm.APIFormatOpenAIResponse, nil); err == nil {
		t.Fatal("nil tool result was projected")
	}
	if _, err := inlineCompactionToolResultMessage(&llm.ToolResult{ProviderData: json.RawMessage(`{"private":true}`)}, llm.APIFormatOpenAIResponse, nil); err == nil {
		t.Fatal("private-only tool result was projected")
	}
	if _, err := inlineCompactionToolResultMessage(&llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "x", SourceResidual: json.RawMessage(`{"future":true}`)}}}, llm.APIFormatOpenAIResponse, nil); err == nil {
		t.Fatal("tool result residual was projected")
	}
	if _, err := inlineCompactionToolResultMessage(&llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", maxInlineCompactionRetainedItemBytes+1)}}}}, llm.APIFormatOpenAIResponse, nil); err == nil {
		t.Fatal("oversized tool result image was projected")
	}
	if _, err := inlineCompactionToolResultMessage(&llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceURL, URL: "https://example.test/file.pdf"}}}}, llm.APIFormatOpenAIChatCompletion, nil); err == nil {
		t.Fatal("unsupported tool-result document target was projected")
	}
	if _, err := inlineCompactionToolResultMessage(&llm.ToolResult{Content: []llm.ContentBlock{{Kind: llm.ContentKindUnknown}}}, llm.APIFormatOpenAIResponse, nil); err == nil {
		t.Fatal("unknown tool-result content was projected")
	}
	if !strings.Contains(inlineCompactionToolCallHistory(nil), "tool call") || !strings.Contains(inlineCompactionToolResultHistory(nil), "tool outcome") || !strings.Contains(inlineCompactionMCPCallHistory(nil), "MCP outcome") {
		t.Fatal("nil history helpers lost closed fallback markers")
	}

	validContinuation := validInlineCompactionStateForTest().Continuation
	if err := validateInlineContinuation(validContinuation); err != nil {
		t.Fatalf("valid continuation rejected: %v", err)
	}
	for _, item := range []llm.Item{
		{},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: inlineCompactionSummaryPrefix + strings.Repeat("x", maxInlineCompactionVisibleTextBytes+1)}}, ProtocolHints: validContinuation.ProtocolHints},
	} {
		if err := validateInlineContinuation(item); err == nil {
			t.Fatalf("unsafe continuation accepted: %#v", item)
		}
	}
}

func TestInlineCompactionSummaryAdmissionBranchesStayFailClosed(t *testing.T) {
	if _, err := inlineCompactionStateBytes(CompactionState{Retained: []llm.Item{{Kind: llm.ItemKindMessage, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, UnknownRaw: json.RawMessage(`{`)}}}}}); err == nil {
		t.Fatal("malformed raw state must not be silently serialized")
	}
	if _, _, _, err := inlineCompactionSummaryMessages([]llm.Item{{Kind: llm.ItemKindUnknown}}, llm.APIFormatOpenAIResponse); err == nil {
		t.Fatal("unknown history item must fail summary admission")
	}
	if messages, drops, _, err := inlineCompactionSummaryMessages([]llm.Item{{Kind: llm.ItemKindReasoning, Reasoning: &llm.ReasoningItem{}}}, llm.APIFormatOpenAIResponse); err == nil || len(messages) != 0 || drops.Reasoning != 1 {
		t.Fatalf("reasoning-only history must be safely dropped then rejected: messages=%#v drops=%#v err=%v", messages, drops, err)
	}
	if messages, _, _, err := inlineCompactionSummaryMessages([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser}}, llm.APIFormatOpenAIResponse); err == nil || len(messages) != 0 {
		t.Fatalf("empty model-visible message must not become an empty summary request: messages=%#v err=%v", messages, err)
	}

	latestPDF := strings.Repeat("p", 300<<10)
	history := []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: strings.Repeat("i", 300<<10)}}}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: strings.Repeat("d", 300<<10)}}},
		{Kind: llm.ItemKindToolDeclaration, ToolDeclaration: &llm.ToolDeclarationItem{}},
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: latestPDF, Filename: "latest.pdf", MIMEType: "application/pdf"}}}},
	}
	messages, _, stats, err := inlineCompactionSummaryMessages(history, llm.APIFormatOpenAIResponse)
	if err != nil || len(messages) == 0 || stats.Truncated == 0 {
		t.Fatalf("summary admission lost its bounded fallback: messages=%#v stats=%#v err=%v", messages, stats, err)
	}
	bytes := 0
	seenLatestPDF := false
	for _, message := range messages {
		bytes += inlineCompactionSummaryMessageBudget(message)
		for _, part := range message.Content.MultipleContent {
			seenLatestPDF = seenLatestPDF || part.Document != nil && part.Document.Data == latestPDF
		}
	}
	if bytes > maxInlineCompactionSummaryVisibleBytes || !seenLatestPDF {
		t.Fatalf("summary budget=%d latest_pdf=%v messages=%#v", bytes, seenLatestPDF, messages)
	}
}

func TestProjectInlineCompactionRejectsPendingDurableStateBeforeSummary(t *testing.T) {
	image := "data:image/png;base64," + strings.Repeat("A", 500<<10-len("data:image/png;base64,"))
	request := &llm.Request{APIFormat: llm.APIFormatOpenAIResponse, Input: make([]llm.Item, 0, 10)}
	for index := 0; index < 9; index++ {
		request.Input = append(request.Input, llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindImage, Image: &llm.ImageURL{URL: image}}}})
	}
	request.Input = append(request.Input, llm.Item{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}})
	plan, err := NewPlanner().Plan(request, llm.APIFormatOpenAIResponse)
	if err != nil || plan == nil || !plan.Complete() {
		t.Fatalf("plan=%#v err=%v", plan, err)
	}
	if execution, state, err := projectInlineCompactionExecution(request, plan, &recordingInlineCompactionCodec{}); execution != nil || state != nil {
		t.Fatalf("oversized pending state must not build summary execution: execution=%#v state=%#v err=%v", execution, state, err)
	} else {
		var inlineErr *InlineCompactionError
		if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionOversize {
			t.Fatalf("err=%v inline=%#v", err, inlineErr)
		}
	}
}

func TestInlineCompactionFairSummaryReservesActualNewestExternalItem(t *testing.T) {
	developer := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "DEVELOPER_MARKER " + strings.Repeat("d", 400<<10)}}}
	pdf := strings.Repeat("p", 200<<10)
	external := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: pdf, Filename: "latest.pdf", MIMEType: "application/pdf"}}}}
	messages, drops, stats, err := inlineCompactionSummaryMessages([]llm.Item{developer, external}, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatal(err)
	}
	bytes := 0
	seenDeveloper := false
	seenPDF := false
	for index := range messages {
		bytes += inlineCompactionSummaryMessageBudget(messages[index])
		seenDeveloper = seenDeveloper || inlineSummaryMessageContains(messages[index], "DEVELOPER_MARKER")
		for partIndex := range messages[index].Content.MultipleContent {
			part := messages[index].Content.MultipleContent[partIndex]
			seenPDF = seenPDF || part.Document != nil && part.Document.Data == pdf
		}
	}
	if bytes > maxInlineCompactionSummaryVisibleBytes || !seenDeveloper || !seenPDF || stats.Truncated == 0 || drops.DocumentProjected != 1 {
		t.Fatalf("summary bytes=%d developer=%v pdf=%v drops=%#v stats=%#v messages=%#v", bytes, seenDeveloper, seenPDF, drops, stats, messages)
	}
}

func TestInlineCompactionSummaryDoesNotClaimPartiallyOmittedDocumentProjection(t *testing.T) {
	developer := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleDeveloper, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: strings.Repeat("d", 400<<10)}}}
	olderDocument := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceBase64, Data: strings.Repeat("p", 200<<10), Filename: "older.pdf", MIMEType: "application/pdf"}}}}
	latestText := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: strings.Repeat("n", 100<<10)}}}
	messages, drops, _, err := inlineCompactionSummaryMessages([]llm.Item{developer, olderDocument, latestText}, llm.APIFormatOpenAIResponse)
	if err != nil {
		t.Fatal(err)
	}
	if drops.DocumentProjected != 0 {
		t.Fatalf("omitted document was incorrectly recorded as projected: %#v", drops)
	}
	for _, message := range messages {
		for _, part := range message.Content.MultipleContent {
			if part.Document != nil {
				t.Fatalf("partially admitted document reached summary provider: %#v", part.Document)
			}
		}
	}
}

func inlineSummaryMessageContains(message llm.Message, needle string) bool {
	if message.Content.Content != nil && strings.Contains(*message.Content.Content, needle) {
		return true
	}
	for index := range message.Content.MultipleContent {
		if message.Content.MultipleContent[index].Text != nil && strings.Contains(*message.Content.MultipleContent[index].Text, needle) {
			return true
		}
	}
	return false
}

func TestInlineCompactionExecutionCheckedBoundariesPropagatePureFailures(t *testing.T) {
	t.Parallel()
	canonicalFailure := errors.New("canonicalizer failure")
	_, err := checkedInlineCompactionSummaryItems(nil, func([]llm.Message) ([]llm.Item, error) {
		return nil, canonicalFailure
	})
	var inlineErr *InlineCompactionError
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, canonicalFailure) {
		t.Fatalf("canonical checked boundary err=%v inline=%#v", err, inlineErr)
	}

	byteFailure := errors.New("state byte counter failure")
	_, err = checkedInlineCompactionStateBytes(CompactionState{}, func(CompactionState) (uint32, error) {
		return 0, byteFailure
	})
	if !errors.Is(err, byteFailure) {
		t.Fatalf("state-byte checked boundary err=%v", err)
	}

	plan := &Plan{Target: CapabilityProfile{APIFormat: llm.APIFormatOpenAIResponse}, Actions: []Action{{Ref: ObjectRef{Kind: ObjectRequestControl}, Kind: ActionEmulate, Strategy: StrategyInlineCompactionGateway}}}
	request := &llm.Request{Input: []llm.Item{
		{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained"}}},
		{Kind: llm.ItemKindCompactionTrigger, CompactionTrigger: &llm.CompactionTriggerItem{}},
	}}
	_, _, err = projectInlineCompactionExecutionWithDependencies(request, plan, &recordingInlineCompactionCodec{}, func([]llm.Message) ([]llm.Item, error) {
		return nil, canonicalFailure
	}, inlineCompactionStateBytes)
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, canonicalFailure) {
		t.Fatalf("execution canonical boundary err=%v inline=%#v", err, inlineErr)
	}

	session := &Session{inlineCompaction: &inlineCompactionState{codec: &recordingInlineCompactionCodec{}, generation: 1, retained: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained"}}}}}}
	_, err = restoreInlineCompactionWithStateBytes(context.Background(), &llm.Response{Status: llm.ResponseStatusCompleted, Output: []llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleAssistant, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "summary"}}}}}, session, func(CompactionState) (uint32, error) {
		return 0, byteFailure
	})
	if !errors.Is(err, byteFailure) {
		t.Fatalf("restore byte boundary err=%v", err)
	}
}

func TestInlineCompactionProjectionCheckedBoundariesPropagatePureFailures(t *testing.T) {
	t.Parallel()
	documentFailure := errors.New("document projector failure")
	_, err := checkedInlineCompactionDocumentProjection(nil, llm.APIFormatOpenAIResponse, func(*llm.DocumentURL, llm.APIFormat) (inlineCompactionDocumentProjectionResult, error) {
		return inlineCompactionDocumentProjectionResult{}, documentFailure
	})
	if !errors.Is(err, documentFailure) {
		t.Fatalf("document checked boundary err=%v", err)
	}

	agentFailure := errors.New("agent envelope failure")
	_, err = checkedInlineCompactionAgentEnvelope(nil, func(*llm.AgentMessage) (string, error) {
		return "", agentFailure
	})
	var inlineErr *InlineCompactionError
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, agentFailure) {
		t.Fatalf("agent checked boundary err=%v inline=%#v", err, inlineErr)
	}

	documentItem := llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindDocument, Document: &llm.DocumentURL{SourceType: llm.DocumentSourceText, Data: "document"}}}}
	_, err = inlineCompactionSummaryMessageWithProjectors(documentItem, llm.APIFormatOpenAIResponse, nil, func(*llm.DocumentURL, llm.APIFormat) (inlineCompactionDocumentProjectionResult, error) {
		return inlineCompactionDocumentProjectionResult{}, documentFailure
	}, func(message *llm.AgentMessage) (string, error) { return message.LegacyInterAgentMessageJSON() })
	if !errors.Is(err, documentFailure) {
		t.Fatalf("summary document boundary err=%v", err)
	}

	agentItem := llm.Item{Kind: llm.ItemKindAgentMessage, AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "route"}}}}
	_, err = inlineCompactionSummaryMessageWithProjectors(agentItem, llm.APIFormatOpenAIResponse, nil, inlineCompactionDocumentProjection, func(*llm.AgentMessage) (string, error) {
		return "", agentFailure
	})
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, agentFailure) {
		t.Fatalf("summary agent boundary err=%v inline=%#v", err, inlineErr)
	}
}

func TestInlineCompactionRetentionSerializerPropagatesPureFailure(t *testing.T) {
	t.Parallel()
	failure := errors.New("retained serializer failure")
	_, err := checkedInlineCompactionRetainedEncoding(llm.Item{}, func(any) ([]byte, error) {
		return nil, failure
	})
	var inlineErr *InlineCompactionError
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, failure) {
		t.Fatalf("retained checked boundary err=%v inline=%#v", err, inlineErr)
	}
	_, err = retainedInlineCompactionItemWithMarshal(llm.Item{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "retained"}}}, func(any) ([]byte, error) {
		return nil, failure
	})
	if !errors.As(err, &inlineErr) || inlineErr == nil || inlineErr.Code != InlineCompactionInvalidState || !errors.Is(err, failure) {
		t.Fatalf("retained item boundary err=%v inline=%#v", err, inlineErr)
	}
}

func TestInlineCompactionSummaryProjectorPropagatesPureFailure(t *testing.T) {
	t.Parallel()
	failure := errors.New("summary projector failure")
	_, _, _, err := inlineCompactionSummaryMessagesWithProjector([]llm.Item{{Kind: llm.ItemKindMessage, Role: llm.RoleUser, Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "history"}}}}, llm.APIFormatOpenAIResponse, func(llm.Item, llm.APIFormat, *inlineCompactionDrops) (*llm.Message, error) {
		return nil, failure
	})
	if !errors.Is(err, failure) {
		t.Fatalf("summary projector boundary err=%v", err)
	}
}

package responses

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/conversion"
	"github.com/looplj/axonhub/llm/httpclient"
)

func TestResponsesObjectEvidenceIsSelectedAndFutureUnknownOnly(t *testing.T) {
	t.Parallel()
	messageRaw := json.RawMessage(`{"id":"msg_1","type":"message","role":"user","content":[{"type":"input_text","text":"ordinary"}]}`)
	functionRaw := json.RawMessage(`{"id":"call_1","type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}`)
	futureRaw := json.RawMessage(`{"id":"future_1","type":"future_behavior","tools":[],"content":null,"arguments":"","status":null,"future":{"mode":"opaque"}}`)
	contextRaw := json.RawMessage(`{"id":"ctx_1","type":"context_compaction","encrypted_content":"opaque"}`)
	for _, test := range []struct {
		name       string
		raw        json.RawMessage
		wantDigest bool
	}{
		{name: "ordinary message", raw: messageRaw},
		{name: "ordinary function", raw: functionRaw},
		{name: "future unknown", raw: futureRaw, wantDigest: true},
		{name: "context compaction", raw: contextRaw, wantDigest: true},
	} {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			var wire Item
			if err := json.Unmarshal(test.raw, &wire); err != nil {
				t.Fatalf("decode fixture: %v", err)
			}
			canonical, err := responseItemToCanonical(&wire, test.raw, 0)
			if err != nil || canonical == nil {
				t.Fatalf("canonical item=%#v err=%v", canonical, err)
			}
			if !test.wantDigest {
				if canonical.ProtocolHints.SourceBytes != 0 || canonical.ProtocolHints.SourceDigest != "" {
					t.Fatalf("ordinary item unexpectedly received object evidence: %#v", canonical.ProtocolHints)
				}
				return
			}
			wantDigest := fmt.Sprintf("%x", sha256.Sum256(test.raw))
			if canonical.ProtocolHints.SourceBytes != uint32(len(test.raw)) || canonical.ProtocolHints.SourceDigest != wantDigest {
				t.Fatalf("selected evidence = %#v, want bytes=%d digest=%s", canonical.ProtocolHints, len(test.raw), wantDigest)
			}
		})
	}
}

func TestResponsesSelectedEvidenceUsesExactWireBytesAndSurvivesStreamSnapshots(t *testing.T) {
	t.Parallel()
	// Spaces and field order are intentional: this assertion must not derive the
	// digest from re-marshaled semantic JSON.
	raw := json.RawMessage(`{ "future": { "z": 1 }, "tools": [], "type": "future_behavior", "content": null, "arguments": "", "status": null }`)
	var item Item
	if err := json.Unmarshal(raw, &item); err != nil {
		t.Fatalf("decode future wire: %v", err)
	}
	canonical, err := responseItemToCanonical(&item, raw, 3)
	if err != nil || canonical == nil {
		t.Fatalf("canonical future=%#v err=%v", canonical, err)
	}
	want := fmt.Sprintf("%x", sha256.Sum256(raw))
	if canonical.ProtocolHints.SourceDigest != want || canonical.ProtocolHints.SourceBytes != uint32(len(raw)) {
		t.Fatalf("exact input evidence=%#v want=%s", canonical.ProtocolHints, want)
	}

	var event StreamEvent
	streamRaw := []byte(`{"type":"response.output_item.added","output_index":0,"item":` + string(raw) + `}`)
	if err := json.Unmarshal(streamRaw, &event); err != nil {
		t.Fatalf("decode stream snapshot: %v", err)
	}
	streamItem, err := canonicalStreamItem(event.Item, event.ItemRaw, 0, true)
	if err != nil || streamItem == nil || streamItem.ProtocolHints.SourceDigest != want || streamItem.ProtocolHints.SourceBytes != uint32(len(raw)) {
		t.Fatalf("exact stream evidence=%#v err=%v", streamItem, err)
	}
}

func TestResponsesContextCompactionTypedUnionRoundTripsAndPreservesEvidence(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{"id":"ctx_1","type":"context_compaction","encrypted_content":"opaque","future_context":{"version":1}}`)
	var wire Item
	if err := json.Unmarshal(raw, &wire); err != nil {
		t.Fatalf("decode context wire: %v", err)
	}
	canonical, err := responseItemToCanonical(&wire, raw, 2)
	if err != nil || canonical == nil || canonical.Kind != llm.ItemKindContextCompaction || canonical.ContextCompaction == nil || canonical.ContextCompaction.EncryptedContent == nil {
		t.Fatalf("decode context canonical=%#v err=%v", canonical, err)
	}
	wantDigest := fmt.Sprintf("%x", sha256.Sum256(raw))
	if *canonical.ContextCompaction.EncryptedContent != "opaque" || canonical.ProtocolHints.SourceBytes != uint32(len(raw)) || canonical.ProtocolHints.SourceDigest != wantDigest ||
		!bytes.Contains(canonical.ProtocolHints.SourceResidual, []byte(`"future_context"`)) {
		t.Fatalf("context evidence/residual=%#v", canonical)
	}
	restored, ok := canonicalItemToResponses(canonical)
	if !ok {
		t.Fatal("context union has no Responses encoder")
	}
	encoded, err := json.Marshal(restored)
	if err != nil || !bytes.Contains(encoded, []byte(`"future_context":{"version":1}`)) || !bytes.Contains(encoded, []byte(`"encrypted_content":"opaque"`)) {
		t.Fatalf("context identity=%s err=%v", encoded, err)
	}

	var streamWire StreamEvent
	if err := json.Unmarshal([]byte(`{"type":"response.output_item.added","output_index":0,"item":`+string(raw)+`}`), &streamWire); err != nil {
		t.Fatalf("decode context stream snapshot: %v", err)
	}
	streamCanonical, err := canonicalStreamItem(streamWire.Item, streamWire.ItemRaw, 0, true)
	if err != nil || streamCanonical == nil || streamCanonical.Kind != llm.ItemKindContextCompaction || streamCanonical.ProtocolHints.SourceDigest != wantDigest {
		t.Fatalf("context stream canonical=%#v err=%v", streamCanonical, err)
	}
	clone := cloneCanonicalItem(streamCanonical)
	if clone == nil || clone.ContextCompaction == nil || clone.ContextCompaction.EncryptedContent == streamCanonical.ContextCompaction.EncryptedContent || clone.ProtocolHints.SourceDigest != wantDigest {
		t.Fatalf("context clone=%#v source=%#v", clone, streamCanonical)
	}
}

func TestResponsesStatuslessTypedUnionsNeverEncodeInjectedCanonicalStatus(t *testing.T) {
	t.Parallel()
	for _, item := range []*llm.Item{
		{Kind: llm.ItemKindAgentMessage, ID: "agent", Status: llm.ItemStatusCompleted, AgentMessage: &llm.AgentMessage{
			Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{Kind: llm.AgentMessageContentInputText, Text: "handoff"}},
		}},
		{Kind: llm.ItemKindContextCompaction, ID: "context", Status: llm.ItemStatusCompleted, ContextCompaction: &llm.ContextCompactionItem{}},
		{Kind: llm.ItemKindCompaction, ID: "compact", Status: llm.ItemStatusCompleted, Compaction: &llm.CompactionItem{}},
	} {
		item := item
		wire, ok := canonicalItemToResponses(item)
		if !ok || wire.Status != nil {
			t.Fatalf("statusless canonical item=%#v wire=%#v ok=%v", item, wire, ok)
		}
		encoded, err := json.Marshal(wire)
		if err != nil || bytes.Contains(encoded, []byte(`"status"`)) {
			t.Fatalf("statusless wire=%s err=%v", encoded, err)
		}
	}
}

func TestResponsesEvidenceHintHelpersStayPayloadFreeForEmptyInputs(t *testing.T) {
	t.Parallel()

	for _, set := range []func(*llm.ProtocolHints, json.RawMessage){setResponsesUnknownEvidenceHints, setResponsesRawEvidenceHints} {
		set(nil, json.RawMessage(`{"type":"future"}`))
		hints := llm.ProtocolHints{SourceBytes: 99, SourceDigest: "keep"}
		set(&hints, nil)
		if hints.SourceBytes != 99 || hints.SourceDigest != "keep" {
			t.Fatalf("empty raw unexpectedly changed evidence hints: %#v", hints)
		}
	}

	value := "opaque checkpoint"
	clone := stringPointerClone(&value)
	if clone == nil || clone == &value || *clone != value {
		t.Fatalf("string pointer clone=%#v", clone)
	}
	*clone = "mutated"
	if value != "opaque checkpoint" || stringPointerClone(nil) != nil {
		t.Fatalf("string pointer clone aliases input: value=%q", value)
	}
}

func TestCanonicalUnknownResponsesUsesFullRawIdentityOverTypedFields(t *testing.T) {
	t.Parallel()
	raw := json.RawMessage(`{ "type": "future_behavior", "id":"original", "tools": [], "content": null, "arguments": "", "status": null }`)
	item := &llm.Item{Kind: llm.ItemKindUnknown, ID: "mutated", Status: llm.ItemStatusCompleted, Unknown: &llm.UnknownItem{Type: "future_behavior", Raw: raw, Behavioral: true}, ProtocolHints: llm.ProtocolHints{SourceFormat: llm.APIFormatOpenAIResponse}}
	wire, ok := canonicalItemToResponses(item)
	if !ok {
		t.Fatal("future unknown has no same-protocol identity encoder")
	}
	if !bytes.Equal(wire.Raw, raw) {
		t.Fatalf("unknown raw identity was reconstructed: wire=%s raw=%s", wire.Raw, raw)
	}
	encoded, err := json.Marshal(wire)
	var encodedObject, rawObject any
	if err != nil || json.Unmarshal(encoded, &encodedObject) != nil || json.Unmarshal(raw, &rawObject) != nil || !reflect.DeepEqual(encodedObject, rawObject) {
		t.Fatalf("unknown full raw contract encoded=%s raw=%s err=%v", encoded, raw, err)
	}
}

func TestInboundBuildsOrderedCanonicalDirectlyFromResponsesItems(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"instructions":"be precise",
		"input":[
			{"type":"message","id":"msg_user","role":"user","content":[{"type":"input_text","id":"text_1","text":"before"},{"type":"input_image","id":"image_1","image_url":"https://example.invalid/image.png","detail":"low"}]},
			{"type":"reasoning","id":"reason_1","summary":[{"type":"summary_text","text":"think"}],"encrypted_content":"opaque-signature"},
			{"type":"function_call","id":"item_call","call_id":"call_1","name":"lookup","arguments":{"city":"Singapore"},"status":"completed"},
			{"type":"message","id":"msg_after","role":"assistant","content":[{"type":"output_text","text":"after"}]},
			{"type":"function_call_output","id":"item_result","call_id":"call_1","name":"lookup","output":[{"type":"input_text","text":"sunny"}]},
			{"type":"compaction","id":"compact_1","encrypted_content":"opaque-compaction","created_by":"server"},
			{"type":"future_behavior","id":"future_1","mode":"must_preserve","nested":{"enabled":true}}
		],
		"tools":[{"type":"function","name":"lookup","description":"weather","parameters":{"type":"object"},"strict":true}]
	}`)

	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method:  http.MethodPost,
		URL:     "/v1/responses",
		Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body:    body,
	})
	if err != nil {
		t.Fatalf("decode Responses request: %v", err)
	}

	wantKinds := []llm.ItemKind{
		llm.ItemKindMessage,
		llm.ItemKindMessage,
		llm.ItemKindReasoning,
		llm.ItemKindToolCall,
		llm.ItemKindMessage,
		llm.ItemKindToolResult,
		llm.ItemKindCompaction,
		llm.ItemKindUnknown,
	}
	if len(request.Input) != len(wantKinds) {
		t.Fatalf("canonical item count = %d, want %d: %#v", len(request.Input), len(wantKinds), request.Input)
	}
	for index, want := range wantKinds {
		if request.Input[index].Kind != want {
			t.Fatalf("canonical item %d kind = %q, want %q: %#v", index, request.Input[index].Kind, want, request.Input[index])
		}
	}

	if request.Input[0].Role != llm.RoleSystem || request.Input[0].Content[0].Text != "be precise" {
		t.Fatalf("instructions were not represented canonically: %#v", request.Input[0])
	}
	message := request.Input[1]
	if message.ID != "msg_user" || len(message.Content) != 2 || message.Content[0].ID != "text_1" ||
		message.Content[1].ID != "image_1" || message.Content[1].Image == nil || message.Content[1].Image.URL != "https://example.invalid/image.png" {
		t.Fatalf("message content identity/order degraded: %#v", message)
	}
	reasoning := request.Input[2]
	if reasoning.ID != "reason_1" || reasoning.Reasoning == nil || reasoning.Reasoning.Content != "think" || reasoning.Reasoning.Signature != "opaque-signature" {
		t.Fatalf("reasoning item degraded: %#v", reasoning)
	}
	call := request.Input[3]
	if call.ID != "item_call" || call.ToolCall == nil || call.ToolCall.ID != "item_call" || call.ToolCall.CallID != "call_1" ||
		string(call.ToolCall.ArgumentsJSON) != `{"city":"Singapore"}` {
		t.Fatalf("function item/call identity degraded: %#v", call)
	}
	result := request.Input[5]
	if result.ID != "item_result" || result.ToolResult == nil || result.ToolResult.CallID != "call_1" ||
		len(result.ToolResult.Content) != 1 || result.ToolResult.Content[0].Text != "sunny" {
		t.Fatalf("function result degraded: %#v", result)
	}
	compaction := request.Input[6]
	if compaction.ID != "compact_1" || compaction.Compaction == nil || compaction.Compaction.EncryptedContent != "opaque-compaction" || compaction.Compaction.CreatedBy != "server" {
		t.Fatalf("compaction item degraded: %#v", compaction)
	}
	unknown := request.Input[7]
	if unknown.ID != "future_1" || unknown.Unknown == nil || !unknown.Unknown.Behavioral || unknown.Unknown.Type != "future_behavior" {
		t.Fatalf("unknown behavioral item degraded: %#v", unknown)
	}
	var unknownObject map[string]any
	if err := json.Unmarshal(unknown.Unknown.Raw, &unknownObject); err != nil || unknownObject["mode"] != "must_preserve" {
		t.Fatalf("unknown raw item was not preserved: raw=%s err=%v", unknown.Unknown.Raw, err)
	}
}

func TestInboundAgentMessageUsesExactWireTypeAndTypedContentUnion(t *testing.T) {
	t.Parallel()
	plain := "visible handoff"
	encrypted := "opaque"
	tests := []struct {
		name    string
		input   *Input
		wantErr string
		known   bool
		wantLen int
	}{
		{name: "nil input", wantErr: "non-empty typed item array"},
		{name: "string input", input: &Input{Text: &plain}, wantErr: "non-empty typed item array"},
		{name: "empty items", input: &Input{}, wantErr: "non-empty typed item array"},
		{name: "input text missing text", input: &Input{Items: []Item{{Type: "input_text"}}}, wantErr: "requires text"},
		{name: "input text mixed", input: &Input{Items: []Item{{Type: "input_text", Text: &plain, EncryptedContent: &encrypted}}}, wantErr: "cannot contain encrypted_content"},
		{name: "encrypted missing value", input: &Input{Items: []Item{{Type: "encrypted_content"}}}, wantErr: "requires encrypted_content"},
		{name: "encrypted empty", input: &Input{Items: []Item{{Type: "encrypted_content", EncryptedContent: new(string)}}}, wantErr: "requires encrypted_content"},
		{name: "encrypted whitespace", input: &Input{Items: []Item{{Type: "encrypted_content", EncryptedContent: stringPointer(" ")}}}, wantErr: "requires encrypted_content"},
		{name: "encrypted mixed", input: &Input{Items: []Item{{Type: "encrypted_content", Text: &plain, EncryptedContent: &encrypted}}}, wantErr: "cannot contain text"},
		{name: "future type", input: &Input{Items: []Item{{Type: "future_agent_content"}}}, known: false},
		{name: "future before malformed known", input: &Input{Items: []Item{{Type: "future_agent_content"}, {Type: "input_text", Text: &plain, EncryptedContent: &encrypted}}}, wantErr: "cannot contain encrypted_content"},
		{name: "malformed known before future", input: &Input{Items: []Item{{Type: "input_text", Text: &plain, EncryptedContent: &encrypted}, {Type: "future_agent_content"}}}, wantErr: "cannot contain encrypted_content"},
		{name: "typed union", input: &Input{Items: []Item{
			{Type: "input_text", Text: &plain, Residual: json.RawMessage(`{"future":true}`)},
			{Type: "encrypted_content", EncryptedContent: &encrypted},
		}}, known: true, wantLen: 2},
	}
	for _, test := range tests {
		test := test
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			content, known, err := responseAgentMessageContent(test.input)
			if test.wantErr != "" {
				if err == nil || !known || !strings.Contains(err.Error(), test.wantErr) || content != nil {
					t.Fatalf("agent content = %#v known=%v err=%v, want error containing %q", content, known, err, test.wantErr)
				}
				return
			}
			if !test.known {
				if err != nil || known || content != nil {
					t.Fatalf("future agent content = %#v known=%v err=%v", content, known, err)
				}
				return
			}
			if err != nil || !known || len(content) != test.wantLen || content[0].Kind != llm.AgentMessageContentInputText ||
				content[0].Text != plain || string(content[0].SourceResidual) != `{"future":true}` ||
				content[1].Kind != llm.AgentMessageContentEncryptedContent || content[1].EncryptedContent != encrypted {
				t.Fatalf("agent content = %#v err=%v", content, err)
			}
		})
	}

	const futureChildWire = `{ "recipient":"/root/worker", "content":[{"payload":{"private":true},"type":"future_agent_content"}], "id":"am_future_child", "author":"/root", "type":"agent_message" }`
	var futureChild Item
	if err := json.Unmarshal([]byte(futureChildWire), &futureChild); err != nil {
		t.Fatalf("decode future agent child wire: %v", err)
	}
	opaque, err := responseItemToCanonical(&futureChild, json.RawMessage(futureChildWire), 3)
	if err != nil || opaque == nil || opaque.Kind != llm.ItemKindUnknown || opaque.Unknown == nil || !opaque.Unknown.Behavioral ||
		opaque.Unknown.Type != "agent_message" || !bytes.Equal(opaque.Unknown.Raw, []byte(futureChildWire)) || opaque.ProtocolHints.SourceType != "agent_message" ||
		opaque.ProtocolHints.SourceBytes != uint32(len(futureChildWire)) || len(opaque.ProtocolHints.SourceDigest) != 64 {
		t.Fatalf("future agent child opaque canonical=%#v err=%v", opaque, err)
	}

	// A future outer type may contain all current agent-message-shaped fields,
	// but exact type dispatch keeps it opaque rather than guessing its semantics.
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{"model":"fixture-model","input":[{"id":"future_1","type":"future_agent_message","author":"/root","recipient":"/root/worker","content":[{"type":"input_text","text":"not typed yet"}]}]}`),
	})
	if err != nil || len(request.Input) != 1 || request.Input[0].Kind != llm.ItemKindUnknown || request.Input[0].AgentMessage != nil ||
		request.Input[0].Unknown == nil || request.Input[0].Unknown.Type != "future_agent_message" {
		t.Fatalf("future agent-like item classification = %#v err=%v", request, err)
	}
}

func TestCanonicalAgentMessageResponsesIdentityKeepsOuterAndContentResiduals(t *testing.T) {
	t.Parallel()
	text := "typed handoff"
	item := &llm.Item{
		Kind: llm.ItemKindAgentMessage, ID: "am_1", Status: llm.ItemStatusCompleted,
		AgentMessage: &llm.AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []llm.AgentMessageContentPart{{
			Kind: llm.AgentMessageContentInputText, Text: text,
			ResidualOwnerType: "input_text", SourceResidual: json.RawMessage(`{"future_content":true}`),
		}}},
		ProtocolHints: llm.ProtocolHints{
			SourceFormat: llm.APIFormatOpenAIResponse, SourceType: "agent_message", ResidualOwnerType: "agent_message",
			SourceResidual: json.RawMessage(`{"future_outer":true}`),
		},
	}
	wire, ok := canonicalItemToResponses(item)
	if !ok {
		t.Fatal("typed agent message has no Responses identity encoding")
	}
	encoded, err := json.Marshal(wire)
	if err != nil || !bytes.Contains(encoded, []byte(`"future_outer":true`)) || !bytes.Contains(encoded, []byte(`"future_content":true`)) {
		t.Fatalf("agent Responses identity encoding=%s err=%v", encoded, err)
	}
	var decoded Item
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("decode agent Responses identity wire: %v", err)
	}
	restored, err := responseItemToCanonical(&decoded, encoded, 1)
	if err != nil || restored.Kind != llm.ItemKindAgentMessage || restored.AgentMessage == nil ||
		string(restored.ProtocolHints.SourceResidual) != `{"future_outer":true}` || len(restored.AgentMessage.Content) != 1 ||
		string(restored.AgentMessage.Content[0].SourceResidual) != `{"future_content":true}` {
		t.Fatalf("agent Responses identity restoration=%#v err=%v", restored, err)
	}
}

func TestInboundTypesResponsesHostedToolLifecycleWithoutDroppingRawState(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"search and draw"}]},
			{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"axon","sources":[{"type":"url","url":"https://example.com","title":"Example"}]}},
			{"type":"image_generation_call","id":"ig_1","status":"completed","action":"generate","output_format":"webp","result":"aW1hZ2U="}
		]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode hosted Responses history: %v", err)
	}
	if len(request.Input) != 3 || request.Input[1].Kind != llm.ItemKindHostedCall || request.Input[2].Kind != llm.ItemKindHostedCall {
		t.Fatalf("hosted canonical order = %#v", request.Input)
	}
	web := request.Input[1].HostedCall
	if web == nil || web.Invocation.Kind != llm.ToolKindWebSearch || web.Invocation.CallID != "ws_1" ||
		web.Invocation.Execution != llm.ExecutionOwnerProvider || web.Result == nil || len(web.Result.Content) != 1 ||
		web.Result.Content[0].Citation == nil || web.Result.Content[0].Citation.URL != "https://example.com" ||
		!json.Valid(web.Invocation.ProviderData) {
		t.Fatalf("web search hosted lifecycle = %#v", web)
	}
	image := request.Input[2].HostedCall
	if image == nil || image.Invocation.Kind != llm.ToolKindImageGeneration || image.Invocation.CallID != "ig_1" ||
		image.Result == nil || len(image.Result.Content) != 1 || image.Result.Content[0].Image == nil ||
		image.Result.Content[0].Image.URL != "data:image/webp;base64,aW1hZ2U=" || !json.Valid(image.Result.ProviderData) {
		t.Fatalf("image hosted lifecycle = %#v", image)
	}
	if err := llm.ValidateItemStructure(request.Input); err != nil {
		t.Fatalf("validate hosted canonical items: %v", err)
	}
}

func TestResponsesCanonicalCoversNativeComputerFileCodeShellAndPatchFamilies(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"computer_call","id":"cu_1","call_id":"computer_1","status":"completed","action":{"type":"click","x":12,"y":34,"button":"left"},"pending_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}]},
			{"type":"computer_call_output","id":"cuo_1","call_id":"computer_1","status":"completed","output":{"type":"computer_screenshot","image_url":"data:image/png;base64,aW1hZ2U="},"acknowledged_safety_checks":[{"id":"safe_1","code":"external_side_effect","message":"confirm"}]},
			{"type":"apply_patch_call","id":"ap_1","call_id":"patch_1","status":"completed","operation":{"type":"update_file","path":"main.go","diff":"@@ -1 +1 @@"}},
			{"type":"apply_patch_call_output","id":"apo_1","call_id":"patch_1","status":"completed","output":"Done"},
			{"type":"file_search_call","id":"fs_1","status":"completed","queries":["axon tool conversion"],"results":[{"file_id":"file_1","filename":"design.md","score":0.91,"text":"canonical"}]},
			{"type":"code_interpreter_call","id":"ci_1","status":"completed","container_id":"ctr_1","code":"print(42)","outputs":[{"type":"logs","logs":"42\\n"}]},
			{"type":"shell_call","id":"sh_1","call_id":"shell_1","status":"completed","action":{"commands":["go test ./conversion"],"timeout_ms":30000,"max_output_length":4096}},
			{"type":"shell_call_output","id":"sho_1","call_id":"shell_1","status":"completed","max_output_length":4096,"output":[{"stdout":"ok\\n","stderr":"","outcome":{"type":"exit","exit_code":0}}]}
		],
		"tools":[
			{"type":"computer_use_preview","display_width":1024,"display_height":768,"environment":"browser","future_computer_option":true},
			{"type":"apply_patch"},
			{"type":"file_search","vector_store_ids":["vs_1"],"max_num_results":4,"ranking_options":{"ranker":"auto","score_threshold":0.2},"future_file_option":"keep"},
			{"type":"code_interpreter","container":{"type":"auto","file_ids":["file_1"]},"future_code_option":7},
			{"type":"shell"}
		]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Responses native families: %v", err)
	}
	wantDefinitionKinds := []llm.ToolKind{
		llm.ToolKindComputer, llm.ToolKindApplyPatch, llm.ToolKindFileSearch, llm.ToolKindCodeInterpreter, llm.ToolKindShell,
	}
	if len(request.ToolDefinitions) != len(wantDefinitionKinds) {
		t.Fatalf("definition count = %d, want %d: %#v", len(request.ToolDefinitions), len(wantDefinitionKinds), request.ToolDefinitions)
	}
	for index, want := range wantDefinitionKinds {
		definition := &request.ToolDefinitions[index]
		if definition.Kind != want || definition.Hosted == nil || !json.Valid(definition.Hosted.Configuration) {
			t.Fatalf("definition %d = %#v, want kind %q with raw configuration", index, definition, want)
		}
	}
	if !bytes.Contains(request.ToolDefinitions[0].Hosted.Configuration, []byte(`"future_computer_option":true`)) ||
		!bytes.Contains(request.ToolDefinitions[2].Hosted.Configuration, []byte(`"future_file_option":"keep"`)) ||
		!bytes.Contains(request.ToolDefinitions[3].Hosted.Configuration, []byte(`"future_code_option":7`)) {
		t.Fatalf("native tool configuration lost unknown fields: %#v", request.ToolDefinitions)
	}
	if len(request.Input) != 7 {
		t.Fatalf("canonical item count = %d, want 7: %#v", len(request.Input), request.Input)
	}
	computerCall, computerResult := request.Input[0].ToolCall, request.Input[1].ToolResult
	if computerCall == nil || computerCall.Kind != llm.ToolKindComputer || computerCall.Execution != llm.ExecutionOwnerClient ||
		len(computerCall.PendingSafetyChecks) != 1 || computerCall.PendingSafetyChecks[0].ID != "safe_1" ||
		computerResult == nil || computerResult.Kind != llm.ToolKindComputer || len(computerResult.AcknowledgedSafetyChecks) != 1 ||
		len(computerResult.Content) != 1 || computerResult.Content[0].Image == nil {
		t.Fatalf("computer lifecycle degraded: call=%#v result=%#v", computerCall, computerResult)
	}
	if request.Input[2].ToolCall == nil || request.Input[2].ToolCall.Kind != llm.ToolKindApplyPatch ||
		request.Input[3].ToolResult == nil || request.Input[3].ToolResult.Kind != llm.ToolKindApplyPatch {
		t.Fatalf("apply_patch lifecycle degraded: %#v", request.Input[2:4])
	}
	for _, index := range []int{4, 5, 6} {
		if request.Input[index].HostedCall == nil || request.Input[index].HostedCall.Invocation.Execution != llm.ExecutionOwnerProvider ||
			request.Input[index].HostedCall.Result == nil {
			t.Fatalf("hosted lifecycle %d degraded: %#v", index, request.Input[index])
		}
	}
	if request.Input[6].HostedCall.Invocation.Kind != llm.ToolKindShell ||
		!bytes.Contains(request.Input[6].HostedCall.Result.ProviderData, []byte(`"id":"sho_1"`)) {
		t.Fatalf("shell call/output adjacency was not merged: %#v", request.Input[6])
	}
	if err := llm.ValidateItems(request.Input); err != nil {
		t.Fatalf("validate native Responses canonical items: %v", err)
	}

	tools, ok, err := canonicalRequestTools(request)
	if err != nil || !ok || len(tools) != len(wantDefinitionKinds) {
		t.Fatalf("encode native Responses tools: ok=%v err=%v tools=%#v", ok, err, tools)
	}
	encodedTools, err := json.Marshal(tools)
	if err != nil || !bytes.Contains(encodedTools, []byte(`"future_file_option":"keep"`)) ||
		!bytes.Contains(encodedTools, []byte(`"future_code_option":7`)) {
		t.Fatalf("native Responses tool identity round-trip failed: err=%v body=%s", err, encodedTools)
	}
	input, _, ok, err := canonicalRequestInput(request)
	if err != nil || !ok || len(input.Items) != 8 {
		t.Fatalf("encode native Responses input: ok=%v err=%v input=%#v", ok, err, input)
	}
	encodedInput, err := json.Marshal(input)
	if err != nil || !bytes.Contains(encodedInput, []byte(`"type":"computer_call_output"`)) ||
		!bytes.Contains(encodedInput, []byte(`"type":"shell_call_output"`)) ||
		!bytes.Contains(encodedInput, []byte(`"type":"apply_patch_call_output"`)) {
		t.Fatalf("native Responses lifecycle round-trip failed: err=%v body=%s", err, encodedInput)
	}
}

func TestResponsesSpecificBuiltinToolChoiceDoesNotRequireName(t *testing.T) {
	t.Parallel()
	choiceType := "computer"
	choice := convertToolChoiceToLLM(&ToolChoice{Type: &choiceType})
	if choice == nil || choice.NamedToolChoice == nil || choice.NamedToolChoice.Type != choiceType || choice.NamedToolChoice.Function.Name != "" {
		t.Fatalf("decode type-only Responses tool choice: %#v", choice)
	}
	wire := convertToolChoice(choice)
	if wire == nil || wire.Type == nil || *wire.Type != choiceType || wire.Name != nil {
		t.Fatalf("encode type-only Responses tool choice: %#v", wire)
	}
}

func TestResponsesAllowedToolsChoiceCanonicalRoundTrip(t *testing.T) {
	t.Parallel()
	choiceType := "allowed_tools"
	mode := "required"
	wire := &ToolChoice{
		Type: &choiceType, Mode: &mode,
		Tools: []ToolOption{
			{Type: "function", Name: "lookup"},
			{Type: "mcp", ServerLabel: "docs", Name: "search"},
			{Type: "shell"},
		},
	}
	canonical := convertToolChoiceToLLM(wire)
	if canonical == nil || canonical.AllowedTools == nil || canonical.AllowedTools.Mode != mode ||
		len(canonical.AllowedTools.Tools) != 3 || canonical.AllowedTools.Tools[1].ServerLabel != "docs" {
		t.Fatalf("decode Responses allowed_tools: %#v", canonical)
	}
	encoded, err := json.Marshal(convertToolChoice(canonical))
	if err != nil {
		t.Fatalf("encode Responses allowed_tools: %v", err)
	}
	want := `{"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"lookup"},{"type":"mcp","name":"search","server_label":"docs"},{"type":"shell"}]}`
	var wantJSON, gotJSON any
	if json.Unmarshal([]byte(want), &wantJSON) != nil || json.Unmarshal(encoded, &gotJSON) != nil || !reflect.DeepEqual(wantJSON, gotJSON) {
		t.Fatalf("Responses allowed_tools round-trip = %s, want %s", encoded, want)
	}
}

func TestResponsesComputerFileOutputIsStructuredAndCrossProtocolPortable(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"computer_call","id":"cu_1","call_id":"computer_1","action":{"type":"screenshot"},"status":"completed"},
			{"type":"computer_call_output","id":"cuo_1","call_id":"computer_1","output":{"type":"computer_screenshot","file_id":"file_screen_1"},"status":"completed"}
		],
		"tools":[{"type":"computer","display_width":1024,"display_height":768,"environment":"browser"}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode Responses computer file output: %v", err)
	}
	if len(request.Input) != 2 || request.Input[1].ToolResult == nil || len(request.Input[1].ToolResult.Content) != 1 ||
		request.Input[1].ToolResult.Content[0].Kind != llm.ContentKindText ||
		!bytes.Contains(request.Input[1].ToolResult.StructuredContent, []byte(`"file_id":"file_screen_1"`)) {
		t.Fatalf("canonical computer file output = %#v", request.Input)
	}
	plan, err := conversion.NewPlanner().Plan(request, llm.APIFormatOpenAIChatCompletion)
	if err != nil || !plan.Complete() || plan.Summary.Unknown != 0 {
		t.Fatalf("computer file output cross-protocol plan = %#v, err=%v", plan, err)
	}
	input, _, ok, err := canonicalRequestInput(request)
	if err != nil || !ok || len(input.Items) != 2 || input.Items[1].ComputerOutput == nil ||
		input.Items[1].ComputerOutput.FileID != "file_screen_1" {
		t.Fatalf("computer file output identity restoration: input=%#v ok=%v err=%v", input, ok, err)
	}
}

func TestInboundTypesResponsesLocalShellLifecycle(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"local_shell_call","id":"lsh_1","call_id":"shell_1","status":"completed","action":{
				"type":"exec","command":["cat","README.md"],"timeout_ms":5000,"working_directory":"/workspace",
				"env":{"LANG":"C"},"user":"sandbox"
			}},
			{"type":"function_call_output","call_id":"shell_1","output":"contents"}
		],
		"tools":[{"type":"local_shell"}]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode local shell lifecycle: %v", err)
	}
	if len(request.ToolDefinitions) != 1 || request.ToolDefinitions[0].Kind != llm.ToolKindLocalShell ||
		len(request.Input) != 2 || request.Input[0].ToolCall == nil || request.Input[1].ToolResult == nil {
		t.Fatalf("local shell canonical shape = %#v", request)
	}
	call, result := request.Input[0].ToolCall, request.Input[1].ToolResult
	if call.Kind != llm.ToolKindLocalShell || call.CallID != "shell_1" || call.Execution != llm.ExecutionOwnerClient ||
		!bytes.Contains(call.ArgumentsJSON, []byte(`"working_directory":"/workspace"`)) ||
		result.Kind != llm.ToolKindLocalShell || result.CallID != "shell_1" || result.Content[0].Text != "contents" {
		t.Fatalf("local shell lifecycle degraded: call=%#v result=%#v", call, result)
	}
	wire, ok := canonicalItemToResponses(&request.Input[0])
	if !ok || wire.Type != "local_shell_call" || wire.CallID != "shell_1" || wire.Action == nil || wire.Action.LocalShell == nil ||
		wire.Action.LocalShell.Type != "exec" || len(wire.Action.LocalShell.Command) != 2 ||
		wire.Action.LocalShell.Command[0] != "cat" || wire.Action.LocalShell.Command[1] != "README.md" ||
		wire.Action.LocalShell.TimeoutMS == nil || *wire.Action.LocalShell.TimeoutMS != 5000 ||
		wire.Action.LocalShell.WorkingDirectory == nil || *wire.Action.LocalShell.WorkingDirectory != "/workspace" ||
		wire.Action.LocalShell.Env["LANG"] != "C" || wire.Action.LocalShell.User == nil || *wire.Action.LocalShell.User != "sandbox" {
		t.Fatalf("local shell Responses identity projection degraded: %#v", wire)
	}
}

func TestInboundTypesResponsesClientToolSearchLifecycle(t *testing.T) {
	t.Parallel()
	body := []byte(`{
		"model":"fixture-model",
		"input":[
			{"type":"tool_search_call","id":"tsc_1","call_id":"search_1","status":"completed","execution":"client","arguments":{"query":"calendar","limit":2}},
			{"type":"tool_search_output","id":"tso_1","call_id":"search_1","status":"completed","execution":"client","tools":[
				{"type":"function","name":"create_event","description":"Create an event","defer_loading":true,"parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}
			]}
		],
		"tools":[
			{"type":"tool_search","execution":"client","description":"Search deferred tools","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"],"additionalProperties":false}},
			{"type":"function","name":"create_event","description":"Create an event","defer_loading":true,"parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}
		]
	}`)
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}}, Body: body,
	})
	if err != nil {
		t.Fatalf("decode client tool search lifecycle: %v", err)
	}
	if len(request.ToolDefinitions) != 2 || request.ToolDefinitions[0].Kind != llm.ToolKindToolSearch ||
		request.ToolDefinitions[0].Execution != llm.ExecutionOwnerClient || len(request.Input) != 2 ||
		request.Input[0].ToolCall == nil || request.Input[1].ToolResult == nil {
		t.Fatalf("tool search canonical shape = %#v", request)
	}
	if request.ToolDefinitions[1].LogicalName != "create_event" || request.ToolDefinitions[1].DeferLoading == nil ||
		!*request.ToolDefinitions[1].DeferLoading {
		t.Fatalf("top-level deferred tool metadata degraded: %#v", request.ToolDefinitions[1])
	}
	call, result := request.Input[0].ToolCall, request.Input[1].ToolResult
	if call.Kind != llm.ToolKindToolSearch || call.CallID != "search_1" || call.Execution != llm.ExecutionOwnerClient ||
		!bytes.Contains(call.ArgumentsJSON, []byte(`"query":"calendar"`)) || result.Kind != llm.ToolKindToolSearch ||
		result.CallID != "search_1" || result.Execution != llm.ExecutionOwnerClient || len(result.DiscoveredTools) != 1 ||
		result.DiscoveredTools[0].Kind != llm.ToolKindFunction || result.DiscoveredTools[0].LogicalName != "create_event" ||
		result.DiscoveredTools[0].DeferLoading == nil || !*result.DiscoveredTools[0].DeferLoading {
		t.Fatalf("tool search lifecycle degraded: call=%#v result=%#v", call, result)
	}
	if err := llm.ValidateItems(request.Input); err != nil {
		t.Fatalf("validate tool search lifecycle: %v", err)
	}
	callWire, callOK := canonicalItemToResponses(&request.Input[0])
	resultWire, resultOK := canonicalItemToResponses(&request.Input[1])
	if !callOK || !resultOK || callWire.Type != "tool_search_call" || callWire.Execution != "client" ||
		resultWire.Type != "tool_search_output" || resultWire.Execution != "client" ||
		len(resultWire.AdditionalTools) != 1 || resultWire.AdditionalTools[0].Name != "create_event" ||
		resultWire.AdditionalTools[0].DeferLoading == nil || !*resultWire.AdditionalTools[0].DeferLoading {
		t.Fatalf("tool search Responses identity projection degraded: call=%#v result=%#v", callWire, resultWire)
	}
	encoded, err := json.Marshal(resultWire)
	if err != nil || !bytes.Contains(encoded, []byte(`"type":"tool_search_output"`)) ||
		!bytes.Contains(encoded, []byte(`"name":"create_event"`)) {
		t.Fatalf("encode tool search output: err=%v body=%s", err, encoded)
	}
}

func TestInboundPreservesCanonicalConversationAndLifecycleIntent(t *testing.T) {
	t.Parallel()
	request, err := NewInboundTransformer().TransformRequest(context.Background(), &httpclient.Request{
		Method: http.MethodPost, URL: "/v1/responses", Headers: http.Header{"Content-Type": []string{"application/json"}},
		Body: []byte(`{
			"model":"fixture-model",
			"input":"continue",
			"conversation":{"id":"conv_123"},
			"previous_response_id":"resp_parent_123",
			"background":false,
			"store":false
		}`),
	})
	if err != nil {
		t.Fatalf("decode Responses conversation request: %v", err)
	}
	if request.Conversation == nil || request.Conversation.ID != "conv_123" || request.Conversation.ParentID != "resp_parent_123" {
		t.Fatalf("canonical conversation = %#v", request.Conversation)
	}
	if request.Lifecycle.Background == nil || *request.Lifecycle.Background || request.Lifecycle.Store == nil || *request.Lifecycle.Store {
		t.Fatalf("canonical lifecycle intent = %#v", request.Lifecycle)
	}
	if request.PreviousResponseID == nil || *request.PreviousResponseID != "resp_parent_123" {
		t.Fatalf("legacy previous_response_id view = %#v", request.PreviousResponseID)
	}
}

func TestOutboundUsesCanonicalConversationAndLifecycleIntent(t *testing.T) {
	t.Parallel()
	transformer, err := NewOutboundTransformer("https://example.invalid", "fixture-key")
	if err != nil {
		t.Fatalf("create Responses outbound: %v", err)
	}
	explicitFalse := false
	legacyTrue := true
	request, err := transformer.TransformRequest(context.Background(), &llm.Request{
		Model: "fixture-model",
		Input: []llm.Item{{
			Kind: llm.ItemKindMessage, Role: llm.RoleUser,
			Content: []llm.ContentBlock{{Kind: llm.ContentKindText, Text: "continue"}},
		}},
		Conversation:       &llm.ConversationRef{ID: "conv_456", ParentID: "resp_parent_456"},
		PreviousResponseID: stringPointer("legacy_parent"),
		Store:              &legacyTrue,
		Lifecycle: llm.LifecycleOptions{
			Background: &explicitFalse,
			Store:      &explicitFalse,
		},
	})
	if err != nil {
		t.Fatalf("encode canonical Responses conversation request: %v", err)
	}
	var wire Request
	if err := json.Unmarshal(request.Body, &wire); err != nil {
		t.Fatalf("decode outbound Responses body: %v body=%s", err, request.Body)
	}
	if wire.Conversation == nil || wire.Conversation.ID == nil || *wire.Conversation.ID != "conv_456" {
		t.Fatalf("wire conversation = %#v", wire.Conversation)
	}
	if wire.PreviousResponseID == nil || *wire.PreviousResponseID != "resp_parent_456" {
		t.Fatalf("wire previous_response_id = %#v", wire.PreviousResponseID)
	}
	if wire.Background == nil || *wire.Background || wire.Store == nil || *wire.Store {
		t.Fatalf("wire lifecycle = background=%#v store=%#v", wire.Background, wire.Store)
	}
}

func stringPointer(value string) *string { return &value }

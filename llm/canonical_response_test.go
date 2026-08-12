package llm

import (
	"encoding/json"
	"testing"
)

func TestCanonicalResponseAccumulatorMaterializesInterleavedParallelTools(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	events := parallelToolLifecycleEvents()
	for index := range events {
		response := &Response{ID: "provider_response_1", Model: "fixture-model", Events: []Event{events[index]}}
		if err := accumulator.Observe(response); err != nil {
			t.Fatalf("observe event %d %s: %v", index, events[index].Kind, err)
		}
	}
	if !accumulator.IsTerminal() || accumulator.TerminalKind() != EventKindResponseCompleted {
		t.Fatalf("terminal = %q", accumulator.TerminalKind())
	}
	snapshot := accumulator.Snapshot()
	if snapshot.ID != "provider_response_1" || len(snapshot.Output) != 2 {
		t.Fatalf("snapshot = %#v", snapshot)
	}
	if got := string(snapshot.Output[0].ToolCall.ArgumentsJSON); got != `{"q":"hello"}` {
		t.Fatalf("first arguments = %q", got)
	}
	if got := string(snapshot.Output[1].ToolCall.ArgumentsJSON); got != `{"x":1}` {
		t.Fatalf("second arguments = %q", got)
	}

	// Snapshot must not retain mutable decoder state.
	snapshot.Output[0].ToolCall.ArgumentsJSON[0] = '['
	if got := string(accumulator.Snapshot().Output[0].ToolCall.ArgumentsJSON); got != `{"q":"hello"}` {
		t.Fatalf("snapshot mutation leaked into accumulator: %q", got)
	}
}

func TestCanonicalResponseAccumulatorRejectsDuplicateObservation(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	event := Event{Kind: EventKindResponseStarted, Sequence: 0}
	if err := accumulator.Observe(&Response{Events: []Event{event}}); err != nil {
		t.Fatalf("observe first event: %v", err)
	}
	if err := accumulator.Observe(&Response{Events: []Event{event}}); err == nil {
		t.Fatal("duplicate event observation was accepted")
	}
}

func TestCanonicalResponseAccumulatorConsumesTerminalAttachedToDoneMarker(t *testing.T) {
	accumulator := NewCanonicalResponseAccumulator()
	events := []Event{
		{Kind: EventKindResponseStarted, Sequence: 0},
		{Kind: EventKindResponseInProgress, Sequence: 1},
	}
	if err := accumulator.Observe(&Response{Events: events}); err != nil {
		t.Fatalf("observe start: %v", err)
	}
	if err := accumulator.Observe(&Response{
		Object: "[DONE]",
		Events: []Event{{Kind: EventKindResponseCompleted, Sequence: 2}},
	}); err != nil {
		t.Fatalf("observe done terminal: %v", err)
	}
	if !accumulator.IsTerminal() || accumulator.TerminalKind() != EventKindResponseCompleted {
		t.Fatalf("terminal = %q", accumulator.TerminalKind())
	}
}

func TestCanonicalTerminalReasonSurvivesMaterializeAndAggregate(t *testing.T) {
	response := &Response{
		ID: "provider_pause", Status: ResponseStatusIncomplete, TerminalReason: "pause_turn",
		Output: []Item{{
			Kind: ItemKindMessage, Role: RoleAssistant, Status: ItemStatusIncomplete,
			Content: []ContentBlock{{Kind: ContentKindText, Text: "continuing"}},
		}},
	}
	events, err := CanonicalEventsFromResponse(response)
	if err != nil {
		t.Fatalf("render pause_turn response: %v", err)
	}
	terminal := events[len(events)-1]
	if terminal.Kind != EventKindResponseIncomplete || terminal.TerminalReason != "pause_turn" {
		t.Fatalf("terminal event = %#v", terminal)
	}
	accumulator := NewCanonicalResponseAccumulator()
	for index := range events {
		if err := accumulator.Observe(&Response{Events: []Event{events[index]}}); err != nil {
			t.Fatalf("observe pause_turn event %d: %v", index, err)
		}
	}
	snapshot := accumulator.Snapshot()
	if snapshot.Status != ResponseStatusIncomplete || snapshot.TerminalReason != "pause_turn" {
		t.Fatalf("pause_turn snapshot = %#v", snapshot)
	}

	invalid := Event{Kind: EventKindTextDelta, ItemRef: ItemRef{ItemID: "msg"}, Delta: Delta{Text: "x"}, TerminalReason: "pause_turn"}
	if err := invalid.Validate(); err == nil {
		t.Fatal("non-terminal event accepted a terminal reason")
	}
}

func TestCanonicalResponseResidualSurvivesEventMaterializationAndIsIsolated(t *testing.T) {
	response := &Response{
		ID: "resp_residual", Status: ResponseStatusCompleted,
		ProviderExtensions: &ResponseProviderExtensions{
			OpenAIResponses: &OpenAIResponsesResponseExtensions{
				ResidualFields: json.RawMessage(`{"conversation":{"future":1}}`),
			},
		},
	}
	events, err := CanonicalEventsFromResponse(response)
	if err != nil {
		t.Fatalf("render response residual: %v", err)
	}
	if len(events) != 3 {
		t.Fatalf("events = %d", len(events))
	}
	for index := range events {
		if string(events[index].ResponseSourceResidual) != `{"conversation":{"future":1}}` {
			t.Fatalf("event %d response residual = %s", index, events[index].ResponseSourceResidual)
		}
	}

	accumulator := NewCanonicalResponseAccumulator()
	for index := range events {
		if err := accumulator.Observe(&Response{Events: []Event{events[index]}}); err != nil {
			t.Fatalf("observe residual event %d: %v", index, err)
		}
	}
	snapshot := accumulator.Snapshot()
	if snapshot.ProviderExtensions == nil || snapshot.ProviderExtensions.OpenAIResponses == nil {
		t.Fatal("response residual extensions were lost")
	}
	if got := string(snapshot.ProviderExtensions.OpenAIResponses.ResidualFields); got != `{"conversation":{"future":1}}` {
		t.Fatalf("snapshot response residual = %s", got)
	}
	snapshot.ProviderExtensions.OpenAIResponses.ResidualFields[0] = '['
	if got := string(accumulator.Snapshot().ProviderExtensions.OpenAIResponses.ResidualFields); got != `{"conversation":{"future":1}}` {
		t.Fatalf("snapshot mutation leaked into accumulator: %s", got)
	}
}

func TestCanonicalAgentMessageStreamKeepsTypedSnapshotWithoutTextProjection(t *testing.T) {
	t.Parallel()

	response := &Response{
		Status: ResponseStatusCompleted,
		Output: []Item{{
			Kind: ItemKindAgentMessage, ID: "agent_output_1", Role: RoleAssistant,
			AgentMessage: &AgentMessage{
				Author: "/root", Recipient: "/root/worker",
				Content: []AgentMessageContentPart{{Kind: AgentMessageContentInputText, Text: "typed handoff"}},
			},
		}},
	}
	events, err := CanonicalEventsFromResponse(response)
	if err != nil {
		t.Fatalf("render agent message response: %v", err)
	}
	for index := range events {
		if events[index].Kind == EventKindTextDelta {
			t.Fatalf("agent_message was incorrectly projected as text_delta: %#v", events[index])
		}
		if events[index].Kind == EventKindItemAdded || events[index].Kind == EventKindItemDone {
			if events[index].Snapshot == nil || events[index].Snapshot.Status != "" {
				t.Fatalf("agent_message snapshot gained illegal lifecycle status: %#v", events[index].Snapshot)
			}
		}
	}

	accumulator := NewCanonicalResponseAccumulator()
	if err := accumulator.Observe(&Response{Events: events}); err != nil {
		t.Fatalf("accumulate typed agent message stream: %v", err)
	}
	snapshot := accumulator.Snapshot()
	if snapshot == nil || len(snapshot.Output) != 1 || snapshot.Output[0].Kind != ItemKindAgentMessage ||
		snapshot.Output[0].AgentMessage == nil || snapshot.Output[0].AgentMessage.Content[0].Text != "typed handoff" {
		t.Fatalf("typed agent stream snapshot degraded: %#v", snapshot)
	}
}

func TestCanonicalStatuslessResponsesUnionsNeverGainItemStatus(t *testing.T) {
	t.Parallel()
	response := &Response{Status: ResponseStatusCompleted, Output: []Item{
		{Kind: ItemKindAgentMessage, ID: "agent", Status: ItemStatusCompleted, AgentMessage: &AgentMessage{
			Author: "/root", Recipient: "/root/worker", Content: []AgentMessageContentPart{{Kind: AgentMessageContentInputText, Text: "handoff"}},
		}},
		{Kind: ItemKindContextCompaction, ID: "context", Status: ItemStatusCompleted, ContextCompaction: &ContextCompactionItem{}},
		{Kind: ItemKindCompaction, ID: "compact", Status: ItemStatusCompleted, Compaction: &CompactionItem{}},
		{Kind: ItemKindUnknown, ID: "future", Status: ItemStatusCompleted, Unknown: &UnknownItem{Type: "future_behavior", Raw: json.RawMessage(`{"type":"future_behavior"}`), Behavioral: true}},
	}}
	events, err := CanonicalEventsFromResponse(response)
	if err != nil {
		t.Fatalf("render statusless unions: %v", err)
	}
	for index := range events {
		event := &events[index]
		if (event.Kind != EventKindItemAdded && event.Kind != EventKindItemDone) || event.Snapshot == nil {
			continue
		}
		if event.Snapshot.Status != "" {
			t.Fatalf("%s snapshot %q retained invalid status %q", event.Kind, event.Snapshot.Kind, event.Snapshot.Status)
		}
	}
}

func TestAgentMessageValidationRejectsWhitespaceOnlyBoundaries(t *testing.T) {
	t.Parallel()
	for name, message := range map[string]*AgentMessage{
		"blank author":    {Author: " ", Recipient: "/root/worker", Content: []AgentMessageContentPart{{Kind: AgentMessageContentInputText, Text: "handoff"}}},
		"blank recipient": {Author: "/root", Recipient: "\t", Content: []AgentMessageContentPart{{Kind: AgentMessageContentInputText, Text: "handoff"}}},
		"blank encrypted": {Author: "/root", Recipient: "/root/worker", Content: []AgentMessageContentPart{{Kind: AgentMessageContentEncryptedContent, EncryptedContent: " "}}},
	} {
		message := message
		t.Run(name, func(t *testing.T) {
			t.Parallel()
			if err := message.Validate(); err == nil {
				t.Fatalf("whitespace-only agent-message boundary was accepted: %#v", message)
			}
		})
	}
}

func TestLegacyInterAgentMessageJSONMatchesCodexPlaintextParts(t *testing.T) {
	t.Parallel()
	multiple := &AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []AgentMessageContentPart{
		{Kind: AgentMessageContentInputText, Text: "first"},
		{Kind: AgentMessageContentInputText, Text: "second"},
	}}
	envelope, err := multiple.LegacyInterAgentMessageJSON()
	var legacy struct {
		Author          string   `json:"author"`
		Recipient       string   `json:"recipient"`
		OtherRecipients []string `json:"other_recipients"`
		Content         string   `json:"content"`
		TriggerTurn     bool     `json:"trigger_turn"`
	}
	if err != nil || json.Unmarshal([]byte(envelope), &legacy) != nil || legacy.Author != "/root" || legacy.Recipient != "/root/worker" ||
		len(legacy.OtherRecipients) != 0 || legacy.Content != "first\nsecond" || !legacy.TriggerTurn {
		t.Fatalf("multi-part legacy inter-agent JSON=%q decoded=%#v err=%v", envelope, legacy, err)
	}

	blank := &AgentMessage{Author: "/root", Recipient: "/root/worker", Content: []AgentMessageContentPart{
		{Kind: AgentMessageContentInputText, Text: " \t"},
		{Kind: AgentMessageContentInputText, Text: "\n"},
	}}
	if envelope, err := blank.LegacyInterAgentMessageJSON(); err == nil || envelope != "" {
		t.Fatalf("blank plaintext legacy inter-agent message was accepted: envelope=%q err=%v", envelope, err)
	}
}

func TestValidateCodexAgentPathClosedGrammar(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		path  string
		valid bool
	}{
		{path: "/morpheus", valid: true},
		{path: "/root", valid: true},
		{path: "/root/worker_2/child", valid: true},
		{path: "/root/Worker"},
		{path: "/root/worker-2"},
		{path: "/root/worker/"},
		{path: "/root//worker"},
		{path: "/root/root"},
		{path: "/root/."},
		{path: "/root/.."},
		{path: "/other/worker"},
	} {
		test := test
		t.Run(test.path, func(t *testing.T) {
			err := ValidateCodexAgentPath(test.path)
			if (err == nil) != test.valid {
				t.Fatalf("AgentPath %q err=%v valid=%v", test.path, err, test.valid)
			}
		})
	}
}

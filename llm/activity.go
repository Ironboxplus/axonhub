package llm

// HasModelActivity reports whether a unified response contains provider output
// that is safe to expose to the client. Protocol lifecycle metadata, usage and
// terminal markers are deliberately excluded so relays can retain failover
// safety until the first text, reasoning, media, or tool invocation exists.
//
// Canonical Events and Output are authoritative. The legacy response fields
// remain covered while provider adapters complete their migration.
func (response *Response) HasModelActivity() bool {
	if response == nil || response == DoneResponse || response.Object == "[DONE]" {
		return false
	}
	for index := range response.Events {
		if response.Events[index].HasModelActivity() {
			return true
		}
	}
	for index := range response.Output {
		if canonicalItemHasModelActivity(&response.Output[index]) {
			return true
		}
	}
	if response.Embedding != nil && len(response.Embedding.Data) > 0 ||
		response.Rerank != nil && len(response.Rerank.Results) > 0 ||
		response.Image != nil && len(response.Image.Data) > 0 ||
		response.ImageStreamEvent != nil && (response.ImageStreamEvent.B64JSON != "" || response.ImageStreamEvent.URL != "") ||
		response.Video != nil || response.Compact != nil || response.Speech != nil ||
		response.Transcription != nil || response.Moderation != nil ||
		response.SpeechAudioChunk != nil || response.SpeechStreamEvent != nil || response.TranscriptionStreamEvent != nil {
		return true
	}
	if response.Completion != nil {
		for _, choice := range response.Completion.Choices {
			if choice.Text != "" {
				return true
			}
		}
	}
	for index := range response.Choices {
		choice := &response.Choices[index]
		if legacyMessageHasModelActivity(choice.Message) || legacyMessageHasModelActivity(choice.Delta) {
			return true
		}
	}
	return false
}

// HasModelActivity reports whether one canonical stream event carries actual
// provider output rather than transport or lifecycle metadata.
func (event Event) HasModelActivity() bool {
	switch event.Kind {
	case EventKindTextDelta, EventKindReasoningDelta, EventKindRefusalDelta:
		return event.Delta.Text != ""
	case EventKindToolInputDelta:
		return event.Delta.ArgumentsJSON != "" || event.Delta.InputText != ""
	case EventKindItemAdded, EventKindItemDone:
		return canonicalItemHasModelActivity(event.Snapshot)
	default:
		return false
	}
}

func canonicalItemHasModelActivity(item *Item) bool {
	if item == nil {
		return false
	}
	switch item.Kind {
	case ItemKindMessage:
		for index := range item.Content {
			block := &item.Content[index]
			if block.Text != "" || block.Image != nil || block.Audio != nil || block.Document != nil ||
				block.Citation != nil || len(block.UnknownRaw) > 0 {
				return true
			}
		}
		return false
	case ItemKindReasoning:
		return item.Reasoning != nil && (item.Reasoning.Content != "" || item.Reasoning.Signature != "")
	case ItemKindAgentMessage:
		if item.AgentMessage == nil {
			return false
		}
		for index := range item.AgentMessage.Content {
			part := &item.AgentMessage.Content[index]
			if part.Text != "" || part.EncryptedContent != "" {
				return true
			}
		}
		return false
	case ItemKindToolCall:
		return item.ToolCall != nil
	case ItemKindHostedCall:
		return item.HostedCall != nil
	case ItemKindMCPListTools:
		return item.MCPListTools != nil
	case ItemKindMCPApprovalRequest:
		return item.MCPApprovalRequest != nil
	case ItemKindMCPCall:
		return item.MCPCall != nil
	case ItemKindCompaction:
		return item.Compaction != nil
	case ItemKindContextCompaction:
		return item.ContextCompaction != nil
	case ItemKindUnknown:
		return item.Unknown != nil
	default:
		return false
	}
}

func legacyMessageHasModelActivity(message *Message) bool {
	if message == nil {
		return false
	}
	if message.Content.Content != nil && *message.Content.Content != "" || len(message.Content.MultipleContent) > 0 ||
		len(message.ToolCalls) > 0 || message.Refusal != "" || message.Audio != nil ||
		message.ReasoningContent != nil && *message.ReasoningContent != "" ||
		message.Reasoning != nil && *message.Reasoning != "" ||
		message.ReasoningSignature != nil && *message.ReasoningSignature != "" ||
		message.RedactedReasoningContent != nil && *message.RedactedReasoningContent != "" ||
		len(message.InlineToolResults) > 0 {
		return true
	}
	for index := range message.ReasoningItems {
		if message.ReasoningItems[index].Content != "" || message.ReasoningItems[index].Signature != "" {
			return true
		}
	}
	return false
}

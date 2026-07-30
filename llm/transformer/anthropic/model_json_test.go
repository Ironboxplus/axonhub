package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"
)

func TestMessageContentBlockMarshalJSON_PreservesEmptyThinkingSignature(t *testing.T) {
	data, err := json.Marshal(MessageContentBlock{
		Type:      "thinking",
		Thinking:  lo.ToPtr(""),
		Signature: lo.ToPtr(""),
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"thinking","thinking":"","signature":""}`, string(data))
}

func TestMessageContentBlockMarshalJSON_PreservesNilThinkingSignature(t *testing.T) {
	data, err := json.Marshal(MessageContentBlock{
		Type: "thinking",
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"thinking","thinking":"","signature":""}`, string(data))
}

func TestMessageContentBlockMarshalJSON_WithCitations(t *testing.T) {
	data, err := json.Marshal(MessageContentBlock{
		Type: "text",
		Text: lo.ToPtr("hello"),
		Citations: []TextCitation{
			{
				Type:           "url_citation",
				URL:            "https://example.com/a",
				Title:          "Example A",
				EncryptedIndex: lo.ToPtr("enc-1"),
				CitedText:      lo.ToPtr("quote"),
			},
		},
	})
	require.NoError(t, err)
	require.JSONEq(t, `{
		"type":"text",
		"text":"hello",
		"citations":[{
			"type":"url_citation",
			"url":"https://example.com/a",
			"title":"Example A",
			"encrypted_index":"enc-1",
			"cited_text":"quote"
		}]
	}`, string(data))
}

func TestMessageContentBlockUnmarshalJSON_WithCitations(t *testing.T) {
	var block MessageContentBlock
	err := json.Unmarshal([]byte(`{
		"type":"text",
		"text":"hello",
		"citations":[{
			"type":"url_citation",
			"url":"https://example.com/a",
			"title":"Example A",
			"encrypted_index":"enc-1",
			"cited_text":"quote"
		}]
	}`), &block)
	require.NoError(t, err)
	require.Equal(t, "text", block.Type)
	require.Equal(t, "hello", lo.FromPtr(block.Text))
	require.Equal(t, []TextCitation{
		{
			Type:           "url_citation",
			URL:            "https://example.com/a",
			Title:          "Example A",
			EncryptedIndex: lo.ToPtr("enc-1"),
			CitedText:      lo.ToPtr("quote"),
		},
	}, block.Citations)
}

func TestMessageContentBlockJSON_DocumentCitationsObject(t *testing.T) {
	raw := `{
		"type":"document",
		"source":{"type":"content","content":[{"type":"text","text":"source text"}]},
		"title":"report.txt",
		"context":"quarterly report",
		"citations":{"enabled":true},
		"cache_control":{"type":"ephemeral"}
	}`
	var block MessageContentBlock
	require.NoError(t, json.Unmarshal([]byte(raw), &block))
	require.Equal(t, "document", block.Type)
	require.NotNil(t, block.Source)
	require.Equal(t, "content", block.Source.Type)
	require.JSONEq(t, `[{"type":"text","text":"source text"}]`, string(block.Source.Content))
	require.Equal(t, "report.txt", block.Title)
	require.NotNil(t, block.DocumentCitations)
	require.True(t, block.DocumentCitations.Enabled)
	require.Empty(t, block.Citations)

	roundTrip, err := json.Marshal(block)
	require.NoError(t, err)
	require.JSONEq(t, raw, string(roundTrip))
}

func TestMessageContentUnmarshalJSON_PreservesObjectServerToolResult(t *testing.T) {
	raw := []byte(`{
		"type":"web_fetch_result",
		"url":"https://example.com/article",
		"retrieved_at":"2026-07-30T00:00:00Z",
		"content":{
			"type":"document",
			"source":{"type":"text","media_type":"text/plain","data":"Article content"}
		}
	}`)

	var content MessageContent
	require.NoError(t, json.Unmarshal(raw, &content))
	require.NotNil(t, content.ObjectContent)
	require.Equal(t, "web_fetch_result", content.ObjectContent.Type)
	require.JSONEq(t, string(raw), string(content.Raw))

	roundTrip, err := json.Marshal(content)
	require.NoError(t, err)
	require.JSONEq(t, string(raw), string(roundTrip))
}

func TestStreamDeltaMarshalJSON_OmitsSignatureForThinkingDelta(t *testing.T) {
	data, err := json.Marshal(StreamDelta{
		Type:      lo.ToPtr("thinking_delta"),
		Thinking:  lo.ToPtr("Thinking..."),
		Signature: lo.ToPtr(""),
	})
	require.NoError(t, err)
	require.JSONEq(t, `{"type":"thinking_delta","thinking":"Thinking..."}`, string(data))
}

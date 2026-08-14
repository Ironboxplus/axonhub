package pipeline

import (
	"context"
	"errors"
	"testing"

	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

func TestHasResponseContent_ReasoningSignature(t *testing.T) {
	signature := "gAAAA_reasoning"

	require.True(t, hasResponseContent(&llm.Response{
		Object: "chat.completion.chunk",
		Choices: []llm.Choice{{
			Delta: &llm.Message{ReasoningSignature: &signature},
		}},
	}))
}

func TestHasResponseContent_ResponseFailedIsClientVisibleTerminal(t *testing.T) {
	responseErr := &llm.ResponseError{StatusCode: 429, Detail: llm.ErrorDetail{Code: "rate_limited", Message: "provider rejected request"}}
	require.True(t, hasResponseContent(&llm.Response{
		Status: llm.ResponseStatusFailed,
		Error:  responseErr,
		Events: []llm.Event{{Kind: llm.EventKindResponseFailed, Error: responseErr}, {Kind: llm.EventKindUsage, Usage: &llm.Usage{TotalTokens: 7}}},
	}))
	require.True(t, hasResponseContent(&llm.Response{Events: []llm.Event{{Kind: llm.EventKindResponseFailed, Error: responseErr}}}))
	require.True(t, hasResponseContent(&llm.Response{Events: []llm.Event{{Kind: llm.EventKindResponseIncomplete}}}))
	require.True(t, hasResponseContent(&llm.Response{Events: []llm.Event{{Kind: llm.EventKindResponseCancelled}}}))
	require.False(t, hasResponseContent(&llm.Response{Status: llm.ResponseStatusFailed}))
	require.True(t, hasResponseContent(&llm.Response{Status: llm.ResponseStatusIncomplete, Usage: &llm.Usage{TotalTokens: 7}}))
	require.True(t, hasResponseContent(&llm.Response{Status: llm.ResponseStatusCancelled, Usage: &llm.Usage{TotalTokens: 7}}))
}

func TestPreReadLlmStreamPreservesStructuredResponseFailedWithoutEmptyRetry(t *testing.T) {
	responseErr := &llm.ResponseError{StatusCode: 429, Detail: llm.ErrorDetail{Code: "rate_limited", Message: "provider rejected request"}}
	failed := &llm.Response{
		Status: llm.ResponseStatusFailed,
		Error:  responseErr,
		Usage:  &llm.Usage{TotalTokens: 7},
		Events: []llm.Event{{Kind: llm.EventKindResponseFailed, Error: responseErr}, {Kind: llm.EventKindUsage, Usage: &llm.Usage{TotalTokens: 7}}},
	}
	p := &pipeline{emptyResponseDetection: true, maxSameChannelRetries: 1}
	stream, err := p.preReadLlmStream(context.Background(), streams.SliceStream([]*llm.Response{failed}), nil)
	require.NoError(t, err)
	responses, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, responses, 1)
	require.Same(t, failed, responses[0])
	require.Equal(t, llm.EventKindResponseFailed, responses[0].Events[0].Kind)
	require.Same(t, responseErr, responses[0].Events[0].Error)
	require.Equal(t, int64(7), responses[0].Usage.TotalTokens)
}

func TestPreReadLlmStreamPreservesMaterializedIncompleteAndCancelledWithoutEmptyRetry(t *testing.T) {
	for _, status := range []llm.ResponseStatus{llm.ResponseStatusIncomplete, llm.ResponseStatusCancelled} {
		status := status
		t.Run(string(status), func(t *testing.T) {
			terminal := &llm.Response{Status: status, Usage: &llm.Usage{TotalTokens: 7}}
			p := &pipeline{emptyResponseDetection: true, maxSameChannelRetries: 2}
			stream, err := p.preReadLlmStream(context.Background(), streams.SliceStream([]*llm.Response{terminal}), nil)
			require.NoError(t, err)
			responses, err := streams.All(stream)
			require.NoError(t, err)
			require.Len(t, responses, 1)
			require.Same(t, terminal, responses[0])
			require.Equal(t, int64(7), responses[0].Usage.TotalTokens)
		})
	}
}

func TestHasResponseContent(t *testing.T) {
	t.Run("empty response", func(t *testing.T) {
		require.False(t, hasResponseContent(&llm.Response{}))
	})

	t.Run("message text content", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Choices: []llm.Choice{{
				Message: &llm.Message{
					Content: llm.MessageContent{Content: lo.ToPtr("hello")},
				},
			}},
		}))
	})

	t.Run("tool calls", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Choices: []llm.Choice{{
				Message: &llm.Message{
					ToolCalls: []llm.ToolCall{{ID: "call_1"}},
				},
			}},
		}))
	})

	t.Run("reasoning content", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Choices: []llm.Choice{{
				Message: &llm.Message{
					ReasoningContent: lo.ToPtr("thinking"),
				},
			}},
		}))
	})

	t.Run("speech response audio", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Speech: &llm.SpeechResponse{Audio: []byte{0x01, 0x02}},
		}))
	})

	t.Run("empty speech response", func(t *testing.T) {
		require.False(t, hasResponseContent(&llm.Response{
			Speech: &llm.SpeechResponse{},
		}))
	})

	t.Run("transcription response text", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Transcription: &llm.TranscriptionResponse{Text: "hello"},
		}))
	})

	t.Run("transcription response raw", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Transcription: &llm.TranscriptionResponse{Raw: []byte("1\n00:00:00,000 --> 00:00:01,000\nhi\n")},
		}))
	})

	t.Run("empty transcription response", func(t *testing.T) {
		require.False(t, hasResponseContent(&llm.Response{
			Transcription: &llm.TranscriptionResponse{},
		}))
	})

	t.Run("embedding response data", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{
			Embedding: &llm.EmbeddingResponse{
				Object: "list",
				Data: []llm.EmbeddingData{{
					Object: "embedding",
					Embedding: llm.Embedding{
						Embedding: []float64{0.1, 0.2, 0.3},
					},
					Index: 0,
				}},
			},
		}))
	})

	t.Run("rerank response results", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Rerank: &llm.RerankResponse{Results: []llm.RerankResult{{Index: 0}}}}))
	})

	t.Run("image response data", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Image: &llm.ImageResponse{Data: []llm.ImageData{{URL: "https://image.example/test.png"}}}}))
	})

	t.Run("image stream payload", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{ImageStreamEvent: &llm.ImageStreamEvent{B64JSON: "aW1hZ2U="}}))
		require.True(t, hasResponseContent(&llm.Response{ImageStreamEvent: &llm.ImageStreamEvent{URL: "https://image.example/test.png"}}))
	})

	t.Run("video response lifecycle", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Video: &llm.VideoResponse{ID: "video_1"}}))
		require.True(t, hasResponseContent(&llm.Response{Video: &llm.VideoResponse{Status: "queued"}}))
		require.True(t, hasResponseContent(&llm.Response{Video: &llm.VideoResponse{VideoURL: "https://video.example/test.mp4"}}))
		require.True(t, hasResponseContent(&llm.Response{Video: &llm.VideoResponse{Error: &llm.VideoError{Code: "failed"}}}))
	})

	t.Run("compact response output", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Compact: &llm.CompactResponse{Output: []llm.Message{{Role: "user"}}}}))
	})

	t.Run("streaming speech and transcription content", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{SpeechStreamEvent: &llm.SpeechStreamEvent{AudioBase64: "YXVkaW8="}}))
		require.True(t, hasResponseContent(&llm.Response{SpeechAudioChunk: &llm.SpeechAudioChunk{Audio: []byte("audio")}}))
		require.True(t, hasResponseContent(&llm.Response{TranscriptionStreamEvent: &llm.TranscriptionStreamEvent{Delta: "delta"}}))
		require.True(t, hasResponseContent(&llm.Response{TranscriptionStreamEvent: &llm.TranscriptionStreamEvent{Text: "final"}}))
		require.True(t, hasResponseContent(&llm.Response{TranscriptionStreamEvent: &llm.TranscriptionStreamEvent{Type: "transcript.text.done"}}))
	})

	t.Run("completion response text", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Completion: &llm.CompletionResponse{Choices: []llm.CompletionChoice{{Text: "completion"}}}}))
	})

	t.Run("valid gateway compaction is client-visible", func(t *testing.T) {
		require.True(t, hasResponseContent(&llm.Response{Output: []llm.Item{{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{EncryptedContent: "gateway-owned-opaque"}}}}))
	})

	t.Run("malformed gateway compaction remains empty", func(t *testing.T) {
		require.False(t, hasResponseContent(&llm.Response{Output: []llm.Item{{Kind: llm.ItemKindCompaction}}}))
		require.False(t, hasResponseContent(&llm.Response{Output: []llm.Item{{Kind: llm.ItemKindCompaction, Compaction: &llm.CompactionItem{}}}}))
	})
}

func TestPipeline_Process_StreamEmptyResponseDetection(t *testing.T) {
	ctx := context.Background()

	t.Run("retries on empty stream response", func(t *testing.T) {
		streamCalls := 0
		executor := &mockExecutor{
			doStream: func(ctx context.Context, req *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
				streamCalls++
				// Return a minimal stream that the outbound TransformStream will convert
				return streams.SliceStream([]*httpclient.StreamEvent{{}}), nil
			},
		}

		prepareCalls := 0
		outbound := &mockOutbound{
			transformStream: func(ctx context.Context, req *httpclient.Request, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
				if streamCalls == 1 {
					// First call: empty response (finish reason, no content)
					return streams.SliceStream([]*llm.Response{
						{Choices: []llm.Choice{{FinishReason: lo.ToPtr("stop"), Message: &llm.Message{}}}},
					}), nil
				}
				// Second call: response with content
				return streams.SliceStream([]*llm.Response{
					{Choices: []llm.Choice{{
						Message: &llm.Message{
							Content: llm.MessageContent{Content: lo.ToPtr("ok")},
						},
					}}},
					llm.DoneResponse,
				}), nil
			},
			canRetry: func(err error) bool { return errors.Is(err, ErrEmptyResponse) },
			prepareForRetry: func(ctx context.Context) error {
				prepareCalls++
				return nil
			},
		}

		streamFlag := true
		streamInbound := &mockInbound{
			transformRequest: func(ctx context.Context, req *httpclient.Request) (*llm.Request, error) {
				return &llm.Request{Stream: &streamFlag}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                streamInbound,
			Outbound:               outbound,
			maxSameChannelRetries:  1,
			emptyResponseDetection: true,
		}

		res, err := p.Process(ctx, &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.True(t, res.Stream)
		require.Equal(t, 2, streamCalls)
		require.Equal(t, 1, prepareCalls)
	})

	t.Run("retries on empty binary speech stream", func(t *testing.T) {
		// TTS binary streams emit a terminal *llm.Response{Object:"[DONE]"} that is
		// not the shared llm.DoneResponse sentinel. Empty-response detection must
		// still recognize it so providers returning 200 + zero audio chunks trigger
		// a retry instead of being recorded as completed.
		streamCalls := 0
		executor := &mockExecutor{
			doStream: func(ctx context.Context, req *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
				streamCalls++
				return streams.SliceStream([]*httpclient.StreamEvent{{}}), nil
			},
		}

		prepareCalls := 0
		outbound := &mockOutbound{
			transformStream: func(ctx context.Context, req *httpclient.Request, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
				if streamCalls == 1 {
					// Empty binary speech stream: only the [DONE] terminator, no SpeechAudioChunk.
					return streams.SliceStream([]*llm.Response{
						{Object: "[DONE]", RequestType: llm.RequestTypeSpeech, APIFormat: llm.APIFormatOpenAISpeech},
					}), nil
				}

				return streams.SliceStream([]*llm.Response{
					{
						RequestType: llm.RequestTypeSpeech,
						APIFormat:   llm.APIFormatOpenAISpeech,
						SpeechAudioChunk: &llm.SpeechAudioChunk{
							Audio:       []byte{0x01, 0x02},
							ContentType: "audio/mpeg",
						},
					},
					{Object: "[DONE]", RequestType: llm.RequestTypeSpeech, APIFormat: llm.APIFormatOpenAISpeech},
				}), nil
			},
			canRetry: func(err error) bool { return errors.Is(err, ErrEmptyResponse) },
			prepareForRetry: func(ctx context.Context) error {
				prepareCalls++
				return nil
			},
		}

		streamFlag := true
		streamInbound := &mockInbound{
			transformRequest: func(ctx context.Context, req *httpclient.Request) (*llm.Request, error) {
				return &llm.Request{Stream: &streamFlag}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                streamInbound,
			Outbound:               outbound,
			maxSameChannelRetries:  1,
			emptyResponseDetection: true,
		}

		res, err := p.Process(ctx, &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 2, streamCalls)
		require.Equal(t, 1, prepareCalls)
	})

	t.Run("retries on empty TTS SSE stream with only done event", func(t *testing.T) {
		// TTS stream_format=sse providers that emit only "speech.audio.done" (no
		// speech.audio.delta) must still be treated as empty so empty-response
		// detection retries instead of completing a request with audio_bytes=0.
		streamCalls := 0
		executor := &mockExecutor{
			doStream: func(ctx context.Context, req *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
				streamCalls++
				return streams.SliceStream([]*httpclient.StreamEvent{{}}), nil
			},
		}

		prepareCalls := 0
		outbound := &mockOutbound{
			transformStream: func(ctx context.Context, req *httpclient.Request, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
				if streamCalls == 1 {
					return streams.SliceStream([]*llm.Response{
						{
							RequestType:       llm.RequestTypeSpeech,
							APIFormat:         llm.APIFormatOpenAISpeech,
							SpeechStreamEvent: &llm.SpeechStreamEvent{Type: "speech.audio.done"},
						},
						llm.DoneResponse,
					}), nil
				}

				return streams.SliceStream([]*llm.Response{
					{
						RequestType: llm.RequestTypeSpeech,
						APIFormat:   llm.APIFormatOpenAISpeech,
						SpeechStreamEvent: &llm.SpeechStreamEvent{
							Type:        "speech.audio.delta",
							AudioBase64: "YWJjZA==",
						},
					},
					llm.DoneResponse,
				}), nil
			},
			canRetry: func(err error) bool { return errors.Is(err, ErrEmptyResponse) },
			prepareForRetry: func(ctx context.Context) error {
				prepareCalls++
				return nil
			},
		}

		streamFlag := true
		streamInbound := &mockInbound{
			transformRequest: func(ctx context.Context, req *httpclient.Request) (*llm.Request, error) {
				return &llm.Request{Stream: &streamFlag}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                streamInbound,
			Outbound:               outbound,
			maxSameChannelRetries:  1,
			emptyResponseDetection: true,
		}

		res, err := p.Process(ctx, &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 2, streamCalls)
		require.Equal(t, 1, prepareCalls)
	})

	t.Run("accepts stream response with content", func(t *testing.T) {
		streamCalls := 0
		executor := &mockExecutor{
			doStream: func(ctx context.Context, req *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
				streamCalls++
				return streams.SliceStream([]*httpclient.StreamEvent{{}}), nil
			},
		}

		outbound := &mockOutbound{
			transformStream: func(ctx context.Context, req *httpclient.Request, stream streams.Stream[*httpclient.StreamEvent]) (streams.Stream[*llm.Response], error) {
				return streams.SliceStream([]*llm.Response{
					{Choices: []llm.Choice{{
						Delta: &llm.Message{
							Content: llm.MessageContent{Content: lo.ToPtr("hello")},
						},
					}}},
					llm.DoneResponse,
				}), nil
			},
		}

		streamFlag := true
		streamInbound := &mockInbound{
			transformRequest: func(ctx context.Context, req *httpclient.Request) (*llm.Request, error) {
				return &llm.Request{Stream: &streamFlag}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                streamInbound,
			Outbound:               outbound,
			emptyResponseDetection: true,
		}

		res, err := p.Process(ctx, &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.True(t, res.Stream)
		require.Equal(t, 1, streamCalls)
	})
}

func TestPipeline_Process_NonStreamEmptyResponseDetection(t *testing.T) {
	ctx := context.Background()

	t.Run("retries on empty non-stream response", func(t *testing.T) {
		execCalls := 0
		executor := &mockExecutor{
			do: func(ctx context.Context, req *httpclient.Request) (*httpclient.Response, error) {
				execCalls++
				return &httpclient.Response{}, nil
			},
		}

		prepareCalls := 0
		outbound := &mockOutbound{
			transformResponse: func(ctx context.Context, resp *httpclient.Response) (*llm.Response, error) {
				if execCalls == 1 {
					return &llm.Response{
						Choices: []llm.Choice{{Message: &llm.Message{}}},
					}, nil
				}

				return &llm.Response{
					Choices: []llm.Choice{{
						Message: &llm.Message{
							Content: llm.MessageContent{Content: lo.ToPtr("ok")},
						},
					}},
				}, nil
			},
			canRetry: func(err error) bool { return errors.Is(err, ErrEmptyResponse) },
			prepareForRetry: func(ctx context.Context) error {
				prepareCalls++
				return nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                &mockInbound{},
			Outbound:               outbound,
			maxSameChannelRetries:  1,
			emptyResponseDetection: true,
		}

		res, err := p.Process(ctx, &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 2, execCalls)
		require.Equal(t, 1, prepareCalls)
	})

	t.Run("accepts non-stream tool-call response", func(t *testing.T) {
		executor := &mockExecutor{
			do: func(ctx context.Context, req *httpclient.Request) (*httpclient.Response, error) {
				return &httpclient.Response{}, nil
			},
		}

		outbound := &mockOutbound{
			transformResponse: func(ctx context.Context, resp *httpclient.Response) (*llm.Response, error) {
				return &llm.Response{
					Choices: []llm.Choice{{
						Message: &llm.Message{
							ToolCalls: []llm.ToolCall{{ID: "call_1"}},
						},
					}},
				}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                &mockInbound{},
			Outbound:               outbound,
			emptyResponseDetection: true,
		}

		res, err := p.Process(context.Background(), &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
	})

	t.Run("accepts non-stream embedding response", func(t *testing.T) {
		execCalls := 0
		executor := &mockExecutor{
			do: func(ctx context.Context, req *httpclient.Request) (*httpclient.Response, error) {
				execCalls++
				return &httpclient.Response{}, nil
			},
		}

		outbound := &mockOutbound{
			transformResponse: func(ctx context.Context, resp *httpclient.Response) (*llm.Response, error) {
				return &llm.Response{
					RequestType: llm.RequestTypeEmbedding,
					Embedding: &llm.EmbeddingResponse{
						Object: "list",
						Data: []llm.EmbeddingData{{
							Object: "embedding",
							Embedding: llm.Embedding{
								Embedding: []float64{0.1, 0.2, 0.3},
							},
							Index: 0,
						}},
					},
				}, nil
			},
		}

		p := &pipeline{
			Executor:               executor,
			Inbound:                &mockInbound{},
			Outbound:               outbound,
			emptyResponseDetection: true,
		}

		res, err := p.Process(context.Background(), &httpclient.Request{})
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, 1, execCalls)
	})
}

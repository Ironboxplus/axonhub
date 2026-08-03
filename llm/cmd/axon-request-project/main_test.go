package main

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"
)

func TestProjectResponsesRequestUsesProductionPipeline(t *testing.T) {
	t.Parallel()
	request := []byte(`{
		"model":"old-model",
		"input":[
			{"type":"additional_tools","role":"developer","tools":[
				{"type":"namespace","name":"collaboration","description":"Tools for spawning and managing sub-agents.","tools":[{"type":"function","name":"spawn_agent","description":"Spawn an agent.","parameters":{"type":"object","properties":{},"additionalProperties":false}}]}
			]},
			{"type":"message","role":"user","content":[
				{"type":"input_text","text":"CORPUS_MARKER_A41C"},
				{"type":"input_image","image_url":"data:image/png;base64,iVBORw0KGgo="},
				{"type":"input_file","file_data":"data:application/pdf;base64,JVBERi0xLjQ=","filename":"report.pdf"}
			]}
		],
		"tools":[{"type":"function","name":"lookup","description":"Lookup a marker.","parameters":{"type":"object","properties":{},"additionalProperties":false}}],
		"stream":true
	}`)

	for _, target := range []string{"chat", "anthropic"} {
		target := target
		t.Run(target, func(t *testing.T) {
			t.Parallel()
			projected, err := projectResponsesRequest(context.Background(), request, target, "axon-live-chat-out")
			if err != nil {
				t.Fatalf("project Responses to %s: %v", target, err)
			}
			if !json.Valid(projected) {
				t.Fatalf("invalid projected JSON: %s", projected)
			}
			for _, marker := range [][]byte{[]byte("axon-live-chat-out"), []byte("CORPUS_MARKER_A41C"), []byte("report.pdf")} {
				if !bytes.Contains(projected, marker) {
					t.Fatalf("projected %s request lost %q: %s", target, marker, projected)
				}
			}
		})
	}
}

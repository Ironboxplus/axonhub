package llm

import (
	"context"
	"testing"
)

func TestConversionTraceConstructorsAndAppendCoverCriticalBranches(t *testing.T) {
	if ConversionTraceEnabled(nil) || ConversionTraceEnabled(context.Background()) {
		t.Fatal("conversion timing must be disabled without opt-in")
	}
	if !ConversionTraceEnabled(WithConversionTrace(context.Background())) {
		t.Fatal("conversion timing opt-in was ignored")
	}

	if NewConversionDebugTrace(nil, 1) != nil || NewConversionDebugTrace(context.Background(), 1) != nil {
		t.Fatal("sampled trace must require a debug context")
	}
	defaulted := NewConversionDebugTrace(WithConversionDebugTrace(nil, []byte("default"), 0), -1)
	if defaulted == nil || defaulted.Mode != ConversionEvidenceSampled || cap(defaulted.Actions) != 0 {
		t.Fatalf("default sampled trace = %#v", defaulted)
	}
	clamped := NewConversionDebugTrace(
		WithConversionDebugTrace(context.Background(), []byte("clamped"), MaxConversionDebugActions+1),
		MaxConversionDebugActions+1,
	)
	if clamped == nil || cap(clamped.Actions) != MaxConversionDebugActions {
		t.Fatalf("sampled cap was not clamped: %#v", clamped)
	}

	required := NewRequiredConversionDebugTrace(-1)
	if required == nil || required.Mode != ConversionEvidenceRequired || cap(required.Actions) != 0 {
		t.Fatalf("required trace = %#v", required)
	}
	var nilTrace *ConversionDebugTrace
	nilTrace.Append(ConversionActionTrace{}, nil)
	if nilTrace.Clone() != nil {
		t.Fatal("nil trace clone must stay nil")
	}

	action := ConversionActionTrace{
		Direction: ConversionDirectionRequest, ObjectKind: "input_item", ObjectID: "input[0]",
		FieldPath: "input[0]", Stage: ConversionStageRequestPlanning, Action: "unknown",
		Strategy: "unavailable", Reason: "no_strategy", Result: ConversionResultUnknown,
		Severity: ConversionSeverityCritical,
	}
	for index := 0; index < DefaultRequiredConversionActions+1; index++ {
		required.Append(action, []byte("input[0]"))
	}
	if len(required.Actions) != DefaultRequiredConversionActions || required.Truncated != 1 ||
		required.Actions[0].Seq != 1 || required.Actions[0].LogicalIDHash != "" {
		t.Fatalf("required append bound = %#v", required)
	}

	sampled := NewConversionDebugTrace(
		WithConversionDebugTrace(context.Background(), []byte("sampled"), 1),
		1,
	)
	sampled.Append(action, []byte("input[0]"))
	sampled.Append(action, []byte("input[1]"))
	if len(sampled.Actions) != 1 || sampled.Truncated != 1 || len(sampled.Actions[0].LogicalIDHash) != 16 {
		t.Fatalf("sampled append = %#v", sampled)
	}

	snapshot := sampled.Clone()
	if snapshot == nil || snapshot.Mode != ConversionEvidenceSampled || len(snapshot.Actions) != 1 ||
		snapshot.Truncated != 1 || snapshot.maxActions != 0 || snapshot.hashLogicalID {
		t.Fatalf("persistence-safe clone = %#v", snapshot)
	}
	snapshot.Actions[0].ObjectID = "changed"
	if sampled.Actions[0].ObjectID == "changed" {
		t.Fatal("clone shares action storage with live trace")
	}
}

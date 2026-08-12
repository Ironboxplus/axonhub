package llm

import (
	"context"
	"strconv"
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

func TestConversionDebugTraceRetainsTailCriticalEvidenceBySeverity(t *testing.T) {
	trace := NewConversionDebugTrace(
		WithConversionDebugTrace(context.Background(), []byte("priority"), 2),
		2,
	)
	if trace == nil {
		t.Fatal("sampled trace is nil")
	}
	info := ConversionActionTrace{ObjectID: "input[0]", Severity: ConversionSeverityInfo}
	warning := ConversionActionTrace{ObjectID: "input[1]", Severity: ConversionSeverityWarning}
	critical := ConversionActionTrace{ObjectID: "input[2000]", Severity: ConversionSeverityCritical, SourceType: "agent_message", SemanticClass: "agent_message_encrypted", RawBytes: 73}
	trace.Append(info, []byte("info"))
	trace.Append(warning, []byte("warning"))
	trace.Append(critical, []byte("critical"))
	if len(trace.Actions) != 2 || trace.Truncated != 1 || trace.InfoTruncated != 1 ||
		trace.WarningTruncated != 0 || trace.CriticalTruncated != 0 {
		t.Fatalf("priority displacement accounting = %#v", trace)
	}
	if trace.Actions[0].Severity != ConversionSeverityWarning || trace.Actions[1].Severity != ConversionSeverityCritical ||
		trace.Actions[1].ObjectID != "input[2000]" || trace.Actions[1].SourceType != "agent_message" ||
		trace.Actions[1].SemanticClass != "agent_message_encrypted" || trace.Actions[1].RawBytes != 73 {
		t.Fatalf("tail critical evidence was not retained: %#v", trace.Actions)
	}

	// Equal/higher-priority retained actions are never displaced. This covers
	// warning and default-info truncation classification as well as the hard cap.
	trace.Append(ConversionActionTrace{Severity: ConversionSeverityWarning}, []byte("later-warning"))
	trace.Append(ConversionActionTrace{}, []byte("later-info"))
	if trace.Truncated != 3 || trace.WarningTruncated != 1 || trace.InfoTruncated != 2 || trace.CriticalTruncated != 0 {
		t.Fatalf("priority rejection accounting = %#v", trace)
	}

	allCritical := NewConversionDebugTrace(
		WithConversionDebugTrace(context.Background(), []byte("critical"), 1),
		1,
	)
	allCritical.Append(ConversionActionTrace{Severity: ConversionSeverityCritical}, []byte("first"))
	allCritical.Append(ConversionActionTrace{Severity: ConversionSeverityCritical}, []byte("second"))
	if len(allCritical.Actions) != 1 || allCritical.Truncated != 1 || allCritical.CriticalTruncated != 1 {
		t.Fatalf("critical hard-cap accounting = %#v", allCritical)
	}

	snapshot := trace.Clone()
	if snapshot == nil || snapshot.InfoTruncated != trace.InfoTruncated ||
		snapshot.WarningTruncated != trace.WarningTruncated || snapshot.CriticalTruncated != trace.CriticalTruncated {
		t.Fatalf("priority counters not cloned: snapshot=%#v original=%#v", snapshot, trace)
	}
}

func TestRequiredConversionEvidenceKeepsBlockerAfterTwoThousandActions(t *testing.T) {
	trace := NewRequiredConversionDebugTrace(2001)
	for index := 0; index < 2000; index++ {
		trace.Append(ConversionActionTrace{
			ObjectID: "input[" + strconv.Itoa(index) + "]", Severity: ConversionSeverityWarning,
		}, []byte("warning"))
	}
	trace.Append(ConversionActionTrace{
		ObjectID: "input[2000]", Severity: ConversionSeverityCritical,
		SourceType: "agent_message", SemanticClass: "agent_message_encrypted",
	}, []byte("agent-message-blocker"))
	if len(trace.Actions) != DefaultRequiredConversionActions || trace.Truncated != 1969 ||
		trace.WarningTruncated != 1969 || trace.CriticalTruncated != 0 {
		t.Fatalf("large required trace accounting = %#v", trace)
	}
	for index := range trace.Actions {
		if trace.Actions[index].ObjectID == "input[2000]" && trace.Actions[index].Severity == ConversionSeverityCritical {
			return
		}
	}
	t.Fatalf("tail blocker disappeared from bounded trace: %#v", trace.Actions)
}

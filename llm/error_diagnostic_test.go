package llm

import (
	"errors"
	"fmt"
	"testing"
)

type safeDiagnosticFixture struct {
	diagnostic ErrorDiagnostic
}

func (fixture safeDiagnosticFixture) Error() string { return "private error text" }

func (fixture safeDiagnosticFixture) SafeDiagnostic() ErrorDiagnostic {
	return fixture.diagnostic
}

func TestErrorDiagnosticFromRequiresExplicitCompleteSafeContract(t *testing.T) {
	want := ErrorDiagnostic{Component: "fixture", Code: "fixture_failed", Message: "Fixture failed safely.", StatusCode: 418}
	wrapped := fmt.Errorf("outer private context: %w", safeDiagnosticFixture{diagnostic: want})
	got := ErrorDiagnosticFrom(wrapped)
	if got == nil || *got != want {
		t.Fatalf("diagnostic = %#v, want %#v", got, want)
	}
	got.Message = "mutated"
	if next := ErrorDiagnosticFrom(wrapped); next == nil || *next != want {
		t.Fatalf("retained diagnostic shared mutable state: %#v", next)
	}

	for name, err := range map[string]error{
		"nil":                nil,
		"ordinary":           errors.New("must stay private"),
		"missing component":  safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Code: "code", Message: "message"}},
		"missing code":       safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Component: "component", Message: "message"}},
		"missing message":    safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Component: "component", Code: "code"}},
		"component too long": safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Component: string(make([]byte, maxErrorDiagnosticComponentBytes+1)), Code: "code", Message: "message"}},
		"code too long":      safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Component: "component", Code: string(make([]byte, maxErrorDiagnosticCodeBytes+1)), Message: "message"}},
		"message too long":   safeDiagnosticFixture{diagnostic: ErrorDiagnostic{Component: "component", Code: "code", Message: string(make([]byte, maxErrorDiagnosticMessageBytes+1))}},
	} {
		t.Run(name, func(t *testing.T) {
			if diagnostic := ErrorDiagnosticFrom(err); diagnostic != nil {
				t.Fatalf("incomplete or unsafe error produced diagnostic %#v", diagnostic)
			}
		})
	}
}

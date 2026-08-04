package llm

import "errors"

const (
	maxErrorDiagnosticComponentBytes = 64
	maxErrorDiagnosticCodeBytes      = 128
	maxErrorDiagnosticMessageBytes   = 512
)

// ErrorDiagnostic is a bounded, payload-free description that an error may
// explicitly opt into exposing to operators. Implementations must never place
// endpoint URLs, headers, credentials, request content, tool arguments, model
// output, or remote free-form messages in these fields.
type ErrorDiagnostic struct {
	Component  string `json:"component"`
	Code       string `json:"code"`
	Message    string `json:"message"`
	StatusCode int    `json:"status_code,omitempty"`
}

// SafeDiagnosticError marks an error whose diagnostic is safe to persist and
// render after ordinary wrapping. Errors that do not implement this interface
// remain classified only by pipeline stage and error class.
type SafeDiagnosticError interface {
	SafeDiagnostic() ErrorDiagnostic
}

// ErrorDiagnosticFrom returns the first explicitly safe diagnostic in an
// error chain. The returned value is copied so callers may retain it without
// sharing mutable state with the source error.
func ErrorDiagnosticFrom(err error) *ErrorDiagnostic {
	if err == nil {
		return nil
	}
	var diagnosticError SafeDiagnosticError
	if !errors.As(err, &diagnosticError) {
		return nil
	}
	diagnostic := diagnosticError.SafeDiagnostic()
	if diagnostic.Component == "" || len(diagnostic.Component) > maxErrorDiagnosticComponentBytes ||
		diagnostic.Code == "" || len(diagnostic.Code) > maxErrorDiagnosticCodeBytes ||
		diagnostic.Message == "" || len(diagnostic.Message) > maxErrorDiagnosticMessageBytes {
		return nil
	}
	return &diagnostic
}

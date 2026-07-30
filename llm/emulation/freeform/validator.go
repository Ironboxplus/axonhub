package freeform

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/looplj/axonhub/llm"
)

const (
	maxGrammarBytes = 256 * 1024
	maxInputBytes   = 8 * 1024 * 1024
)

var (
	ErrInvalidDefinition = errors.New("invalid custom tool grammar definition")
	ErrInvalidGrammar    = errors.New("custom tool grammar cannot be compiled")
	ErrResourceLimit     = errors.New("custom tool grammar resource limit exceeded")
	ErrConstraintEngine  = errors.New("custom tool grammar engine failed")
	ErrValidatorClosed   = errors.New("custom tool grammar validator is closed")
)

// Validator is a compiled, request-scoped custom-tool grammar. The grammar is
// retained only inside the embedded matcher and is never exposed to tracing.
type Validator struct {
	engine *wasmEngine
	handle uint32

	mu     sync.Mutex
	closed bool
}

// Compile creates a validator for an OpenAI custom-tool grammar. Text-format
// custom tools have no constraint and should not call Compile.
func Compile(ctx context.Context, definition *llm.FreeformDefinition) (*Validator, error) {
	if definition == nil || definition.Format != "grammar" ||
		(definition.Syntax != "lark" && definition.Syntax != "regex") || definition.Definition == "" {
		return nil, ErrInvalidDefinition
	}
	if len(definition.Definition) > maxGrammarBytes {
		return nil, ErrResourceLimit
	}
	engine, err := sharedEngine(ctx)
	if err != nil {
		return nil, err
	}
	handle, err := engine.compile(ctx, definition.Syntax, definition.Definition)
	if err != nil {
		return nil, err
	}
	return &Validator{engine: engine, handle: handle}, nil
}

// Validate reports whether the entire input is accepted by the compiled
// grammar. A mismatch is a normal false result; engine failures are errors.
func (validator *Validator) Validate(ctx context.Context, input string) (bool, error) {
	if validator == nil {
		return false, ErrInvalidDefinition
	}
	if len(input) > maxInputBytes {
		return false, ErrResourceLimit
	}
	validator.mu.Lock()
	defer validator.mu.Unlock()
	if validator.closed {
		return false, ErrValidatorClosed
	}
	return validator.engine.validate(ctx, validator.handle, input)
}

// Close releases the compiled matcher. It is safe to call more than once.
func (validator *Validator) Close(ctx context.Context) error {
	if validator == nil {
		return nil
	}
	validator.mu.Lock()
	defer validator.mu.Unlock()
	if validator.closed {
		return nil
	}
	if err := validator.engine.free(ctx, validator.handle); err != nil {
		return fmt.Errorf("release custom tool grammar: %w", err)
	}
	validator.closed = true
	validator.handle = 0
	return nil
}

package freeform

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/looplj/axonhub/llm"
	"github.com/stretchr/testify/require"
	"github.com/tetratelabs/wazero"
)

const officialMathLark = `
start: expr
expr: term (SP ADD SP term)* -> add
    | term
term: factor (SP MUL SP factor)* -> mul
    | factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
%import common.INT
`

const codexApplyPatchLark = `
start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
`

const officialTimestampRegex = `^(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+(?P<year>\d{4})\s+at\s+(?P<hour>0?[1-9]|1[0-2])(?P<ampm>AM|PM)$`

func TestOfficialLarkGrammarValidatesEntireInput(t *testing.T) {
	validator := compileForTest(t, "lark", officialMathLark)
	requireValidation(t, validator, "4 + 4", true)
	requireValidation(t, validator, "4 + 4 trailing", false)
	requireValidation(t, validator, "4++4", false)
}

func TestOfficialRustRegexGrammarValidatesEntireInput(t *testing.T) {
	validator := compileForTest(t, "regex", officialTimestampRegex)
	requireValidation(t, validator, "August 7th 2025 at 10AM", true)
	requireValidation(t, validator, "August 7th 2025 at 10AM trailing", false)
	requireValidation(t, validator, "2025-08-07 10:00", false)
}

func TestCodexApplyPatchGrammarUsesRealLarkFeatures(t *testing.T) {
	validator := compileForTest(t, "lark", codexApplyPatchLark)
	valid := "*** Begin Patch\n*** Add File: nested/new.txt\n+created\n*** Delete File: obsolete.txt\n*** Update File: existing.txt\n@@\n-old\n+new\n*** End Patch"
	invalid := "*** Begin Patch\n*** Frobnicate File: existing.txt\n*** End Patch"
	requireValidation(t, validator, valid, true)
	requireValidation(t, validator, invalid, false)
}

func TestInvalidGrammarIsRejectedWithoutEchoingDefinition(t *testing.T) {
	definition := &llm.FreeformDefinition{Format: "grammar", Syntax: "lark", Definition: "start: ("}
	validator, err := Compile(context.Background(), definition)
	require.Nil(t, validator)
	require.ErrorIs(t, err, ErrInvalidGrammar)
	require.NotContains(t, err.Error(), definition.Definition)
}

func TestValidatorCanBeUsedConcurrentlyAndClosed(t *testing.T) {
	validator := compileForTest(t, "regex", `^[a-z]+$`)
	const workers = 16
	var wait sync.WaitGroup
	errorsSeen := make(chan error, workers)
	for index := 0; index < workers; index++ {
		index := index
		wait.Add(1)
		go func() {
			defer wait.Done()
			valid, err := validator.Validate(context.Background(), "axon")
			if err != nil {
				errorsSeen <- err
				return
			}
			if !valid {
				errorsSeen <- errors.New("valid input was rejected")
			}
			_ = index
		}()
	}
	wait.Wait()
	close(errorsSeen)
	for err := range errorsSeen {
		require.NoError(t, err)
	}
	require.NoError(t, validator.Close(context.Background()))
	require.NoError(t, validator.Close(context.Background()))
	_, err := validator.Validate(context.Background(), "axon")
	require.ErrorIs(t, err, ErrValidatorClosed)
}

func compileForTest(t *testing.T, syntax, grammar string) *Validator {
	t.Helper()
	validator, err := Compile(context.Background(), &llm.FreeformDefinition{
		Format: "grammar", Syntax: syntax, Definition: grammar,
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, validator.Close(context.Background())) })
	return validator
}

func requireValidation(t *testing.T, validator *Validator, input string, expected bool) {
	t.Helper()
	valid, err := validator.Validate(context.Background(), input)
	require.NoError(t, err)
	require.Equal(t, expected, valid)
}

func BenchmarkEngineColdStart(b *testing.B) {
	ctx := context.Background()
	b.ReportAllocs()
	b.SetBytes(int64(len(embeddedEngine)))
	for b.Loop() {
		engine, err := newWASMEngine(ctx)
		if err != nil {
			b.Fatal(err)
		}
		if err := engine.runtime.Close(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEngineColdStartInterpreter(b *testing.B) {
	ctx := context.Background()
	config := wazero.NewRuntimeConfigInterpreter().
		WithMemoryLimitPages(1024).
		WithDebugInfoEnabled(false)
	b.ReportAllocs()
	b.SetBytes(int64(len(embeddedEngine)))
	for b.Loop() {
		engine, err := newWASMEngineWithConfig(ctx, config)
		if err != nil {
			b.Fatal(err)
		}
		if err := engine.runtime.Close(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInterpreterValidateCodexApplyPatch(b *testing.B) {
	ctx := context.Background()
	config := wazero.NewRuntimeConfigInterpreter().
		WithMemoryLimitPages(1024).
		WithDebugInfoEnabled(false)
	engine, err := newWASMEngineWithConfig(ctx, config)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = engine.runtime.Close(ctx) })
	handle, err := engine.compile(ctx, "lark", codexApplyPatchLark)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = engine.free(ctx, handle) })
	input := "*** Begin Patch\n*** Update File: existing.txt\n@@\n-old\n+new\n*** End Patch"
	b.ReportAllocs()
	b.SetBytes(int64(len(input)))
	b.ResetTimer()
	for b.Loop() {
		valid, validateErr := engine.validate(ctx, handle, input)
		if validateErr != nil || !valid {
			b.Fatalf("valid=%v err=%v", valid, validateErr)
		}
	}
}

func BenchmarkCompileCodexApplyPatchGrammar(b *testing.B) {
	ctx := context.Background()
	definition := &llm.FreeformDefinition{Format: "grammar", Syntax: "lark", Definition: codexApplyPatchLark}
	warmup, err := Compile(ctx, definition)
	if err != nil {
		b.Fatal(err)
	}
	if err := warmup.Close(ctx); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		validator, err := Compile(ctx, definition)
		if err != nil {
			b.Fatal(err)
		}
		if err := validator.Close(ctx); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkValidateCodexApplyPatch(b *testing.B) {
	ctx := context.Background()
	validator, err := Compile(ctx, &llm.FreeformDefinition{
		Format: "grammar", Syntax: "lark", Definition: codexApplyPatchLark,
	})
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = validator.Close(ctx) })
	input := "*** Begin Patch\n*** Update File: existing.txt\n@@\n-old\n+new\n*** End Patch"
	b.ReportAllocs()
	b.SetBytes(int64(len(input)))
	b.ResetTimer()
	for b.Loop() {
		valid, validateErr := validator.Validate(ctx, input)
		if validateErr != nil || !valid {
			b.Fatalf("valid=%v err=%v", valid, validateErr)
		}
	}
}

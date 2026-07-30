package freeform

import (
	"context"
	"crypto/sha256"
	_ "embed"
	"errors"
	"fmt"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

const embeddedEngineSHA256 = "890ef480d4f732806099b4f7cd25254c7a4fc25a84bf34499662b66c9b169dcc"

//go:embed llguidance.wasm
var embeddedEngine []byte

type wasmEngine struct {
	mu sync.Mutex

	runtime          wazero.Runtime
	module           api.Module
	memory           api.Memory
	alloc            api.Function
	dealloc          api.Function
	compileFunction  api.Function
	validateFunction api.Function
	freeFunction     api.Function
}

var engineSingleton struct {
	once   sync.Once
	engine *wasmEngine
	err    error
}

func sharedEngine(ctx context.Context) (*wasmEngine, error) {
	engineSingleton.once.Do(func() {
		// Engine initialization is process-scoped and must not be permanently
		// poisoned by cancellation of the first request that happens to use it.
		engineSingleton.engine, engineSingleton.err = newWASMEngine(context.WithoutCancel(ctx))
	})
	return engineSingleton.engine, engineSingleton.err
}

func newWASMEngine(ctx context.Context) (*wasmEngine, error) {
	config := wazero.NewRuntimeConfigCompiler().
		WithMemoryLimitPages(1024).
		WithDebugInfoEnabled(false)
	return newWASMEngineWithConfig(ctx, config)
}

func newWASMEngineWithConfig(ctx context.Context, config wazero.RuntimeConfig) (*wasmEngine, error) {
	digest := fmt.Sprintf("%x", sha256.Sum256(embeddedEngine))
	if digest != embeddedEngineSHA256 {
		return nil, errors.New("custom tool grammar engine integrity check failed")
	}
	runtime := wazero.NewRuntimeWithConfig(ctx, config)
	closeRuntime := true
	defer func() {
		if closeRuntime {
			_ = runtime.Close(context.Background())
		}
	}()
	if _, err := wasi_snapshot_preview1.Instantiate(ctx, runtime); err != nil {
		return nil, fmt.Errorf("initialize custom tool grammar WASI: %w", err)
	}
	compiled, err := runtime.CompileModule(ctx, embeddedEngine)
	if err != nil {
		return nil, fmt.Errorf("compile custom tool grammar engine: %w", err)
	}
	module, err := runtime.InstantiateModule(ctx, compiled, wazero.NewModuleConfig().WithName("axon_llguidance"))
	if err != nil {
		return nil, fmt.Errorf("instantiate custom tool grammar engine: %w", err)
	}
	engine := &wasmEngine{
		runtime:          runtime,
		module:           module,
		memory:           module.Memory(),
		alloc:            module.ExportedFunction("axon_alloc"),
		dealloc:          module.ExportedFunction("axon_dealloc"),
		compileFunction:  module.ExportedFunction("axon_compile"),
		validateFunction: module.ExportedFunction("axon_validate"),
		freeFunction:     module.ExportedFunction("axon_free"),
	}
	if engine.memory == nil || engine.alloc == nil || engine.dealloc == nil || engine.compileFunction == nil ||
		engine.validateFunction == nil || engine.freeFunction == nil {
		return nil, errors.New("custom tool grammar engine ABI is incomplete")
	}
	closeRuntime = false
	return engine, nil
}

func (engine *wasmEngine) compile(ctx context.Context, syntax, grammar string) (uint32, error) {
	engine.mu.Lock()
	defer engine.mu.Unlock()
	syntaxPointer, syntaxCleanup, err := engine.writeString(ctx, syntax)
	if err != nil {
		return 0, err
	}
	defer syntaxCleanup()
	grammarPointer, grammarCleanup, err := engine.writeString(ctx, grammar)
	if err != nil {
		return 0, err
	}
	defer grammarCleanup()
	result, err := engine.compileFunction.Call(ctx,
		uint64(syntaxPointer), uint64(len(syntax)), uint64(grammarPointer), uint64(len(grammar)),
	)
	if err != nil {
		return 0, fmt.Errorf("compile custom tool grammar: %w", ErrConstraintEngine)
	}
	code := resultCode(result)
	if code <= 0 {
		return 0, engineResultError(code)
	}
	return uint32(code), nil
}

func (engine *wasmEngine) validate(ctx context.Context, handle uint32, input string) (bool, error) {
	engine.mu.Lock()
	defer engine.mu.Unlock()
	inputPointer, cleanup, err := engine.writeString(ctx, input)
	if err != nil {
		return false, err
	}
	defer cleanup()
	result, err := engine.validateFunction.Call(ctx, uint64(handle), uint64(inputPointer), uint64(len(input)))
	if err != nil {
		return false, fmt.Errorf("validate custom tool input: %w", ErrConstraintEngine)
	}
	code := resultCode(result)
	switch code {
	case 1:
		return true, nil
	case 0:
		return false, nil
	default:
		return false, engineResultError(code)
	}
}

func (engine *wasmEngine) free(ctx context.Context, handle uint32) error {
	engine.mu.Lock()
	defer engine.mu.Unlock()
	result, err := engine.freeFunction.Call(ctx, uint64(handle))
	if err != nil {
		return ErrConstraintEngine
	}
	if code := resultCode(result); code != 0 {
		return engineResultError(code)
	}
	return nil
}

func (engine *wasmEngine) writeString(ctx context.Context, value string) (uint32, func(), error) {
	if value == "" {
		return 0, func() {}, nil
	}
	result, err := engine.alloc.Call(ctx, uint64(len(value)))
	if err != nil || len(result) != 1 {
		return 0, nil, ErrConstraintEngine
	}
	pointer := uint32(result[0])
	if pointer == 0 || !engine.memory.WriteString(pointer, value) {
		if pointer != 0 {
			_, _ = engine.dealloc.Call(context.Background(), uint64(pointer), uint64(len(value)))
		}
		return 0, nil, ErrResourceLimit
	}
	return pointer, func() {
		_, _ = engine.dealloc.Call(context.Background(), uint64(pointer), uint64(len(value)))
	}, nil
}

func resultCode(result []uint64) int32 {
	if len(result) != 1 {
		return -4
	}
	return int32(uint32(result[0]))
}

func engineResultError(code int32) error {
	switch code {
	case -1:
		return ErrInvalidDefinition
	case -2:
		return ErrInvalidGrammar
	case -3:
		return ErrResourceLimit
	default:
		return ErrConstraintEngine
	}
}

# Custom-tool grammar validation

This package validates OpenAI Responses custom-tool inputs when Axon must lower a custom tool to Chat Completions or Anthropic Messages.

## Boundary

- `validator.go` is the request-scoped Go API.
- `engine.go` owns one process-wide wazero module and serializes its small handle ABI.
- `engine/` is a reproducible Rust wrapper pinned to `llguidance 1.7.6`.
- `llguidance.wasm` is generated and checked in. Normal Go builds do not require Rust, cgo, a subprocess, or a network fetch.

Protocol planning, correction rounds, stream holding, fallback policy, counters, and timing remain in Go. The embedded engine receives only a grammar at compile time and a candidate input at validation time. Neither is returned from this package or admitted to observation records.

## Resource and concurrency policy

- The WASM module is lazy: requests without a grammar never initialize it.
- Module integrity is checked once with a pinned SHA-256.
- WASM linear memory is capped at 64 MiB.
- Grammar definitions are capped at 256 KiB; candidate inputs at 8 MiB.
- A grammar compiles once per request-scoped validator. Validation clones the compiled matcher.
- Engine calls are serialized because the module owns a handle table. Validator methods are safe for concurrent callers.
- Call `Validator.Close` at request completion to release its matcher handle.

## Updating

1. Review the upstream LLGuidance release and OpenAI custom-tool documentation changes.
2. Change the exact dependency version in `engine/Cargo.toml` and regenerate `Cargo.lock`.
3. Run `engine/build.ps1` or `engine/build.sh`.
4. Update the artifact hash in `engine.go` and `engine/THIRD_PARTY.md`.
5. Run the official Lark, official Rust-regex, Codex apply-patch, concurrency, resource-limit, and benchmark tests.

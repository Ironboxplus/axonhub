# LLGuidance engine provenance

The embedded grammar matcher is built from `llguidance` version `1.7.6`, pinned exactly in `Cargo.toml` and `Cargo.lock`.

- Upstream: https://github.com/guidance-ai/llguidance
- Package: https://crates.io/crates/llguidance/1.7.6
- License: MIT
- Role: compile and validate the same Lark-variant and Rust-regex grammar formats documented for OpenAI custom tools.

The Rust wrapper in this directory only supplies a bounded handle-based ABI for Go. Protocol conversion, retries, observability, and lifecycle policy remain in Go.

The generated `../llguidance.wasm` is checked in so normal Axon builds do not require Rust. Rebuild it with `build.ps1` or `build.sh`; review the emitted SHA-256 when updating the pinned engine.

Current generated artifact:

- Rust toolchain: `1.97.1`
- Size: `1,854,215` bytes
- SHA-256: `890ef480d4f732806099b4f7cd25254c7a4fc25a84bf34499662b66c9b169dcc`

Go executes the module with `github.com/tetratelabs/wazero` version `1.12.0`. The runtime is pure Go; production hosts do not load a native library or launch a helper process.

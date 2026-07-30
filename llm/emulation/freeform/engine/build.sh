#!/usr/bin/env sh
set -eu

engine_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_path=$(dirname -- "$engine_root")/llguidance.wasm

rustup target add wasm32-wasip1
cargo build --locked --release --target wasm32-wasip1 --manifest-path "$engine_root/Cargo.toml"
cp "$engine_root/target/wasm32-wasip1/release/axon_llguidance.wasm" "$output_path"
sha256sum "$output_path"

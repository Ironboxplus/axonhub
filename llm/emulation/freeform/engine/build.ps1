$ErrorActionPreference = 'Stop'

$engineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputPath = Join-Path (Split-Path -Parent $engineRoot) 'llguidance.wasm'

rustup target add wasm32-wasip1
cargo build --locked --release --target wasm32-wasip1 --manifest-path (Join-Path $engineRoot 'Cargo.toml')
Copy-Item -Force -LiteralPath (Join-Path $engineRoot 'target\wasm32-wasip1\release\axon_llguidance.wasm') -Destination $outputPath
Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath

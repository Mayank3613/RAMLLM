# llama.cpp Backend Compatibility Notes

Verified on:
- Host: macOS Darwin arm64 (Apple Silicon)
- `llama-server` / `llama-cli`: version 0.6.0 (build 11429, commit d81235049) built with AppleClang 21.0.0.21000334

## Flag Discoveries & Adaptations

1. **Memory Mapping & Locking**:
   - Older documents cite `--no-mmap` and `--mlock`.
   - Verified flag: `-lm, --load-mode MODE`
     - `auto`: default mmap
     - `none`: no mmap (fully buffer in RAM)
     - `mmap`: standard memory-map
     - `mlock`: pin pages to RAM (prevent swap/compression)
     - `mmap+mlock`: mmap with pinned pages
     - `dio`: direct I/O where supported
   - RAMLLM adaptation: When `mmap=False` or `no_mmap=True`, use `--load-mode none`. When `mlock=True`, use `--load-mode mlock`.

2. **GPU Layer Offload**:
   - Flag: `-ngl, --gpu-layers, --n-gpu-layers N`
   - Description: Maximum number of layers to store in VRAM.

3. **KV Cache Control**:
   - Offload: `-kvo, --kv-offload` / `-nkvo, --no-kv-offload`
   - Quantization types:
     - `-ctk, --cache-type-k TYPE` (supports `f16`, `q8_0`, `q4_0`, etc.)
     - `-ctv, --cache-type-v TYPE` (supports `f16`, `q8_0`, `q4_0`, etc.)
   - Context size: `-c, --ctx-size N`

4. **MoE Expert Routing & Offload**:
   - Flag: `-cmoe, --cpu-moe`: keep all Mixture of Experts (MoE) weights in CPU RAM.
   - Flag: `-ncmoe, --n-cpu-moe N`: keep MoE weights of the first N layers in CPU RAM.
   - Flag: `-ot, --override-tensor <tensor pattern>=<buffer type>,...` for fine-grained tensor placement.

5. **Multi-GPU Placement**:
   - `-sm, --split-mode {none,layer,row,tensor}`
   - `-ts, --tensor-split N0,N1,N2,...`
   - `-mg, --main-gpu INDEX`

6. **Flash Attention**:
   - `-fa, --flash-attn [on|off|auto]` (default: `auto`)

7. **Server Process Endpoints**:
   - `--host HOST`: listen host (default `127.0.0.1`)
   - `--port PORT`: listen port (default `8080`)
   - `-m, --model FNAME`: model path

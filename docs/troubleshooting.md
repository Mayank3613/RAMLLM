# RAMLLM Troubleshooting Guide

Every error in RAMLLM is classified with a unique code (e.g., `RAM-E012`) and accompanied by clear causes, suggested remedies, and deterministic exit codes.

---

## Quick Diagnostics

Before troubleshooting specific error codes, try running RAMLLM's automatic diagnostics:

```bash
# Run comprehensive system diagnostics
ramllm doctor

# Automatically fix missing backends or permissions
ramllm doctor --fix

# Look up any error code offline
ramllm explain RAM-E012
```

Sample diagnostics report:
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│ RAMLLM Diagnostic Doctor                                                     │
╰──────────────────────────────────────────────────────────────────────────────╯
✔ Storage: Directories
  RAMLLM storage initialized at ~/.local/share/ramllm
✔ Backend: llama.cpp Engine
  Found via path at `/usr/local/bin/llama-server` (GPU Offload, KV Quantization, MoE CPU Offload, Flash Attention)
! OS & Enforcement: Memory Governor Enforcement
  Soft monitoring only — macOS lacks kernel cgroups; RAMLLM uses 1 Hz RSS sampling and proactive tiered recovery.
✔ Hardware: Available RAM
  14.2 GB available (32.0 GB total)
✔ Catalog: Model Catalog Entries
  3 model(s) registered and verified on disk

✔ Everything looks good. Try `ramllm chat`.
```

To see complete Python tracebacks and detailed debug logs:
```bash
ramllm --debug <command>
```
Debug logs are automatically written to `<data_dir>/logs/ramllm.log`.

---

## Exit Codes Reference

| Exit Code | Constant | Meaning |
|---|---|---|
| `0` | `EXIT_OK` | Success |
| `1` | `EXIT_GENERIC` | Generic runtime error or file corruption |
| `2` | `EXIT_USAGE` | Invalid command arguments or invalid configuration |
| `3` | `EXIT_INFEASIBLE` | Model is infeasible under memory budgets |
| `4` | `EXIT_BACKEND` | Inference engine (`llama-server`) missing, outdated, or crashed |
| `5` | `EXIT_NETWORK` | Network connection error, timeout, or gated model auth failure |
| `6` | `EXIT_INTERRUPTED` | Execution interrupted by user (`Ctrl-C` / `SIGINT`) |

---

## Error Catalog (RAM-E001 – RAM-E015)

### RAM-E001: Backend Engine Missing
- **Title:** Backend Engine Missing
- **Exit Code:** `4` (`EXIT_BACKEND`)
- **Cause:** The underlying `llama-server` engine executable could not be found in system PATH or RAMLLM managed backend storage.
- **Remedies:**
  1. Install the managed native engine automatically:
     ```bash
     ramllm backend install
     ```
  2. Or register an existing local binary:
     ```bash
     ramllm backend use /path/to/llama-server
     ```
  3. Or run doctor auto-repair:
     ```bash
     ramllm doctor --fix
     ```

---

### RAM-E002: Backend Engine Outdated
- **Title:** Backend Engine Outdated
- **Exit Code:** `4` (`EXIT_BACKEND`)
- **Cause:** The installed engine build does not meet the minimum version or feature requirements needed for requested operations (such as KV cache quantization or expert offloading).
- **Remedies:**
  1. Update your managed engine:
     ```bash
     ramllm backend update
     ```
  2. Repair with doctor:
     ```bash
     ramllm doctor --fix
     ```

---

### RAM-E005: Download Failed
- **Title:** Download Failed
- **Exit Code:** `5` (`EXIT_NETWORK`)
- **Cause:** Network connection dropped, HTTP request timed out, or remote server returned an error while downloading model weights.
- **Remedies:**
  1. Check your internet connection and proxy/firewall settings.
  2. Verify you have enough disk space before downloading:
     ```bash
     ramllm models du
     ```
  3. If downloading a gated Hugging Face repository, set your token:
     ```bash
     export HF_TOKEN="hf_..."
     ```
  4. Resume the download (RAMLLM automatically resumes partial `.part` downloads):
     ```bash
     ramllm models pull <model>
     ```

---

### RAM-E006: File Integrity Verification Failed
- **Title:** File Integrity Verification Failed
- **Exit Code:** `1` (`EXIT_GENERIC`)
- **Cause:** The SHA-256 checksum of the downloaded model file does not match the expected hash published in the curated registry.
- **Remedies:**
  1. The download may have been corrupted in transit or interrupted.
  2. Re-download the model:
     ```bash
     ramllm models pull <model>
     ```
  3. Delete the corrupted file:
     ```bash
     ramllm models remove <model>
     ```

---

### RAM-E007: Port Already In Use
- **Title:** Port Already In Use
- **Exit Code:** `2` (`EXIT_USAGE`)
- **Cause:** The requested network port (default: 8080) is currently occupied by another process or an existing background RAMLLM server.
- **Remedies:**
  1. Check running RAMLLM engines:
     ```bash
     ramllm ps
     ```
  2. Stop any existing background servers:
     ```bash
     ramllm stop <id>
     # Or stop all
     ramllm stop --all
     ```
  3. Or specify a different port:
     ```bash
     ramllm serve <model> --port 8081
     ```

---

### RAM-E008: Insufficient Disk Space
- **Title:** Insufficient Disk Space
- **Exit Code:** `1` (`EXIT_GENERIC`)
- **Cause:** Target drive does not have enough free space to store model weights or engine binaries.
- **Remedies:**
  1. Inspect model disk usage:
     ```bash
     ramllm models du
     ```
  2. Clean temporary files, logs, or unneeded models:
     ```bash
     ramllm clean
     ```
  3. Configure RAMLLM to store models on a larger external drive:
     ```bash
     ramllm config set models_dir /Volumes/ExternalDrive/models
     ```

---

### RAM-E009: Hugging Face Authentication Required
- **Title:** Hugging Face Authentication Required
- **Exit Code:** `5` (`EXIT_NETWORK`)
- **Cause:** The requested model is gated on Hugging Face (e.g., Llama-3, Gemma-2) and requires accepting user license agreements.
- **Remedies:**
  1. Visit the model repository on Hugging Face and accept terms:
     - `https://huggingface.co/<repo>`
  2. Export your Hugging Face user access token:
     ```bash
     export HF_TOKEN="hf_..."
     ```
  3. Or configure your environment variable in RAMLLM:
     ```bash
     ramllm config set hf_token_env HF_TOKEN
     ```

---

### RAM-E010: Model File Corrupted
- **Title:** Model File Corrupted
- **Exit Code:** `1` (`EXIT_GENERIC`)
- **Cause:** The GGUF file header or metadata is truncated, invalid, or cannot be parsed.
- **Remedies:**
  1. Verify the model file:
     ```bash
     ramllm models verify <model>
     ```
  2. Clean broken files:
     ```bash
     ramllm clean --partials
     ```
  3. Re-download the model:
     ```bash
     ramllm models pull <model>
     ```

---

### RAM-E011: Unsupported Model Architecture
- **Title:** Unsupported Model Architecture
- **Exit Code:** `2` (`EXIT_USAGE`)
- **Cause:** The GGUF model architecture is not supported by the current backend build.
- **Remedies:**
  1. Check models verified to work on RAMLLM:
     ```bash
     ramllm models search
     ```
  2. Upgrade your engine backend to the latest version:
     ```bash
     ramllm backend update
     ```

---

### RAM-E012: Not Enough Memory For This Model
- **Title:** Not Enough Memory For This Model
- **Exit Code:** `3` (`EXIT_INFEASIBLE`)
- **Cause:** The model's weights and required KV cache exceed your computer's total available RAM and GPU VRAM headroom.
- **Remedies:**
  1. Find models that fit your machine comfortably:
     ```bash
     ramllm recommend
     ```
  2. Use a more aggressive memory mode or lower quantization:
     ```bash
     ramllm plan <model> --mode max
     ```
  3. Reduce the target context window:
     ```bash
     ramllm chat <model> --context 2048
     ```

---

### RAM-E013: GPU Driver Communication Error
- **Title:** GPU Driver Communication Error
- **Exit Code:** `4` (`EXIT_BACKEND`)
- **Cause:** Failed to initialize GPU acceleration via Apple Metal, NVIDIA CUDA, or Vulkan.
- **Remedies:**
  1. Run diagnostics to probe GPU status:
     ```bash
     ramllm doctor
     ```
  2. Run temporarily in CPU-only mode:
     ```bash
     ramllm run <model> --cpu-only
     ```
  3. Check GPU drivers and reinstall backend:
     ```bash
     ramllm backend install --force
     ```

---

### RAM-E014: Permission Denied
- **Title:** Permission Denied
- **Exit Code:** `1` (`EXIT_GENERIC`)
- **Cause:** The operating system denied read or write access to a model file, configuration file, or log directory.
- **Remedies:**
  1. Verify ownership and permissions of your `RAMLLM_HOME` directory.
  2. Run `ramllm doctor` to verify file permissions across all managed paths.

---

### RAM-E015: Invalid Configuration
- **Title:** Invalid Configuration
- **Exit Code:** `2` (`EXIT_USAGE`)
- **Cause:** `config.toml` contains syntax errors or invalid values for known settings.
- **Remedies:**
  1. View your active configuration:
     ```bash
     ramllm config list
     ```
  2. Reset configuration to defaults:
     ```bash
     ramllm config reset
     ```
  3. Open configuration in your editor to fix syntax:
     ```bash
     ramllm config edit
     ```

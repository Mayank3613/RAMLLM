"""Glossary terms, explanations, and error code descriptions in plain language."""

GLOSSARY_TERMS: dict[str, dict[str, str]] = {
    "vram": {
        "title": "Video RAM (VRAM)",
        "explanation": "High-speed memory located directly on your graphics card (GPU). Weights placed in VRAM run at maximum speed.",
        "tip": "Fitting all model layers into VRAM gives the fastest responses (typically 30–100+ words per second).",
    },
    "ram": {
        "title": "System RAM",
        "explanation": "Your computer's main system memory. It is much larger than GPU VRAM but about 5–10x slower for large language models.",
        "tip": "RAMLLM offloads warm layers or sparse MoE experts to RAM so you can run models larger than your GPU.",
    },
    "quantization": {
        "title": "Quantization (e.g. Q4_K_M, Q8_0)",
        "explanation": "A technique that compresses model numbers from 16-bit to 4-bit or 8-bit integers, reducing size by up to 75% with barely noticeable changes in answer quality.",
        "tip": "Q4_K_M is the community standard balance of high quality and compact size.",
    },
    "kv-cache": {
        "title": "KV Cache (Context Memory)",
        "explanation": "Memory reserved for the model to remember what was said earlier in the conversation. As conversations grow longer, the KV cache grows in size.",
        "tip": "If memory is tight, RAMLLM can compress the KV cache (e.g. q8_0 or q4_0) or reduce context size.",
    },
    "context": {
        "title": "Context Window",
        "explanation": "The total number of words/tokens the model can read and remember in a single session (e.g. 4096, 8192, or 32768 tokens).",
        "tip": "A larger context requires more memory. 4096 or 8192 is ideal for daily chat.",
    },
    "moe": {
        "title": "Mixture-of-Experts (MoE)",
        "explanation": "Models that contain many specialized sub-networks ('experts') but only activate a few per token (e.g. Mixtral, Qwen MoE).",
        "tip": "RAMLLM can keep active attention in fast VRAM while streaming sparse experts from RAM.",
    },
    "layers": {
        "title": "Model Layers",
        "explanation": "The building blocks of a neural network. A 7B model usually has 32 layers. Offloading layers to GPU is proportional.",
        "tip": "Even offloading 16 of 32 layers to GPU provides a significant speedup over pure CPU execution.",
    },
    "verdicts": {
        "title": "Feasibility Verdicts",
        "explanation": (
            "RAMLLM rates plans honestly into four verdicts:\n"
            "  • COMFORTABLE: Model runs fast with plenty of memory headroom (interactive chat).\n"
            "  • CONSTRAINED: Model fits and runs, but with tight memory or moderate speed (usable).\n"
            "  • BATCH_ONLY: Very slow (<1 word/sec); acceptable for background jobs, not live chat.\n"
            "  • INFEASIBLE: Model exceeds physical memory limits and cannot be safely run."
        ),
        "tip": "Always aim for COMFORTABLE or CONSTRAINED plans for interactive usage.",
    },
    "bottleneck": {
        "title": "Performance Bottleneck",
        "explanation": "The exact physical component limiting generation speed. LLM token generation is almost always memory-bandwidth bound.",
        "tip": "VRAM bandwidth gives >50 tok/s; system RAM bandwidth gives 5–15 tok/s; PCIe or disk gives <2 tok/s.",
    },
}

ERROR_EXPLANATIONS: dict[str, str] = {
    "ram-e001": "RAM-E001: Backend engine executable (llama-server) could not be located. Run `ramllm backend install` to automatically download it.",
    "ram-e002": "RAM-E002: Installed backend engine is outdated. Run `ramllm backend update` to download the latest supported build.",
    "ram-e005": "RAM-E005: Model download failed due to network disruption or invalid repository path. Verify internet connection and retry.",
    "ram-e006": "RAM-E006: Model file failed SHA-256 integrity verification. The file is corrupt or incomplete and should be re-downloaded.",
    "ram-e007": "RAM-E007: API server port is occupied. Another RAMLLM server or application is using this port. Use `ramllm ps` or `--port <N>`.",
    "ram-e008": "RAM-E008: Insufficient disk space to download or run model. Check free disk space with `ramllm models du`.",
    "ram-e009": "RAM-E009: Hugging Face authentication required for gated repository. Set HF_TOKEN environment variable.",
    "ram-e010": "RAM-E010: GGUF model file is corrupted or could not be parsed. Re-download or verify with `ramllm models verify`.",
    "ram-e011": "RAM-E011: Unsupported GGUF architecture for llama.cpp engine. Search for supported alternatives with `ramllm models search`.",
    "ram-e012": "RAM-E012: The requested model exceeds available memory budgets. Pick a smaller model with `ramllm recommend` or use a lower quantization.",
    "ram-e013": "RAM-E013: GPU accelerator driver error. Check drivers with `ramllm doctor` or run CPU-only with `--cpu-only`.",
    "ram-e014": "RAM-E014: Operating system permission denied accessing files or directories. Check file permissions or run `ramllm doctor`.",
    "ram-e015": "RAM-E015: Configuration file syntax or value error. Inspect with `ramllm config list` or reset with `ramllm config reset`.",
}


def get_glossary_entry(term: str) -> dict[str, str] | None:
    t = term.strip().lower()
    if t in GLOSSARY_TERMS:
        return GLOSSARY_TERMS[t]
    if t in ERROR_EXPLANATIONS:
        return {"title": f"Error Code {t.upper()}", "explanation": ERROR_EXPLANATIONS[t], "tip": ""}
    return None

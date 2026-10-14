"""Script to populate src/ramllm/data/models_index.json from verified Hugging Face GGUF repositories."""

import datetime
import json
import re
import urllib.request
from pathlib import Path

MODEL_DEFS = [
    {
        "id": "smollm2-135m",
        "name": "SmolLM2 135M Instruct",
        "family": "SmolLM",
        "params_b": 0.135,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["small", "fast", "testing"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/SmolLM2-135M-Instruct-GGUF",
        "notes": "Minimal test model, instant downloads and minimal RAM consumption.",
        "recommended_min_memory_gb": 1,
    },
    {
        "id": "smollm2-360m",
        "name": "SmolLM2 360M Instruct",
        "family": "SmolLM",
        "params_b": 0.36,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["small", "fast", "chat"],
        "license": "Apache-2.0",
        "hf_repo": "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
        "notes": "Lightweight conversational model tailored for resource-constrained devices.",
        "recommended_min_memory_gb": 2,
    },
    {
        "id": "smollm2-1.7b",
        "name": "SmolLM2 1.7B Instruct",
        "family": "SmolLM",
        "params_b": 1.7,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "small", "fast"],
        "license": "Apache-2.0",
        "hf_repo": "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
        "notes": "Compact everyday model capable of reasoning and instruction following.",
        "recommended_min_memory_gb": 4,
    },
    {
        "id": "qwen2.5-0.5b",
        "name": "Qwen 2.5 0.5B Instruct",
        "family": "Qwen",
        "params_b": 0.5,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "small", "fast", "multilingual"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
        "notes": "Ultra-compact multilingual assistant, runs fast on any laptop or phone.",
        "recommended_min_memory_gb": 2,
    },
    {
        "id": "qwen2.5-1.5b",
        "name": "Qwen 2.5 1.5B Instruct",
        "family": "Qwen",
        "params_b": 1.5,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "fast", "multilingual", "coding"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
        "notes": "Fast and versatile small model with strong multilingual skills.",
        "recommended_min_memory_gb": 4,
    },
    {
        "id": "llama-3.2-1b",
        "name": "Llama 3.2 1B Instruct",
        "family": "Llama",
        "params_b": 1.2,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "small", "fast"],
        "license": "Llama 3.2 Community",
        "hf_repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "notes": "Meta's efficient edge model for quick queries and summarize tasks.",
        "recommended_min_memory_gb": 4,
    },
    {
        "id": "llama-3.2-3b",
        "name": "Llama 3.2 3B Instruct",
        "family": "Llama",
        "params_b": 3.2,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "balanced", "general"],
        "license": "Llama 3.2 Community",
        "hf_repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "notes": "High quality-to-size ratio from Meta, ideal for laptops without discrete GPUs.",
        "recommended_min_memory_gb": 6,
    },
    {
        "id": "gemma-2-2b",
        "name": "Gemma 2 2B Instruct",
        "family": "Gemma",
        "params_b": 2.6,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "fast", "general"],
        "license": "Gemma Terms of Use",
        "hf_repo": "bartowski/gemma-2-2b-it-GGUF",
        "notes": "Google's lightweight research-derived assistant for everyday dialogues.",
        "recommended_min_memory_gb": 4,
    },
    {
        "id": "phi-3.5-mini",
        "name": "Phi 3.5 Mini Instruct",
        "family": "Phi",
        "params_b": 3.8,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["reasoning", "coding", "long-context"],
        "license": "MIT",
        "hf_repo": "bartowski/Phi-3.5-mini-instruct-GGUF",
        "notes": "Microsoft's compact powerhouse with 128k context support and strong math/logic.",
        "recommended_min_memory_gb": 6,
    },
    {
        "id": "deepseek-r1-distill-qwen-1.5b",
        "name": "DeepSeek R1 Distill Qwen 1.5B",
        "family": "DeepSeek",
        "params_b": 1.5,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["reasoning", "math", "fast"],
        "license": "MIT",
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF",
        "notes": "Miniature reasoning model trained with chain-of-thought distillation.",
        "recommended_min_memory_gb": 4,
    },
    {
        "id": "qwen2.5-7b",
        "name": "Qwen 2.5 7B Instruct",
        "family": "Qwen",
        "params_b": 7.6,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "coding", "multilingual", "quality"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "notes": "One of the best general-purpose 7B models available for chat and reasoning.",
        "recommended_min_memory_gb": 8,
    },
    {
        "id": "qwen2.5-coder-7b",
        "name": "Qwen 2.5 Coder 7B Instruct",
        "family": "Qwen",
        "params_b": 7.6,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["coding", "quality"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
        "notes": "Dedicated code generation and debugging specialist.",
        "recommended_min_memory_gb": 8,
    },
    {
        "id": "deepseek-r1-distill-qwen-7b",
        "name": "DeepSeek R1 Distill Qwen 7B",
        "family": "DeepSeek",
        "params_b": 7.6,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["reasoning", "math", "quality"],
        "license": "MIT",
        "hf_repo": "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "notes": "Remarkable math, logic, and coding reasoning capabilities from DeepSeek.",
        "recommended_min_memory_gb": 8,
    },
    {
        "id": "llama-3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "family": "Llama",
        "params_b": 8.0,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "general", "quality"],
        "license": "Llama 3.1 Community",
        "hf_repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "notes": "Industry-standard open weight model with broad tool-use and dialogue proficiency.",
        "recommended_min_memory_gb": 10,
    },
    {
        "id": "mistral-7b-v0.3",
        "name": "Mistral 7B Instruct v0.3",
        "family": "Mistral",
        "params_b": 7.2,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "general"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        "notes": "Proven efficient 7B foundation with function calling and 32k context.",
        "recommended_min_memory_gb": 8,
    },
    {
        "id": "gemma-2-9b",
        "name": "Gemma 2 9B Instruct",
        "family": "Gemma",
        "params_b": 9.2,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "quality", "general"],
        "license": "Gemma Terms of Use",
        "hf_repo": "bartowski/gemma-2-9b-it-GGUF",
        "notes": "Top-tier open weights from Google DeepMind with high benchmark scores.",
        "recommended_min_memory_gb": 12,
    },
    {
        "id": "qwen2.5-14b",
        "name": "Qwen 2.5 14B Instruct",
        "family": "Qwen",
        "params_b": 14.7,
        "architecture": "dense",
        "active_params_b": None,
        "tags": ["chat", "coding", "quality"],
        "license": "Apache-2.0",
        "hf_repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
        "notes": "Substantial 14B model balancing deep reasoning with reasonable memory requirements.",
        "recommended_min_memory_gb": 16,
    },
    {
        "id": "mixtral-8x7b-instruct",
        "name": "Mixtral 8x7B Instruct v0.1",
        "family": "Mixtral",
        "params_b": 46.7,
        "architecture": "moe",
        "active_params_b": 12.9,
        "tags": ["moe", "quality", "multilingual"],
        "license": "Apache-2.0",
        "hf_repo": "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
        "notes": "High capability Sparse Mixture-of-Experts with 12.9B active params during generation.",
        "recommended_min_memory_gb": 32,
    },
]

PREFERRED_QUANTS = ["Q4_K_M", "Q5_K_M", "Q8_0", "Q4_K_S", "Q4_0", "Q5_0", "Q6_K", "Q3_K_M"]


def extract_quant_from_filename(filename: str) -> str | None:
    match = re.search(r"[-._]([iI]?[qQ][0-9]_[A-Za-z0-9_]+)\.gguf$", filename)
    if match:
        return match.group(1).upper()
    return None


def fetch_repo_files(repo: str) -> list[dict]:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "RAMLLM/0.2.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        items = json.loads(resp.read().decode())

    files = []
    for item in items:
        if item.get("type") == "file" and item["path"].endswith(".gguf"):
            fname = item["path"]
            quant = extract_quant_from_filename(fname)
            if not quant:
                continue
            sha = item.get("lfs", {}).get("oid", "")
            size = item.get("size", 0)
            files.append(
                {
                    "quant": quant,
                    "size_bytes": size,
                    "hf_repo": repo,
                    "hf_file": fname,
                    "sha256": sha,
                }
            )
    return files


def main() -> None:
    print(f"Fetching real files for {len(MODEL_DEFS)} model repositories from Hugging Face...")
    index_entries = []
    today = datetime.date.today().isoformat()

    for m in MODEL_DEFS:
        repo = m["hf_repo"]
        print(f" -> Querying {repo}...")
        try:
            repo_files = fetch_repo_files(repo)
            # Filter to relevant quants or sort by preference
            filtered_files = [f for f in repo_files if f["quant"] in PREFERRED_QUANTS]
            if not filtered_files:
                filtered_files = repo_files[:6]
            else:
                # Sort according to PREFERRED_QUANTS
                filtered_files.sort(
                    key=lambda f: (
                        PREFERRED_QUANTS.index(f["quant"]) if f["quant"] in PREFERRED_QUANTS else 99
                    )
                )

            entry = {
                "id": m["id"],
                "name": m["name"],
                "family": m["family"],
                "params_b": m["params_b"],
                "architecture": m["architecture"],
                "active_params_b": m["active_params_b"],
                "tags": m["tags"],
                "license": m["license"],
                "files": filtered_files,
                "notes": m["notes"],
                "recommended_min_memory_gb": m["recommended_min_memory_gb"],
                "last_verified": today,
            }
            index_entries.append(entry)
            print(f"    ✔ Found {len(filtered_files)} verified files for {m['id']}")
        except Exception as e:
            print(f"    ✘ Error querying {repo}: {e}")

    output_path = Path("src/ramllm/data/models_index.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": "1.0.0",
                "updated_at": today,
                "models": index_entries,
            },
            f,
            indent=2,
        )
    print(f"\nSuccessfully wrote {len(index_entries)} verified models to {output_path}")


if __name__ == "__main__":
    main()

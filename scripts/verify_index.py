"""Validation script for models_index.json to verify Hugging Face links and detect drift."""

import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    index_file = Path("src/ramllm/data/models_index.json")
    if not index_file.exists():
        print(f"Error: {index_file} does not exist.")
        return 1

    with open(index_file, encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", [])
    print(f"Verifying {len(models)} indexed models against Hugging Face...")

    failed = 0
    total_files = 0
    verified_files = 0

    for m in models:
        model_id = m.get("id")
        files = m.get("files", [])
        if not files:
            print(f"✘ [EMPTY] {model_id} has no indexed files")
            failed += 1
            continue

        for f_entry in files:
            total_files += 1
            repo = f_entry.get("hf_repo")
            fname = f_entry.get("hf_file")
            url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
            req = urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "RAMLLM-Verifier/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status in (200, 302):
                        verified_files += 1
                    else:
                        print(f"✘ [HTTP {resp.status}] {model_id}: {url}")
                        failed += 1
            except Exception as e:
                print(f"✘ [FAIL] {model_id}: {url} -> {e}")
                failed += 1

    print(
        f"\nVerification summary: {verified_files}/{total_files} files verified across {len(models)} models."
    )
    if failed > 0:
        print(f"FAILED: {failed} files encountered errors.")
        return 1

    print("ALL OK: models_index.json is fully valid and reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

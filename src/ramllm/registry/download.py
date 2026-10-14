"""Explicit, resumable model downloading with SHA-256 verification (Offline-First)."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ramllm.registry.catalog import REGISTRY_DIR, add_model


def resolve_model_url(target: str) -> tuple[str, str]:
    """Parse target into download URL and destination filename.

    Supports:
    - Direct HTTP/HTTPS URLs
    - Hugging Face shorthand: `org/repo:file.gguf` or `org/repo/file.gguf`
    """
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        filename = Path(parsed.path).name
        if not filename.endswith(".gguf"):
            filename = f"{filename}.gguf"
        return target, filename

    # Hugging Face shorthand
    if ":" in target:
        repo, filename = target.split(":", 1)
    elif "/" in target:
        parts = target.strip("/").split("/")
        if len(parts) >= 3:
            repo = f"{parts[0]}/{parts[1]}"
            filename = "/".join(parts[2:])
        else:
            raise ValueError(
                f"Invalid Hugging Face target '{target}'. Use format 'org/repo:filename.gguf'."
            )
    else:
        raise ValueError(
            f"Cannot resolve '{target}'. Must be a URL or Hugging Face shorthand (e.g. 'org/repo:file.gguf')."
        )

    if not filename.endswith(".gguf"):
        filename = f"{filename}.gguf"

    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    return url, filename


def pull_model(
    target: str,
    dest_dir: Path | str | None = None,
    expected_sha256: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    client: httpx.Client | None = None,
) -> Path:
    """Download a model file with resume support and integrity verification.

    Offline-first rule: Never called implicitly by RAMLLM; only invoked by explicit user command.
    """
    url, filename = resolve_model_url(target)
    models_dir = Path(dest_dir).resolve() if dest_dir else (REGISTRY_DIR / "models")
    models_dir.mkdir(parents=True, exist_ok=True)

    dest_path = models_dir / filename
    part_path = models_dir / f"{filename}.part"

    # Check for existing completed file
    if dest_path.exists():
        if expected_sha256:
            from ramllm.registry.catalog import compute_file_sha256

            existing_hash = compute_file_sha256(dest_path)
            if existing_hash.lower() == expected_sha256.lower():
                add_model(dest_path, sha256=existing_hash, source_url=url)
                return dest_path
        else:
            add_model(dest_path, source_url=url)
            return dest_path

    # Check for resumable partial download
    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={existing_bytes}-"} if existing_bytes > 0 else {}

    http_client = client or httpx.Client(follow_redirects=True, timeout=120.0)
    should_close_client = client is None

    try:
        with http_client.stream("GET", url, headers=headers) as resp:
            if resp.status_code not in (200, 206):
                raise RuntimeError(
                    f"Failed to download model from {url}: HTTP {resp.status_code} {resp.reason_phrase}"
                )

            is_resumed = resp.status_code == 206
            total_bytes = int(resp.headers.get("content-length", 0))
            if is_resumed:
                total_bytes += existing_bytes

            file_mode = "ab" if is_resumed else "wb"
            downloaded = existing_bytes if is_resumed else 0

            with open(part_path, file_mode) as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total_bytes)

    finally:
        if should_close_client:
            http_client.close()

    # Compute final SHA-256 hash
    hasher = hashlib.sha256()
    with open(part_path, "rb") as f:
        while True:
            c = f.read(4 * 1024 * 1024)
            if not c:
                break
            hasher.update(c)
    computed_sha256 = hasher.hexdigest()

    if expected_sha256 and computed_sha256.lower() != expected_sha256.lower():
        part_path.unlink(missing_ok=True)
        raise ValueError(
            f"Integrity check failed: Expected SHA-256 {expected_sha256}, computed {computed_sha256}."
        )

    # Atomic move to final destination
    part_path.rename(dest_path)

    # Register in catalog
    add_model(dest_path, sha256=computed_sha256, source_url=url)
    return dest_path

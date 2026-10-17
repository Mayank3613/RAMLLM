"""Explicit, resumable model downloading with SHA-256 verification (Offline-First)."""

import hashlib
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

import httpx

from ramllm.registry.catalog import REGISTRY_DIR, add_model

logger = logging.getLogger("ramllm.registry")


def resolve_model_url(target: str) -> tuple[str, str]:
    """Parse target into download URL and destination filename.

    Supports:
    - Direct HTTP/HTTPS URLs
    - Hugging Face shorthand: `org/repo:file.gguf` or `org/repo/file.gguf`
    """
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        raw_name = Path(parsed.path).name
        # Sanitize against path traversal or embedded separators
        filename = Path(raw_name).name.replace("..", "").replace("/", "").replace(chr(92), "")
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

    # Security: Validate filename against path traversal
    if (
        not filename
        or ".." in filename
        or "/" in filename
        or chr(92) in filename
        or chr(0) in filename
    ):
        raise ValueError(f"Unsafe model filename '{filename}': Potential path traversal detected.")

    dest_path = (models_dir / filename).resolve()
    if not dest_path.is_relative_to(models_dir) or dest_path.parent != models_dir:
        raise ValueError(f"Unsafe destination path for '{filename}': Path traversal detected.")

    part_path = models_dir / f"{filename}.part"

    # Security: Warn loudly if no cryptographic integrity hash is provided
    if not expected_sha256:
        logger.warning(
            "Security Notice: No SHA-256 integrity hash provided for '%s'. Download cannot be cryptographically verified.",
            filename,
        )

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

    # Support HF_TOKEN without leaking to logs or exceptions
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and "huggingface.co" in url.lower():
        headers["Authorization"] = f"Bearer {hf_token}"

    def _check_redirect(resp: httpx.Response) -> None:
        if resp.is_redirect:
            loc = resp.headers.get("location", "")
            if loc.startswith("http://") and str(resp.url).startswith("https://"):
                raise ValueError(
                    "Insecure redirect rejected: HTTPS to HTTP downgrade is forbidden."
                )

    http_client = client or httpx.Client(
        follow_redirects=True,
        max_redirects=5,
        timeout=120.0,
        event_hooks={"response": [_check_redirect]},
    )
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

            # Preflight disk space verification (requires file size * 1.15 headroom)
            if total_bytes > 0:
                needed_bytes = int(total_bytes * 1.15)
                try:
                    free_bytes = shutil.disk_usage(models_dir).free
                    if free_bytes < needed_bytes:
                        needed_gb = needed_bytes / (1024**3)
                        free_gb = free_bytes / (1024**3)
                        raise ValueError(
                            f"Insufficient disk space on {models_dir}: {needed_gb:.2f} GB required "
                            f"(including 15% safety buffer), but only {free_gb:.2f} GB available."
                        )
                except OSError:
                    pass

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

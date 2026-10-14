"""Model reference resolver: resolves paths, catalog names, index IDs, and HF shorthand."""

import difflib
import os
import shutil
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ramllm.cli.ui.console import get_console, print_info
from ramllm.cli.ui.errors import (
    ChecksumMismatchError,
    DiskSpaceExhaustedError,
    DownloadFailedError,
    GatedModelAuthError,
)
from ramllm.core.profiler import probe_hardware
from ramllm.core.schemas import HardwareProfile
from ramllm.registry.catalog import list_models
from ramllm.registry.download import pull_model
from ramllm.services.model_service import (
    IndexedFile,
    get_curated_model,
    load_curated_index,
    pick_best_quant,
)
from ramllm.services.paths import get_models_dir


def _download_with_progress(
    hf_target: str,
    dest_dir: Path,
    expected_sha256: str | None = None,
    expected_size: int = 0,
    console: Console | None = None,
) -> Path:
    """Download model with Rich progress bar, speed, ETA, and disk preflight checks."""
    c = console or get_console()

    # Pre-flight disk space check: size + 10% safety margin
    margin_bytes = int(expected_size * 1.1) if expected_size > 0 else 2 * 1024**3
    disk_free = shutil.disk_usage(dest_dir).free
    if disk_free < margin_bytes:
        raise DiskSpaceExhaustedError(
            required_bytes=margin_bytes,
            available_bytes=disk_free,
            path=str(dest_dir),
        )

    # Check for Hugging Face auth token if needed
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=c,
    ) as progress:
        task_id = progress.add_task("Downloading model...", total=expected_size or 100)

        def _update_progress(downloaded: int, total: int) -> None:
            if total > 0 and progress.tasks[task_id].total != total:
                progress.update(task_id, total=total)
            progress.update(task_id, completed=downloaded)

        try:
            client = None
            if hf_token:
                import httpx

                client = httpx.Client(
                    headers={"Authorization": f"Bearer {hf_token}"},
                    follow_redirects=True,
                    timeout=180.0,
                )

            dest_path = pull_model(
                target=hf_target,
                dest_dir=dest_dir,
                expected_sha256=expected_sha256,
                progress_callback=_update_progress,
                client=client,
            )
            return dest_path
        except ValueError as e:
            if "Integrity check failed" in str(e):
                raise ChecksumMismatchError(hf_target, expected_sha256 or "", "") from e
            raise
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "403" in err_str:
                repo_name = hf_target.split(":")[0] if ":" in hf_target else hf_target
                raise GatedModelAuthError(repo_name) from e
            raise DownloadFailedError(hf_target, err_str) from e


def resolve_model_reference(
    ref: str | Path,
    hw: HardwareProfile | None = None,
    auto_download: bool = False,
    confirm: bool = True,
    console: Console | None = None,
) -> Path:
    """Resolve a model reference string into an absolute path to a local GGUF file.

    Resolution order:
    1. Local file path
    2. Catalog registered model name
    3. Curated index ID (with optional @QUANT) -> download if missing
    4. Hugging Face shorthand (repo:file.gguf) -> download if missing
    5. Fuzzy match against catalog and index
    """
    c = console or get_console()
    sref = str(ref).strip()

    # 1. Local file path check
    candidate_path = Path(sref).expanduser()
    if candidate_path.exists() and candidate_path.is_file():
        return candidate_path.resolve()
    if candidate_path.is_absolute() or sref.startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Model file not found: {sref}")

    # 2. Local catalog match
    models_dir = get_models_dir()
    for cat_entry in list_models():
        if cat_entry.name.lower() == sref.lower():
            p = Path(cat_entry.path)
            if p.exists():
                return p.resolve()
        if Path(cat_entry.path).name.lower() == sref.lower():
            p = Path(cat_entry.path)
            if p.exists():
                return p.resolve()

    # Check directly inside models_dir
    direct_in_models = models_dir / sref
    if direct_in_models.exists() and direct_in_models.is_file():
        return direct_in_models.resolve()

    # 3. Curated index ID (e.g. "qwen2.5-7b" or "qwen2.5-7b@Q5_K_M")
    req_quant: str | None = None
    model_id = sref
    if "@" in sref:
        model_id, req_quant = sref.split("@", 1)

    curated = get_curated_model(model_id)
    if curated:
        hardware = hw or probe_hardware(quick=True)
        # Select target file
        target_file: IndexedFile
        if req_quant:
            matched = next((f for f in curated.files if f.quant.upper() == req_quant.upper()), None)
            if not matched:
                available = ", ".join(f.quant for f in curated.files)
                raise ValueError(
                    f"Quantization '{req_quant}' not available for '{curated.id}'. Available: {available}"
                )
            target_file = matched
        else:
            target_file = pick_best_quant(curated, hardware)

        # Check if already downloaded
        local_dest = models_dir / target_file.hf_file
        if local_dest.exists() and local_dest.stat().st_size > 0:
            return local_dest.resolve()

        # Not downloaded: prompt user or auto_download
        size_gb = target_file.size_bytes / (1024**3)
        hf_target = f"{target_file.hf_repo}:{target_file.hf_file}"

        if not auto_download:
            is_interactive = sys.stdin.isatty()
            if confirm and is_interactive:
                import typer

                print_info(
                    f"Model '{curated.name}' ({target_file.quant}, {size_gb:.1f} GB) is not stored locally."
                )
                do_download = typer.confirm(f"Download to {models_dir}?", default=True)
                if not do_download:
                    raise FileNotFoundError(f"Model '{curated.id}' download was cancelled by user.")
            elif confirm and not is_interactive:
                raise FileNotFoundError(
                    f"Model '{curated.id}' is not downloaded. Run `ramllm models pull {curated.id}` first."
                )

        print_info(
            f"Downloading '{curated.name}' ({target_file.quant}, {size_gb:.1f} GB) from Hugging Face..."
        )
        return _download_with_progress(
            hf_target=hf_target,
            dest_dir=models_dir,
            expected_sha256=target_file.sha256 or None,
            expected_size=target_file.size_bytes,
            console=c,
        )

    # 4. Hugging Face shorthand: "org/repo:filename.gguf"
    if ":" in sref or sref.endswith(".gguf") and "/" in sref:
        repo, filename = (
            sref.split(":", 1) if ":" in sref else (sref.rsplit("/", 1)[0], sref.rsplit("/", 1)[1])
        )
        local_file = models_dir / filename
        if local_file.exists() and local_file.stat().st_size > 0:
            return local_file.resolve()

        hf_target = f"{repo}:{filename}"
        if not auto_download:
            is_interactive = sys.stdin.isatty()
            if confirm and is_interactive:
                import typer

                do_download = typer.confirm(
                    f"Download '{filename}' from Hugging Face ({repo})?", default=True
                )
                if not do_download:
                    raise FileNotFoundError(f"Download cancelled for {hf_target}.")
            elif confirm and not is_interactive:
                raise FileNotFoundError(
                    f"Model file '{filename}' not downloaded. Run `ramllm models pull {hf_target}` first."
                )

        return _download_with_progress(
            hf_target=hf_target,
            dest_dir=models_dir,
            console=c,
        )

    # 5. Fuzzy match against catalog and curated index
    idx = load_curated_index()
    all_names = [m.id for m in idx.models] + [c.name for c in list_models()]
    matches = difflib.get_close_matches(sref.lower(), all_names, n=3, cutoff=0.5)

    if matches:
        suggestions = ", ".join(f"'{m}'" for m in matches)
        raise FileNotFoundError(f"Model '{sref}' not found. Did you mean one of: {suggestions}?")

    raise FileNotFoundError(
        f"Model '{sref}' not found as a local file, catalog model, or curated index ID.\n"
        f"Search available models with `ramllm models search`."
    )

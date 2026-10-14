"""Unit tests for model reference resolver and auto-download behavior."""

from pathlib import Path

import pytest
from tests.fixtures.make_gguf import create_tiny_dense_gguf

from ramllm.registry import add_model
from ramllm.services.model_resolver import resolve_model_reference
from ramllm.services.model_service import get_curated_model


def test_resolve_local_file(tmp_path: Path) -> None:
    model_file = tmp_path / "custom_model.gguf"
    create_tiny_dense_gguf(model_file)

    resolved = resolve_model_reference(str(model_file))
    assert resolved == model_file.resolve()


def test_resolve_explicit_path_not_found() -> None:
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        resolve_model_reference("/nonexistent/path/to/model.gguf")

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        resolve_model_reference("./nonexistent_relative.gguf")


def test_resolve_catalog_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_catalog = tmp_path / "test_catalog.json"
    monkeypatch.setattr("ramllm.registry.catalog.REGISTRY_FILE", fake_catalog)

    model_file = tmp_path / "catalog_unique_test.gguf"
    create_tiny_dense_gguf(model_file)
    entry = add_model(model_file)

    resolved = resolve_model_reference(entry.name)
    assert resolved == model_file.resolve()


def test_resolve_curated_index_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAMLLM_MODELS_DIR", str(tmp_path))
    model = get_curated_model("qwen2.5-0.5b")
    assert model is not None

    target_file = model.files[0]
    local_cached = tmp_path / target_file.hf_file
    create_tiny_dense_gguf(local_cached)

    resolved = resolve_model_reference(f"{model.id}@{target_file.quant}")
    assert resolved == local_cached.resolve()


def test_resolve_fuzzy_match() -> None:
    with pytest.raises(FileNotFoundError, match="Did you mean one of:"):
        resolve_model_reference("qwen2.5-7bb")


def test_download_resume_local_http_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ramllm.registry.download import pull_model

    # Generate a real tiny valid GGUF file
    src_gguf = tmp_path / "valid_src.gguf"
    create_tiny_dense_gguf(src_gguf)
    content = src_gguf.read_bytes()

    class RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            if range_header and range_header.startswith("bytes="):
                start = int(range_header.replace("bytes=", "").split("-")[0])
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(content) - 1}/{len(content)}"
                )
                self.send_header("Content-Length", str(len(content) - start))
                self.end_headers()
                self.wfile.write(content[start:])
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    fake_catalog = tmp_path / "catalog.json"
    monkeypatch.setattr("ramllm.registry.catalog.REGISTRY_FILE", fake_catalog)

    try:
        # Pre-seed a partial file of 30,000 bytes
        dest_dir = tmp_path / "models"
        dest_dir.mkdir(parents=True, exist_ok=True)
        part_file = dest_dir / "test_download.gguf.part"
        part_file.write_bytes(content[:30000])

        url = f"http://127.0.0.1:{port}/test_download.gguf"
        saved = pull_model(url, dest_dir=dest_dir)

        assert saved.exists()
        assert saved.stat().st_size == len(content)
        assert saved.read_bytes() == content
    finally:
        server.shutdown()

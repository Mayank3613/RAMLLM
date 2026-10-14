"""Unit tests for model catalog, checksums, and explicit downloading."""

from pathlib import Path
from unittest.mock import MagicMock

from tests.fixtures.make_gguf import create_tiny_dense_gguf

from ramllm.registry.catalog import (
    add_model,
    compute_file_sha256,
    get_model,
    remove_model,
    verify_model,
)
from ramllm.registry.download import pull_model, resolve_model_url


def test_compute_sha256_and_verify(tmp_path: Path) -> None:
    """Test SHA-256 calculation and catalog verification."""
    model_path = tmp_path / "test_model.gguf"
    create_tiny_dense_gguf(model_path)

    sha = compute_file_sha256(model_path)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)

    # Add to catalog
    entry = add_model(model_path)
    assert entry.name == "test_model.gguf"

    # Verify model without pre-existing hash in catalog (should register it)
    valid, computed_sha, msg = verify_model(entry.name)
    assert valid
    assert computed_sha == sha

    # Re-verify against registered hash
    valid2, computed_sha2, msg2 = verify_model(entry.name)
    assert valid2
    assert "matches recorded" in msg2

    # Query model
    found = get_model("test_model.gguf")
    assert found is not None
    assert found.sha256 == sha

    # Remove model
    removed = remove_model("test_model.gguf")
    assert removed
    assert get_model("test_model.gguf") is None


def test_resolve_model_url() -> None:
    """Test URL parsing for direct and Hugging Face URLs."""
    # 1. Direct URL
    url, filename = resolve_model_url("https://example.com/models/custom.gguf")
    assert url == "https://example.com/models/custom.gguf"
    assert filename == "custom.gguf"

    # 2. Hugging Face shorthand with colon
    url, filename = resolve_model_url("Qwen/Qwen2.5-0.5B-GGUF:qwen2.5-0.5b-q4_k_m.gguf")
    assert (
        url == "https://huggingface.co/Qwen/Qwen2.5-0.5B-GGUF/resolve/main/qwen2.5-0.5b-q4_k_m.gguf"
    )
    assert filename == "qwen2.5-0.5b-q4_k_m.gguf"

    # 3. Hugging Face shorthand with slash
    url, filename = resolve_model_url("bartowski/Llama-3.2-1B-GGUF/Llama-3.2-1B-Q4_K_M.gguf")
    assert (
        url
        == "https://huggingface.co/bartowski/Llama-3.2-1B-GGUF/resolve/main/Llama-3.2-1B-Q4_K_M.gguf"
    )
    assert filename == "Llama-3.2-1B-Q4_K_M.gguf"


def test_pull_model_mock_http(tmp_path: Path) -> None:
    """Test pull_model with simulated HTTP response and checksum validation."""
    dest_dir = tmp_path / "models"
    dummy_content = b"GGUF_MOCK_MODEL_DATA"
    import hashlib

    expected_sha = hashlib.sha256(dummy_content).hexdigest()

    # Mock HTTP response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-length": str(len(dummy_content))}
    mock_response.iter_bytes.return_value = [dummy_content]

    # Context manager mock for stream
    mock_client = MagicMock()
    mock_client.stream.return_value.__enter__.return_value = mock_response

    # Mock add_model to avoid analyzing non-GGUF bytes in this mock test
    import ramllm.registry.download as dl_mod

    original_add = dl_mod.add_model
    dl_mod.add_model = MagicMock()  # type: ignore[assignment]

    try:
        path = pull_model(
            target="https://example.com/model.gguf",
            dest_dir=dest_dir,
            expected_sha256=expected_sha,
            client=mock_client,
        )
        assert path.exists()
        assert path.read_bytes() == dummy_content
    finally:
        dl_mod.add_model = original_add

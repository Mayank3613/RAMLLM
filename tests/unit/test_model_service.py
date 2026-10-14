"""Unit tests for model service: curated index, hardware fit, and disk usage."""

from pathlib import Path

from ramllm.core.profiler import load_synthetic_profile
from ramllm.services.model_service import (
    clean_models_dir,
    estimate_model_fit,
    get_curated_model,
    get_models_disk_usage,
    load_curated_index,
    pick_best_quant,
    search_curated_models,
)


def test_load_curated_index() -> None:
    idx = load_curated_index()
    assert len(idx.models) >= 15
    assert idx.version == "1.0.0"

    qwen = next((m for m in idx.models if m.id == "qwen2.5-7b"), None)
    assert qwen is not None
    assert qwen.name == "Qwen 2.5 7B Instruct"
    assert qwen.params_b >= 7.0
    assert len(qwen.files) > 0
    assert any(f.quant == "Q4_K_M" for f in qwen.files)


def test_search_curated_models() -> None:
    # Query search
    qwen_models = search_curated_models(query="qwen")
    assert len(qwen_models) >= 3

    # Tag search
    coding_models = search_curated_models(tag="coding")
    assert any("coding" in m.tags for m in coding_models)

    # Empty match
    no_models = search_curated_models(query="nonexistent_xyz_model_123")
    assert len(no_models) == 0


def test_get_curated_model() -> None:
    m = get_curated_model("llama-3.2-1b")
    assert m is not None
    assert m.family == "Llama"
    assert m.params_b >= 1.0

    none_m = get_curated_model("nonexistent-model-xyz")
    assert none_m is None


def test_estimate_model_fit() -> None:
    hw_laptop = load_synthetic_profile("tests/fixtures/hardware/laptop_8gb_no_gpu.json")
    hw_gpu = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")

    small_m = get_curated_model("smollm2-135m")
    assert small_m is not None
    verdict, _, _ = estimate_model_fit(small_m, hw_laptop)
    assert verdict == "COMFORTABLE"

    large_m = get_curated_model("mixtral-8x7b-instruct")
    assert large_m is not None
    verdict_large, _, _ = estimate_model_fit(large_m, hw_laptop)
    assert verdict_large == "INFEASIBLE"

    qwen7b = get_curated_model("qwen2.5-7b")
    assert qwen7b is not None
    verdict_gpu, _, _ = estimate_model_fit(qwen7b, hw_gpu)
    assert verdict_gpu in ("COMFORTABLE", "CONSTRAINED")


def test_pick_best_quant() -> None:
    hw = load_synthetic_profile("tests/fixtures/hardware/laptop_16gb_gpu_6gb.json")
    model = get_curated_model("qwen2.5-0.5b")
    assert model is not None
    best_file = pick_best_quant(model, hw)
    assert best_file is not None
    assert best_file.quant in ("Q4_K_M", "Q5_K_M", "Q8_0")


def test_models_disk_usage_and_clean(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAMLLM_MODELS_DIR", str(tmp_path))

    # Create dummy model and partial files
    (tmp_path / "test_model.gguf").write_bytes(b"A" * 1024)
    (tmp_path / "download.part").write_bytes(b"B" * 2048)

    usage = get_models_disk_usage()
    assert len(usage["model_files"]) == 1
    assert len(usage["partial_files"]) == 1
    assert usage["total_model_bytes"] == 1024
    assert usage["total_partial_bytes"] == 2048

    count, reclaimed = clean_models_dir(remove_partials=True)
    assert count == 1
    assert reclaimed == 2048

    after_usage = get_models_disk_usage()
    assert len(after_usage["partial_files"]) == 0
    assert len(after_usage["model_files"]) == 1


def test_search_fits_deterministic_ordering_4_profiles() -> None:
    profiles = [
        "tests/fixtures/hardware/laptop_8gb_no_gpu.json",
        "tests/fixtures/hardware/laptop_16gb_gpu_6gb.json",
        "tests/fixtures/hardware/apple_m_16gb_unified.json",
        "tests/fixtures/hardware/desktop_32gb_gpu_24gb.json",
    ]
    models = search_curated_models()
    assert len(models) >= 15

    for prof_path in profiles:
        hw = load_synthetic_profile(prof_path)
        # Check stability across repeated evaluation
        run1 = [estimate_model_fit(m, hw)[0] for m in models]
        run2 = [estimate_model_fit(m, hw)[0] for m in models]
        assert run1 == run2
        assert len(run1) == len(models)

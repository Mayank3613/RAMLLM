"""GGUF header and metadata parsing without tensor data loading."""

import math
import re
from pathlib import Path
from typing import Any

import gguf

from ramllm.core.analyzer.moe import is_expert_tensor, is_shared_expert_tensor
from ramllm.core.schemas import ModelInfo

LAYER_REGEX = re.compile(r"blk\.(\d+)\.")

# Security bounds to guard against malicious/corrupted GGUF metadata
MAX_TENSORS = 50_000
MAX_LAYERS = 2_048
MAX_HIDDEN_SIZE = 1_000_000
MAX_HEADS = 65_536
MAX_CONTEXT_LENGTH = 16_777_216
MAX_STR_LEN = 256


def _get_field_val(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    field = reader.get_field(key)
    if field is None:
        return default
    if field.data and len(field.data) > 0:
        val = field.parts[-1]
        # String data in GGUF is stored as uint8 ascii bytes array
        if hasattr(val, "dtype") and str(val.dtype).startswith("uint8"):
            try:
                return bytes(val).decode("utf-8").strip("\x00")
            except Exception:
                pass
        if isinstance(val, memoryview | bytes):
            try:
                return bytes(val).decode("utf-8").strip("\x00")
            except Exception:
                return default
        try:
            return val.item() if hasattr(val, "item") else val[0]
        except Exception:
            return val
    return default


def _extract_quant_string(reader: gguf.GGUFReader) -> str:
    # First check general.file_type name
    file_type = _get_field_val(reader, "general.file_type")
    if file_type is not None:
        try:
            # Map GGUF file_type integer enum to string
            ft_enum = getattr(gguf, "LlamaFileType", None)
            if ft_enum:
                return str(ft_enum(int(file_type)).name)
        except Exception:
            pass

    # Inspect the first tensor type
    if reader.tensors:
        t_type = reader.tensors[0].tensor_type
        try:
            q_enum = getattr(gguf, "GGMLQuantizationType", None)
            if q_enum:
                return str(q_enum(int(t_type)).name)
        except Exception:
            pass
    return "UNKNOWN"


def analyze_gguf(file_path: Path | str) -> ModelInfo:
    """Analyze a GGUF model file and return its ModelInfo structure.

    Reads only headers and metadata; never loads tensor array weights into memory.
    Enforces strict security bounds on header sizes and metadata values to prevent DoS.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Target path is not a regular file: {path}")
    if path.stat().st_size < 16:
        raise ValueError(
            f"Corrupted or invalid GGUF file '{path}': file size ({path.stat().st_size} bytes) is too small to contain a valid header."
        )

    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError(
            f"Invalid GGUF file '{path}': Invalid magic header ({magic!r} != b'GGUF')."
        )

    path_str = str(path)
    try:
        reader = gguf.GGUFReader(path_str)
    except Exception as e:
        raise ValueError(f"Failed to parse GGUF header for '{path}': {e}") from e

    arch = _get_field_val(reader, "general.architecture", "llama")
    if isinstance(arch, bytes):
        arch = arch.decode("utf-8")
    arch = str(arch).lower()
    if not arch or len(arch) > MAX_STR_LEN:
        raise ValueError(
            f"Corrupted GGUF: architecture string is invalid or too long ({len(arch)} chars)."
        )

    n_layers = int(_get_field_val(reader, f"{arch}.block_count", 32))
    if n_layers < 1 or n_layers > MAX_LAYERS:
        raise ValueError(
            f"Corrupted GGUF: layer count {n_layers} out of valid range [1, {MAX_LAYERS}]."
        )

    hidden_size = int(_get_field_val(reader, f"{arch}.embedding_length", 4096))
    if hidden_size < 1 or hidden_size > MAX_HIDDEN_SIZE:
        raise ValueError(
            f"Corrupted GGUF: hidden size {hidden_size} out of valid range [1, {MAX_HIDDEN_SIZE}]."
        )

    n_heads = int(_get_field_val(reader, f"{arch}.attention.head_count", 32))
    if n_heads < 1 or n_heads > MAX_HEADS:
        raise ValueError(
            f"Corrupted GGUF: head count {n_heads} out of valid range [1, {MAX_HEADS}]."
        )

    n_kv_heads = int(_get_field_val(reader, f"{arch}.attention.head_count_kv", n_heads))
    if n_kv_heads < 1 or n_kv_heads > MAX_HEADS:
        raise ValueError(
            f"Corrupted GGUF: KV head count {n_kv_heads} out of valid range [1, {MAX_HEADS}]."
        )

    head_dim_val = _get_field_val(reader, f"{arch}.attention.key_length")
    head_dim = int(head_dim_val) if head_dim_val else (hidden_size // n_heads)

    ctx_max = int(_get_field_val(reader, f"{arch}.context_length", 4096))
    if ctx_max < 1 or ctx_max > MAX_CONTEXT_LENGTH:
        raise ValueError(
            f"Corrupted GGUF: context length {ctx_max} out of valid range [1, {MAX_CONTEXT_LENGTH}]."
        )

    quant_str = _extract_quant_string(reader)

    if len(reader.tensors) > MAX_TENSORS:
        raise ValueError(
            f"Corrupted or malicious GGUF: tensor count {len(reader.tensors)} exceeds maximum threshold of {MAX_TENSORS}."
        )

    # Attention / KV architecture metadata
    sw_size = _get_field_val(reader, f"{arch}.attention.sliding_window_size")
    if sw_size is None:
        sw_size = _get_field_val(reader, f"{arch}.sliding_window")
    sw_size_int = int(sw_size) if sw_size is not None and int(sw_size) > 0 else None

    mla_rank = _get_field_val(reader, f"{arch}.attention.kv_lora_rank")
    mla_rank_int = int(mla_rank) if mla_rank is not None and int(mla_rank) > 0 else None

    ssm_state = _get_field_val(reader, f"{arch}.ssm.d_state")
    is_ssm = arch in ("mamba", "jamba") or ssm_state is not None
    ssm_state_int = int(ssm_state) if ssm_state is not None else (16 if is_ssm else None)

    # MoE metadata
    n_experts = _get_field_val(reader, f"{arch}.expert_count")
    n_experts_used = _get_field_val(reader, f"{arch}.expert_used_count")
    is_moe = n_experts is not None and int(n_experts) > 0

    n_exp_int = int(n_experts) if is_moe and n_experts else None
    n_exp_used_int = int(n_experts_used) if is_moe and n_experts_used else None

    # Aggregate tensors
    per_layer_bytes = [0] * n_layers
    expert_bytes_per_layer = [0] * n_layers if is_moe else None
    shared_expert_bytes_per_layer = [0] * n_layers if is_moe else None
    non_layer_bytes = 0
    total_params = 0
    total_weight_bytes = 0

    for t in reader.tensors:
        if len(t.shape) > 8 or any(dim < 0 for dim in t.shape):
            raise ValueError(f"Corrupted GGUF: tensor '{t.name}' has invalid shape {t.shape}.")
        num_elems = math.prod(t.shape)
        total_params += num_elems
        n_bytes = int(t.n_bytes)
        total_weight_bytes += n_bytes

        match = LAYER_REGEX.search(t.name)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < n_layers:
                per_layer_bytes[layer_idx] += n_bytes
                if is_moe and expert_bytes_per_layer is not None:
                    if is_expert_tensor(t.name):
                        expert_bytes_per_layer[layer_idx] += n_bytes
                    elif (
                        is_shared_expert_tensor(t.name)
                        and shared_expert_bytes_per_layer is not None
                    ):
                        shared_expert_bytes_per_layer[layer_idx] += n_bytes
            else:
                non_layer_bytes += n_bytes
        else:
            non_layer_bytes += n_bytes

    return ModelInfo(
        path=path_str,
        arch=arch,
        n_params=total_params,
        n_layers=n_layers,
        hidden_size=hidden_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        context_length_max=ctx_max,
        quant=quant_str,
        total_weight_bytes=total_weight_bytes,
        per_layer_bytes=per_layer_bytes,
        non_layer_bytes=non_layer_bytes,
        is_moe=is_moe,
        n_experts=n_exp_int,
        n_experts_used=n_exp_used_int,
        expert_bytes_per_layer=expert_bytes_per_layer,
        shared_expert_bytes_per_layer=shared_expert_bytes_per_layer,
        sliding_window_size=sw_size_int,
        kv_lora_rank=mla_rank_int,
        is_hybrid_ssm=is_ssm,
        ssm_d_state=ssm_state_int,
    )

"""Generate synthetic tiny GGUF fixture files for headless tests and CI."""

from pathlib import Path

import gguf
import numpy as np


def create_tiny_dense_gguf(path: Path) -> Path:
    """Create a tiny dense model GGUF fixture."""
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_block_count(4)
    writer.add_embedding_length(128)
    writer.add_head_count(8)
    writer.add_head_count_kv(8)
    writer.add_context_length(4096)
    writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_Q4_K_M))

    for i in range(4):
        writer.add_tensor(f"blk.{i}.attn_q.weight", np.zeros((128, 128), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_k.weight", np.zeros((128, 128), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_v.weight", np.zeros((128, 128), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.attn_output.weight", np.zeros((128, 128), dtype=np.float32))

    writer.add_tensor("token_embd.weight", np.zeros((128, 256), dtype=np.float32))
    writer.add_tensor("output.weight", np.zeros((128, 256), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def create_tiny_moe_gguf(path: Path) -> Path:
    """Create a tiny MoE model GGUF fixture."""
    writer = gguf.GGUFWriter(str(path), "qwen3moe")
    writer.add_block_count(2)
    writer.add_embedding_length(64)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_context_length(2048)
    writer.add_uint32("qwen3moe.expert_count", 4)
    writer.add_uint32("qwen3moe.expert_used_count", 2)
    writer.add_uint32("general.file_type", int(gguf.LlamaFileType.MOSTLY_Q8_0))

    for i in range(2):
        writer.add_tensor(f"blk.{i}.attn_q.weight", np.zeros((64, 64), dtype=np.float32))
        # Expert tensors
        writer.add_tensor(f"blk.{i}.ffn_gate_exps.weight", np.zeros((4, 64, 64), dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_down_exps.weight", np.zeros((4, 64, 64), dtype=np.float32))
        # Shared expert tensor
        writer.add_tensor(f"blk.{i}.ffn_gate_shexp.weight", np.zeros((64, 64), dtype=np.float32))

    writer.add_tensor("output.weight", np.zeros((64, 128), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path

"""Helpers for detecting the source safetensors format of a model directory.

Used by mflux-save-advanced to enforce PRD rules: a Task 1 source must be
bf16 pytorch-safetensors; a Task 2 source must be nvfp4 pytorch-safetensors.

Heuristics are intentionally light. The existing mflux loaders are already
defensive; these functions are guards for clearer error messages at the CLI
boundary, not a perfect format linter.
"""
import json
from pathlib import Path

import mlx.core as mx


def _index_and_first_shard(path: str) -> tuple[dict, Path] | None:
    """Return (index_dict, first_shard_path) if `path` looks like a valid
    safetensors model directory, else None.
    """
    root = Path(path)
    index_file = root / "model.safetensors.index.json"
    if not index_file.exists():
        return None
    try:
        index = json.loads(index_file.read_text())
    except json.JSONDecodeError:
        return None
    weight_map = index.get("weight_map", {})
    if not weight_map:
        return None
    first_file = next(iter(set(weight_map.values())), None)
    if first_file is None:
        return None
    shard_path = root / first_file
    if not shard_path.exists():
        return None
    return index, shard_path


def _shard_meta_and_tensors(shard_path: Path) -> tuple[dict, dict] | None:
    """Return (metadata_dict, tensors_dict) for a safetensors shard, or None on
    read failure.
    """
    try:
        data = mx.load(str(shard_path), return_metadata=True)
    except Exception:
        return None
    if not data or len(data) < 2:
        return None
    weights, meta = data[0], data[1]
    return meta or {}, dict(weights)


def detect_bf16_pytorch_safetensors(path: str) -> bool:
    """Return True if `path` looks like a bf16 pytorch-safetensors model directory.

    Heuristic:
    - has model.safetensors.index.json with a non-empty weight_map
    - the first referenced shard exists and contains only bfloat16 tensors
    - neither the shard's safetensors metadata nor the index's metadata block
      contains an `mflux_version` key (mflux-saved models have one)
    """
    parsed = _index_and_first_shard(path)
    if parsed is None:
        return False
    index, shard_path = parsed

    # Reject if the directory already looks mflux-saved.
    index_meta = index.get("metadata", {}) or {}
    if "mflux_version" in index_meta:
        return False

    loaded = _shard_meta_and_tensors(shard_path)
    if loaded is None:
        return False
    md, tensors = loaded
    if "mflux_version" in md:
        return False
    for tensor in tensors.values():
        if tensor.dtype != mx.bfloat16:
            return False
    return True


def detect_nvfp4_pytorch_safetensors(path: str) -> bool:
    """Return True if `path` looks like a nvfp4 (4-bit) pytorch-safetensors model.

    Heuristic:
    - has model.safetensors.index.json with a non-empty weight_map
    - the first referenced shard exists and has quantization_level == "4"
      in its safetensors metadata
    - no mflux_version key in either the safetensors or index metadata
      (mflux-saved models have one)
    """
    parsed = _index_and_first_shard(path)
    if parsed is None:
        return False
    index, shard_path = parsed

    index_meta = index.get("metadata", {}) or {}
    if "mflux_version" in index_meta:
        return False

    loaded = _shard_meta_and_tensors(shard_path)
    if loaded is None:
        return False
    md, _ = loaded
    if "mflux_version" in md:
        return False
    if str(md.get("quantization_level")) != "4":
        return False
    return True

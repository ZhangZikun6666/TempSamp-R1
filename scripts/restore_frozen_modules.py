"""Backfill missing module weights into a verl-merged HF checkpoint.

Defensive guardrail for downstream eval/inference. With recent verl the
``scripts/model_merger.py`` already writes both the new (``model.
language_model.visual.*``) AND the old (``model.visual.*``) HF Qwen3.5
layouts for the visual tower, so frozen-module restoration is normally
a no-op even when training set ``worker.actor.model.freeze_vision_tower
=true``. This script remains useful for:

* Older verl whose merger only wrote one layout and dropped frozen modules.
* Interrupted ``model_merger.py`` runs.
* Manual edits to the HF folder that accidentally remove tensors.
* New ``freeze_*`` flags that future-you adds and forgets to handle.

Behaviour:
* First does a CHEAP key-only scan (index.json or safetensors headers
  only — no tensor data loaded). If every base key is already present
  in the merged folder, exits as a no-op without reading tensor data.
* Otherwise copy-merges the missing tensors from the base model into
  the merged HF folder (trained weights take precedence over base,
  missing weights come from base).

Usage:
    python scripts/restore_frozen_modules.py \\
        --hf_dir /path/to/global_step_NNN/actor/huggingface \\
        --base_model /path/to/Qwen3.5-4B
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def _find_safetensors(directory: Path) -> list[Path]:
    """Return all ``*.safetensors`` files in ``directory`` (non-recursive).

    Tolerates both the standard ``model-00001-of-00002.safetensors`` naming
    AND the unusual ``model.safetensors-00001-of-00002.safetensors`` form
    that some Qwen mirrors use.
    """
    files = sorted(p for p in directory.iterdir() if p.suffix == ".safetensors")
    if not files:
        # Fall back to anything containing the safetensors extension as a
        # substring (handles the broken `*.safetensors-XXXXX.safetensors`).
        files = sorted(p for p in directory.iterdir() if ".safetensors" in p.name)
    return files


def _load_all_safetensors(directory: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for fp in _find_safetensors(directory):
        chunk = load_file(str(fp), device="cpu")
        for k, v in chunk.items():
            if k in state:
                # Sanity: should not happen unless directory is corrupt.
                raise RuntimeError(f"duplicate tensor key {k!r} across shards in {directory}")
            state[k] = v
    return state


def _list_keys(directory: Path) -> set[str]:
    """Return all parameter keys under ``directory`` without loading tensor data.

    Fast path: a single read of ``model.safetensors.index.json`` if present
    (sharded checkpoints). Fallback: open every safetensors file with
    ``safe_open`` and read its JSON header (no tensor data is decoded).
    Either path is on the order of milliseconds even for ~10 GiB checkpoints.
    """
    idx = directory / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            data = json.load(f)
        return set(data.get("weight_map", {}).keys())

    keys: set[str] = set()
    for fp in _find_safetensors(directory):
        with safe_open(str(fp), framework="pt") as h:
            keys.update(h.keys())
    return keys


def _save_sharded(
    state: dict[str, torch.Tensor],
    out_dir: Path,
    max_shard_bytes: int = 5 * 1024**3,
) -> None:
    """Write ``state`` to ``out_dir`` as sharded safetensors + index.

    Mirrors HuggingFace's default sharding (5 GiB per shard). Old shards
    in ``out_dir`` are removed first so we don't leave stale files behind.
    """
    for old in _find_safetensors(out_dir):
        old.unlink()
    old_index = out_dir / "model.safetensors.index.json"
    if old_index.exists():
        old_index.unlink()

    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes: list[int] = [0]
    for k in sorted(state):
        t = state[k].contiguous()
        nbytes = t.numel() * t.element_size()
        if shards[-1] and sizes[-1] + nbytes > max_shard_bytes:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = t
        sizes[-1] += nbytes

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0
    for i, shard in enumerate(shards, start=1):
        if n_shards == 1:
            fname = "model.safetensors"
        else:
            fname = f"model-{i:05d}-of-{n_shards:05d}.safetensors"
        save_file(shard, str(out_dir / fname), metadata={"format": "pt"})
        for k, t in shard.items():
            weight_map[k] = fname
            total_size += t.numel() * t.element_size()

    if n_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": dict(sorted(weight_map.items())),
        }
        with open(out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)


def restore(hf_dir: Path, base_model: Path, dry_run: bool = False) -> tuple[int, int, int]:
    """Backfill missing weights. Returns (kept, added, base_total).

    Two-stage to keep the common (no-op) case nearly free:

    1. CHEAP: list keys in both folders without loading tensor data.
       If every base key is already in the merged folder, return immediately.
    2. EXPENSIVE: only when missing keys exist, load tensors from both
       folders, copy-merge, and rewrite the merged folder's shards.
    """
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"hf_dir does not exist: {hf_dir}")
    if not base_model.is_dir():
        raise FileNotFoundError(f"base_model does not exist: {base_model}")

    print(f"[restore] scanning keys in merged HF dir  {hf_dir}")
    merged_keys = _list_keys(hf_dir)
    print(f"[restore] scanning keys in base model dir {base_model}")
    base_keys = _list_keys(base_model)

    added_keys = sorted(base_keys - merged_keys)
    kept = len(merged_keys & base_keys)
    extra_in_merged = len(merged_keys - base_keys)

    print(f"[restore] merged keys      : {len(merged_keys)}")
    print(f"[restore] base keys        : {len(base_keys)}")
    print(f"[restore]   shared (kept)  : {kept}    (trained weights win)")
    print(f"[restore]   missing (add)  : {len(added_keys)}    (taken from base)")
    print(f"[restore]   extra in merged: {extra_in_merged}    (kept as-is)")
    if added_keys:
        sample = ", ".join(added_keys[:5])
        print(f"[restore]   sample missing : {sample}{' ...' if len(added_keys) > 5 else ''}")

    if not added_keys:
        # Truly idempotent: nothing missing, so do not touch the safetensors
        # files. Avoids bumping mtime on the HF dir, which downstream
        # launchers (e.g. eval/task/temporal_grounding/run_eval_verl_vllm.sh)
        # use to decide whether the vLLM-compatible sibling dir needs a
        # rebuild. Also avoids the multi-second I/O of loading and rewriting
        # ~9 GiB of safetensors in the common case (current verl already
        # writes both Qwen3.5 layouts, so this branch is the norm).
        print("[restore] no missing tensors; HF dir is already complete (no-op).")
        return kept, 0, len(base_keys)

    if dry_run:
        print("[restore] --dry-run set; not writing.")
        return kept, len(added_keys), len(base_keys)

    print(f"[restore] loading merged HF state from  {hf_dir}")
    merged = _load_all_safetensors(hf_dir)
    print(f"[restore] loading base model state from {base_model}")
    base = _load_all_safetensors(base_model)

    final = dict(merged)
    for k in added_keys:
        final[k] = base[k]
    print(f"[restore] writing {len(final)} tensors back to {hf_dir} ...")
    _save_sharded(final, hf_dir)
    print("[restore] done.")
    return kept, len(added_keys), len(base_keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dir", required=True, type=Path,
                    help="The verl-merged HF folder (will be modified in place).")
    ap.add_argument("--base_model", required=True, type=Path,
                    help="Original base model folder providing the frozen "
                         "weights (e.g. /path/to/Qwen3.5-4B).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would change without writing.")
    args = ap.parse_args()
    restore(args.hf_dir.resolve(), args.base_model.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

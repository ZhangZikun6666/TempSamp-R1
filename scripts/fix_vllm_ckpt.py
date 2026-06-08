"""Build a vLLM-compatible copy of a merged HF Qwen3.5 checkpoint.

Problem
-------
verl's model_merger produces an HF checkpoint whose state_dict keys follow the
new HF Qwen3.5 convention:

    model.language_model.visual.blocks.*.attn.qkv.weight
    model.language_model.embed_tokens.weight
    model.language_model.layers.*.*
    lm_head.weight

HuggingFace's `AutoModelForImageTextToText.from_pretrained` loads these keys
directly because the HF model defines `self.model.language_model.visual`, so
the HF eval script works out of the box.

vLLM's `Qwen3_5ForConditionalGeneration`, however, flattens the module tree:

    self.visual           = Qwen3_VisionTransformer(prefix="visual")
    self.language_model   = Qwen3_5ForCausalLM(prefix="language_model")

and relies on `hf_to_vllm_mapper` (inherited from Qwen3VLForConditionalGeneration)
to translate HF names to vLLM names. That mapper only knows the OLD layout:

    "model.visual."          -> "visual."
    "model.language_model."  -> "language_model."  (catch-all)

So the new-layout visual keys `model.language_model.visual.*` are eaten by the
catch-all rule and routed to the LM subtree, where they don't exist, and the
engine dies with:

    ValueError: Following weights were not initialized from checkpoint:
    {'visual.blocks.0.attn.qkv.weight', ...}

Fix
---
Rename just the visual prefix in a fresh checkpoint directory so it matches
what vLLM's existing mapper already handles correctly:

    model.language_model.visual.*  ->  model.visual.*

Everything else stays identical. HF eval continues to use the original
`huggingface/` directory unchanged; vLLM eval points at `huggingface_vllm/`.

Implementation notes
--------------------
safetensors files are [header_size u64][json header][raw tensor blob]. Since
only KEY NAMES change (no data, no dtype, no shape, no offsets), we rewrite
only the JSON header and stream-copy the tensor blob byte-for-byte via
sendfile / copyfileobj. No tensor is deserialized into Python/Torch. Memory
footprint is a few KB regardless of checkpoint size.

Runtime on a ~10 GB checkpoint is basically bounded by sequential disk I/O
(tens of seconds on local SSD, a minute or two on network FS).

Usage
-----
    python eval/fix_vllm_ckpt.py \
        --hf_dir  /path/to/ckpt/huggingface \
        --out_dir /path/to/ckpt/huggingface_vllm
"""

import argparse
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Callable


# Rename rules. Insertion order matters: more specific rules first.
PREFIX_RENAME_RULES = [
    ("model.language_model.visual.", "model.visual."),
]


def rename_key(name: str) -> str:
    for old, new in PREFIX_RENAME_RULES:
        if name.startswith(old):
            return new + name[len(old) :]
    return name


def rewrite_safetensors_header(src: Path, dst: Path, renamer: Callable[[str], str]) -> int:
    """Rewrite safetensors file `src` -> `dst`, applying `renamer` to keys.

    The tensor data blob is stream-copied unchanged. Returns number of renamed
    keys (for logging).
    """
    with open(src, "rb") as f:
        (header_size,) = struct.unpack("<Q", f.read(8))
        header_bytes = f.read(header_size)
        data_offset = 8 + header_size

    header = json.loads(header_bytes.decode("utf-8"))

    new_header = {}
    renamed = 0
    for k, v in header.items():
        if k == "__metadata__":
            new_header[k] = v
            continue
        nk = renamer(k)
        if nk != k:
            renamed += 1
        new_header[nk] = v

    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # safetensors requires 8-byte alignment of the header length.
    pad = (-len(new_header_bytes)) % 8
    new_header_bytes += b" " * pad

    with open(src, "rb") as src_f, open(dst, "wb") as dst_f:
        dst_f.write(struct.pack("<Q", len(new_header_bytes)))
        dst_f.write(new_header_bytes)
        src_f.seek(data_offset)
        shutil.copyfileobj(src_f, dst_f, length=64 * 1024 * 1024)

    return renamed


def mirror_aux_files(hf_dir: Path, out_dir: Path) -> None:
    """Symlink (fallback: copy) every non-weight file so vLLM sees a complete model dir."""
    weight_suffixes = {".safetensors"}
    weight_names = {"model.safetensors.index.json"}

    for item in hf_dir.iterdir():
        if item.is_dir():
            continue
        if item.suffix in weight_suffixes or item.name in weight_names:
            continue
        dst = out_dir / item.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(item.resolve(), dst)
        except OSError:
            shutil.copy2(item, dst)


def maybe_rewrite_index(hf_dir: Path, out_dir: Path, renamer: Callable[[str], str]) -> None:
    idx = hf_dir / "model.safetensors.index.json"
    if not idx.exists():
        return
    with open(idx) as f:
        data = json.load(f)
    data["weight_map"] = {renamer(k): v for k, v in data["weight_map"].items()}
    with open(out_dir / idx.name, "w") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rewrite HF checkpoint for vLLM key convention")
    ap.add_argument("--hf_dir", required=True, help="Merged HF checkpoint dir")
    ap.add_argument("--out_dir", required=True, help="Output dir for vLLM-compatible checkpoint")
    ap.add_argument("--force", action="store_true", help="Rewrite even if out_dir already has weights")
    args = ap.parse_args()

    hf_dir = Path(args.hf_dir)
    out_dir = Path(args.out_dir)
    assert hf_dir.is_dir(), f"hf_dir not found: {hf_dir}"

    shards = sorted(hf_dir.glob("*.safetensors"))
    assert shards, f"No .safetensors files found in {hf_dir}"

    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("*.safetensors"))
    if existing and not args.force:
        print(f"[skip] vLLM checkpoint already exists at {out_dir} ({len(existing)} shard(s)); use --force to rewrite.")
        return

    mirror_aux_files(hf_dir, out_dir)

    total_renamed = 0
    for src in shards:
        dst = out_dir / src.name
        renamed = rewrite_safetensors_header(src, dst, rename_key)
        total_renamed += renamed
        print(f"  {src.name}: renamed {renamed} keys")

    maybe_rewrite_index(hf_dir, out_dir, rename_key)

    print(f"[done] wrote vLLM checkpoint to {out_dir} ({total_renamed} keys renamed total)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline video preprocessing for faster RL training.

The trainer's default rollout path decodes every video on every epoch.
This script does the decode once: it reads a trainer-format JSONL, runs the
same ``process_video`` contract that the training data loader uses, and writes
the decoded frames + metadata into ``.pt`` artifacts in
``--output_dir``. It also emits a new JSONL in which every row carries a
``preprocessed_video`` field pointing at the artifact filename.

At training time, set in the YAML:

    data:
      use_preprocessed_videos: true
      video_source_mode: prefer_preprocessed   # or preprocessed_only
      preprocessed_video_dir: /path/to/preprocessed_videos

Preprocessing parameters (fps / pixel caps / max_frames) MUST match the
training-time values, otherwise the loader will re-decode at runtime.

Typical use::

    python scripts/preprocess_videos.py \
        --input_file  data/timelens_grpo_train.jsonl \
        --output_dir  data/preprocessed_videos \
        --output_file data/timelens_grpo_train.preprocessed.jsonl \
        --video_fps 2.0 --video_max_frames 768 \
        --video_min_pixels 4096 --video_max_pixels 786432 \
        --video_total_pixels 14680064 \
        --workers 16
"""

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from tqdm import tqdm

from verl.utils.multimodal_contract import process_video

PREPROCESS_VERSION = "qwen3vl_patch16_v1"


def process_single_video(
    video_path: str,
    min_pixels: int,
    max_pixels: int,
    max_frames: int,
    video_fps: float,
    total_pixels: Optional[int] = None,
    image_dir: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any], float]:
    """Decode one video with the same contract as the training pipeline."""
    if image_dir is not None and not os.path.isabs(video_path):
        full_video_path = os.path.join(image_dir, video_path)
    else:
        full_video_path = video_path

    if not os.path.exists(full_video_path):
        raise FileNotFoundError(f"Video file not found: {full_video_path}")

    video_data, sample_fps = process_video(
        full_video_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        max_frames=max_frames,
        video_fps=video_fps,
        total_pixels=total_pixels,
        return_fps=True,
    )

    if isinstance(video_data, tuple) and len(video_data) == 2:
        frames, metadata = video_data
    else:
        frames = video_data
        frame_count = len(frames) if hasattr(frames, "__len__") else 0
        metadata = {
            "fps": float(sample_fps),
            "frames_indices": list(range(frame_count)),
            "total_num_frames": frame_count,
        }

    return frames, dict(metadata), float(sample_fps)


def save_preprocessed_video(
    frames: Any,
    metadata: Dict[str, Any],
    sample_fps: float,
    output_path: str,
) -> None:
    """Save one preprocessed video artifact in the training-compatible format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "frames": frames,
            "metadata": metadata,
            "sample_fps": sample_fps,
            "preprocess_version": PREPROCESS_VERSION,
        },
        output_path,
    )


def get_video_hash(video_path: str, params: Dict[str, Any]) -> str:
    """Build a stable artifact name from source path and preprocessing params."""
    param_parts = [
        PREPROCESS_VERSION,
        video_path,
        str(params["min_pixels"]),
        str(params["max_pixels"]),
        str(params["max_frames"]),
        str(params["fps"]),
    ]
    if params.get("total_pixels") is not None:
        param_parts.append(str(params["total_pixels"]))
    param_str = "_".join(param_parts)
    return hashlib.md5(param_str.encode()).hexdigest()[:16]


def process_video_worker(args: Tuple[int, dict, dict, str, Optional[str]]) -> Tuple[int, dict, Optional[str]]:
    """Worker entry for multiprocessing."""
    index, item, params, output_dir, image_dir = args

    try:
        videos = item.get("videos")
        if not videos:
            return index, item, None

        video_path = videos[0]
        video_hash = get_video_hash(video_path, params)
        output_filename = f"{video_hash}.pt"
        output_path = os.path.join(output_dir, output_filename)

        if os.path.exists(output_path):
            item["preprocessed_video"] = output_filename
            return index, item, None

        frames, metadata, sample_fps = process_single_video(
            video_path=video_path,
            min_pixels=params["min_pixels"],
            max_pixels=params["max_pixels"],
            max_frames=params["max_frames"],
            video_fps=params["fps"],
            total_pixels=params.get("total_pixels"),
            image_dir=image_dir,
        )

        save_preprocessed_video(
            frames=frames,
            metadata=metadata,
            sample_fps=sample_fps,
            output_path=output_path,
        )

        item["preprocessed_video"] = output_filename
        return index, item, None
    except Exception as exc:  # pragma: no cover - surfaced in worker logs
        return index, item, str(exc)


def _load_dataset_records(input_file: str) -> list[dict]:
    with open(input_file, "r", encoding="utf-8") as handle:
        first_char = handle.read(1)
        handle.seek(0)

        if first_char == "[":
            print("Detected JSON array format")
            return json.load(handle)

        print("Detected JSONL format")
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(output_file: str, items: list[dict]) -> None:
    with open(output_file, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def preprocess_dataset(
    input_file: str,
    output_dir: str,
    output_file: str,
    video_fps: float = 2.0,
    video_max_frames: int = 128,
    video_min_pixels: int = 5120,
    video_max_pixels: int = 131072,
    video_total_pixels: Optional[int] = None,
    image_dir: Optional[str] = None,
    workers: int = 8,
    skip_errors: bool = False,
) -> None:
    """Preprocess a dataset and write an updated JSONL file."""
    print(f"Loading dataset from {input_file}...")
    data = _load_dataset_records(input_file)
    print(f"Total samples: {len(data)}")

    os.makedirs(output_dir, exist_ok=True)

    video_count = sum(1 for item in data if item.get("videos"))
    print(f"Samples with videos: {video_count}")

    if video_count == 0:
        print("Warning: No videos found in dataset. Writing the original dataset unchanged.")
        _write_jsonl(output_file, data)
        _write_jsonl(f"{output_file}.failed.jsonl", [])
        return

    params = {
        "min_pixels": video_min_pixels,
        "max_pixels": video_max_pixels,
        "max_frames": video_max_frames,
        "fps": video_fps,
        "total_pixels": video_total_pixels,
    }

    print("\nProcessing parameters:")
    print(f"  - video_fps: {video_fps}")
    print(f"  - video_max_frames: {video_max_frames}")
    print(f"  - video_min_pixels: {video_min_pixels}")
    print(f"  - video_max_pixels: {video_max_pixels}")
    print(f"  - video_total_pixels: {video_total_pixels}")
    print(f"  - workers: {workers}")
    print(f"  - output_dir: {output_dir}")
    print()

    tasks = [(i, item.copy(), params, output_dir, image_dir) for i, item in enumerate(data)]
    results: list[Optional[dict]] = [None] * len(data)
    errors: list[tuple[int, str]] = []

    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(process_video_worker, task): task[0] for task in tasks}

        with tqdm(total=len(tasks), desc="Processing videos") as pbar:
            for future in as_completed(futures):
                index, updated_item, error = future.result()

                if error:
                    errors.append((index, error))
                    if not skip_errors:
                        print(f"\nError processing item {index}: {error}")
                    # On error keep the original record; do not propagate the
                    # worker-side mutated copy.
                    results[index] = data[index]
                else:
                    results[index] = updated_item

                pbar.update(1)

    if errors:
        print(f"\n{len(errors)} errors encountered:")
        for idx, err in errors[:10]:
            print(f"  - Item {idx}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")

    # Only write samples whose video was successfully preprocessed; bad videos
    # would otherwise repeatedly hit the realtime-decode fallback at train time.
    preprocessed_items = [item for item in results if item is not None and "preprocessed_video" in item]
    failed_items = [item for item in results if item is not None and "preprocessed_video" not in item]

    print(f"\nSaving preprocessed dataset to {output_file}...")
    _write_jsonl(output_file, preprocessed_items)

    failed_output_file = f"{output_file}.failed.jsonl"
    print(f"Saving failed samples to {failed_output_file}...")
    _write_jsonl(failed_output_file, failed_items)

    total_size = 0
    for file in Path(output_dir).glob("*.pt"):
        total_size += file.stat().st_size

    print("\nPreprocessing complete!")
    print(f"  - Total samples: {len(results)}")
    print(f"  - Successfully preprocessed: {len(preprocessed_items)}")
    print(f"  - Errors: {len(errors)}")
    print(f"  - Skipped samples (no preprocessed_video): {len(failed_items)}")
    print(f"  - Output file: {output_file}")
    print(f"  - Failed samples file: {failed_output_file}")
    print(f"  - Preprocessed videos directory: {output_dir}")
    print(f"  - Total preprocessed data size: {total_size / (1024**3):.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-decode videos for RL training (writes .pt artifacts).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/preprocess_videos.py \\
      --input_file  data/timelens_grpo_train.jsonl \\
      --output_dir  data/preprocessed_videos \\
      --output_file data/timelens_grpo_train.preprocessed.jsonl \\
      --video_fps 2.0 --video_max_frames 768 \\
      --video_min_pixels 4096 --video_max_pixels 786432 \\
      --video_total_pixels 14680064 \\
      --workers 16
        """,
    )
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input JSON / JSONL produced by data/convert_timelens_to_verl.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the per-video .pt artifacts (one file per source video).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSONL: same rows as --input_file with an extra 'preprocessed_video' field.")
    parser.add_argument("--video_fps", type=float, default=2.0, help="Sampling FPS (default: 2.0). Must match training.")
    parser.add_argument("--video_max_frames", type=int, default=128, help="Max frames per video (default: 128).")
    parser.add_argument("--video_min_pixels", type=int, default=4 * 32 * 32,
                        help="Per-frame min pixels (default: 4096).")
    parser.add_argument("--video_max_pixels", type=int, default=64 * 32 * 32,
                        help="Per-frame max pixels (default: 65536).")
    parser.add_argument("--video_total_pixels", type=int, default=None,
                        help="Total pixel budget across sampled frames (default: None, no cap).")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Root directory used to resolve relative video paths in the input JSONL.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel decode workers (default: 8).")
    parser.add_argument("--skip_errors", action="store_true",
                        help="Continue past per-sample decode errors (failed rows go to <output_file>.failed.jsonl).")
    args = parser.parse_args()

    preprocess_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_file=args.output_file,
        video_fps=args.video_fps,
        video_max_frames=args.video_max_frames,
        video_min_pixels=args.video_min_pixels,
        video_max_pixels=args.video_max_pixels,
        video_total_pixels=args.video_total_pixels,
        image_dir=args.image_dir,
        workers=args.workers,
        skip_errors=args.skip_errors,
    )


if __name__ == "__main__":
    main()

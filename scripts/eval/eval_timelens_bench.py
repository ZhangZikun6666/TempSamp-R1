"""
Evaluate temporal grounding on TimeLens-Bench with vLLM AsyncLLMEngine.

Drop-in accelerated replacement for eval_timelens_hf.py, sharing the same CLI,
prompt templates, metric definitions, and output file layout so that
run_eval_verl_vllm.sh can reuse the exact same shard + merge logic.

Key speedups over the HF version:
  1. vLLM continuous batching: many samples share prefill / decode steps.
  2. Async pipeline: CPU video decoding (qwen_vl_utils) and GPU inference
     run in parallel via ThreadPoolExecutor + asyncio.
  3. Chunked prefill: long video prompts are split so decode can proceed.

Usage (single GPU):
    python eval/eval_timelens_vllm.py \
        --model_path /path/to/hf_model \
        --bench_dir /path/to/TimeLens-Bench \
        --dataset charades-timelens \
        --output_dir outputs/eval_run

Multi-GPU (launched by run_eval_verl_vllm.sh):
    CUDA_VISIBLE_DEVICES=0 python eval/eval_timelens_vllm.py \
        --model_path /path/to/hf_model ... --chunk 8 --index 0 &
    ...
"""

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

# vLLM env flags must be set BEFORE importing vllm / torch
os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("DECORD_EOF_RETRY_MAX", "20480")

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

warnings.filterwarnings("ignore", message=".*pad_token_id.*")
logging.getLogger("transformers").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Prompts (aligned with TimeLens evaluation, kept identical to HF version)
# ---------------------------------------------------------------------------

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', "
    "determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

PROMPT_WO_THINK = (
    'To accurately pinpoint the event "{}" in the video, '
    "determine the precise time period of the event. "
    "Provide the start and end times (in seconds, precise to two decimal places) "
    'in the format "start time to end time" within <answer> </answer> tags. '
    "For example: <answer> 12.5 to 17.8 </answer>."
)

# Variant of PROMPT_WO_THINK without the <answer></answer> wrapper instruction
# and without the wrapper around the few-shot example — the model is asked to
# emit a plain "X to Y". Toggle via --no_answer_wrap or env NO_ANSWER_WRAP=1
# when evaluating a checkpoint trained on the corresponding no-answer-wrap
# training data variant.
PROMPT_WO_THINK_NO_ANSWER_WRAP = (
    'To accurately pinpoint the event "{}" in the video, '
    "determine the precise time period of the event. "
    "Provide the start and end times (in seconds, precise to two decimal places) "
    'in the format "start time to end time". '
    "For example: 12.5 to 17.8."
)

PROMPT_THINK = (
    'To accurately pinpoint the event "{}" in the video, '
    "determine the precise time period of the event.\n\n"
    "First, analyze the video content and reason about when the described event happens "
    "inside <think> </think> tags. Then provide the start and end times "
    "(in seconds, precise to two decimal places) "
    'in the format "start time to end time" within <answer> </answer> tags. '
    "For example: <think> reasoning </think><answer> 12.54 to 17.83 </answer>."
)


# ---------------------------------------------------------------------------
# Timestamp extraction / IoU (identical to HF version)
# ---------------------------------------------------------------------------

def extract_time(paragraph):
    paragraph = paragraph.lower()

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", paragraph)
    if answer_match:
        paragraph = answer_match.group(1)

    timestamps = []

    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        time_matches_converted = []
        for t in time_matches:
            parts = t.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            elif len(parts) == 2:
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            time_matches_converted.append(float(time_in_sec))
        timestamps = [
            (time_matches_converted[i], time_matches_converted[i + 1])
            for i in range(0, len(time_matches_converted), 2)
        ]

    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]
        for time_pattern in patterns:
            time_matches = re.findall(time_pattern, paragraph)
            if time_matches:
                timestamps = [(float(s), float(e)) for s, e in time_matches]
                break

    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    return timestamps


def compute_iou(a, b):
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    return max(min1 - max0, 0) / (max1 - min0) if (max1 - min0) > 0 else 0.0


# ---------------------------------------------------------------------------
# Data loading (identical to HF version)
# ---------------------------------------------------------------------------

def parse_query(query):
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


DATASET_CONFIGS = {
    "charades-timelens": {
        "anno": "charades-timelens.json",
        "video_subdir": "video_shards/charades",
    },
    "activitynet-timelens": {
        "anno": "activitynet-timelens.json",
        "video_subdir": "video_shards/activitynet",
    },
    "qvhighlights-timelens": {
        "anno": "qvhighlights-timelens.json",
        "video_subdir": "video_shards/qvhighlights",
    },
}


def load_annotations(bench_dir, dataset_name):
    cfg = DATASET_CONFIGS[dataset_name]
    anno_path = os.path.join(bench_dir, cfg["anno"])
    video_root = os.path.join(bench_dir, cfg["video_subdir"])

    with open(anno_path, "r") as f:
        raw = json.load(f)

    annos = []
    for vid, info in raw.items():
        video_path = os.path.join(video_root, vid + ".mp4")
        duration = info.get("duration", 0)
        queries = info.get("queries", info.get("sentences", []))
        spans = info.get("spans", info.get("timestamps", []))
        for span, query in zip(spans, queries):
            annos.append(dict(
                video_path=video_path,
                duration=duration,
                query=parse_query(query),
                span=[span] if not isinstance(span[0], (list, tuple)) else span,
            ))
    return annos


# ---------------------------------------------------------------------------
# Per-sample prompt + vision preparation
# ---------------------------------------------------------------------------

def select_prompt_and_dr(
    model_path: str,
    enable_thinking: bool,
    no_answer_wrap: bool = False,
) -> Tuple[str, int, str]:
    """Return (prompt_template, downsample_rate, model_family) matching HF logic.

    When ``no_answer_wrap`` is True, swaps PROMPT_WO_THINK for the
    *_no_answer_wrap variant — keep this in sync with eval_timelens_hf.py's
    GroundingDataset.__init__.
    """
    model_lower = model_path.lower()

    if "timelens-7b" in model_lower:
        prompt = PROMPT_WO_THINK
    elif "timelens" in model_lower:
        prompt = PROMPT_WO_THINK
    else:
        if enable_thinking:
            prompt = PROMPT_THINK
        elif no_answer_wrap:
            prompt = PROMPT_WO_THINK_NO_ANSWER_WRAP
        else:
            prompt = PROMPT_WO_THINK

    if "qwen3" in model_lower or "timelens-8b" in model_lower:
        dr = 32
        family = "qwen3"
    elif "qwen2" in model_lower or "timelens-7b" in model_lower:
        dr = 28
        family = "qwen2"
    else:
        dr = 32
        family = "qwen3"

    return prompt, dr, family


def build_messages(anno: dict, prompt_tpl: str, dr: int, args) -> List[dict]:
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": anno["video_path"],
                "min_pixels": args.min_tokens * dr * dr,
                "total_pixels": args.total_tokens * dr * dr,
                "fps": args.fps,
            },
            {"type": "text", "text": prompt_tpl.format(anno["query"])},
        ],
    }]


def prepare_llm_input(
    idx: int,
    anno: dict,
    prompt_tpl: str,
    dr: int,
    patch_size: int,
    processor,
    args,
) -> Tuple[int, dict, Optional[dict]]:
    """CPU-bound: apply chat template + run decord/qwen_vl_utils.

    Returns (idx, anno, llm_input or None if failed)."""
    try:
        msg = build_messages(anno, prompt_tpl, dr, args)

        prompt_text = processor.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )

        _images, video_inputs, video_kwargs = process_vision_info(
            msg,
            image_patch_size=patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        video_kwargs = video_kwargs or {}
        video_kwargs["do_resize"] = False

        llm_input = {
            "prompt": prompt_text,
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs,
        }
        return idx, anno, llm_input
    except Exception as e:
        print(f"  [LOAD FAIL] idx={idx} video={anno['video_path']} err={e}", flush=True)
        return idx, anno, None


# ---------------------------------------------------------------------------
# Async pipeline
# ---------------------------------------------------------------------------

async def run_async_eval(
    engine,
    sampling_params,
    annos: List[dict],
    prompt_tpl: str,
    dr: int,
    patch_size: int,
    processor,
    args,
    output_path: str,
    summary_path: str,
    dataset_name: str,
):
    """Main async loop: loader ↔ submitter ↔ collector."""
    total = len(annos)
    pbar = tqdm(total=total, desc=f"shard-{args.index}")

    loaded_queue: asyncio.Queue = asyncio.Queue(maxsize=args.queue_size)
    loop = asyncio.get_running_loop()

    loading_done = False

    async def data_loader_task():
        nonlocal loading_done
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            for idx in range(total):
                try:
                    result = await loop.run_in_executor(
                        executor,
                        prepare_llm_input,
                        idx, annos[idx], prompt_tpl, dr, patch_size, processor, args,
                    )
                    await loaded_queue.put(result)
                except Exception as e:
                    print(f"  [LOADER ERR] idx={idx}: {e}", flush=True)
                    await loaded_queue.put((idx, annos[idx], None))
        await loaded_queue.put(None)
        loading_done = True

    async def process_request(gen, idx, anno):
        """Drain a single engine.generate async generator."""
        final_output = None
        try:
            async for request_output in gen:
                if request_output.finished:
                    final_output = request_output
        except Exception as e:
            print(f"  [GEN ERR] idx={idx}: {e}", flush=True)
        return idx, anno, final_output

    results: List[dict] = []
    ious: List[float] = []
    recall = {0.3: 0, 0.5: 0, 0.7: 0}

    pending_tasks: Dict[str, asyncio.Task] = {}
    completed = 0
    t_start = time.time()

    loader_task = asyncio.create_task(data_loader_task())

    def _save_results_incremental():
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [SAVE WARN] {e}", flush=True)

    reached_end_of_queue = False

    while completed < total:
        # 1. Submit as many ready samples as allowed
        while len(pending_tasks) < args.max_concurrent and not reached_end_of_queue:
            try:
                item = loaded_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if item is None:
                reached_end_of_queue = True
                break

            idx, anno, llm_input = item
            if llm_input is None:
                duration = anno.get("duration", 0)
                span = anno["span"][0] if isinstance(anno["span"][0], (list, tuple)) else anno["span"]
                pred = (duration + 10, duration + 20)
                iou_val = compute_iou(span, pred)
                results.append({
                    "video": os.path.basename(anno["video_path"]),
                    "query": anno["query"],
                    "gt_span": list(span),
                    "pred_span": list(pred),
                    "iou": round(iou_val, 4),
                    "answer": "",
                    "duration": duration,
                    "error": "load_failed",
                })
                ious.append(iou_val)
                completed += 1
                pbar.update(1)
                continue

            request_id = f"req_{args.index}_{idx}"
            gen = engine.generate(
                llm_input,
                sampling_params,
                request_id=request_id,
            )
            task = asyncio.create_task(process_request(gen, idx, anno))
            pending_tasks[request_id] = task

        # 2. Wait for any in-flight task to finish
        if pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.values(),
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED,
            )
            finished_ids = []
            for task in done:
                idx, anno, request_output = task.result()
                for rid, t in pending_tasks.items():
                    if t is task:
                        finished_ids.append(rid)
                        break

                duration = anno.get("duration", 0)
                span = anno["span"][0] if isinstance(anno["span"][0], (list, tuple)) else anno["span"]

                if request_output is None:
                    answer = ""
                    pred = (duration + 10, duration + 20)
                else:
                    answer = request_output.outputs[0].text
                    ts = extract_time(answer)
                    if not ts:
                        pred = (duration + 10, duration + 20)
                    else:
                        pred = (round(ts[0][0]), round(ts[0][1]))

                iou_val = compute_iou(span, pred)
                ious.append(iou_val)
                for t_th in recall:
                    if iou_val >= t_th:
                        recall[t_th] += 1

                results.append({
                    "video": os.path.basename(anno["video_path"]),
                    "query": anno["query"],
                    "gt_span": list(span),
                    "pred_span": list(pred),
                    "iou": round(iou_val, 4),
                    "answer": answer,
                    "duration": duration,
                })
                completed += 1
                pbar.update(1)

                if completed <= 5 or completed % 50 == 0:
                    print(
                        f"  [{completed}/{total}] iou={iou_val:.3f} "
                        f"in_flight={len(pending_tasks)-1} | {answer[:80]}",
                        flush=True,
                    )

                if completed % 50 == 0 or completed == total:
                    n_cur = len(ious)
                    cur = {
                        "num_samples": n_cur,
                        "mIoU": round(sum(ious) / n_cur * 100, 2),
                    }
                    for t_th in [0.3, 0.5, 0.7]:
                        cur[f"R@{t_th}"] = round(recall[t_th] / n_cur * 100, 2)
                    print(
                        f"  [checkpoint {completed}/{total}] "
                        f"mIoU={cur['mIoU']:.2f}%  R@0.3={cur['R@0.3']:.2f}%  "
                        f"R@0.5={cur['R@0.5']:.2f}%  R@0.7={cur['R@0.7']:.2f}%",
                        flush=True,
                    )
                    _save_results_incremental()

            for rid in finished_ids:
                pending_tasks.pop(rid, None)

        else:
            # No tasks in flight yet; let the loader run
            await asyncio.sleep(0.01)

        # Safety: if loader sent sentinel but something went wrong
        if reached_end_of_queue and not pending_tasks and completed < total:
            print(
                f"  [WARN] queue drained but completed={completed}<total={total}; "
                f"remaining samples will be skipped",
                flush=True,
            )
            break

    pbar.close()

    # Ensure loader finished
    await loader_task

    elapsed = time.time() - t_start
    n = len(ious)
    print(f"\nInference done: {n} samples in {elapsed:.1f}s ({n/max(elapsed,1e-6):.2f} samples/s)")

    metrics = {
        "num_samples": n,
        "mIoU": round(sum(ious) / n * 100, 2) if n else 0,
    }
    for t_th in [0.3, 0.5, 0.7]:
        metrics[f"R@{t_th}"] = round(recall[t_th] / n * 100, 2) if n else 0

    n_parsed = sum(
        1 for r in results
        if r["pred_span"] != [round(r["duration"] + 10), round(r["duration"] + 20)]
    )
    metrics["parse_rate"] = round(n_parsed / n * 100, 2) if n else 0

    print(
        f"  mIoU={metrics['mIoU']:.2f}%  R@0.3={metrics['R@0.3']:.2f}%  "
        f"R@0.5={metrics['R@0.5']:.2f}%  R@0.7={metrics['R@0.7']:.2f}%  "
        f"Parse={metrics['parse_rate']:.2f}%"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({dataset_name: metrics}, f, ensure_ascii=False, indent=2)
    print(f"Results -> {output_path}")
    print(f"Summary -> {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="TimeLens-Bench eval with vLLM AsyncLLMEngine")
    # Same as eval_timelens_hf.py
    p.add_argument("--model_path", required=True)
    p.add_argument("--bench_dir", required=True)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--enable_thinking", default="false", choices=["true", "false"])
    p.add_argument("--min_tokens", type=int, default=64)
    p.add_argument("--total_tokens", type=int, default=14336)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--chunk", type=int, default=1)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=10)
    p.add_argument("--processor_path", default=None,
                   help="Processor path (defaults to model_path)")
    p.add_argument(
        "--no_answer_wrap",
        action="store_true",
        help="Use the no-answer-wrapper prompt: drops the "
             "'within <answer></answer> tags' instruction and the wrapper "
             "around the few-shot example, so the model emits a plain 'X to Y'. "
             "Set this when evaluating ckpts trained on the matching "
             "no-answer-wrap training data variant. Can also be enabled via "
             "env NO_ANSWER_WRAP=1.",
    )

    # vLLM-specific
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--max_model_len", type=int, default=24000)
    p.add_argument("--gpu_mem_util", type=float, default=0.9)
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--max_num_batched_tokens", type=int, default=32768)
    p.add_argument("--enable_chunked_prefill", action="store_true", default=True)
    p.add_argument("--enforce_eager", action="store_true", default=False)
    p.add_argument("--max_concurrent", type=int, default=16,
                   help="Max concurrent in-flight requests fed to the engine")
    p.add_argument("--queue_size", type=int, default=16,
                   help="Preloaded sample queue size (controls CPU-side memory)")

    args = p.parse_args()
    args.enable_thinking = args.enable_thinking == "true"
    if not args.no_answer_wrap and os.getenv("NO_ANSWER_WRAP", "0") == "1":
        args.no_answer_wrap = True
    if args.processor_path is None:
        args.processor_path = args.model_path
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"results_{args.dataset}_shard{args.index}.json"
    )
    summary_path = os.path.join(
        args.output_dir, f"summary_shard{args.index}.json"
    )

    print(f"Model:     {args.model_path}")
    print(f"Processor: {args.processor_path}")
    print(f"Dataset:   {args.dataset} | shard {args.index}/{args.chunk}")
    print(f"Thinking:  {args.enable_thinking} | FPS: {args.fps}")
    print(f"Tokens:    min={args.min_tokens}, total={args.total_tokens}")
    print(f"Output:    {output_path}")

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.processor_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.processor_path, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer
    patch_size = processor.image_processor.patch_size
    print(f"Patch size: {patch_size}")

    prompt_tpl, dr, family = select_prompt_and_dr(
        args.model_path, args.enable_thinking, args.no_answer_wrap,
    )
    if args.no_answer_wrap:
        print("[INFO] Using NO_ANSWER_WRAP prompt "
              "(matches no-answer-wrap training data — emits plain 'X to Y')")
    print(f"Family: {family} | downsample_rate={dr}")

    # Annotations: same sort + shard as HF version
    annos = load_annotations(args.bench_dir, args.dataset)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]
    print(f"Loaded {len(annos)} samples (shard {args.index}/{args.chunk})")

    if len(annos) == 0:
        print("No samples to process, exiting.")
        with open(output_path, "w") as f:
            json.dump([], f)
        with open(summary_path, "w") as f:
            json.dump({args.dataset: {"num_samples": 0}}, f)
        return

    # vLLM engine
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    print(f"[AsyncLLMEngine] TP={args.tensor_parallel_size}, "
          f"max_num_seqs={args.max_num_seqs}, max_concurrent={args.max_concurrent}")

    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        limit_mm_per_prompt={"image": 1, "video": 1},
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Greedy to match HF's do_sample=False
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        stop_token_ids=[],
    )

    asyncio.run(
        run_async_eval(
            engine=engine,
            sampling_params=sampling_params,
            annos=annos,
            prompt_tpl=prompt_tpl,
            dr=dr,
            patch_size=patch_size,
            processor=processor,
            args=args,
            output_path=output_path,
            summary_path=summary_path,
            dataset_name=args.dataset,
        )
    )


if __name__ == "__main__":
    main()

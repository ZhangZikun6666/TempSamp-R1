"""Reward functions for TimeLens temporal grounding.

Supports both ground_truth formats:
  - JSON: '{"time": [7.0, 11.0]}'
  - Text: "7.0 to 11.0"

Answer format: model outputs <answer>...</answer> with parseable timestamps.

Reward: overall = iou * 1.0 + format * FORMAT_WEIGHT

TempSamp-R1 extras (opt-in via env vars):
  - GTPO_REWARD_SHAPING=1 : apply `transform_rewards` to the IoU term before
    assembling `overall`. High IoU (>= threshold) is squashed to a near-flat
    "success" plateau and low IoU is exponentially penalised, so the
    group-relative advantage sharpens around the success threshold.
  - GTPO_REWARD_SHAPING_THRESHOLD : float, default 0.8.
  - GTPO_REWARD_SHAPING_ALPHA : float, default 1.0.

`build_gt_response` is used by `RayPPOTrainer._inject_gt_rollout_in_gen_output`
when `algorithm.use_gt_injection=True`; it converts a raw ground_truth string
into the exact response text that will be tokenized in place of one rollout.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


REWARD_NAME = "timelens"
REWARD_TYPE = "batch"

FORMAT_WEIGHT = 0.1

# GTPO reward shaping (env-controlled so RewardManager workers inherit it).
_SHAPING_ENABLED = os.getenv("GTPO_REWARD_SHAPING", "0") == "1"
_SHAPING_THRESHOLD = float(os.getenv("GTPO_REWARD_SHAPING_THRESHOLD", "0.8"))
_SHAPING_ALPHA = float(os.getenv("GTPO_REWARD_SHAPING_ALPHA", "1.0"))
_REWARD_DEBUG = os.getenv("TIMELENS_REWARD_DEBUG", "0") == "1"
_SAMPLE_IO_DEBUG = os.getenv("TIMELENS_SAMPLE_IO_DEBUG", "1") == "1"
_SAMPLE_IO_MAX_CHARS = int(os.getenv("TIMELENS_SAMPLE_IO_MAX_CHARS", "4000"))
_SAMPLE_IO_PRINTED = False


def _clip_for_log(text: Any, max_chars: int = _SAMPLE_IO_MAX_CHARS) -> str:
    text = "" if text is None else str(text)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + f"\n... [truncated {len(text) - max_chars} chars] ...\n"
        + text[-tail_chars:]
    )


def _claim_sample_io_print() -> bool:
    """Return True once across reward worker processes when a sentinel is configured."""
    global _SAMPLE_IO_PRINTED
    if _SAMPLE_IO_PRINTED:
        return False

    once_file = os.getenv("TIMELENS_SAMPLE_IO_ONCE_FILE")
    if once_file:
        try:
            os.makedirs(os.path.dirname(once_file), exist_ok=True)
            fd = os.open(once_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            _SAMPLE_IO_PRINTED = True
            return False
        except OSError:
            # Fall back to once per process if the shared sentinel is unavailable.
            pass

    _SAMPLE_IO_PRINTED = True
    return True


def _maybe_print_sample_io(item: Dict[str, Any], answer: Optional[str], score: float) -> None:
    if not _SAMPLE_IO_DEBUG or not _claim_sample_io_print():
        return

    prompt = item.get("prompt") or item.get("problem") or ""
    response = item.get("response", "") or ""
    print("=" * 80)
    print("[TIMELENS SAMPLE IO] first rollout sample for format check")
    print(f"problem_id: {item.get('problem_id')}")
    print(f"problem_type: {item.get('problem_type')}")
    print(f"ground_truth: {item.get('ground_truth')!r}")
    print(f"score: {score:.4f}")
    print("-" * 80)
    print("[PROMPT]")
    print(_clip_for_log(prompt))
    print("-" * 80)
    print("[RAW MODEL RESPONSE]")
    print(_clip_for_log(response))
    print("-" * 80)
    print("[EXTRACTED ANSWER]")
    print(_clip_for_log(answer))
    print("=" * 80)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_timestamp_from_text(text: str) -> Optional[Tuple[float, float]]:
    """Parse timestamps from text.

    Supports: <answer> tags, JSON {"time": [X, Y]}, 'X to Y', 'X and Y',
    'X - Y', fallback to number pairs.
    """
    if not text:
        return None
    search_text = text
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_match:
        search_text = answer_match.group(1)

    # "X to Y" or "X and Y"
    matches = re.findall(r'(\d+\.?\d*)\s*(?:to|and)\s*(\d+\.?\d*)', search_text)
    if matches:
        last = matches[-1]
        try:
            return float(last[0]), float(last[1])
        except ValueError:
            pass

    # "X - Y"
    matches = re.findall(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', search_text)
    if matches:
        last = matches[-1]
        try:
            return float(last[0]), float(last[1])
        except ValueError:
            pass

    # Fallback: any pair of numbers
    nums = re.findall(r'\b(\d+\.?\d*)\b', search_text)
    if len(nums) >= 2:
        try:
            return float(nums[0]), float(nums[1])
        except ValueError:
            pass

    return None


def parse_bbox_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract (x1, y1, x2, y2) from text."""
    if not text:
        return None
    m = re.search(
        r'\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)\s*,?\s*\((\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\)',
        text,
    )
    if m:
        return tuple(float(x) for x in m.groups())
    nums = re.findall(r'(\d+(?:\.\d+)?)', text)
    if len(nums) >= 4:
        return tuple(float(x) for x in nums[:4])
    return None


def extract_answer(text: str) -> Optional[str]:
    """Extract content from the last <answer>...</answer> block."""
    if not text:
        return None
    matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None


# ---------------------------------------------------------------------------
# IoU computations
# ---------------------------------------------------------------------------

def compute_temporal_iou(pred, gt) -> float:
    inter = max(0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-task reward functions
# ---------------------------------------------------------------------------

def _timestamp_iou_reward(solution_str: str, ground_truth: str) -> float:
    gt_span = parse_timestamp_from_text(str(ground_truth))
    if gt_span is None:
        return 0.0
    pred_span = parse_timestamp_from_text(solution_str)
    if pred_span is None:
        return 0.0
    return compute_temporal_iou(pred_span, gt_span)


def _temporal_format_reward(solution_str: str) -> float:
    """Loose temporal-grounding format check.

      - Returns 1.0 if any ``<answer>...</answer>`` block contains a parseable
        ``X to Y`` / ``X and Y`` / ``X-Y`` pair.
      - Does NOT penalise multiple answer blocks, trailing text, or internal
        newlines — this keeps the format-reward signal dense in early training
        when the base model has not yet learned the EOS pattern.

    Set ``TIMELENS_STRICT_FORMAT=1`` to enable a strict variant (single
    ``<answer></answer>`` block, no leftover text, no internal newlines).
    """
    if not solution_str:
        return 0.0

    if os.getenv("TIMELENS_STRICT_FORMAT", "0") == "1":
        stripped = solution_str.rstrip()
        answer_blocks = re.findall(r'<answer>(.*?)</answer>', stripped, re.DOTALL)
        if len(answer_blocks) != 1:
            return 0.0
        if not stripped.endswith('</answer>'):
            return 0.0
        content = answer_blocks[0]
        if '\n' in content or '\r' in content:
            return 0.0
        if re.search(r'\d+\.?\d*\s*(?:to|and|-)\s*\d+\.?\d*', content):
            return 1.0
        return 0.0

    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if answer_match:
        content = answer_match.group(1)
        if re.search(r'\d+\.?\d*\s*(?:to|and|-)\s*\d+\.?\d*', content):
            return 1.0
    return 0.0


def _bbox_iou_reward(solution_str: str, ground_truth: str) -> float:
    gt_bbox = parse_bbox_from_text(str(ground_truth))
    if gt_bbox is None:
        return 0.0
    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    pred_bbox = parse_bbox_from_text(answer)
    if pred_bbox is None:
        return 0.0
    return compute_iou(pred_bbox, gt_bbox)


def _grounding_format_reward(solution_str: str) -> float:
    """Loose bbox-grounding format check.

    Returns 1.0 if any ``<answer>`` block contains a parseable bbox. Set
    ``TIMELENS_STRICT_FORMAT=1`` for the same strict variant as the temporal
    version.
    """
    if not solution_str:
        return 0.0

    if os.getenv("TIMELENS_STRICT_FORMAT", "0") == "1":
        stripped = solution_str.rstrip()
        answer_blocks = re.findall(r'<answer>(.*?)</answer>', stripped, re.DOTALL)
        if len(answer_blocks) != 1:
            return 0.0
        if not stripped.endswith('</answer>'):
            return 0.0
        content = answer_blocks[0]
        if '\n' in content or '\r' in content:
            return 0.0
        if parse_bbox_from_text(content.strip()) is None:
            return 0.0
        return 1.0

    answer = extract_answer(solution_str)
    if answer is None:
        return 0.0
    if parse_bbox_from_text(answer) is not None:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# TempSamp-R1: reward shaping + GT response builder
# ---------------------------------------------------------------------------

def transform_rewards(
    iou: float,
    threshold: float = _SHAPING_THRESHOLD,
    alpha: float = _SHAPING_ALPHA,
) -> float:
    """Non-linear IoU shaping used by TempSamp-R1.

    - iou >= threshold: `threshold + log(iou - threshold + 1) * 0.01`
      → squashed into a near-flat plateau right above `threshold` (e.g. for
      threshold=0.8, IoU 0.8→0.8, IoU 1.0→~0.8018). "Success is success;
      don't obsess over tiny IoU wins."
    - iou <  threshold: `threshold - (exp(alpha*(threshold-iou)) - 1) / (exp(alpha) - 1)`
      → smooth exponential penalty ramp, 0.0 IoU → ~0.087 when threshold=0.8.

    The combined effect is to give a sharp signal for crossing the success
    threshold while keeping gradients below it. Returns a plain float.
    """
    iou = float(iou)
    if iou >= threshold:
        return threshold + math.log(iou - threshold + 1.0) * 0.01
    diff = threshold - iou
    denom = math.exp(alpha) - 1.0
    if denom <= 0:
        return max(0.0, iou)
    return threshold - (math.exp(alpha * diff) - 1.0) / denom


def build_gt_response(ground_truth: str, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Convert raw ground_truth (e.g. '{"time": [7.0, 11.0]}' or '7.0 to 11.0')
    into the literal response text that GTPO will tokenize in place of a rollout.

    The shape of the returned string must match what `compute_score` expects
    (so the GT sample reliably scores IoU=1.0, fmt=1.0). For this TimeLens
    reward that is: `"<answer> {start:.2f} to {end:.2f} </answer>"`.

    For bbox-grounding GT we return None (GT injection is opt-in per sample);
    callers of the trainer hook skip samples where the builder returns None.
    """
    if not ground_truth:
        return None
    gt_span = parse_timestamp_from_text(str(ground_truth))
    if gt_span is not None:
        start, end = gt_span
        return f"<answer> {start:.2f} to {end:.2f} </answer>"
    # bbox grounding: we don't currently materialise a canonical answer string
    # for GT injection. Extend here if you want GT injection for bbox too.
    return None


# ---------------------------------------------------------------------------
# Unified compute_score (batch reward interface)
# ---------------------------------------------------------------------------

def compute_score(
    reward_inputs: List[Dict[str, Any]],
    **kwargs,
) -> List[Dict[str, float]]:
    """Batch reward interface for the verl trainer.

    Each item in reward_inputs has:
        response, response_length, ground_truth, data_type, problem_type
    Returns:
        List of {"overall", "iou", "format"}
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Use reward_type=batch for this reward function.")

    results: List[Dict[str, float]] = []

    for item in reward_inputs:
        solution_str = item.get("response", "") or ""
        gt_str = str(item.get("ground_truth", "") or "")
        answer = extract_answer(solution_str)

        if parse_bbox_from_text(gt_str) is not None:
            iou = _bbox_iou_reward(solution_str, gt_str)
            fmt = _grounding_format_reward(solution_str)
            task = "bbox"
        elif parse_timestamp_from_text(gt_str) is not None:
            iou = _timestamp_iou_reward(solution_str, gt_str)
            fmt = _temporal_format_reward(solution_str)
            task = "temporal"
        else:
            if _REWARD_DEBUG:
                print(f"[REWARD DEBUG] task=unknown | gt={gt_str!r} | answer={answer!r}")
            results.append({"overall": 0.0, "iou": 0.0, "iou_raw": 0.0, "format": 0.0})
            continue

        raw_iou = iou
        if _SHAPING_ENABLED:
            iou = transform_rewards(iou, _SHAPING_THRESHOLD, _SHAPING_ALPHA)

        score = iou * 1.0 + fmt * FORMAT_WEIGHT
        _maybe_print_sample_io(item, answer, score)

        if _REWARD_DEBUG:
            pred_span = parse_timestamp_from_text(solution_str) if task == "temporal" else None
            gt_span = parse_timestamp_from_text(gt_str) if task == "temporal" else None
            shaping_tag = f" | iou_raw={raw_iou:.4f}" if _SHAPING_ENABLED else ""
            print(
                f"[REWARD DEBUG] task={task} | gt={gt_str!r} | answer={answer!r} "
                f"| pred_span={pred_span} | gt_span={gt_span} "
                f"| iou={iou:.4f}{shaping_tag} | fmt={fmt:.1f} | score={score:.4f}"
            )

        results.append({
            "overall": float(score),
            "iou": float(iou),
            "iou_raw": float(raw_iou),
            "format": float(fmt),
        })

    return results

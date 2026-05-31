"""Convert raw TimeLens-100K JSONL records to the trainer's input format.

Default mode:
  - problem: messages[0].content as-is (keeps the full instruction + few-shot
    example so the model sees the same prompt at train and eval).
  - answer:  '<answer> X to Y </answer>' string.
  - videos:  absolute paths (no file:// scheme).

Optional --extract_question and --answer_json flags reshape the record for
other downstream consumers that expect a question-only prompt and a
JSON-formatted ground truth.

Source format (each line):
{
    "messages": [{"role": "user", "content": "<video>...question..."}],
    "videos": ["datasets/TimeLens-100K/video_shards/...mp4"],
    "solution": "7.0 to 11.0",
    "difficulty": 0.8778,
    ...
}

Target format (each line):
{
    "problem": "<video>To accurately pinpoint the event \"...\" ...",
    "answer": "<answer> 7.0 to 11.0 </answer>",
    "problem_type": "temporal grounding",
    "data_type": "video",
    "videos": ["/abs/path/to/video.mp4"],
    "has_offline_trajectory": false,
    "offline_output": ""
}

Usage:
    python data/convert_timelens_to_verl.py \
        --input  /path/to/timelens.jsonl \
        --output /path/to/timelens_train.jsonl \
        --video_root /path/to/TimeLens-100K
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys


def _video_basename(path: str) -> str:
    return os.path.basename(str(path))


def load_duration_index(duration_source: str) -> dict[str, float]:
    """Load a {video_basename: duration_seconds} map from a sidecar JSONL.

    The sidecar should be a JSONL where each row carries a video identifier
    (`video_path` or `videos[0]`) and a `duration` field, such as the raw
    TimeLens-100K manifest at /path/to/timelens-100k.jsonl. Lookup is by
    video basename so different path prefixes still match.
    """
    index: dict[str, float] = {}
    with open(duration_source, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            duration = row.get("duration")
            if duration is None:
                continue
            video = row.get("video_path")
            if video is None:
                videos = row.get("videos") or []
                if videos:
                    video = videos[0]
            if video is None:
                continue
            try:
                index[_video_basename(video)] = float(duration)
            except (TypeError, ValueError):
                continue
    return index


def parse_gt_span(solution: str) -> tuple[float, float] | None:
    """Parse '<start> to <end>' style solution into a (start, end) tuple."""
    nums = re.findall(r"(\d+\.?\d*)", solution.strip())
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


def parse_solution_to_answer(solution: str, answer_format: str = "answer_tag") -> str:
    """Convert 'start to end' format to the canonical answer string.

    answer_format:
      - 'answer_tag' (default): '<answer> 7.0 to 11.0 </answer>'
      - 'json'                : '{"time": [7.0, 11.0]}'
    """
    span = parse_gt_span(solution)
    if span is not None:
        start, end = span
        if answer_format == "json":
            return json.dumps({"time": [start, end]})
        return f"<answer> {start} to {end} </answer>"
    return solution


def extract_question(user_msg: str) -> str:
    """Extract the core question from the wrapped prompt and prepend ``<video>``.

    Source format:
      '<video>To accurately pinpoint the event "QUESTION" in the video, ...'
    Target format:
      '<video>QUESTION'
    """
    text = user_msg.replace("<video>", "").strip()

    m = re.search(r'(?:pinpoint|locate|find)\s+(?:the\s+)?(?:event|clip|moment)\s+"([^"]+)"', text, re.IGNORECASE)
    if m:
        return f"<video>{m.group(1)}"

    m = re.search(r'(?:pinpoint|locate|find)\s+(?:the\s+)?(?:event|clip|moment)\s+["\u201c]([^"\u201d]+)["\u201d]', text, re.IGNORECASE)
    if m:
        return f"<video>{m.group(1)}"

    # Fallback: strip format instructions (everything after "Provide the start...")
    cleaned = re.sub(
        r'\s*(?:Provide the start|Please provide|Determine the precise).*$',
        '', text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if cleaned:
        return f"<video>{cleaned}"

    return f"<video>{text}"


# --------------------------------------------------------------------------- #
# Stratified sampling on (gt_len, coverage) buckets
#
# Default targets reflect the *sample-weighted union* of the three TimeLens-Bench
# evaluation sets (Charades 3363 + ActivityNet 4500 + QVHighlights 1541 = 9404
# events). They were measured with scripts/timelens/analyze_timelens_durations.py
# and matter because raw TimeLens-100K vs the bench union differs sharply on:
#   - very short actions (<2s):  bench 13.3% vs train ~3.7%
#   - long actions (>20s):       bench 20.9% vs train ~5.8%
#   - high coverage (>40%):      bench 14.0% vs train ~2.4%
# These three buckets are systematically under-fit by the policy, which is the
# main story behind the Realtime-Video step160 regression on R@0.7.
# --------------------------------------------------------------------------- #
GT_LEN_BUCKETS = [(0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0),
                  (20.0, float("inf"))]
COVERAGE_BUCKETS = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40),
                    (0.40, float("inf"))]
GT_LEN_BUCKET_NAMES = ["<2s", "2-5s", "5-10s", "10-20s", ">20s"]
COVERAGE_BUCKET_NAMES = ["<5%", "5-10%", "10-20%", "20-40%", ">40%"]

# Sample-weighted marginals over Charades+ActivityNet+QVHighlights union (9404 ev).
DEFAULT_GT_LEN_TARGETS = [0.1325, 0.3232, 0.1940, 0.1410, 0.2094]
DEFAULT_COVERAGE_TARGETS = [0.3454, 0.2075, 0.1831, 0.1236, 0.1405]

# True joint distribution (gt_len_bucket x coverage_bucket) measured on the same
# 9404-event bench union. Differs sharply from the outer product because
# (gt_len, coverage) are NOT independent in real datasets:
#   - QV: every video is 150s, so <2s events all live in <5% coverage.
#   - Charades: 30s videos, so <2s events sit in 5-10% coverage.
#   - ActivityNet: long videos, so >20s events tend to dominate 20-40% / >40%.
# Most importantly, cells that are mathematically impossible under the duration
# filter (e.g. <2s × >40% needs dur < 5s) are ~0 here, so we don't waste budget
# on unreachable buckets.
DEFAULT_JOINT_TARGETS = [
    [0.1217, 0.0102, 0.0006, 0.0000, 0.0000],  # <2s
    [0.1560, 0.1069, 0.0536, 0.0066, 0.0001],  # 2-5s
    [0.0644, 0.0364, 0.0482, 0.0410, 0.0039],  # 5-10s
    [0.0033, 0.0525, 0.0339, 0.0267, 0.0246],  # 10-20s
    [0.0000, 0.0015, 0.0468, 0.0492, 0.1119],  # >20s
]


def _bucket_index(value: float, edges: list[tuple[float, float]]) -> int:
    for i, (lo, hi) in enumerate(edges):
        if lo <= value < hi:
            return i
    return len(edges) - 1


def _largest_remainder_alloc(weights: list[float], total: int) -> list[int]:
    """Allocate `total` integer counts to weights summing to 1, minimising rounding bias."""
    raw = [w * total for w in weights]
    base = [int(c) for c in raw]
    deficit = total - sum(base)
    if deficit > 0:
        order = sorted(range(len(weights)), key=lambda i: raw[i] - base[i], reverse=True)
        for i in order[:deficit]:
            base[i] += 1
    elif deficit < 0:
        order = sorted(range(len(weights)), key=lambda i: raw[i] - base[i])
        for i in order[:(-deficit)]:
            base[i] = max(0, base[i] - 1)
    return base


def _parse_targets_arg(value: str | None, dim: str) -> list | None:
    """Parse --stratify_targets (JSON or path to JSON file)."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if os.path.isfile(text):
        with open(text, "r") as f:
            data = json.load(f)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"--stratify_targets must be JSON or a path to a JSON file: {e}")
    if dim == "joint":
        if isinstance(data, dict):
            mat = [[0.0] * len(COVERAGE_BUCKET_NAMES) for _ in GT_LEN_BUCKET_NAMES]
            for i, gn in enumerate(GT_LEN_BUCKET_NAMES):
                for j, cn in enumerate(COVERAGE_BUCKET_NAMES):
                    mat[i][j] = float(data.get(f"{gn}|{cn}", data.get(f"{gn},{cn}", 0.0)))
            return mat
        if isinstance(data, list) and len(data) == len(GT_LEN_BUCKET_NAMES) \
                and all(isinstance(r, list) and len(r) == len(COVERAGE_BUCKET_NAMES) for r in data):
            return [[float(x) for x in r] for r in data]
        raise ValueError("Joint --stratify_targets must be a 5x5 list or a {'<2s|<5%': p, ...} dict.")
    names = GT_LEN_BUCKET_NAMES if dim == "gt_len" else COVERAGE_BUCKET_NAMES
    if isinstance(data, dict):
        return [float(data.get(n, 0.0)) for n in names]
    if isinstance(data, list):
        if len(data) != len(names):
            raise ValueError(f"--stratify_targets list length {len(data)} != {len(names)} buckets for dim={dim}")
        return [float(x) for x in data]
    raise ValueError("--stratify_targets must be a JSON list or dict.")


def _stratified_sample(
    rows: list[dict],
    *,
    stratify_by: str,
    targets: list | None,
    sample_size: int,
    seed: int,
    use_difficulty_weighting: bool = False,
    difficulty_mu: float = 0.75,
    difficulty_sigma: float = 0.2,
    difficulty_bins: int = 100,
    allow_oversample: bool = True,
) -> list[dict]:
    """Stratified sample to match a target distribution over GT-length / coverage buckets.

    stratify_by:
      - "gt_len"   : 5 buckets on GT span length (s)
      - "coverage" : 5 buckets on gt_len/duration
      - "joint"    : 5x5 (gt_len, coverage) buckets
    targets:
      - gt_len   : list of 5 floats, defaults to DEFAULT_GT_LEN_TARGETS
      - coverage : list of 5 floats, defaults to DEFAULT_COVERAGE_TARGETS
      - joint    : 5x5 list, defaults to the *measured* joint distribution
                   (DEFAULT_JOINT_TARGETS) of the bench union, NOT the outer
                   product (which falsely allocates mass to impossible cells)

    Within each bucket we either pick uniformly at random (when
    use_difficulty_weighting is False) or re-weight by the existing TimeLens
    difficulty Gaussian. If a bucket has fewer rows than its target count, we
    oversample with replacement (and log it).
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not rows:
        raise ValueError("Cannot stratified-sample from an empty pool")

    rng = random.Random(seed)

    if stratify_by == "gt_len":
        names = list(GT_LEN_BUCKET_NAMES)
        target_props = list(targets) if targets is not None else list(DEFAULT_GT_LEN_TARGETS)
        if len(target_props) != len(names):
            raise ValueError(f"gt_len targets need {len(names)} entries")
        buckets: list[list[dict]] = [[] for _ in names]
        for row in rows:
            gt = row.get("gt_len")
            if gt is None:
                continue
            buckets[_bucket_index(float(gt), GT_LEN_BUCKETS)].append(row)
    elif stratify_by == "coverage":
        names = list(COVERAGE_BUCKET_NAMES)
        target_props = list(targets) if targets is not None else list(DEFAULT_COVERAGE_TARGETS)
        if len(target_props) != len(names):
            raise ValueError(f"coverage targets need {len(names)} entries")
        buckets = [[] for _ in names]
        for row in rows:
            gt = row.get("gt_len")
            dur = row.get("duration")
            if gt is None or not dur or float(dur) <= 0:
                continue
            buckets[_bucket_index(float(gt) / float(dur), COVERAGE_BUCKETS)].append(row)
    elif stratify_by == "joint":
        if targets is None:
            mat = [list(row) for row in DEFAULT_JOINT_TARGETS]
        else:
            mat = targets
        names = []
        target_props = []
        buckets = []
        for i, gn in enumerate(GT_LEN_BUCKET_NAMES):
            for j, cn in enumerate(COVERAGE_BUCKET_NAMES):
                names.append(f"{gn}|{cn}")
                target_props.append(float(mat[i][j]))
                buckets.append([])
        n_cov = len(COVERAGE_BUCKET_NAMES)
        for row in rows:
            gt = row.get("gt_len")
            dur = row.get("duration")
            if gt is None or not dur or float(dur) <= 0:
                continue
            i = _bucket_index(float(gt), GT_LEN_BUCKETS)
            j = _bucket_index(float(gt) / float(dur), COVERAGE_BUCKETS)
            buckets[i * n_cov + j].append(row)
    else:
        raise ValueError(f"Unknown stratify_by: {stratify_by!r}")

    total_w = sum(target_props)
    if total_w <= 0:
        raise ValueError("Stratify targets must sum to a positive value")
    target_props = [w / total_w for w in target_props]
    target_counts = _largest_remainder_alloc(target_props, sample_size)

    selected: list[dict] = []
    summary: list[tuple[str, int, int, int, str]] = []
    for name, target_count, bucket_rows in zip(names, target_counts, buckets):
        if target_count == 0:
            summary.append((name, 0, len(bucket_rows), 0, ""))
            continue
        pool_size = len(bucket_rows)
        if pool_size == 0:
            summary.append((name, target_count, 0, 0, "EMPTY"))
            print(f"[stratified] WARN: bucket '{name}' has 0 pool rows but needs {target_count}; skipping.")
            continue

        if use_difficulty_weighting and pool_size > 0 and \
                all(r.get("difficulty") is not None for r in bucket_rows):
            unique_take = min(target_count, pool_size)
            picks = _difficulty_gaussian_sample(
                bucket_rows,
                mu=difficulty_mu,
                sigma=difficulty_sigma,
                sample_size=unique_take,
                seed=rng.randint(0, 2**31 - 1),
                bins=difficulty_bins,
            )
        else:
            unique_take = min(target_count, pool_size)
            picks = rng.sample(bucket_rows, unique_take)

        oversampled = 0
        if len(picks) < target_count:
            if not allow_oversample:
                summary.append((name, target_count, pool_size, len(picks), "SHORT"))
                selected.extend(picks)
                print(f"[stratified] WARN: bucket '{name}' short by "
                      f"{target_count - len(picks)} (oversample disabled).")
                continue
            extra = target_count - len(picks)
            picks.extend(rng.choices(bucket_rows, k=extra))
            oversampled = extra

        selected.extend(picks)
        note = f"+{oversampled} dup" if oversampled else ""
        summary.append((name, target_count, pool_size, len(picks), note))

    rng.shuffle(selected)
    print(f"[stratified] dim={stratify_by} -> {len(selected)} samples "
          f"(target {sample_size}, pool {len(rows)}, "
          f"diff_weight={use_difficulty_weighting})")
    print(f"[stratified] {'bucket':<14} {'target':>8} {'pool':>8} {'kept':>8}   note")
    for name, tc, pool, kept, note in summary:
        print(f"[stratified] {name:<14} {tc:>8} {pool:>8} {kept:>8}   {note}")

    return selected


def _difficulty_gaussian_sample(
    rows: list[dict],
    *,
    mu: float,
    sigma: float,
    sample_size: int,
    seed: int,
    bins: int = 100,
) -> list[dict]:
    """Sample rows following TimeLens difficulty-aware Gaussian sampling.

    Weight per sample is g(d; mu, sigma^2) / p_hat(d), where p_hat is the
    empirical difficulty density estimated with fixed bins over [0, 1].
    """
    if sample_size >= len(rows):
        return rows
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    difficulties = [float(row["difficulty"]) for row in rows]
    bin_ids = [min(bins - 1, max(0, int(d * bins))) for d in difficulties]
    counts = [0] * bins
    for bin_id in bin_ids:
        counts[bin_id] += 1

    weights = []
    two_sigma_sq = 2.0 * sigma * sigma
    for d, bin_id in zip(difficulties, bin_ids):
        target = math.exp(-((d - mu) ** 2) / two_sigma_sq)
        empirical = counts[bin_id] / float(len(rows))
        weights.append(target / max(empirical, 1e-12))

    rng = random.Random(seed)
    selected: list[dict] = []
    available = list(range(len(rows)))
    available_weights = weights[:]
    for _ in range(sample_size):
        total = sum(available_weights[i] for i in available)
        if total <= 0:
            chosen_pos = rng.randrange(len(available))
        else:
            threshold = rng.random() * total
            acc = 0.0
            chosen_pos = len(available) - 1
            for pos, idx in enumerate(available):
                acc += available_weights[idx]
                if acc >= threshold:
                    chosen_pos = pos
                    break
        idx = available.pop(chosen_pos)
        selected.append(rows[idx])

    return selected


def convert(input_path: str, output_path: str, video_root: str,
            strip_prefixes: list[str] | None = None,
            keep_original_prompt: bool = True,
            answer_format: str = "answer_tag",
            min_difficulty: float | None = None,
            max_difficulty: float | None = None,
            min_duration: float | None = None,
            max_duration: float | None = None,
            duration_index: dict[str, float] | None = None,
            min_gt_len: float | None = None,
            max_gt_len: float | None = None,
            min_coverage: float | None = None,
            max_coverage: float | None = None,
            max_samples: int | None = None,
            sort_by_difficulty_desc: bool = False,
            difficulty_sampling: str = "filter",
            difficulty_mu: float = 0.75,
            difficulty_sigma: float = 0.2,
            difficulty_bins: int = 100,
            stratify_by: str = "none",
            stratify_targets: list | None = None,
            stratify_with_difficulty: bool = False,
            stratify_allow_oversample: bool = True,
            seed: int = 1):
    if strip_prefixes is None:
        strip_prefixes = ["datasets/TimeLens-100K/", "datasets/"]

    rows = []
    skipped = 0
    skipped_difficulty = 0
    skipped_duration = 0
    skipped_gt_len = 0
    skipped_coverage = 0
    with open(input_path, "r") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            difficulty = item.get("difficulty")
            if difficulty_sampling == "gaussian" and difficulty is None:
                skipped_difficulty += 1
                continue
            if min_difficulty is not None and (difficulty is None or float(difficulty) < min_difficulty):
                skipped_difficulty += 1
                continue
            if max_difficulty is not None and (difficulty is None or float(difficulty) > max_difficulty):
                skipped_difficulty += 1
                continue

            user_msg = item["messages"][0]["content"]
            if keep_original_prompt:
                problem = user_msg
            else:
                problem = extract_question(user_msg)

            video_rel = item["videos"][0]
            for prefix in strip_prefixes:
                if video_rel.startswith(prefix):
                    video_rel = video_rel[len(prefix):]
                    break
            video_abs = f"{video_root.rstrip('/')}/{video_rel}"

            duration = item.get("duration")
            if duration is None and duration_index:
                duration = duration_index.get(_video_basename(video_rel))
            if min_duration is not None or max_duration is not None:
                if duration is None:
                    skipped_duration += 1
                    continue
                dur = float(duration)
                if min_duration is not None and dur < min_duration:
                    skipped_duration += 1
                    continue
                if max_duration is not None and dur > max_duration:
                    skipped_duration += 1
                    continue

            solution = item.get("solution", "")
            stratify_needs_gt = stratify_by in ("gt_len", "coverage", "joint")
            stratify_needs_cov = stratify_by in ("coverage", "joint")
            need_gt_len = (min_gt_len is not None or max_gt_len is not None
                           or stratify_needs_gt)
            need_coverage = (min_coverage is not None or max_coverage is not None
                             or stratify_needs_cov)
            gt_len: float | None = None
            if need_gt_len or need_coverage:
                span = parse_gt_span(solution)
                if span is None:
                    skipped_gt_len += 1
                    continue
                gt_len = max(0.0, span[1] - span[0])
                if min_gt_len is not None and gt_len < min_gt_len:
                    skipped_gt_len += 1
                    continue
                if max_gt_len is not None and gt_len > max_gt_len:
                    skipped_gt_len += 1
                    continue
                if need_coverage:
                    if duration is None or float(duration) <= 0:
                        skipped_coverage += 1
                        continue
                    cov = gt_len / float(duration)
                    if min_coverage is not None and cov < min_coverage:
                        skipped_coverage += 1
                        continue
                    if max_coverage is not None and cov > max_coverage:
                        skipped_coverage += 1
                        continue

            answer = parse_solution_to_answer(solution, answer_format=answer_format)

            row = {
                "problem": problem,
                "answer": answer,
                "problem_type": "temporal grounding",
                "data_type": "video",
                "videos": [video_abs],
                "has_offline_trajectory": False,
                "offline_output": "",
            }

            if difficulty is not None:
                row["difficulty"] = difficulty
            if duration is not None:
                row["duration"] = float(duration)
            if gt_len is not None:
                row["gt_len"] = gt_len

            rows.append(row)

    if sort_by_difficulty_desc:
        rows.sort(key=lambda row: float(row.get("difficulty", -1.0)), reverse=True)

    if stratify_by != "none":
        if max_samples is None:
            raise ValueError("--stratify_by requires --max_samples")
        before_sample = len(rows)
        rows = _stratified_sample(
            rows,
            stratify_by=stratify_by,
            targets=stratify_targets,
            sample_size=max_samples,
            seed=seed,
            use_difficulty_weighting=stratify_with_difficulty,
            difficulty_mu=difficulty_mu,
            difficulty_sigma=difficulty_sigma,
            difficulty_bins=difficulty_bins,
            allow_oversample=stratify_allow_oversample,
        )
        print(
            f"Stratified sampling: kept {len(rows)} / {before_sample} "
            f"(by={stratify_by}, diff_weight={stratify_with_difficulty}, seed={seed})"
        )
    elif difficulty_sampling == "gaussian":
        if max_samples is None:
            raise ValueError("--difficulty_sampling gaussian requires --max_samples")
        before_sample = len(rows)
        rows = _difficulty_gaussian_sample(
            rows,
            mu=difficulty_mu,
            sigma=difficulty_sigma,
            sample_size=max_samples,
            seed=seed,
            bins=difficulty_bins,
        )
        print(
            f"Gaussian difficulty sampling: kept {len(rows)} / {before_sample} "
            f"(mu={difficulty_mu}, sigma={difficulty_sigma}, bins={difficulty_bins}, seed={seed})"
        )
    elif max_samples is not None:
        rows = rows[:max_samples]

    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Converted {len(rows)} samples -> {output_path}")
    if skipped_difficulty:
        print(f"Skipped by difficulty filter: {skipped_difficulty}")
    if skipped_duration:
        print(f"Skipped by duration filter: {skipped_duration}"
              f" (min={min_duration}, max={max_duration})")
    if skipped_gt_len:
        print(f"Skipped by gt_len filter: {skipped_gt_len}"
              f" (min={min_gt_len}, max={max_gt_len})")
    if skipped_coverage:
        print(f"Skipped by coverage filter: {skipped_coverage}"
              f" (min={min_coverage}, max={max_coverage})")
    if skipped:
        print(f"Skipped {skipped} samples")

    sample = rows[0] if rows else {}
    print(f"Sample 0: problem_type={sample.get('problem_type')}, "
          f"answer={sample.get('answer')}, "
          f"video={sample.get('videos', [''])[0][:80]}...")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert TimeLens JSONL to the trainer's input format.")
    p.add_argument("--input", required=True, help="Input TimeLens JSONL file")
    p.add_argument("--output", required=True, help="Output JSONL file")
    p.add_argument("--video_root", required=True,
                    help="Root directory containing the TimeLens-100K videos")
    p.add_argument("--strip_prefixes", nargs="*",
                    default=["datasets/TimeLens-100K/", "datasets/"],
                    help="Prefixes to strip from video paths")
    p.add_argument("--extract_question", action="store_true",
                    help="Extract the bare question from the prompt instead of keeping the "
                         "original full prompt (default: keep original).")
    p.add_argument("--answer_format", choices=["answer_tag", "json"], default="answer_tag",
                    help="Output answer format. 'answer_tag' = '<answer> X to Y </answer>' "
                         "(default). 'json' = '{\"time\": [X, Y]}'.")
    p.add_argument("--min_difficulty", type=float, default=None,
                    help="Keep only samples with difficulty >= this value.")
    p.add_argument("--max_difficulty", type=float, default=None,
                    help="Keep only samples with difficulty <= this value.")
    p.add_argument("--min_duration", type=float, default=None,
                    help="Keep only samples with video duration >= this value (seconds).")
    p.add_argument("--max_duration", type=float, default=None,
                    help="Keep only samples with video duration <= this value (seconds).")
    p.add_argument("--duration_source", type=str, default=None,
                    help="Optional sidecar JSONL providing {video_path|videos[0]: duration} "
                         "lookup by video basename; used as fallback when input rows lack "
                         "a duration field.")
    p.add_argument("--min_gt_len", type=float, default=None,
                    help="Keep only samples with GT span length (end-start, seconds) >= this value.")
    p.add_argument("--max_gt_len", type=float, default=None,
                    help="Keep only samples with GT span length (end-start, seconds) <= this value.")
    p.add_argument("--min_coverage", type=float, default=None,
                    help="Keep only samples with gt_len/duration >= this ratio. Requires duration.")
    p.add_argument("--max_coverage", type=float, default=None,
                    help="Keep only samples with gt_len/duration <= this ratio. Requires duration.")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Optional cap after filtering/sorting.")
    p.add_argument("--sort_by_difficulty_desc", action="store_true",
                    help="Sort kept samples from hardest to easiest before optional --max_samples.")
    p.add_argument("--difficulty_sampling", choices=["filter", "gaussian"], default="filter",
                    help="Difficulty selection mode. 'gaussian' follows TimeLens difficulty-aware sampling.")
    p.add_argument("--difficulty_mu", type=float, default=0.75,
                    help="Target mean difficulty for Gaussian sampling.")
    p.add_argument("--difficulty_sigma", type=float, default=0.2,
                    help="Target std for Gaussian difficulty sampling.")
    p.add_argument("--difficulty_bins", type=int, default=100,
                    help="Number of bins for empirical density correction p_hat(d).")
    p.add_argument("--stratify_by", choices=["none", "gt_len", "coverage", "joint"],
                    default="none",
                    help="Enable stratified sampling on bench-aligned bucket distribution. "
                         "'gt_len' uses 5 GT-length buckets, 'coverage' uses 5 gt_len/duration "
                         "buckets, 'joint' uses 5x5 buckets. Requires --max_samples.")
    p.add_argument("--stratify_targets", type=str, default=None,
                    help="Optional JSON string or path to JSON file overriding default targets. "
                         "For 'gt_len'/'coverage': list of 5 floats (will be normalised) or "
                         "{bucket_name: prop} dict. For 'joint': 5x5 list or "
                         "{'<2s|<5%': p, ...} dict. Default = sample-weighted union of "
                         "Charades+ActivityNet+QVHighlights bench (joint default is the "
                         "*measured* 5x5 distribution, not an outer product).")
    p.add_argument("--stratify_with_difficulty", action="store_true",
                    help="Within each bucket, re-weight by the existing TimeLens difficulty "
                         "Gaussian (--difficulty_mu/--difficulty_sigma).")
    p.add_argument("--no_stratify_oversample", action="store_true",
                    help="If set, buckets short of supply will not be oversampled with "
                         "replacement (final sample count may be < --max_samples).")
    p.add_argument("--seed", type=int, default=1,
                    help="Random seed for Gaussian difficulty sampling.")
    args = p.parse_args()
    duration_index = None
    if args.duration_source:
        duration_index = load_duration_index(args.duration_source)
        print(f"Loaded duration index: {len(duration_index)} entries from {args.duration_source}")
    stratify_targets = _parse_targets_arg(args.stratify_targets, args.stratify_by) \
        if args.stratify_by != "none" else None
    convert(
        args.input,
        args.output,
        args.video_root,
        strip_prefixes=args.strip_prefixes,
        keep_original_prompt=not args.extract_question,
        answer_format=args.answer_format,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        duration_index=duration_index,
        min_gt_len=args.min_gt_len,
        max_gt_len=args.max_gt_len,
        min_coverage=args.min_coverage,
        max_coverage=args.max_coverage,
        max_samples=args.max_samples,
        sort_by_difficulty_desc=args.sort_by_difficulty_desc,
        difficulty_sampling=args.difficulty_sampling,
        difficulty_mu=args.difficulty_mu,
        difficulty_sigma=args.difficulty_sigma,
        difficulty_bins=args.difficulty_bins,
        stratify_by=args.stratify_by,
        stratify_targets=stratify_targets,
        stratify_with_difficulty=args.stratify_with_difficulty,
        stratify_allow_oversample=not args.no_stratify_oversample,
        seed=args.seed,
    )

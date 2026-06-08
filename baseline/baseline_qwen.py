#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline (Qwen-VL two-stage) for the Highlight Video Re-framing task.
基线方案（Qwen-VL 两阶段）：视频高光重构图任务。

This baseline reads ONLY the public test index (video_id + targetRatioWH) and
the input videos. It does NOT use any ground-truth annotation.
本基线仅读取公开的测试索引（video_id + 目标画幅）与输入视频，
不依赖任何真值标注。

Two stages 两阶段:
  Stage 1  Highlight localization (temporal grounding):
           Feed the whole video to a video-LLM and let it predict the time
           interval(s) [start_sec, end_sec] of the most highlight-worthy clip.
           阶段1 高光定位：把整段视频喂给视频大模型，预测高光时间区间（秒）。

  Stage 2  Per-frame re-framing:
           Given the target aspect ratio and source size, the crop box SIZE is
           uniquely determined (largest target-ratio rectangle inside the
           frame). So the model only predicts the SUBJECT CENTER; the script
           places the crop window and clamps it inside the frame, then
           densifies every frame inside the segment by interpolation.
           阶段2 逐帧重构图：目标比例+源尺寸已确定裁剪框尺寸，模型只需预测主体
           中心点，脚本放置裁剪窗并夹紧到画面内，再按关键帧插值得到逐帧框。

Output (submission format, one JSON object per line per video):
输出（提交格式，每行一个视频对象）:
    {"video_id": "0", "targetRatioWH": [16, 9],
     "predictions": [{"frame": 10, "bboxes": [x, y, w]}, ...]}
  bboxes is a TRIPLET [x, y, w]; the height is derived by the evaluator as
  h = w * target_h / target_w.
  bboxes 为三元组 [x, y, w]；高度由评测程序按 h = w * th/tw 自动补算。

Dependencies: torch, transformers, qwen-vl-utils, opencv-python, pillow, numpy
"""

import os
import re
import json
import argparse

import cv2
from PIL import Image


# --------------------------- IO: test index ---------------------------
def load_index(index_path):
    """Read public test index -> [(video_id, (tw, th)), ...].

    Each item: {"video_id": "0", "targetRatioWH": [16, 9]}.
    """
    with open(index_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        vid = str(it["video_id"])
        tr = it.get("targetRatioWH", [16, 9])
        tw, th = float(tr[0]), float(tr[1])
        out.append((vid, (tw, th)))
    return out


def video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, fps, w, h


def extract_frames(video_path, frame_ids):
    """Read BGR frames by id -> {frame: ndarray}. Sequential read avoids seeks."""
    frame_ids = sorted(set(int(f) for f in frame_ids))
    out = {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened() or not frame_ids:
        cap.release()
        return out
    start, last = frame_ids[0], frame_ids[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    want = set(frame_ids)
    idx = start
    while idx <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if idx in want:
            out[idx] = fr
        idx += 1
    cap.release()
    return out


# --------------- Geometry: crop size + center placement ---------------
def compute_crop_size(W, H, tw, th):
    """Largest target-ratio rectangle that fits inside the source frame."""
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    target = float(tw) / float(th)
    if W / float(H) >= target:        # source wider than target -> full height
        ch = H
        cw = min(int(round(H * target)), W)
    else:                              # source taller than target -> full width
        cw = W
        ch = min(int(round(W / target)), H)
    return max(1, cw), max(1, ch)


def center_to_box(cx, cy, W, H, cw, ch):
    """Normalized center (0~1) -> pixel crop box [x, y, cw, ch], clamped."""
    px = cx * W - cw / 2.0
    py = cy * H - ch / 2.0
    px = int(round(max(0, min(px, W - cw))))
    py = int(round(max(0, min(py, H - ch))))
    return [px, py, cw, ch]


# --------------------- Parse model outputs ---------------------
_NUM = re.compile(r"-?\d+\.?\d*")


def parse_focus_norm(text, W=None, H=None):
    """Parse subject center, normalize to 0~1. Returns None on failure.

    Robust to coordinate spaces (0~1 float / 0~1000 int / absolute pixel).
    Takes the LAST valid center (the model may reason before answering).
    """
    cand = None
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("center", "subject_center", "focus", "point", "cxcy"):
            v = obj.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                try:
                    cand = [float(v[0]), float(v[1])]
                except (TypeError, ValueError):
                    pass
    if cand is None:
        nums = _NUM.findall(text)
        if len(nums) >= 2:
            cand = [float(nums[-2]), float(nums[-1])]
    if cand is None:
        return None
    cx, cy = cand

    def _norm(v, size):
        if v <= 1.5:
            n = v
        elif v <= 1000.0:
            n = v / 1000.0
        elif size:
            n = v / float(size)
        else:
            n = v / 1000.0
        return max(0.0, min(1.0, n))

    return [_norm(cx, W), _norm(cy, H)]


def parse_segments_sec(text):
    """Parse highlight intervals [[start_sec, end_sec], ...]. [] on failure."""
    for m in re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        segs = obj.get("segments")
        if isinstance(segs, list):
            out = []
            for s in segs:
                if isinstance(s, (list, tuple)) and len(s) >= 2:
                    try:
                        out.append((float(s[0]), float(s[1])))
                    except (TypeError, ValueError):
                        continue
            if out:
                return out
    idx = text.find("segments")
    if idx >= 0:
        tail = text[idx:]
        pairs = re.findall(
            r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", tail)
        if pairs:
            return [(float(a), float(b)) for a, b in pairs]
    return []


def sec_segments_to_frames(segs_sec, fps, n_frames):
    out = []
    for s, e in segs_sec:
        f0 = int(round(min(s, e) * fps))
        f1 = int(round(max(s, e) * fps))
        f0 = max(0, min(f0, n_frames - 1))
        f1 = max(0, min(f1, n_frames - 1))
        if f1 > f0:
            out.append((f0, f1))
    return out


def merge_segments(segs_frame, n_frames):
    """Clamp to [0, n_frames-1], ensure a<=b, merge overlapping/adjacent."""
    if n_frames <= 0 or not segs_frame:
        return []
    segs = []
    for a, b in segs_frame:
        a, b = int(min(a, b)), int(max(a, b))
        a = max(0, min(a, n_frames - 1))
        b = max(0, min(b, n_frames - 1))
        segs.append([a, b])
    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def lerp_box(b0, b1, t):
    return [int(round(b0[i] + (b1[i] - b0[i]) * t)) for i in range(4)]


def densify_boxes(seg_start, seg_end, key_frames, key_boxes):
    """Per-frame boxes inside a segment by linear interpolation of keyframes."""
    if not key_frames:
        return {}
    pts = sorted(zip(key_frames, key_boxes))
    kfs = [p[0] for p in pts]
    kbs = [p[1] for p in pts]
    out = {}
    for f in range(seg_start, seg_end + 1):
        if f <= kfs[0]:
            out[f] = kbs[0]
        elif f >= kfs[-1]:
            out[f] = kbs[-1]
        else:
            j = 0
            while j + 1 < len(kfs) and kfs[j + 1] < f:
                j += 1
            f0, f1 = kfs[j], kfs[j + 1]
            t = (f - f0) / float(f1 - f0) if f1 > f0 else 0.0
            out[f] = lerp_box(kbs[j], kbs[j + 1], t)
    return out


# --------------------------- Model wrapper ---------------------------
class QwenVL:
    def __init__(self, model_path, device_map="auto", dtype="auto",
                 min_pixels=None, max_pixels=None, enable_thinking=False):
        self.enable_thinking = enable_thinking
        import torch  # noqa
        from transformers import AutoProcessor
        model = None
        try:
            from transformers import AutoModelForImageTextToText
            model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=dtype, device_map=device_map)
        except Exception:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as M
                model = M.from_pretrained(model_path, torch_dtype=dtype,
                                          device_map=device_map)
            except Exception:
                from transformers import AutoModelForVision2Seq as M
                model = M.from_pretrained(model_path, torch_dtype=dtype,
                                          device_map=device_map)
        model.eval()
        self.model = model
        proc_kwargs = {}
        if min_pixels:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels:
            proc_kwargs["max_pixels"] = max_pixels
        self.processor = AutoProcessor.from_pretrained(model_path, **proc_kwargs)

    def _generate(self, messages, max_new_tokens=2560):
        tmpl_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text = self.processor.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **tmpl_kwargs)
        except (TypeError, ValueError):
            text = self.processor.apply_chat_template(messages, **tmpl_kwargs)
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
        except Exception:
            image_inputs, video_inputs = None, None
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        import torch
        with torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=False)
        trimmed = gen[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

    def detect_highlights(self, video_path, fps_sample=2.0, max_new_tokens=2560):
        prompt = (
            "This is a short video. Find the single most highlight-worthy clip "
            "that is worth keeping and re-framing, and give its time interval "
            "in seconds (float, counted from the start of the video).\n"
            "Output ONLY JSON, no extra text, strictly: "
            "{\"segments\": [[start_sec, end_sec]]}"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps_sample},
                {"type": "text", "text": prompt},
            ],
        }]
        raw = self._generate(messages, max_new_tokens=max_new_tokens)
        return parse_segments_sec(raw), raw

    def predict_focus(self, pil_img, target_ratio, max_new_tokens=128):
        tw, th = target_ratio
        prompt = (
            "Below is a video frame to be re-framed (cropped) to %d:%d.\n"
            "Point out the center of the most important subject / region to "
            "keep. Use normalized integer coordinates in range 0~1000: x is "
            "horizontal (0=left, 1000=right), y is vertical (0=top, "
            "1000=bottom).\n"
            "Output ONLY JSON (no other text): {\"center\": [x, y]}"
            % (int(tw), int(th))
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt},
            ],
        }]
        return self._generate(messages, max_new_tokens=max_new_tokens)


# --------------------------- Segment processing ---------------------------
def crop_keyframes(model, video_path, seg, target, stride, W, H, cw, ch,
                   raw_log, vid, max_new_tokens=128):
    """Sample keyframes inside a segment and predict centers -> (frames, boxes)."""
    s, e = seg
    kfs = list(range(s, e + 1, max(1, stride)))
    if kfs[-1] != e:
        kfs.append(e)
    frames_map = extract_frames(video_path, kfs)
    key_frames, key_boxes = [], []
    for f in kfs:
        fr = frames_map.get(f)
        if fr is None:
            continue
        pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        try:
            raw = model.predict_focus(pil, target, max_new_tokens=max_new_tokens)
            if raw_log is not None:
                raw_log.write(json.dumps(
                    {"video_id": vid, "frame": f, "stage": "crop", "raw": raw},
                    ensure_ascii=False) + "\n")
            fc = parse_focus_norm(raw, W, H)
            if fc:
                key_frames.append(f)
                key_boxes.append(center_to_box(fc[0], fc[1], W, H, cw, ch))
        except Exception as ex:
            print("    [crop failed] %s#%d -> %s" % (vid, f, ex))
    if not key_frames:  # fallback: center placement keeps the segment non-empty
        key_frames = [s, e]
        cb = center_to_box(0.5, 0.5, W, H, cw, ch)
        key_boxes = [cb, cb]
    return key_frames, key_boxes


# --------------------------- Main ---------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Qwen-VL two-stage baseline: highlight localization + re-framing")
    ap.add_argument("--model", required=True, help="local VL model weights path")
    ap.add_argument("--index", default=os.path.join(here, "test_index.json"),
                    help="public test index (video_id + targetRatioWH)")
    ap.add_argument("--video-dir", default=os.path.join(here, "video"))
    ap.add_argument("--out", default=os.path.join(here, "predictions.jsonl"))
    ap.add_argument("--num-videos", type=int, default=0,
                    help="process first N videos, 0 = all")
    ap.add_argument("--crop-stride", type=int, default=15,
                    help="stage-2 keyframe stride (frames)")
    ap.add_argument("--detect-fps", type=float, default=2.0,
                    help="stage-1 sampling fps fed to the model")
    ap.add_argument("--max-new-tokens", type=int, default=2560)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--min-pixels", type=int, default=None)
    ap.add_argument("--max-pixels", type=int, default=None)
    args = ap.parse_args()

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[:args.num_videos]
    print("To infer: %d videos" % len(index))

    model = QwenVL(args.model, device_map=args.device_map, dtype=args.dtype,
                   min_pixels=args.min_pixels, max_pixels=args.max_pixels,
                   enable_thinking=args.enable_thinking)

    raw_path = args.out + ".raw.jsonl"
    n_lines = 0
    with open(args.out, "w", encoding="utf-8") as fout, \
            open(raw_path, "w", encoding="utf-8") as raw_log:
        for vi, (vid, target) in enumerate(index, 1):
            video_path = os.path.join(args.video_dir, vid + ".mp4")
            if not os.path.exists(video_path):
                print("  [skip] no video: %s" % video_path)
                fout.write(json.dumps(
                    {"video_id": vid, "targetRatioWH": [int(target[0]),
                     int(target[1])], "predictions": []},
                    ensure_ascii=False) + "\n")
                continue
            n_frames, fps, W, H = video_meta(video_path)
            cw, ch = compute_crop_size(W, H, target[0], target[1])

            # Stage 1: highlight localization (model only).
            segs_sec, raw = model.detect_highlights(
                video_path, fps_sample=args.detect_fps,
                max_new_tokens=args.max_new_tokens)
            raw_log.write(json.dumps(
                {"video_id": vid, "stage": "detect", "raw": raw},
                ensure_ascii=False) + "\n")
            segments = merge_segments(
                sec_segments_to_frames(segs_sec, fps, n_frames), n_frames)

            print("  [%d/%d] %s %dx%d fps=%.2f frames=%d crop=%dx%d segs=%s"
                  % (vi, len(index), vid, W, H, fps, n_frames, cw, ch, segments))

            # Stage 2: per-frame re-framing inside each segment.
            predictions = []
            for seg in segments:
                kfs, kbs = crop_keyframes(
                    model, video_path, seg, target, args.crop_stride,
                    W, H, cw, ch, raw_log, vid)
                dense = densify_boxes(seg[0], seg[1], kfs, kbs)
                for f in range(seg[0], seg[1] + 1):
                    box = dense.get(f)
                    if not box:
                        continue
                    x, y, w, _h = box
                    # submit TRIPLET [x, y, w]; height is derived by evaluator
                    predictions.append({"frame": int(f),
                                        "bboxes": [int(round(x)),
                                                   int(round(y)),
                                                   int(round(w))]})
            predictions.sort(key=lambda r: r["frame"])
            rec = {"video_id": vid,
                   "targetRatioWH": [int(target[0]), int(target[1])],
                   "predictions": predictions}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_lines += len(predictions)
            fout.flush()
            raw_log.flush()
    print("Done: %d videos / %d frame predictions -> %s"
          % (len(index), n_lines, args.out))
    print("Raw model outputs -> %s" % raw_path)


if __name__ == "__main__":
    main()

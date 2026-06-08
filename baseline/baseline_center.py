#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trivial baseline (no model): center crop for every frame.
极简基线（无需模型）：对每一帧输出居中裁剪框。

This is the simplest runnable baseline. For each video it places the
largest target-ratio crop box at the center of the frame, for ALL frames.
It needs no GPU / no model weights -- only OpenCV to read video size.
这是最简单的可运行基线：对每个视频，把最大目标比例裁剪框放在画面中心，
覆盖所有帧。无需 GPU / 模型权重，仅用 OpenCV 读取视频尺寸。

It serves as a format reference / starting point. A real solution must
localize the highlight clip (so it does not predict every frame) and track
the subject (so the box is not always centered).
它仅作为格式参考与起点。真正的方案需要定位高光片段（避免对每帧都预测）
并跟踪主体（裁剪框不应恒居中）。

Output (submission format): one JSON object per line per video
    {"video_id": "0", "targetRatioWH": [16, 9],
     "predictions": [{"frame": f, "bboxes": [x, y, w]}, ...]}
  bboxes is a TRIPLET [x, y, w]; height is derived from the target ratio.

Dependencies: opencv-python
"""

import os
import json
import argparse

import cv2


def load_index(index_path):
    with open(index_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    out = []
    for it in items:
        vid = str(it["video_id"])
        tr = it.get("targetRatioWH", [16, 9])
        out.append((vid, (float(tr[0]), float(tr[1]))))
    return out


def video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def compute_crop_size(W, H, tw, th):
    if tw <= 0 or th <= 0 or W <= 0 or H <= 0:
        return W, H
    target = float(tw) / float(th)
    if W / float(H) >= target:
        ch = H
        cw = min(int(round(H * target)), W)
    else:
        cw = W
        ch = min(int(round(W / target)), H)
    return max(1, cw), max(1, ch)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Trivial center-crop baseline")
    ap.add_argument("--index", default=os.path.join(here, "test_index.json"))
    ap.add_argument("--video-dir", default=os.path.join(here, "video"))
    ap.add_argument("--out", default=os.path.join(here, "predictions_center.jsonl"))
    ap.add_argument("--num-videos", type=int, default=0, help="first N, 0=all")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="predict every k-th frame (1 = every frame)")
    args = ap.parse_args()

    index = load_index(args.index)
    if args.num_videos > 0:
        index = index[:args.num_videos]

    n_lines = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for vi, (vid, (tw, th)) in enumerate(index, 1):
            video_path = os.path.join(args.video_dir, vid + ".mp4")
            preds = []
            if os.path.exists(video_path):
                n_frames, W, H = video_meta(video_path)
                cw, ch = compute_crop_size(W, H, tw, th)
                x = int(round((W - cw) / 2.0))
                y = int(round((H - ch) / 2.0))
                step = max(1, args.frame_stride)
                for f in range(0, n_frames, step):
                    preds.append({"frame": int(f), "bboxes": [x, y, int(cw)]})
            else:
                print("  [skip] no video: %s" % video_path)
            rec = {"video_id": vid, "targetRatioWH": [int(tw), int(th)],
                   "predictions": preds}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_lines += len(preds)
            if vi % 20 == 0:
                print("  %d/%d" % (vi, len(index)))
    print("Done: %d videos / %d predictions -> %s"
          % (len(index), n_lines, args.out))


if __name__ == "__main__":
    main()

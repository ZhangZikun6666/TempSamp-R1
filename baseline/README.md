# AIC Highlight Video Re-framing — Baseline Kit / 视频高光重构图 · 基线工具包


> Put the official videos in `baseline/video/` as `<video_id>.mp4`
> (folder not shipped). 官方视频请放入 `baseline/video/`，命名为 `<video_id>.mp4`。

---

## 1. Task / 任务

Given a source video and a target aspect ratio, for **each frame of the
highlight clip** predict a crop box that follows the main subject:

1. **Localize** the highlight clip (you do NOT predict every frame).
2. **Re-frame** each kept frame with a subject-tracking crop box.

任务分两步：先**定位高光片段**（不必对全片每一帧都输出），再对保留下来的每一帧
输出**跟随主体的裁剪框**。

> The crop **size** is fixed by the target ratio + source size (largest
> target-ratio rectangle inside the frame). You only decide *which* frames
> are highlights and *where* the main subject lies (the crop window then
> tracks that subject, clamped to the frame).
> 裁剪框**尺寸**已由目标比例与源尺寸唯一确定，参赛者只需决定哪些帧入选、
> 以及画面**主体的位置**——裁剪窗会以主体为目标、再夹紧到画面内（贴边时
> 主体不一定恰在裁剪框正中）。

---

## 2. Data / 数据

`test_index.json` — 174 entries: 119 × **16:9** + 55 × **9:16**.

```json
[{"video_id": "0", "targetRatioWH": [16, 9]}, ...]
```

---

## 3. Submission format / 提交格式

JSONL, **one line per `video_id`**, covering **every** entry in
`test_index.json` (use `"predictions": []` for videos with no highlight).

```json
{"video_id": "0", "targetRatioWH": [16, 9], "predictions": [{"frame": 0, "bboxes": [0, 15, 720]}, {"frame": 1, "bboxes": [0, 15, 720]}]}
{"video_id": "1", "targetRatioWH": [16, 9], "predictions": []}
```

- `frame` — original (global) frame index in the source video.
- `bboxes` — **triplet `[x, y, w]`** of integers in source pixels.
  **Do NOT submit height**; the evaluator derives `h = w * target_h / target_w`.

不要提交高度。`bboxes` 为源像素坐标的整数三元组，高度由评测程序按
`h = w * target_h / target_w` 自动补算。

A real example produced by the Qwen baseline: [`predictions.jsonl`](predictions.jsonl).

---

## 4. Baselines / 基线

Both scripts default to reading `test_index.json` next to the script and
expect videos under `./video/`. 两个脚本默认读同目录的 `test_index.json`，
从 `./video/` 读取视频。

### 4.1 Center crop (no model) / 极简居中裁剪

OpenCV-only, every frame gets a centered crop. Use it as a format reference.

```bash
pip install opencv-python
python3 baseline_center.py --out predictions_center.jsonl
```

### 4.2 Qwen-VL two-stage (recommended) / Qwen-VL 两阶段（推荐）

- **Stage 1** — feed the whole video to a video-LLM, parse
  `{"segments": [[start_sec, end_sec]]}` for the highlight interval.
- **Stage 2** — sample keyframes, ask the model for a normalized **subject
  point** `{"center": [x, y]}`, place the size-fixed crop window so the
  subject sits at its center (clamped to the frame), then interpolate every
  frame.

阶段1由视频大模型预测高光时间区间；阶段2在片段内抽关键帧让模型输出**主体点**
`{"center": [x, y]}`，将尺寸已定的裁剪窗以该主体点为目标中心放置并夹紧到画面内
（贴边时主体不一定在框正中），再线性插值到逐帧。

```bash
pip install torch transformers qwen-vl-utils opencv-python pillow numpy

python3 baseline_qwen.py \
    --model /path/to/local/Qwen-VL \
    --out predictions.jsonl \
    --crop-stride 15 --detect-fps 2.0 \
    --dtype bfloat16 --device-map auto
```

| Arg | Default | Meaning |
|---|---|---|
| `--model` | *required* | local path to Qwen-VL weights |
| `--index` / `--video-dir` / `--out` | `./test_index.json` / `./video` / `./predictions.jsonl` | IO paths |
| `--num-videos` | `0` | first N only (0 = all) |
| `--crop-stride` | `15` | stage-2 keyframe stride (smaller = finer & slower) |
| `--detect-fps` | `2.0` | stage-1 sampling fps |

---

## 5. FAQ / 常见问题

- **`bboxes` length error?** Submit `[x, y, w]`, three ints in source pixels;
  the evaluator computes `h` itself.
- **Predict every frame?** No, only frames inside the highlight clip; the rest
  count as discarded.
- **No highlight in a video?** Still output a line with `"predictions": []`.

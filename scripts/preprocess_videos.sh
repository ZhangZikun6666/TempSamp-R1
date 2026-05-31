#!/bin/bash
# =============================================================================
# Offline video preprocessing wrapper.
#
# Runs scripts/preprocess_videos.py on a trainer-format JSONL (produced by
# data/convert_timelens_to_verl.py), writing .pt artifacts so the trainer can
# skip realtime decode at every epoch.
#
# Override via env vars; defaults match the TimeLens training YAML
# (scripts/timelens/timelens_tempsamp.yaml).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"

INPUT_FILE="${INPUT_FILE:-${PROJECT_DIR}/data/timelens_grpo_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data/preprocessed_videos}"
OUTPUT_FILE="${OUTPUT_FILE:-${PROJECT_DIR}/data/timelens_grpo_train.preprocessed.jsonl}"
IMAGE_DIR="${IMAGE_DIR:-}"

# Must match the training YAML (scripts/timelens/timelens_*.yaml).
VIDEO_FPS="${VIDEO_FPS:-2.0}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-768}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-4096}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-786432}"
VIDEO_TOTAL_PIXELS="${VIDEO_TOTAL_PIXELS:-14680064}"

WORKERS="${WORKERS:-16}"
SKIP_ERRORS="${SKIP_ERRORS:-1}"

mkdir -p "$OUTPUT_DIR" "$(dirname "$OUTPUT_FILE")"

cat <<EOF
================================================================================
  Offline video preprocessing
================================================================================
  Input JSONL:           $INPUT_FILE
  Output JSONL:          $OUTPUT_FILE
  Preprocessed .pt dir:  $OUTPUT_DIR
  Image root (relative): ${IMAGE_DIR:-<unset, paths must be absolute>}
  fps / max_frames:      $VIDEO_FPS / $VIDEO_MAX_FRAMES
  pixels (min/max/total):$VIDEO_MIN_PIXELS / $VIDEO_MAX_PIXELS / $VIDEO_TOTAL_PIXELS
  Workers:               $WORKERS
================================================================================
EOF

cmd=(
    "$PYTHON_BIN" "${SCRIPT_DIR}/preprocess_videos.py"
    --input_file       "$INPUT_FILE"
    --output_dir       "$OUTPUT_DIR"
    --output_file      "$OUTPUT_FILE"
    --video_fps        "$VIDEO_FPS"
    --video_max_frames "$VIDEO_MAX_FRAMES"
    --video_min_pixels "$VIDEO_MIN_PIXELS"
    --video_max_pixels "$VIDEO_MAX_PIXELS"
    --workers          "$WORKERS"
)
[ -n "$VIDEO_TOTAL_PIXELS" ] && cmd+=(--video_total_pixels "$VIDEO_TOTAL_PIXELS")
[ -n "$IMAGE_DIR" ]           && cmd+=(--image_dir "$IMAGE_DIR")
[ "$SKIP_ERRORS" = "1" ]      && cmd+=(--skip_errors)

"${cmd[@]}"

echo
echo "Done. To use the preprocessed cache at train time, set in the YAML:"
echo "  data:"
echo "    train_files:             $OUTPUT_FILE"
echo "    use_preprocessed_videos: true"
echo "    video_source_mode:       prefer_preprocessed"
echo "    preprocessed_video_dir:  $OUTPUT_DIR"

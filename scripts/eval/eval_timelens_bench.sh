#!/bin/bash
# =============================================================================
# TimeLens-Bench Evaluation — verl FSDP checkpoint -> vLLM AsyncLLMEngine.
#
# Pipeline:
#   1. Merge FSDP checkpoint into a single HF dir (scripts/model_merger.py).
#   2. Verify all base parameters are present (scripts/restore_frozen_modules.py).
#   3. Build a vLLM-compatible checkpoint dir (scripts/fix_vllm_ckpt.py)
#      — renames `model.language_model.visual.*` to `model.visual.*` so vLLM's
#      Qwen3.5 mapper finds the ViT.
#   4. For each dataset in DATASETS, launch one shard per GPU and merge.
#
# Usage:
#   bash scripts/eval/eval_timelens_bench.sh <CHECKPOINT_PATH> [ENABLE_THINKING] [OUTPUT_DIR]
#
# Example:
#   bash scripts/eval/eval_timelens_bench.sh /path/to/exp/global_step_125/actor
#
# Env overrides:
#   DATASETS           charades-timelens,activitynet-timelens,qvhighlights-timelens
#   BASE_MODEL         path to the base HF model (for tokenizer/processor + missing weights)
#   BENCH_DIR          path to TimeLens-Bench root
#   MAX_CONCURRENT     in-flight requests per GPU (default 16)
#   MAX_NUM_SEQS       vLLM running queue (default 8)
#   MAX_MODEL_LEN      total ctx len per request (default 24000)
#   MAX_NUM_BATCHED_TOKENS  chunked prefill chunk size (default 32768)
#   GPU_MEM_UTIL       vLLM GPU memory fraction (default 0.9)
#   TP                 tensor parallel size per process (default 1)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# scripts/eval/eval_timelens_bench.sh  ->  Tempsamp-R1-verl/  (up 2 levels)
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------- paths ----------
MODEL_PATH="${1:?Usage: $0 <CHECKPOINT_PATH> [ENABLE_THINKING] [OUTPUT_DIR]}"
ENABLE_THINKING="${2:-false}"
BENCH_DIR="${BENCH_DIR:-/path/to/TimeLens-Bench}"
BASE_MODEL="${BASE_MODEL:-/path/to/Qwen3.5-4B}"

# ---------- eval settings ----------
DATASETS="${DATASETS:-charades-timelens,activitynet-timelens,qvhighlights-timelens}"
MIN_TOKENS="${MIN_TOKENS:-64}"
TOTAL_TOKENS="${TOTAL_TOKENS:-14336}"
FPS="${FPS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
NUM_WORKERS="${NUM_WORKERS:-10}"
NO_ANSWER_WRAP="${NO_ANSWER_WRAP:-0}"

# ---------- vLLM-specific settings ----------
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
TP="${TP:-1}"
QUEUE_SIZE="${QUEUE_SIZE:-16}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-$((29500 + (RANDOM % 100) * 16))}"

# ---------- resolve verl FSDP checkpoint to HF format ----------
HF_MODEL_PATH="$MODEL_PATH/huggingface"
HAS_WEIGHTS=false
if [ -d "$HF_MODEL_PATH" ]; then
    if ls "$HF_MODEL_PATH"/*.safetensors 1>/dev/null 2>&1 || ls "$HF_MODEL_PATH"/*.bin 1>/dev/null 2>&1; then
        HAS_WEIGHTS=true
    fi
fi

if [ "$HAS_WEIGHTS" = false ]; then
    echo "Converting FSDP checkpoint to HF format ..."
    mkdir -p "$HF_MODEL_PATH"
    for f in config.json tokenizer_config.json tokenizer.json preprocessor_config.json \
             generation_config.json special_tokens_map.json chat_template.json chat_template.jinja \
             vocab.json merges.txt added_tokens.json; do
        if [ -f "$BASE_MODEL/$f" ] && [ ! -f "$HF_MODEL_PATH/$f" ]; then
            cp "$BASE_MODEL/$f" "$HF_MODEL_PATH/$f"
        fi
    done
    python "${PROJECT_DIR}/scripts/model_merger.py" --local_dir "$MODEL_PATH"
    echo "Conversion done: $HF_MODEL_PATH"
else
    echo "HF checkpoint already exists: $HF_MODEL_PATH"
fi

# Restore any tensors from the base model that the merger dropped (e.g. when
# freeze_vision_tower=true ViT weights are not in the FSDP shards). Fast no-op
# in the common case (scans safetensors keys, exits if nothing is missing).
echo "Verifying merged HF dir contains all base parameters ..."
python "${PROJECT_DIR}/scripts/restore_frozen_modules.py" \
    --hf_dir "$HF_MODEL_PATH" --base_model "$BASE_MODEL"

# ---------- build vLLM-compatible checkpoint ----------
# Qwen3.5's HF state dict uses `model.language_model.visual.*` but vLLM's
# built-in hf_to_vllm_mapper expects `model.visual.*`. fix_vllm_ckpt.py writes
# a sibling dir with the renamed keys so vLLM loads the ViT correctly.
VLLM_MODEL_PATH="$MODEL_PATH/huggingface_vllm"

NEED_VLLM_REBUILD=true
if [ -d "$VLLM_MODEL_PATH" ] && ls "$VLLM_MODEL_PATH"/*.safetensors 1>/dev/null 2>&1; then
    HF_LATEST=$(ls -t "$HF_MODEL_PATH"/*.safetensors 2>/dev/null | head -1)
    VLLM_LATEST=$(ls -t "$VLLM_MODEL_PATH"/*.safetensors 2>/dev/null | head -1)
    if [ -n "$HF_LATEST" ] && [ -n "$VLLM_LATEST" ] && [ "$VLLM_LATEST" -nt "$HF_LATEST" ]; then
        NEED_VLLM_REBUILD=false
    fi
fi

if [ "$NEED_VLLM_REBUILD" = true ]; then
    echo "Building vLLM-compatible checkpoint ..."
    python "${PROJECT_DIR}/scripts/fix_vllm_ckpt.py" \
        --hf_dir "$HF_MODEL_PATH" \
        --out_dir "$VLLM_MODEL_PATH" \
        --force
    echo "vLLM checkpoint ready: $VLLM_MODEL_PATH"
else
    echo "vLLM checkpoint is up-to-date: $VLLM_MODEL_PATH"
fi

# ---------- output dir ----------
MODEL_TAG=$(basename "${MODEL_PATH%/}")
EXP_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/\([^/]*\)/global_step_[0-9]*.*|\1|p')
CKPT_STEP=$(echo "$MODEL_PATH" | sed -n 's|.*global_step_\([0-9]*\).*|\1|p')
if [ -n "$EXP_NAME" ]; then
    MODEL_TAG="${EXP_NAME}"
    [ -n "$CKPT_STEP" ] && MODEL_TAG="${MODEL_TAG}-step${CKPT_STEP}"
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/eval_timelens-${MODEL_TAG}}}"
RUN_TAG="${TIMESTAMP}"
[ "$ENABLE_THINKING" = "true" ] && RUN_TAG="${RUN_TAG}-think"
[ "${NO_ANSWER_WRAP:-0}" = "1" ] && RUN_TAG="${RUN_TAG}-naw"

# ---------- hardware + env ----------
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_RPC_TIMEOUT=200000
export VLLM_USE_V1="1"
export FORCE_QWENVL_VIDEO_READER=decord

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $(($(nvidia-smi -L | wc -l)-1)))}"
NUM_GPUS=${#GPULIST[@]}

if [ "$TP" -gt 1 ]; then
    if [ $((NUM_GPUS % TP)) -ne 0 ]; then
        echo "[ERR] NUM_GPUS=$NUM_GPUS not divisible by TP=$TP"
        exit 1
    fi
    NUM_SHARDS=$((NUM_GPUS / TP))
else
    NUM_SHARDS=$NUM_GPUS
fi

echo "=============================================="
echo "TimeLens-Bench Evaluation (vLLM)"
echo "=============================================="
echo "Checkpoint:    $MODEL_PATH"
echo "vLLM Model:    $VLLM_MODEL_PATH"
echo "Base Model:    $BASE_MODEL"
echo "Bench Dir:     $BENCH_DIR"
echo "Thinking:      $ENABLE_THINKING"
echo "Datasets:      $DATASETS"
echo "GPUs:          ${GPULIST[*]} (${NUM_GPUS} total, TP=$TP, shards=$NUM_SHARDS)"
echo "MaxModelLen:   $MAX_MODEL_LEN"
echo "GpuMemUtil:    $GPU_MEM_UTIL"
echo "Output root:   $OUTPUT_ROOT"
echo "=============================================="

# ---------- cleanup handler ----------
PIDS=()
cleanup() {
    echo ""
    echo "Caught interrupt, killing all workers ..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All workers killed."
    exit 1
}
trap cleanup INT TERM

gpu_slice_for_shard() {
    local IDX=$1
    local START=$((IDX * TP))
    local END=$((START + TP - 1))
    local SLICE=""
    for j in $(seq $START $END); do
        if [ -z "$SLICE" ]; then SLICE="${GPULIST[$j]}"; else SLICE="${SLICE},${GPULIST[$j]}"; fi
    done
    echo "$SLICE"
}

# ---------- run evaluation for each dataset ----------
IFS=',' read -ra DATASET_LIST <<< "$DATASETS"
NO_ANSWER_WRAP_FLAG=""
[ "$NO_ANSWER_WRAP" = "1" ] && NO_ANSWER_WRAP_FLAG="--no_answer_wrap"

for DATASET in "${DATASET_LIST[@]}"; do
    OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_TAG}"
    mkdir -p "$OUTPUT_DIR"
    echo ""
    echo ">>> Evaluating: $DATASET"
    echo "    Output:    $OUTPUT_DIR"
    PIDS=()

    for IDX in $(seq 0 $((NUM_SHARDS - 1))); do
        GPU_SLICE=$(gpu_slice_for_shard $IDX)
        SHARD_PORT=$((VLLM_BASE_PORT + IDX * 16))

        CUDA_VISIBLE_DEVICES=$GPU_SLICE \
        VLLM_PORT=$SHARD_PORT \
        VLLM_HOST_IP=127.0.0.1 \
        MASTER_PORT=$SHARD_PORT \
        MASTER_ADDR=127.0.0.1 \
        PYTHONUNBUFFERED=1 \
            python "${SCRIPT_DIR}/eval_timelens_bench.py" \
                --model_path "$VLLM_MODEL_PATH" \
                --processor_path "$BASE_MODEL" \
                --bench_dir "$BENCH_DIR" \
                --dataset "$DATASET" \
                --output_dir "$OUTPUT_DIR" \
                --enable_thinking "$ENABLE_THINKING" \
                --min_tokens "$MIN_TOKENS" \
                --total_tokens "$TOTAL_TOKENS" \
                --fps "$FPS" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                --num_workers "$NUM_WORKERS" \
                --chunk "$NUM_SHARDS" \
                --index "$IDX" \
                --tensor_parallel_size "$TP" \
                --max_model_len "$MAX_MODEL_LEN" \
                --gpu_mem_util "$GPU_MEM_UTIL" \
                --max_num_seqs "$MAX_NUM_SEQS" \
                --max_num_batched_tokens "$MAX_NUM_BATCHED_TOKENS" \
                --max_concurrent "$MAX_CONCURRENT" \
                --queue_size "$QUEUE_SIZE" \
                $NO_ANSWER_WRAP_FLAG \
                > "$OUTPUT_DIR/worker_${DATASET}_${IDX}.log" 2>&1 &
        PIDS+=($!)
        echo "  Launched shard $IDX on GPU(s) $GPU_SLICE (PID ${PIDS[-1]})"
    done

    echo "  Waiting for $NUM_SHARDS workers ..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        wait ${PIDS[$i]}
        RC=$?
        if [ $RC -ne 0 ]; then
            echo "  [FAIL] shard $i (PID ${PIDS[$i]}) exited with code $RC"
            FAILED=1
        else
            echo "  [DONE] shard $i (PID ${PIDS[$i]})"
        fi
    done

    if [ $FAILED -ne 0 ]; then
        echo "  Some workers failed. Check logs: $OUTPUT_DIR/worker_${DATASET}_*.log"
        continue
    fi

    # ---------- merge shard results ----------
    echo "  Merging results ..."
    python -c "
import json, os, sys

output_dir = sys.argv[1]
dataset = sys.argv[2]
num_shards = int(sys.argv[3])

all_samples = []
for sid in range(num_shards):
    path = os.path.join(output_dir, f'results_{dataset}_shard{sid}.json')
    if os.path.isfile(path):
        with open(path) as f:
            all_samples.extend(json.load(f))

n = len(all_samples)
if n == 0:
    print(f'  No results for {dataset}')
    sys.exit(0)

ious = [s['iou'] for s in all_samples]
thresholds = [0.3, 0.5, 0.7]
metrics = {
    'num_samples': n,
    'mIoU': round(sum(ious) / n * 100, 2),
    'R@0.3': round(sum(1 for v in ious if v >= 0.3) / n * 100, 2),
    'R@0.5': round(sum(1 for v in ious if v >= 0.5) / n * 100, 2),
    'R@0.7': round(sum(1 for v in ious if v >= 0.7) / n * 100, 2),
    'parse_rate': round(sum(1 for s in all_samples if s.get('parsed', False)) / n * 100, 2),
}
with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(f'  {dataset}: ' + ' | '.join(f'{k}={v}' for k, v in metrics.items()))
" "$OUTPUT_DIR" "$DATASET" "$NUM_SHARDS"

done

echo ""
echo "All datasets done. Results under: $OUTPUT_ROOT"

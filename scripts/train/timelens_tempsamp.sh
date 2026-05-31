#!/bin/bash
# =============================================================================
# TimeLens — TempSamp-R1 (GT injection + reward shaping)
#   on top of vanilla GRPO with a Qwen3-VL backbone (verl trainer).
#
# What this run turns on (vs scripts/train/timelens_grpo.sh):
#   1. GT injection (yaml algorithm.use_gt_injection=true, gt_replace_index=-1)
#        Every group of n=8 rollouts has its last slot replaced with a GT-built
#        "<answer> start to end </answer>"; group mean/std for GRPO advantage
#        runs on all 8 rows (the GT row anchors the baseline).
#   2. Non-linear reward shaping (env GTPO_REWARD_SHAPING=1)
#        IoU >= threshold squashed to a near-flat success plateau;
#        IoU <  threshold exponentially penalised.
#        threshold / alpha are configurable below.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RAY_BIN="${RAY_BIN:-$(command -v ray)}"

# ---------- model / data (override via env) ----------
export TIMELENS_MODEL_PATH="${TIMELENS_MODEL_PATH:-/path/to/Qwen3.5-VL-4B}"
export TIMELENS_TRAIN_FILES="${TIMELENS_TRAIN_FILES:-${PROJECT_DIR}/data/timelens_grpo_train.jsonl}"
export TIMELENS_VAL_FILES="${TIMELENS_VAL_FILES:-${TIMELENS_TRAIN_FILES}}"

export TIMELENS_USE_PREPROCESSED="${TIMELENS_USE_PREPROCESSED:-false}"
export TIMELENS_VIDEO_SOURCE_MODE="${TIMELENS_VIDEO_SOURCE_MODE:-realtime_only}"

# ---------- reward shaping (TempSamp-R1) ----------
export GTPO_REWARD_SHAPING="${GTPO_REWARD_SHAPING:-1}"
export GTPO_REWARD_SHAPING_THRESHOLD="${GTPO_REWARD_SHAPING_THRESHOLD:-0.8}"
export GTPO_REWARD_SHAPING_ALPHA="${GTPO_REWARD_SHAPING_ALPHA:-1.0}"

# ---------- training ----------
ROLLOUT_N="${ROLLOUT_N:-8}"
CONFIG_PATH="${PROJECT_DIR}/scripts/timelens/timelens_tempsamp.yaml"
FORMAT_PROMPT="scripts/timelens/format_prompt/timelens.jinja"
REWARD_FUNCTION="scripts/timelens/timelens_reward.py:compute_score"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-timelens-tempsamp-n${ROLLOUT_N}-$(date +%Y%m%d_%H%M%S)}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${PROJECT_DIR}/outputs/${EXPERIMENT_NAME}}"
FIND_LAST_CHECKPOINT="${FIND_LAST_CHECKPOINT:-false}"

# ---------- runtime ----------
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800000

export VERL_DISABLE_TQDM="${VERL_DISABLE_TQDM:-1}"
export VERL_PRINT_STEP_SUMMARY="${VERL_PRINT_STEP_SUMMARY:-1}"
# Safe under ppo_epochs=1 + one mini-batch / rollout step.
export VERL_SKIP_OLD_LOGPROBS="${VERL_SKIP_OLD_LOGPROBS:-1}"

WORLD_SIZE="${WORLD_SIZE:-1}"
RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"

# ---------- logs ----------
LOG_DIR="${PROJECT_DIR}/logs/tempsamp/timelens"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}.log"

cat <<EOF
================================================================================
  TimeLens TempSamp-R1 (GT injection + reward shaping)
================================================================================
  GT injection:    ON  (gt_replace_index=-1, gt_log_exclude=true)
  Reward shaping:  GTPO_REWARD_SHAPING=$GTPO_REWARD_SHAPING
                   threshold=$GTPO_REWARD_SHAPING_THRESHOLD alpha=$GTPO_REWARD_SHAPING_ALPHA
  Rollout n:       $ROLLOUT_N
  Model:           $TIMELENS_MODEL_PATH
  Train data:      $TIMELENS_TRAIN_FILES
  GPUs:            $NPROC_PER_NODE x $WORLD_SIZE nodes
  Experiment:      $EXPERIMENT_NAME
  Save path:       $SAVE_CHECKPOINT_PATH
  Log:             $LOG_FILE
================================================================================
EOF

cleanup_ray() {
    "${RAY_BIN}" stop --force 2>/dev/null || true
    rm -rf /dev/shm/* 2>/dev/null || true
    sleep 3
}

run_trainer() {
    "${PYTHON_BIN}" -m verl.trainer.main \
        config="${CONFIG_PATH}" \
        data.format_prompt="${FORMAT_PROMPT}" \
        worker.actor.model.model_path="${TIMELENS_MODEL_PATH}" \
        worker.reward.reward_function="${REWARD_FUNCTION}" \
        worker.rollout.n="${ROLLOUT_N}" \
        trainer.experiment_name="${EXPERIMENT_NAME}" \
        trainer.n_gpus_per_node="${NPROC_PER_NODE}" \
        trainer.nnodes="${WORLD_SIZE}" \
        trainer.save_checkpoint_path="${SAVE_CHECKPOINT_PATH}" \
        trainer.find_last_checkpoint="${FIND_LAST_CHECKPOINT}" \
        "$@"
}

cd "$PROJECT_DIR"
cleanup_ray
"${RAY_BIN}" start --head \
    --port="${MASTER_PORT}" \
    --dashboard-port="${RAY_DASHBOARD_PORT}" \
    --num-gpus="${NPROC_PER_NODE}" \
    --disable-usage-stats

run_trainer "$@" 2>&1 | tee -a "$LOG_FILE"
echo "Training finished: ${EXPERIMENT_NAME}"

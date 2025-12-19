#!/bin/bash

# ==============================================================================
# Manual Distributed Hyperparameter Search Script for A2C on Breakout
# 设置：normalize_advantage=True, normalize_advantage_mean=True, normalize_advantage_std=False
# ==============================================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || {
  echo "[Manager] Failed to cd to project root: ${ROOT_DIR}"
  exit 1
}

N_WORKERS=2
STORAGE="sqlite:///logs/breakout_a2c_manual.db"
STUDY_NAME="breakout_a2c_manual_mean_noStd_v1"
ALGO="a2c"
ENV="BreakoutNoFrameskip-v4"
N_TIMESTEPS=10000000
N_TRIALS=100000000
MAX_TOTAL_TRIALS=40
WANDB_PROJECT="Breakout-A2C-Search_new"
WANDB_ENTITY="agent-lab-ppo"

mkdir -p logs

pids=()

cleanup() {
    echo -e "\n[Manager] Caught interrupt signal! Killing all workers..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[Manager] Killing worker PID $pid"
            kill "$pid" 2>/dev/null
        fi
    done
    wait
    echo "[Manager] All workers terminated."
    exit 0
}

trap cleanup INT TERM

echo "[Manager] Starting $N_WORKERS distributed workers for study: $STUDY_NAME"
echo "[Manager] Storage: $STORAGE"
echo "[Manager] Logs will be written to logs/worker_${STUDY_NAME}_*.log"

for ((i=1; i<=N_WORKERS; i++)); do
    echo "[Manager] Launching worker $i..."

    GPU_ID=$(( (i - 1) % 2 ))

    CUDA_VISIBLE_DEVICES="$GPU_ID" python train_custom.py \
        --algo "$ALGO" \
        --env "$ENV" \
        --vec-env subproc \
        -n "$N_TIMESTEPS" \
        -optimize \
        --n-trials "$N_TRIALS" \
        --max-total-trials "$MAX_TOTAL_TRIALS" \
        --sampler tpe \
        --pruner median \
        --verbose 1 \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-entity "$WANDB_ENTITY" \
        --storage "$STORAGE" \
        --study-name "$STUDY_NAME" \
        -params normalize_advantage:True \
                normalize_advantage_mean:True \
                normalize_advantage_std:False \
        > "logs/worker_${STUDY_NAME}_$i.log" 2>&1 &

    pids+=($!)

    sleep 3
done

echo "[Manager] All workers launched. PIDs: ${pids[*]}"
echo "[Manager] To view logs, run: tail -f logs/worker_${STUDY_NAME}_*.log"
echo "[Manager] Waiting for workers... (Press Ctrl+C to stop)"

wait



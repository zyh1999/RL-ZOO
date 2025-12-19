#!/bin/bash

# ==============================================================================
# Manual Distributed Hyperparameter Search Script for A2C on Breakout
# ==============================================================================
# Usage (from anywhere):
#   bash /home/yihe/RL-ZOO/scripts/custom_searches/run_breakout_a2c_manual_dist.sh
#
# The script will cd 到 RL-ZOO 根目录，再启动多个 worker。
# ==============================================================================

# 避免线程数过多导致 CPU 打满
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

# cd 到 RL-ZOO 根目录（脚本所在位置的两级上级目录）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || {
  echo "[Manager] Failed to cd to project root: ${ROOT_DIR}"
  exit 1
}

# --- Configuration (默认：noNorm，normalize_advantage=False) ---
N_WORKERS=4
STORAGE="sqlite:///logs/breakout_a2c_manual.db"
STUDY_NAME="breakout_a2c_manual_noNorm_v1"
ALGO="a2c"
ENV="BreakoutNoFrameskip-v4"
N_TIMESTEPS=10000000                      # 1e7，与 a2c.yml Atari 默认一致
N_TRIALS=100000000                        # 单 worker 理论上限，实际由 MAX_TOTAL_TRIALS 截断
MAX_TOTAL_TRIALS=40                       # 全局最多 40 个 trial
WANDB_PROJECT="Breakout-A2C-Search"
WANDB_ENTITY="agent-lab-ppo"

# Ensure log directory exists (相对于项目根目录)
mkdir -p logs

# Array to store worker PIDs
pids=()

# --- Cleanup Handler ---
# Kills all child processes when the script receives SIGINT (Ctrl+C)
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

# Register the cleanup function for SIGINT (Ctrl+C) and SIGTERM
trap cleanup INT TERM

echo "[Manager] Starting $N_WORKERS distributed workers for study: $STUDY_NAME"
echo "[Manager] Storage: $STORAGE"
echo "[Manager] Logs will be written to logs/worker_${STUDY_NAME}_*.log"

# --- Launch Workers ---
for ((i=1; i<=N_WORKERS; i++)); do
    echo "[Manager] Launching worker $i..."

    # 计算 GPU ID: (i-1) % 2 => worker 1->gpu0, worker 2->gpu1, worker 3->gpu0, worker 4->gpu1
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
        -params normalize_advantage:False \
                normalize_advantage_mean:False \
                normalize_advantage_std:False \
        > "logs/worker_${STUDY_NAME}_$i.log" 2>&1 &

    # Save the background process PID
    pids+=($!)

    # 避免多个 worker 同时初始化 SQLite，稍作延迟
    sleep 3
done

echo "[Manager] All workers launched. PIDs: ${pids[*]}"
echo "[Manager] To view logs, run: tail -f logs/worker_${STUDY_NAME}_*.log"
echo "[Manager] Waiting for workers... (Press Ctrl+C to stop)"

# --- Wait ---
# This keeps the script running until all background jobs finish or are interrupted
wait



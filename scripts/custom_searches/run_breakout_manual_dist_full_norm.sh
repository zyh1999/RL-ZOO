#!/bin/bash

# ==============================================================================
# Manual Distributed Hyperparameter Search Script
# ==============================================================================
# Usage (from anywhere):
#   bash /home/yihe/RL-ZOO/scripts/custom_searches/run_breakout_manual_dist.sh
#
# The script will cd 到 RL-ZOO 根目录，再启动多个 worker。
# ==============================================================================

# cd 到 RL-ZOO 根目录（脚本所在位置的两级上级目录）
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

# --- Configuration ---
N_WORKERS=4
STORAGE="sqlite:///logs/breakout_manual.db"          # 复用之前的数据库
STUDY_NAME="breakout_manual_fullNorm_v1"             # 新的 study，强制开启 adv normalization
ALGO="ppo"
ENV="BreakoutNoFrameskip-v4"
N_TIMESTEPS=10000000
N_TRIALS=100000000                                  # 单 worker 理论上限，实际由 MAX_TOTAL_TRIALS 截断
MAX_TOTAL_TRIALS=40                                 # 全局最多 20 个 trial
WANDB_PROJECT="Breakout-PPO-Search"
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
    
    # We execute train_custom.py without --n-jobs (defaulting to 1)
    # The --storage and --study-name arguments ensure they coordinate via the DB.
    # Standard output and error are redirected to individual log files.
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
        > "logs/worker_${STUDY_NAME}_$i.log" 2>&1 &
    
    # Save the background process PID
    pids+=($!)
done

echo "[Manager] All workers launched. PIDs: ${pids[*]}"
echo "[Manager] To view logs, run: tail -f logs/worker_*.log"
echo "[Manager] Waiting for workers... (Press Ctrl+C to stop)"

# --- Wait ---
# This keeps the script running until all background jobs finish or are interrupted
wait


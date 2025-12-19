#!/bin/bash

# Threading control to avoid CPU oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

# ==============================================================================
# Single Environment Hyperparameter Search: Walker2d-v4
# Covers both fullNorm and noNorm settings.
# ==============================================================================

# --- Configuration ---
ENV="Walker2d-v4"
DB_FILE="logs/Walker2d-v4_sweep.db"
N_WORKERS=4
N_TIMESTEPS=10000000        # 10M
N_TRIALS=100000000          # Local budget (inf)
MAX_TOTAL_TRIALS=20         # Global budget
WANDB_PROJECT="Benchmark-PPO-Sweep_new"
WANDB_ENTITY="agent-lab-ppo"

# Settings: (Name  Mean  Std)
SETTINGS=(
  "fullNorm  true  true"
  "noNorm    false false"
)

# --- Prep ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || exit 1
mkdir -p logs

STORAGE="sqlite:///${DB_FILE}"

# Cleanup handler
pids=()
cleanup() {
    echo -e "\n[Manager] Caught interrupt! Killing current workers..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    exit 1
}
trap cleanup INT TERM

echo "=================================================================="
echo "Starting Benchmark for Env: ${ENV}"
echo "Database: ${STORAGE}"
echo "=================================================================="

# --- Main Loop ---
for SETTING in "${SETTINGS[@]}"; do
    read -r SETTING_NAME ADV_MEAN ADV_STD <<< "$SETTING"
    
    STUDY_NAME="${ENV}_${SETTING_NAME}_v1"
    
    echo "------------------------------------------------------------------"
    echo "Starting Search: ${STUDY_NAME}"
    echo "Setting: Mean=${ADV_MEAN}, Std=${ADV_STD}"
    echo "------------------------------------------------------------------"

    export FORCE_ADV_MEAN="$ADV_MEAN"
    export FORCE_ADV_STD="$ADV_STD"

    pids=()

    for ((i=1; i<=N_WORKERS; i++)); do
        GPU_ID=$(( (i - 1) % 2 ))
        
        echo "[Manager] Launching worker $i on GPU $GPU_ID..."
        CUDA_VISIBLE_DEVICES="$GPU_ID" python train_custom.py \
            --algo ppo \
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
        
        pids+=($!)

    # 避免多个 worker 同时初始化 SQLite，稍作延迟
    sleep 3
    done

    echo "Workers launched. PIDs: ${pids[*]}"
    wait
    echo "Finished: ${STUDY_NAME}"
done

echo "Benchmark for ${ENV} completed."

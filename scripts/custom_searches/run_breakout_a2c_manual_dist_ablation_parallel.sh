#!/bin/bash
#
# A2C Adv Normalization Ablation Search (Fully Parallel)
# 
# 任务：
#   在 BreakoutNoFrameskip-v4 上搜索 A2C 超参 (lr, ent_coef)
#   同时跑 4 种 Advantage Normalization 配置：
#     1. Mean + Std (Baseline)
#     2. Mean Only (No Std)
#     3. Std Only (No Mean)
#     4. No Norm (Raw Advantage)
#
# 并行度：
#   4 种配置 × 2 workers/配置 = 8 workers 总并行。
#   GPU 0 和 1 会被轮询分配 (各 4 个 worker)。
#
# 用法:
#   bash scripts/custom_searches/run_breakout_a2c_manual_dist_ablation_parallel.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# cd 到 RL-ZOO 根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || { echo "Failed to cd to ${ROOT_DIR}"; exit 1; }

# --- 全局配置 ---
N_WORKERS_PER_CONFIG=2
DB_FILE="logs/breakout_a2c_manual.db"
STORAGE="sqlite:///${DB_FILE}"
ALGO="a2c"
ENV="BreakoutNoFrameskip-v4"
N_TIMESTEPS=10000000
N_TRIALS=100000000
MAX_TOTAL_TRIALS=40
WANDB_PROJECT="Breakout-A2C-Search_new"
WANDB_ENTITY="agent-lab-ppo"

mkdir -p logs

# 定义 4 种配置 (Name  Normalize  Mean  Std)
CONFIGS=(
  "meanStd     True   True   True"
  "meanNoStd   True   True   False"
  "noMeanStd   True   False  True"
  "noNorm      False  False  False"
)

pids=()
cleanup() {
    echo -e "\n[Manager] Caught interrupt! Killing all workers..."
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait
    echo "[Manager] Terminated."
    exit 0
}
trap cleanup INT TERM

echo "========================================================"
echo "Starting FULLY PARALLEL Ablation Study (8 Workers Total)"
echo "Database: ${STORAGE}"
echo "========================================================"

global_worker_idx=0

for cfg_str in "${CONFIGS[@]}"; do
    read -r NAME NORM MEAN STD <<< "$cfg_str"
    STUDY_NAME="breakout_a2c_manual_${NAME}_v1"
    
    echo "[Manager] Spawning workers for config: ${NAME}..."
    
    for ((i=1; i<=N_WORKERS_PER_CONFIG; i++)); do
        # 轮询分配 GPU (0, 1, 0, 1...)
        GPU_ID=$(( global_worker_idx % 2 ))
        
        echo "  -> Worker ${i}/${N_WORKERS_PER_CONFIG} for ${NAME} on GPU ${GPU_ID} (PID file: logs/worker_${STUDY_NAME}_${i}.log)"
        
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
            -params normalize_advantage:${NORM} \
                    normalize_advantage_mean:${MEAN} \
                    normalize_advantage_std:${STD} \
            > "logs/worker_${STUDY_NAME}_$i.log" 2>&1 &
        
        pids+=($!)
        global_worker_idx=$((global_worker_idx + 1))
        
        # 稍微错开启动时间，避免 SQLite 锁竞争或瞬间 CPU 峰值
        sleep 2
    done
done

echo
echo "[Manager] All 8 workers launched. PIDs: ${pids[*]}"
echo "[Manager] Waiting for completion..."
wait
echo "[Manager] All studies finished."


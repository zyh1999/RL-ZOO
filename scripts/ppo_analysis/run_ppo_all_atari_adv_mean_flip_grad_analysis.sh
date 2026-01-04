#!/bin/bash
#
# PPO Atari: Adv-Mean Flip Grad Analysis (4 envs × 4 seeds × configs)
#
# 目标：
#   - 跑 4 个常用 Atari 环境
#   - 每个环境跑 4 个 seed
#   - 启用 PPO 的 adv-mean flip / clip-shift 梯度占比分析（analysis/* 指标）
#
# 用法：
#   bash RL-ZOO/scripts/ppo_analysis/run_ppo_all_atari_adv_mean_flip_grad_analysis.sh [CONCURRENCY]
#
# CONCURRENCY（默认 2）表示“每次并行跑几个环境”（每个环境固定 4 个 seed -> 4 个进程）。
#   - 1: 每次只跑 1 个环境（4 进程）
#   - 2: 每次跑 2 个环境（8 进程）
#   - 4: 一次跑完 4 个环境（16 进程）
#
# GPU 分配策略：
#   假定有 2 张卡 (GPU 0, 1)，按 (env_idx_in_batch * 4 + seed_idx) % 2 轮询分配。
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

CONCURRENCY=${1:-2}

# cd 到 RL-ZOO 根目录（避免从其它目录启动时找不到 train.py / 相对路径异常）
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || { echo "Failed to cd to ${ROOT_DIR}"; exit 1; }

mkdir -p logs/ppo_analysis

seeds=(9 1 2 3)
atari_envs=(
  "BreakoutNoFrameskip-v4"
  "BeamRiderNoFrameskip-v4"
  "QbertNoFrameskip-v4"
  "SeaquestNoFrameskip-v4"
)

TOTAL_ENVS=${#atari_envs[@]}
pids=()

trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "All runs killed."' INT

echo "Starting PPO Atari Adv-Mean Flip Grad Analysis with CONCURRENCY=$CONCURRENCY"

# 分批次跑环境（环境间串行/分批，避免一次性起太多进程）
for ((i=0; i<TOTAL_ENVS; i+=CONCURRENCY)); do
  batch_envs=("${atari_envs[@]:i:CONCURRENCY}")
  echo "========================================================"
  echo "Running batch envs: ${batch_envs[*]}"
  echo "========================================================"

  pids=()

  for env_idx in "${!batch_envs[@]}"; do
    env_id="${batch_envs[$env_idx]}"
    echo "  -> Env: ${env_id}"

    for s_idx in "${!seeds[@]}"; do
      seed="${seeds[$s_idx]}"
      gpu=$(( (env_idx * 4 + s_idx) % 2 ))

      run_name="ppo_advMeanFlipGrad_${env_id}"
      log_file="logs/ppo_analysis/${env_id}__seed${seed}__gpu${gpu}.log"

      CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
        --seed "${seed}" \
        --algo ppo \
        --env "${env_id}" \
        --vec-env subproc \
        --track \
        --wandb-run-extra-name "${run_name}" \
        --wandb-project-name sb3_ppo_adv_mean_flip_grad_analysis \
        --wandb-entity agent-lab-ppo \
        -params enable_adv_mean_flip_grad_analysis:True \
                normalize_advantage:True \
                normalize_advantage_mean:True \
                normalize_advantage_std:True \
                separate_optimizers:True \
                max_grad_norm:1e9 \
        > "${log_file}" 2>&1 &

      pids+=($!)
    done
  done

  echo "  Waiting for batch to finish..."
  # 不要因为某一个子进程失败导致整批直接退出（set -e 会让 wait 的非 0 返回码中断脚本）
  set +e
  wait "${pids[@]}"
  wait_rc=$?
  set -e
  if [[ $wait_rc -ne 0 ]]; then
    echo "  [WARN] Some runs in this batch exited non-zero. Check logs/ppo_analysis/*.log for details."
  fi
  echo "  Batch finished."
done

echo "All PPO Atari Adv-Mean Flip Grad Analysis runs finished."



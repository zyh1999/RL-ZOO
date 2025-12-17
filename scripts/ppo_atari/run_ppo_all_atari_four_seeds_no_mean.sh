#!/bin/bash

# 使用 PPO 默认 Atari 超参 + 自定义 advantage 归一化：
#   normalize_advantage: True
#   normalize_advantage_mean: False  （不减去 mean）
#   normalize_advantage_std: True   （只除以 std）
# 对 4 个 Atari 任务，每个任务 4 个 seeds，在两张 GPU 上并行跑。
# 任务列表：
#   BreakoutNoFrameskip-v4   （经典基准）
#   BeamRiderNoFrameskip-v4  （中等难度，reward 密集）
#   QbertNoFrameskip-v4      （中高难度，策略复杂度更高）
#   SeaquestNoFrameskip-v4   （探索/稀疏奖励更重，最难）

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# 使用两张 GPU（0 和 1）：
# - 前两个 seed 跑在 GPU 0
# - 后两个 seed 跑在 GPU 1

seeds=(9 1 2 3)
# 这里只跑 4 个代表性的 Atari 任务
atari_envs=(
  "BreakoutNoFrameskip-v4"
  "BeamRiderNoFrameskip-v4"
  "QbertNoFrameskip-v4"
  "SeaquestNoFrameskip-v4"
)

pids=()

trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "All runs killed."' INT

for env_id in "${atari_envs[@]}"; do
  echo "============================"
  echo "Starting env: $env_id (PPO default hyperparams + no-mean adv norm, 4 seeds, GPUs 0/1)"
  echo "============================"

  for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    if [ "$i" -lt 2 ]; then
      gpu=0
    else
      gpu=1
    fi

    run_name="ppo_noMean_adam"
    echo "  Launching seed $seed for $env_id on GPU ${gpu} | Run: $run_name"

    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --seed "${seed}" \
      --algo ppo \
      --env "${env_id}" \
      --vec-env subproc \
      --track \
      --wandb-run-extra-name "${run_name}" \
      --wandb-project-name sb3 \
      --wandb-entity agent-lab-ppo \
      -params normalize_advantage:True \
              normalize_advantage_mean:False \
              normalize_advantage_std:True \
              separate_optimizers:True \
      &

    pids+=($!)
  done

  # 当前 env 的 4 个 seed 全部完成后再切下一个 env
  wait "${pids[@]}"
  pids=()
  echo "Finished env: $env_id"
  echo

done

echo "All Atari PPO no-mean runs finished."


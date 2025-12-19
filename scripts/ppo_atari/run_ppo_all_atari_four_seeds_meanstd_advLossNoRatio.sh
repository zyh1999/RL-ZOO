#!/bin/bash

# 使用 PPO Atari 超参（ppo.yml 中的 atari 段）+ advantage 归一化（mean+std），
# 再额外打开新加的开关：
#   adv_loss_remove_ratio: True
# 含义：clip mask 仍然用带 ratio 的 PPO surrogate，但最终策略损失把 min(...) 除回 ratio，
# 让“真 loss”更接近只依赖 advantage 的形状。
#
# 任务：4 个 Atari 环境 × 4 seeds，在两张 GPU 上并行跑。
# 环境列表：
#   BreakoutNoFrameskip-v4
#   BeamRiderNoFrameskip-v4
#   QbertNoFrameskip-v4
#   SeaquestNoFrameskip-v4

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
  echo "Starting env: $env_id (PPO mean+std + adv_loss_remove_ratio, 4 seeds, GPUs 0/1)"
  echo "============================"

  for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    if [ "$i" -lt 2 ]; then
      gpu=0
    else
      gpu=1
    fi

    run_name="ppo_meanStd_advLossNoRatio"
    echo "  Launching seed $seed for $env_id on GPU ${gpu} | Run: $run_name"

    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --seed "${seed}" \
      --algo ppo \
      --env "${env_id}" \
      --vec-env subproc \
      --track \
      --wandb-run-extra-name "${run_name}" \
      --wandb-project-name sb3_new \
      --wandb-entity agent-lab-ppo \
      -params normalize_advantage:True \
              normalize_advantage_mean:True \
              normalize_advantage_std:True \
              separate_optimizers:True \
              adv_loss_remove_ratio:True \
      &

    pids+=($!)
  done

  # 当前 env 的 4 个 seed 全部完成后再切下一个 env
  wait "${pids[@]}"
  pids=()
  echo "Finished env: $env_id"
  echo

done

echo "All Atari PPO mean+std + adv_loss_remove_ratio runs finished."




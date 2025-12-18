#!/bin/bash

# 使用 PPOAdvDecouple 在 Atari 上跑 4 个环境 × 4 seeds：
# 这个脚本对应解耦组合：
#   clip_mask_use_adv_mean: True   （mask 减 mean）
#   loss_use_adv_mean:      False  （loss 不减 mean）
#   loss_use_adv_std:       True   （loss 除 std）
#
# 其他超参沿用 ppo.yml 中 atari 段（通过 rl-zoo3 的默认回退机制）：
#   normalize_advantage:      True
#   normalize_advantage_mean: True
#   normalize_advantage_std:  True
#   separate_optimizers:      True

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
  echo "Starting env: $env_id (PPOAdvDecouple mask-mean / loss-noMean, 4 seeds, GPUs 0/1)"
  echo "============================"

  for i in "${!seeds[@]}"; do
    seed="${seeds[$i]}"
    if [ "$i" -lt 2 ]; then
      gpu=0
    else
      gpu=1
    fi

    run_name="ppoAdvDecouple_maskMean_lossNoMean_lossStd"
    echo "  Launching seed $seed for $env_id on GPU ${gpu} | Run: $run_name"

    CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
      --seed "${seed}" \
      --algo ppo_adv_decouple \
      --env "${env_id}" \
      --vec-env subproc \
      --track \
      --wandb-run-extra-name "${run_name}" \
      --wandb-project-name sb3 \
      --wandb-entity agent-lab-ppo \
      -params normalize_advantage:True \
              normalize_advantage_mean:True \
              normalize_advantage_std:True \
              separate_optimizers:True \
              loss_use_adv_mean:False \
              loss_use_adv_std:True \
              clip_mask_use_adv_mean:True \
      &

    pids+=($!)
  done

  # 当前 env 的 4 个 seed 全部完成后再切下一个 env
  wait "${pids[@]}"
  pids=()
  echo "Finished env: $env_id"
  echo

done

echo "All Atari PPOAdvDecouple mask-mean / loss-noMean runs finished."




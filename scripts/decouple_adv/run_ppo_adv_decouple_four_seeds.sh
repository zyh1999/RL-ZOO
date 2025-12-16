#!/bin/bash

# PPOAdvDecouple 四种子并行跑法示例
# - 4 seeds: 9, 1, 2, 3
# - 前两个种子用 GPU 0，后两个用 GPU 1
# - 默认超参回退到 ppo.yml；仅在 -params 中显式打开解耦相关开关

env_id="BreakoutNoFrameskip-v4"
seeds=(9 1 2 3)
pids=()

trap 'echo "捕获 Ctrl+C，正在终止所有子进程..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "子进程已全部终止。"' INT

for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  if [ "$i" -lt 2 ]; then
    gpu=0
  else
    gpu=1
  fi

  echo "启动 seed $seed on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    --seed "$seed" \
    --algo ppo_adv_decouple \
    --env "$env_id" \
    --vec-env subproc \
    --track \
    --wandb-run-extra-name "ppo_adv_decouple_mean_mask_no_mean_loss_${env_id}_seed${seed}" \
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

wait "${pids[@]}"


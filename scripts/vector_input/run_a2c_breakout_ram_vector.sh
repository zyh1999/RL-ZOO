#!/bin/bash

# A2C on Breakout-ram-v4 with vector (RAM) input + MLP policy
# - 4 seeds: 9, 1, 2, 3
# - 前两个种子在 GPU 0，后两个在 GPU 1
# - 使用 MlpPolicy + Adam 优化器 + advantage normalization + sep_optim
# - 关键：覆盖 env_wrapper 防止加载 Atari 图像预处理；开启 normalize 对 MLP 输入做标准化

seeds=(9 1 2 3)
pids=()

# 捕获 Ctrl+C，杀掉所有子进程
trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "All runs killed."' INT

# 使用下标区分前一半 / 后一半
for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"

  if [ "$i" -lt 2 ]; then
    gpu=0
  else
    gpu=1
  fi

  echo "Starting seed $seed on GPU $gpu"

  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    --seed "$seed" \
    --algo a2c \
    --env Breakout-ram-v4 \
    --vec-env subproc \
    --track \
    --wandb-run-extra-name a2c_ram_vector_sep_optim_adam_normadv \
    --wandb-project-name sb3_new \
    --wandb-entity agent-lab-ppo \
    -params policy:'MlpPolicy' \
            n_steps:5 \
            sep_optim:True \
            normalize_advantage:True \
            normalize_advantage_mean:True \
            normalize_advantage_std:True \
            use_rms_prop:False \
            normalize:True \
            env_wrapper:None \
            frame_stack:4 \
            "policy_kwargs:dict(optimizer_class=get_class_by_name('torch.optim.Adam'), net_arch=dict(pi=[256,256], vf=[256,256]))" \
    &

  pids+=($!)
done

wait "${pids[@]}"

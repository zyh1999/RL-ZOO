#!/bin/bash

# PPOAdvDecouple 四种子并行跑法示例
# - 4 seeds: 9, 1, 2, 3
# - 前两个种子用 GPU 0，后两个用 GPU 1
# - 默认超参回退到 ppo.yml；仅在 -params 中显式打开解耦相关开关
# - 三个可选开关：MASK_MEAN（clip_mask_use_adv_mean）、LOSS_MEAN（loss_use_adv_mean）、LOSS_STD（loss_use_adv_std）
#   默认都为 True（标准 adv_norm）；可在命令行 export 覆盖，如：
#   MASK_MEAN=False LOSS_MEAN=False LOSS_STD=True ./run_ppo_adv_decouple_four_seeds.sh

# 开关默认值：都开（True）
# 优先级：命令行参数 > 环境变量 > 默认 True
# 用法示例：
#   ./run_ppo_adv_decouple_four_seeds.sh False True   # MASK_MEAN=False, LOSS_MEAN=True, LOSS_STD 默认/环境变量
#   MASK_MEAN=False LOSS_MEAN=False ./run_ppo_adv_decouple_four_seeds.sh  # 仍然兼容原来的环境变量写法
MASK_MEAN=${1:-${MASK_MEAN:-True}}
LOSS_MEAN=${2:-${LOSS_MEAN:-True}}
LOSS_STD=${LOSS_STD:-True}

env_id="BreakoutNoFrameskip-v4"
seeds=(9 1 2 3)
pids=()

echo "当前开关：MASK_MEAN=${MASK_MEAN}, LOSS_MEAN=${LOSS_MEAN}, LOSS_STD=${LOSS_STD}"

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

  run_name="ppo_adv_decouple_maskMean${MASK_MEAN}_lossMean${LOSS_MEAN}_lossStd${LOSS_STD}"

  echo "启动 seed $seed on GPU $gpu | Run: $run_name"
  CUDA_VISIBLE_DEVICES="$gpu" python train.py \
    --seed "$seed" \
    --algo ppo_adv_decouple \
    --env "$env_id" \
    --vec-env subproc \
    --track \
    --wandb-run-extra-name "$run_name" \
    --wandb-project-name sb3_new \
    --wandb-entity agent-lab-ppo \
    -params normalize_advantage:True \
            normalize_advantage_mean:True \
            normalize_advantage_std:True \
            separate_optimizers:True \
            loss_use_adv_mean:${LOSS_MEAN} \
            loss_use_adv_std:${LOSS_STD} \
            clip_mask_use_adv_mean:${MASK_MEAN} \
    &

  pids+=($!)
done

wait "${pids[@]}"



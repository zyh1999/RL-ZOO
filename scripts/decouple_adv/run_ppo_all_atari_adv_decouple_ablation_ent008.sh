#!/bin/bash
#
# PPO Adv Decouple Ablation Study (Entropy Coef = 0.08)
#
# 基于：scripts/decouple_adv/run_ppo_all_atari_adv_decouple_ablation.sh
# 变化：
#   - 额外强制设置 ent_coef = 0.08
#
# 用法：
#   bash run_ppo_all_atari_adv_decouple_ablation_ent008.sh [CONCURRENCY]
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# cd 到 RL-ZOO 根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}" || { echo "Failed to cd to ${ROOT_DIR}"; exit 1; }

# 参数：每次并行的配置数量
CONCURRENCY=${1:-2}

# 定义所有 4 种配置 (Name MASK_MEAN LOSS_MEAN)
# LOSS_STD 默认为 True
ALL_CONFIGS=(
  "baseline      True  True"
  "noMaskMean    False True"
  "noLossMean    True  False"
  "allNoMean     False False"
)

seeds=(9 1 2 3)
atari_envs=(
  "BreakoutNoFrameskip-v4"
  "BeamRiderNoFrameskip-v4"
  "QbertNoFrameskip-v4"
  "SeaquestNoFrameskip-v4"
)

# 计算总配置数
TOTAL_CONFIGS=${#ALL_CONFIGS[@]}

pids=()

trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "All runs killed."' INT

echo "Starting Ablation Study (PPO Adv Decouple, ent_coef=0.08) with CONCURRENCY=$CONCURRENCY"

# 外层循环：遍历环境（环境间串行，保证显存释放）
for env_id in "${atari_envs[@]}"; do
  echo "========================================================"
  echo "Starting Env: $env_id"
  echo "========================================================"

  # 内层循环：分批次执行配置
  for ((i=0; i<TOTAL_CONFIGS; i+=CONCURRENCY)); do
    
    # 获取当前批次的配置
    batch_configs=("${ALL_CONFIGS[@]:i:CONCURRENCY}")
    
    echo "  Running Batch starting at config index $i..."
    pids=() # 清空当前批次的 PID 列表

    # 遍历当前批次的每一个配置
    for config_str in "${batch_configs[@]}"; do
      read -r CONFIG_NAME MASK_MEAN LOSS_MEAN <<< "$config_str"
      LOSS_STD="True"

      echo "    -> Launching Config: $CONFIG_NAME (MASK_MEAN=$MASK_MEAN, LOSS_MEAN=$LOSS_MEAN, ent_coef=0.08)"

      # 为该配置启动 4 个 seed
      for s_idx in "${!seeds[@]}"; do
        seed="${seeds[$s_idx]}"
        
        # 找到 config 在 batch 里的索引
        cfg_idx=-1
        for idx in "${!batch_configs[@]}"; do
           if [[ "${batch_configs[$idx]}" == "$config_str" ]]; then cfg_idx=$idx; break; fi
        done
        
        global_job_id=$(( cfg_idx * 4 + s_idx ))
        gpu=$(( global_job_id % 2 ))

        run_name="ppo_decouple_${CONFIG_NAME}_ent008"
        
        # 启动训练进程
        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo ppo_adv_decouple \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_new \
          --wandb-entity agent-lab-ppo \
          -params ent_coef:0.08 \
                  normalize_advantage:True \
                  normalize_advantage_mean:True \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  loss_use_adv_mean:${LOSS_MEAN} \
                  loss_use_adv_std:${LOSS_STD} \
                  clip_mask_use_adv_mean:${MASK_MEAN} \
          > /dev/null 2>&1 &

        pids+=($!)
      done
    done

    # 等待当前批次的所有配置完成
    echo "    Waiting for batch to finish..."
    wait "${pids[@]}"
    echo "    Batch finished."
  done

  echo "Finished Env: $env_id"
  echo
done

echo "All Ablation Studies Finished."



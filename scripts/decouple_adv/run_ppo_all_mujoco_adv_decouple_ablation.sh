#!/bin/bash
#
# PPO Adv Decouple Ablation Study (4 seeds per config) - Mujoco Version
#
# 任务：
#   遍历 4 个 Mujoco 环境 (Ant-v4, HalfCheetah-v4, Hopper-v4, Walker2d-v4)
#   测试 4 种 Decouple 配置
#   每个配置跑 4 个 seed
#
# 特殊配置:
#   - 禁用 Reward Normalization (norm_reward: False)
#   - max_grad_norm: 1e9 (相当于禁用梯度裁剪)
#
# 运行模式:
#   bash run_ppo_all_mujoco_adv_decouple_ablation.sh [CONCURRENCY]
#   
#   CONCURRENCY (默认 2): 每次并行跑几种配置。
#     - 1: 每次跑 1 个配置 (4 进程), 最稳妥
#     - 2: 每次跑 2 个配置 (8 进程), 默认
#     - 4: 一次跑完所有配置 (16 进程), 需要大显存/多卡
#
# GPU 分配策略：
#   脚本假定有 2 张卡 (GPU 0, 1)。
#   所有并行任务会自动在 0/1 之间轮询分配。
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

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
# Mujoco 环境列表
mujoco_envs=(
  # "Ant-v4"
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Swimmer-v4"
  "Humanoid-v4"
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

echo "Starting Mujoco Ablation Study (No Reward Norm, GradNorm 1e9) with CONCURRENCY=$CONCURRENCY"

# 外层循环：遍历环境（环境间串行，保证显存释放）
for env_id in "${mujoco_envs[@]}"; do
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

      echo "    -> Launching Config: $CONFIG_NAME (MASK_MEAN=$MASK_MEAN, LOSS_MEAN=$LOSS_MEAN)"

      # 为该配置启动 4 个 seed
      for s_idx in "${!seeds[@]}"; do
        seed="${seeds[$s_idx]}"
        
        # 简单的 GPU 轮询分配 (全局轮询)
        # 找到 config 在 batch 里的索引
        cfg_idx=-1
        for idx in "${!batch_configs[@]}"; do
           if [[ "${batch_configs[$idx]}" == "$config_str" ]]; then cfg_idx=$idx; break; fi
        done
        
        global_job_id=$(( cfg_idx * 4 + s_idx ))
        gpu=$(( global_job_id % 2 ))

        run_name="ppo_decouple_mujoco_${CONFIG_NAME}"
        
        # 启动训练进程
        # 使用通用 Mujoco 参数配置 (覆盖 yaml 中的默认值)
        # 1. normalize:"{'norm_obs':True,'norm_reward':False}" 禁用 Reward Normalization
        # 2. max_grad_norm:1e9 放宽梯度裁剪
        # 3. 其他参数采用 PPO 经典通用配置 (n_steps=2048, batch_size=64, lr=3e-4, etc.)
        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo ppo_adv_decouple \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_new_mujoco_gradnorm0.5_normreward \
          --wandb-entity agent-lab-ppo \
          -params normalize:"{'norm_obs':True,'norm_reward':True}" \
                  max_grad_norm:0.5 \
                  policy:'MlpPolicy' \
                  n_steps:2048 \
                  batch_size:64 \
                  learning_rate:3e-4 \
                  gamma:0.99 \
                  gae_lambda:0.95 \
                  ent_coef:0.0 \
                  clip_range:0.2 \
                  n_epochs:10 \
                  normalize_advantage:True \
                  normalize_advantage_mean:True \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  loss_use_adv_mean:${LOSS_MEAN} \
                  loss_use_adv_std:${LOSS_STD} \
                  clip_mask_use_adv_mean:${MASK_MEAN} \
          > /dev/null 2>&1 &  # 减少输出干扰，或重定向到日志文件

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

echo "All Mujoco Ablation Studies Finished."


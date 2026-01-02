#!/bin/bash
#
# A2C Adv Normalization Ablation Study (4 seeds per config)
#
# 任务：
#   在 BreakoutNoFrameskip-v4 上测试 4 种 Advantage Normalization 配置：
#     1. Mean + Std (Baseline)
#     2. Mean Only (No Std)
#     3. Std Only (No Mean)
#     4. No Norm (Raw Advantage)
#
# 超参数选择 (基于 Search 结果的通用配置):
#   - Learning Rate: 4e-4
#   - Ent Coef: 0.01
#   - Optimizer: Adam (默认)
#   - n_envs: 16 (A2C默认)
#
# 运行模式:
#   bash run_a2c_all_atari_adv_norm_ablation.sh [CONCURRENCY]
#   
#   CONCURRENCY (默认 2): 每次并行跑几种配置。
#     - 1: 每次跑 1 个配置 (4 seeds = 4 进程)
#     - 2: 每次跑 2 个配置 (8 seeds = 8 进程), 默认
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

# 定义所有 4 种配置 (Name NORM_MEAN NORM_STD)
# 注意：normalize_advantage (总开关) 必须始终为 True，
#       然后通过 mean/std 的细粒度开关来控制。
#       如果 mean=False 且 std=False，其实际效果等于 normalize_advantage=False。
ALL_CONFIGS=(
  "meanStd     True   True"
  "meanNoStd   True   False"
  "noMeanStd   False  True"
  "noNorm      False  False"
)

seeds=(9 1 2 3)
# 这里只跑 Breakout，如果需要更多环境可以在这里添加
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

echo "Starting A2C Adv Norm Ablation (LR=4e-4, Ent=0.01) with CONCURRENCY=$CONCURRENCY"

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
      read -r CONFIG_NAME ADV_MEAN ADV_STD <<< "$config_str"

      echo "    -> Launching Config: $CONFIG_NAME (Mean=$ADV_MEAN, Std=$ADV_STD)"

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

        run_name="a2c_advNorm_0.08_ent_coef_${CONFIG_NAME}"
        
        # 启动训练进程
        # 使用选定的通用超参数: lr=3e-4, ent_coef=0.01
        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo a2c \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_a2c_adv_norm_ablation \
          -params normalize_advantage:True \
                  normalize_advantage_mean:${ADV_MEAN} \
                  normalize_advantage_std:${ADV_STD} \
                  learning_rate:3e-4 \
                  ent_coef:0.08 \
                  n_envs:16 \
          > /dev/null 2>&1 &  # 减少输出干扰

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

echo "All A2C Adv Norm Ablation Studies Finished."


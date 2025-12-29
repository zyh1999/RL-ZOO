#!/bin/bash
#
# PPO Adv Decouple Ablation Study (Slow Critic Update)
# 
# 任务：
#   遍历 4 个 Atari 环境
#   测试 4 种 Decouple 配置
#   在 slow_critic_update_interval=5 的情况下进行对比
#   每个配置跑 4 个 seed
#
# 运行模式:
#   bash run_ppo_all_atari_adv_decouple_slow_critic.sh [CONCURRENCY]
#   
#   CONCURRENCY (默认 2): 每次并行跑几种配置。
#     - 1: 每次跑 1 个配置 (4 进程), 最稳妥
#     - 2: 每次跑 2 个配置 (8 进程), 默认
#
# GPU 分配策略：
#   脚本假定有 2 张卡 (GPU 0, 1)。
#   所有并行任务会自动在 0/1 之间轮询分配。

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# 参数：每次并行的配置数量
CONCURRENCY=${1:-2}

# 设置 Critic 更新间隔
SLOW_INTERVAL=20

# 定义所有 4 种配置 (Name MASK_MEAN LOSS_MEAN)
# LOSS_STD 默认为 True
ALL_CONFIGS=(
  "baseline_slow${SLOW_INTERVAL}      True  True"
  "noMaskMean_slow${SLOW_INTERVAL}    False True"
  "noLossMean_slow${SLOW_INTERVAL}    True  False"
  "allNoMean_slow${SLOW_INTERVAL}     False False"
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

echo "Starting Slow Critic Ablation Study (Interval=${SLOW_INTERVAL}) with CONCURRENCY=$CONCURRENCY"

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

      echo "    -> Launching Config: $CONFIG_NAME (MASK_MEAN=$MASK_MEAN, LOSS_MEAN=$LOSS_MEAN, INTERVAL=$SLOW_INTERVAL)"

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

        run_name="ppo_decouple_${CONFIG_NAME}"
        
        # 启动训练进程
        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo ppo_adv_decouple \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_critic_slow \
          --wandb-entity agent-lab-ppo \
          -params normalize_advantage:True \
                  normalize_advantage_mean:True \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  loss_use_adv_mean:${LOSS_MEAN} \
                  loss_use_adv_std:${LOSS_STD} \
                  clip_mask_use_adv_mean:${MASK_MEAN} \
                  slow_critic_update_interval:${SLOW_INTERVAL} \
          > /dev/null 2>&1 &  # 减少输出干扰
        
        # 保存 PID
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

echo "All Slow Critic Ablation Studies Finished."


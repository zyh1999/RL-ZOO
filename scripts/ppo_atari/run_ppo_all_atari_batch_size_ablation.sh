#!/bin/bash
#
# PPO Adv-Mean Ablation with Different Batch Sizes (Atari)
#
# 任务：
#   对比在不同 mini-batch size 下，normalize_advantage_mean (是否减均值) 对性能的影响。
#   Atari 默认 n_steps=128, n_envs=8 -> rollout buffer size = 1024。
#   默认 batch_size = 256。
#
# 配置组合：
#   1. BatchSize=32,  AdvMean=True
#   2. BatchSize=32,  AdvMean=False
#   3. BatchSize=256, AdvMean=True (Baseline)
#   4. BatchSize=256, AdvMean=False
#   5. BatchSize=1024, AdvMean=True (Full Batch)
#   6. BatchSize=1024, AdvMean=False (Full Batch)
#
# 运行方式：
#   bash run_ppo_all_atari_batch_size_ablation.sh [CONCURRENCY]
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# 参数：每次并行的配置数量
CONCURRENCY=${1:-2}

# 定义所有配置 (Name BATCH_SIZE NORM_ADV_MEAN)
ALL_CONFIGS=(
  "bs8_withMean    8    True"
  "bs8_noMean      8    False"
  "bs16_withMean   16   True"
  "bs16_noMean     16   False"
  "bs64_withMean   64   True"
  "bs64_noMean     64   False"
  "bs256_withMean  256  True"
  "bs256_noMean    256  False"
  "bs512_withMean  512  True"
  "bs512_noMean    512  False"
  "bs1024_withMean 1024 True"
  "bs1024_noMean   1024 False"
)

seeds=(9 1 2 3)
atari_envs=(
  "BreakoutNoFrameskip-v4"
  #"BeamRiderNoFrameskip-v4"
  #"QbertNoFrameskip-v4"
  #"SeaquestNoFrameskip-v4"
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

echo "Starting PPO Batch Size Ablation with CONCURRENCY=$CONCURRENCY"

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
      read -r CONFIG_NAME BATCH_SIZE NORM_ADV_MEAN <<< "$config_str"

      echo "    -> Launching Config: $CONFIG_NAME (BatchSize=$BATCH_SIZE, AdvMean=$NORM_ADV_MEAN)"

      # 为该配置启动 4 个 seed
      for s_idx in "${!seeds[@]}"; do
        seed="${seeds[$s_idx]}"

        # 简单 GPU 轮询分配
        # 找到 config 在 batch 里的索引
        cfg_idx=-1
        for idx in "${!batch_configs[@]}"; do
           if [[ "${batch_configs[$idx]}" == "$config_str" ]]; then cfg_idx=$idx; break; fi
        done

        global_job_id=$(( cfg_idx * 4 + s_idx ))
        gpu=$(( global_job_id % 2 ))

        run_name="ppo_bs_ablation_${CONFIG_NAME}"

        # 启动训练进程
        # 注意：这里显式传入 batch_size 覆盖 yaml 默认值
        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo ppo \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_ppo_batch_size_ablation \
          -params normalize_advantage:True \
                  normalize_advantage_mean:${NORM_ADV_MEAN} \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  batch_size:${BATCH_SIZE} \
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

echo "All PPO Batch Size Ablation runs finished."


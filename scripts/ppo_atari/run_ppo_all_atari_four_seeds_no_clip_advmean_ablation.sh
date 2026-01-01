#!/bin/bash
#
# PPO No-Clip Ablation:
#   把 clip_range 开到一个极大值，几乎等价于“没有 ratio clipping”
#   对比两种配置：
#     1) 减 mean：normalize_advantage_mean = True
#     2) 不减 mean：normalize_advantage_mean = False
#   其它超参与默认 Atari PPO 保持一致（包括 normalize_advantage_std=True）。
#
# 运行方式：
#   bash run_ppo_all_atari_four_seeds_no_clip_advmean_ablation.sh [CONCURRENCY]
#   CONCURRENCY 默认 2，含义和 decouple_adv/run_ppo_all_atari_adv_decouple_ablation.sh 一致：
#     - 1: 一次只跑 1 个配置（4 进程）
#     - 2: 一次并行 2 个配置（8 进程）
#     - 4: 一次并行更多配置（本脚本只有 2 个配置，其实 2 已经足够）
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# 参数：每次并行的配置数量
CONCURRENCY=${1:-2}

# 定义所有配置 (Name NORM_ADV_MEAN)
ALL_CONFIGS=(
  "withMean True"
  "noMean  False"
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

echo "Starting PPO No-Clip Adv-Mean Ablation with CONCURRENCY=$CONCURRENCY"

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
      read -r CONFIG_NAME NORM_ADV_MEAN <<< "$config_str"

      echo "    -> Launching Config: $CONFIG_NAME (normalize_advantage_mean=$NORM_ADV_MEAN)"

      # 为该配置启动 4 个 seed
      for s_idx in "${!seeds[@]}"; do
        seed="${seeds[$s_idx]}"

        # 简单 GPU 轮询分配：同一批次内，(config_idx_in_batch * 4 + seed_idx) % 2
        cfg_idx=-1
        for idx in "${!batch_configs[@]}"; do
           if [[ "${batch_configs[$idx]}" == "$config_str" ]]; then cfg_idx=$idx; break; fi
        done

        global_job_id=$(( cfg_idx * 4 + s_idx ))
        gpu=$(( global_job_id % 2 ))

        run_name="ppo_noClip_${CONFIG_NAME}"

        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo ppo \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_ppo_no_clip \
          -params normalize_advantage:True \
                  normalize_advantage_mean:${NORM_ADV_MEAN} \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  clip_range:1000000000.0 \
          > /dev/null 2>&1 &  # 可根据需要改成重定向到日志文件

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

echo "All PPO No-Clip Adv-Mean Ablation runs finished."



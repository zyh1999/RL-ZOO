#!/bin/bash
#
# PPO Ablation (Advantage Mean, Norm Reward, Share Features) - Mujoco
#
# 任务：
#   验证 PPO 的以下三个变量的组合效果 (2*2*2 = 8种配置)：
#   1. share_features_extractor (True/False)
#   2. norm_reward (True/False)
#   3. normalize_advantage_mean (True/False)
#      * normalize_advantage_std 始终为 True
#
#   运行模式:
#   bash scripts/ppo_mujoco_ablation/run_ppo_sharing_ablation.sh [CONCURRENCY]
#   
#   CONCURRENCY (默认 2): 每次并行跑几种配置。
#     - 1: 每次跑 1 个配置 (4 进程)
#     - 2: 每次跑 2 个配置 (8 进程) -> 默认
#

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

set -e

# 并行度参数
CONCURRENCY=${1:-2}

# 定义配置组合: Name SHARE_FLAG NORM_REWARD_FLAG ADV_MEAN_FLAG
CONFIGS=(
  "Shared_NR_Mean      True   True   True"
  "Shared_NR_NoMean    True   True   False"
  "Shared_NoNR_Mean    True   False  True"
  "Shared_NoNR_NoMean  True   False  False"
  "Split_NR_Mean       False  True   True"
  "Split_NR_NoMean     False  True   False"
  "Split_NoNR_Mean     False  False  True"
  "Split_NoNR_NoMean   False  False  False"
)

seeds=(1 2 3 9)
mujoco_envs=(
  "Hopper-v4"
  "HalfCheetah-v4"
)

TOTAL_CONFIGS=${#CONFIGS[@]}
pids=()

trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait; \
      echo "All runs killed."' INT

echo "Starting PPO Ablation (Concurrency=$CONCURRENCY)..."

for env_id in "${mujoco_envs[@]}"; do
  echo "========================================================"
  echo "Starting Env: $env_id"
  echo "========================================================"

  # 分批次执行配置
  for ((i=0; i<TOTAL_CONFIGS; i+=CONCURRENCY)); do
    batch_configs=("${CONFIGS[@]:i:CONCURRENCY}")
    echo "  Running Batch starting at config index $i..."
    pids=()

    for config_str in "${batch_configs[@]}"; do
      read -r NAME SHARE_FLAG NORM_REWARD_FLAG ADV_MEAN_FLAG <<< "$config_str"
      echo "    -> Launching Config: $NAME (share=$SHARE_FLAG, norm_rew=$NORM_REWARD_FLAG, adv_mean=$ADV_MEAN_FLAG)"

      for seed in "${seeds[@]}"; do
        run_name="ppo_${env_id}_${NAME}_seed${seed}"
        
        # 简单的 GPU 轮询 (seed % 2) - 如果并发量大，可能需要更精细的分配
        # 这里仍保持 seed 决定 gpu，可能会有一定不均衡，但 8 个进程分摊到 2 卡通常可以接受
        gpu=$(( seed % 2 ))

        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --algo ppo \
          --env "${env_id}" \
          --seed "${seed}" \
          --vec-env subproc \
          --track \
          --wandb-project-name sb3_ppo_mujoco_ablation_3factor \
          --wandb-run-extra-name "${run_name}" \
          --wandb-entity agent-lab-ppo \
          -params policy_kwargs:"dict(share_features_extractor=${SHARE_FLAG})" \
                  normalize:"{'norm_obs':True,'norm_reward':${NORM_REWARD_FLAG}}" \
                  max_grad_norm:1e9 \
                  normalize_advantage:True \
                  normalize_advantage_mean:${ADV_MEAN_FLAG} \
                  normalize_advantage_std:True \
                  separate_optimizers:True \
                  n_steps:2048 \
                  batch_size:64 \
                  learning_rate:3e-4 \
                  gamma:0.99 \
                  gae_lambda:0.95 \
                  ent_coef:0.0 \
                  clip_range:0.2 \
                  n_epochs:10 \
          > /dev/null 2>&1 &

        pids+=($!)
      done
    done
    
    echo "    Waiting for batch to finish..."
    wait "${pids[@]}"
    echo "    Batch finished."
  done
  
  echo "Finished Env: $env_id"
done

echo "All Done."

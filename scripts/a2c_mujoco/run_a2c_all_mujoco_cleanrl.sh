#!/bin/bash
#
# A2C MuJoCo (CleanRL-style params) - 4 seeds per env
#
# 运行模式:
#   bash scripts/a2c_mujoco/run_a2c_all_mujoco_cleanrl.sh [SEED_CONCURRENCY]
#
# SEED_CONCURRENCY (默认 2):
#   - 1: 每次只跑 1 个 seed（最稳）
#   - 2: 每次并行 2 个 seed（默认，适合 2 张 GPU）
#   - 4: 同一个环境 4 个 seed 全并行（更吃 CPU/显存）
#
# 说明:
# - 超参尽量对齐 CleanRL 在 continuous control A2C 的常用设置
# - 训练输出会写到 logs/a2c_mujoco/*.log，避免 /dev/null 静默失败
#

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

SEED_CONCURRENCY=${1:-2}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/a2c_mujoco

seeds=(9 1 2 3)

# 选择一些常见 MuJoCo 任务（避开 Ant/Hopper）
mujoco_envs=(
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Swimmer-v4"
)

# 四种消融：advantage mean/std 组合
ALL_CONFIGS=(
  "meanStd       True   True"
  "meanNoStd     True   False"
  "noMeanStd     False  True"
  "noNorm        False  False"
)

# CleanRL-style / SB3-default-ish A2C continuous params
# NOTE: n_timesteps 这里显式设成 1e6（Humanoid 通常需要更久，可自行改 2e6+）
N_ENVS="8"
N_STEPS="5"
LR="7e-4"
GAMMA="0.99"
GAE_LAMBDA="0.95"
ENT_COEF="0.0"
VF_COEF="0.5"
MAX_GRAD_NORM="0.5"

pids=()
trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait || true; \
      echo "All runs killed."' INT

echo "Starting A2C MuJoCo (CleanRL-style) with SEED_CONCURRENCY=${SEED_CONCURRENCY}"
echo "Params: n_envs=${N_ENVS}, n_steps=${N_STEPS}, lr=${LR}, gamma=${GAMMA}, gae_lambda=${GAE_LAMBDA}, ent_coef=${ENT_COEF}, vf_coef=${VF_COEF}, max_grad_norm=${MAX_GRAD_NORM}, optimizer=RMSProp(eps=1e-5, alpha=0.99)"

for env_id in "${mujoco_envs[@]}"; do
  echo "========================================================"
  echo "Starting Env: ${env_id}"
  echo "========================================================"

  # 每个环境：两种配置逐批跑 seeds（每批 seeds 并行 SEED_CONCURRENCY 个）
  for config_str in "${ALL_CONFIGS[@]}"; do
    read -r CONFIG_NAME ADV_MEAN ADV_STD <<< "${config_str}"

    echo "--------------------------------------------------------"
    echo "Config: ${CONFIG_NAME} (normalize_advantage_mean=${ADV_MEAN}, normalize_advantage_std=${ADV_STD})"
    echo "--------------------------------------------------------"

    for ((i=0; i<${#seeds[@]}; i+=SEED_CONCURRENCY)); do
      batch_seeds=("${seeds[@]:i:SEED_CONCURRENCY}")
      pids=()

      for s_idx in "${!batch_seeds[@]}"; do
        seed="${batch_seeds[$s_idx]}"

        # 2 GPU 轮询（如果你只有 1 张卡，把 %2 改成 %1 即可）
        gpu=$(( s_idx % 2 ))

        run_name="a2c_mujoco_cleanrl_${env_id}_${CONFIG_NAME}_seed${seed}_sepFE_noRewardNorm"
        log_file="logs/a2c_mujoco/${env_id}_${CONFIG_NAME}_seed${seed}.log"

        echo "  -> Launching: ${run_name} on GPU ${gpu} (log: ${log_file})"

        CUDA_VISIBLE_DEVICES="${gpu}" python train.py \
          --seed "${seed}" \
          --algo a2c \
          --env "${env_id}" \
          --vec-env subproc \
          --track \
          --wandb-run-extra-name "${run_name}" \
          --wandb-project-name sb3_a2c_mujoco_no_reward_norm_sep_feature_rmsprop \
          --wandb-entity agent-lab-ppo \
          -params policy:'MlpPolicy' \
                  policy_kwargs:"dict(share_features_extractor=False)" \
                  n_envs:${N_ENVS} \
                  n_steps:${N_STEPS} \
                  learning_rate:${LR} \
                  gamma:${GAMMA} \
                  gae_lambda:${GAE_LAMBDA} \
                  ent_coef:${ENT_COEF} \
                  vf_coef:${VF_COEF} \
                  max_grad_norm:${MAX_GRAD_NORM} \
                  use_rms_prop:True \
                  rms_prop_eps:1e-5 \
                  normalize:"{'norm_obs':True,'norm_reward':False}" \
                  normalize_advantage:True \
                  normalize_advantage_mean:${ADV_MEAN} \
                  normalize_advantage_std:${ADV_STD} \
          > "${log_file}" 2>&1 &

        pids+=($!)
      done

      echo "  Waiting for seed batch to finish..."
      wait "${pids[@]}" || echo "[WARN] Some runs exited with non-zero status in env=${env_id}, config=${CONFIG_NAME}, batch_start=${i}. Check logs/a2c_mujoco/."
      echo "  Seed batch finished."
    done
  done

  echo "Finished Env: ${env_id}"
  echo
done

echo "All A2C MuJoCo runs finished."



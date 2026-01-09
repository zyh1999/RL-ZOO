#!/bin/bash
#
# NPG MuJoCo (default hyperparams from rl_zoo3 default config: hyperparams/npg.yml)
#
# 运行模式:
#   bash scripts/npg_mujoco/run_npg_all_mujoco_default.sh
#
# 也可用环境变量临时覆盖（不用改文件），例如：
#   GAMMA=0.995 USE_POPART=True ACTION_SQUASH=False NORM_OBS=True NORM_REWARD=False bash scripts/npg_mujoco/run_npg_all_mujoco_default.sh
#
# 说明:
# - RL-ZOO 会默认读取 hyperparams/npg.yml（因为 --algo npg）
# - 你可以在脚本顶部改 GAMMA/USE_POPART/ACTION_SQUASH 来覆盖对应超参
# - 5 个环境：HalfCheetah / Hopper / Walker2d / Swimmer / Humanoid
# - 每个环境跑 4 个种子
# - 并发策略：始终保持“每次最多两个环境一起跑”
#   （每批启动 2 个环境 * 4 个 seed = 8 个训练进程并行；等待该批全部结束后再启动下一批）
#

set -e

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

# ============================================================
# 在这里改超参（默认值建议与 hyperparams/npg.yml 保持一致）
# 说明：bool 请用 True/False（Python 可 eval 的字面量）
# ============================================================
GAMMA="${GAMMA:-0.999}"
USE_POPART="${USE_POPART:-True}"
ACTION_SQUASH="${ACTION_SQUASH:-True}"
NORM_OBS="${NORM_OBS:-True}"
NORM_REWARD="${NORM_REWARD:-False}"

# GPU 轮询分配（默认 2 张卡；单卡可设 GPU_COUNT=1）
GPU_COUNT="${GPU_COUNT:-2}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/npg_mujoco

seeds=(9 1 2 3)

mujoco_envs=(
  #"HalfCheetah-v4"
  #"Hopper-v4"
  #"Walker2d-v4"
  "Swimmer-v4"
  #"Humanoid-v4"
)

# W&B（默认对齐 a2c_mujoco 的风格；可用环境变量覆盖）
: "${WANDB_ENTITY:=agent-lab-ppo}"
: "${WANDB_PROJECT:=sb3_npg_mujoco_default}"
: "${WANDB_GROUP:=npg_mujoco_iter_4_add_action_squash}"

pids=()
trap 'echo "Caught Ctrl+C, killing all runs..."; \
      for pid in "${pids[@]}"; do \
        kill "$pid" 2>/dev/null || true; \
      done; \
      wait || true; \
      echo "All runs killed."' INT

echo "Starting NPG MuJoCo (default hyperparams) with env concurrency=2"
echo "Envs: ${mujoco_envs[*]}"
echo "Seeds: ${seeds[*]}"
echo "Params override: gamma=${GAMMA}, use_popart=${USE_POPART}, action_squash=${ACTION_SQUASH}"
echo "Normalize override: norm_obs=${NORM_OBS}, norm_reward=${NORM_REWARD}"

for ((i=0; i<${#mujoco_envs[@]}; i+=2)); do
  batch_envs=("${mujoco_envs[@]:i:2}")
  pids=()

  echo "========================================================"
  echo "Starting Env batch (concurrency=8 runs): ${batch_envs[*]}"
  echo "Config: rl_zoo3 default (hyperparams/npg.yml), seeds: ${seeds[*]}"
  echo "========================================================"

  # 每批 2 个环境 * 4 个 seed = 8 个训练进程同时跑
  for e_idx in "${!batch_envs[@]}"; do
    env_id="${batch_envs[$e_idx]}"
    log_dir="logs/npg_mujoco/${env_id}"
    mkdir -p "${log_dir}"

    for s_idx in "${!seeds[@]}"; do
      seed="${seeds[$s_idx]}"

      # 在当前 batch(2 envs x 4 seeds) 内做全局轮询：0..7 -> GPU 0/1/0/1...
      global_job_id=$(( e_idx * ${#seeds[@]} + s_idx ))
      gpu=$(( global_job_id % GPU_COUNT ))

      run_name="npg_mujoco_default_${env_id}_seed${seed}"
      log_file="${log_dir}/seed${seed}.log"

      echo "  -> Launching env=${env_id}, seed=${seed}, gpu=${gpu} (log: ${log_file})"
      CUDA_VISIBLE_DEVICES="${gpu}" python -u train.py \
        --seed "${seed}" \
        --algo npg \
        --env "${env_id}" \
        -params gamma:${GAMMA} use_popart:${USE_POPART} action_squash:${ACTION_SQUASH} \
                normalize:"{'norm_obs':${NORM_OBS},'norm_reward':${NORM_REWARD}}" \
        --track \
        --wandb-run-extra-name "${run_name}" \
        --wandb-project-name "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-group-name "${WANDB_GROUP}_pa${USE_POPART}_sq${ACTION_SQUASH}_no${NORM_OBS}_nr${NORM_REWARD}_gamma${GAMMA}_${env_id}" \
        > "${log_file}" 2>&1 &
      pids+=($!)
    done
  done

  echo "Waiting for env batch to finish: ${batch_envs[*]}"
  wait "${pids[@]}" || echo "[WARN] Some runs exited with non-zero status in env batch: ${batch_envs[*]}. Check logs/npg_mujoco/."
  echo "Env batch finished: ${batch_envs[*]}"
done

echo "All NPG MuJoCo runs finished."



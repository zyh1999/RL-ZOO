#!/bin/bash
#
# NPG MuJoCo (default hyperparams from hyperparams/npg.yml)
#
# 运行模式:
#   bash scripts/npg_mujoco/run_npg_all_mujoco_default.sh
#
# 说明:
# - 仅使用当前默认超参（hyperparams/npg.yml），不额外传 -params 覆盖
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/npg_mujoco

seeds=(9 1 2 3)

mujoco_envs=(
  "HalfCheetah-v4"
  "Hopper-v4"
  "Walker2d-v4"
  "Swimmer-v4"
  "Humanoid-v4"
)

CONF_FILE="hyperparams/npg.yml"

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

for ((i=0; i<${#mujoco_envs[@]}; i+=2)); do
  batch_envs=("${mujoco_envs[@]:i:2}")
  pids=()

  echo "========================================================"
  echo "Starting Env batch (concurrency=8 runs): ${batch_envs[*]}"
  echo "Config: ${CONF_FILE} (defaults), seeds: ${seeds[*]}"
  echo "========================================================"

  # 每批 2 个环境 * 4 个 seed = 8 个训练进程同时跑
  for e_idx in "${!batch_envs[@]}"; do
    env_id="${batch_envs[$e_idx]}"
    log_dir="logs/npg_mujoco/${env_id}"
    mkdir -p "${log_dir}"

    for seed in "${seeds[@]}"; do
      run_name="npg_mujoco_default_${env_id}_seed${seed}"
      log_file="${log_dir}/seed${seed}.log"

      echo "  -> Launching env=${env_id}, seed=${seed} (log: ${log_file})"
      python -u train.py \
        --seed "${seed}" \
        --algo npg \
        --env "${env_id}" \
        --conf-file "${CONF_FILE}" \
        --track \
        --wandb-run-extra-name "${run_name}" \
        --wandb-project-name "${WANDB_PROJECT}" \
        --wandb-entity "${WANDB_ENTITY}" \
        --wandb-group-name "${WANDB_GROUP}_${env_id}" \
        > "${log_file}" 2>&1 &
      pids+=($!)
    done
  done

  echo "Waiting for env batch to finish: ${batch_envs[*]}"
  wait "${pids[@]}" || echo "[WARN] Some runs exited with non-zero status in env batch: ${batch_envs[*]}. Check logs/npg_mujoco/."
  echo "Env batch finished: ${batch_envs[*]}"
done

echo "All NPG MuJoCo runs finished."



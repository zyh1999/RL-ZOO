#!/bin/bash
#SBATCH -J halfcheetah_sweep
#SBATCH -o logs/sbatch_halfcheetah_%j.out
#SBATCH -e logs/sbatch_halfcheetah_%j.err
#SBATCH -p multicore
#SBATCH -c 42
#SBATCH --ntasks=1

set -e

# 防止 CPU 过订
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export torch_num_threads=1

cd RL-ZOO

# 激活虚拟环境
source ~/.RLvenv/bin/activate

bash scripts/custom_searches/run_sweep_HalfCheetah_v4.sh



#!/bin/bash
#SBATCH -J humanoid_sweep
#SBATCH -o logs/sbatch_humanoid_%j.out
#SBATCH -e logs/sbatch_humanoid_%j.err
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

bash scripts/custom_searches/run_sweep_Humanoid_v4.sh



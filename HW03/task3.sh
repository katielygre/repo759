#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o task3.out -e task3.err
#SBATCH --array=10-29
#SBATCH --gres=gpu:1

nvcc task3.cu vscale.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o task3

n=$((2**SLURM_ARRAY_TASK_ID))

mkdir -p timing_files

./task3 $n > "timing_files/timing_$SLURM_ARRAY_TASK_ID.txt"
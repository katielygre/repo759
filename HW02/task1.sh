#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err
#SBATCH --array=10-30

n=$(python3 -c "print(2**${SLURM_ARRAY_TASK_ID})")

./task1 $n
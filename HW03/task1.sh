#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err
#SBATCH --gres=gpu:1

module load nvidia/cuda

./task1
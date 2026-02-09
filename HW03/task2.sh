#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err
#SBATCH --gres=gpu:1

./task2
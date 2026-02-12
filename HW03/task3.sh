#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o task3.out -e task3.err
#SBATCH --gres=gpu:1

./task3 1024
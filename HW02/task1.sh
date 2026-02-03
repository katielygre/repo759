#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task1
#SBATCH -o task1.out -e task1.err

./task1 1000000

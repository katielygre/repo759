#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task2
#SBATCH -o task2.out -e task2.err

./task2 4 3
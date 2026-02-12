#!/usr/bin/env bash

#SBATCH -p instruction
#SBATCH -J task3
#SBATCH -o task3.out -e task3.err
#SBATCH --gres=gpu:1

mkdir -p timing_files/512threads

for power in {10..29}
do
    n=$((2**power))
    ./task3 $n > "timing_files/512threads/timing_$power.txt"
done


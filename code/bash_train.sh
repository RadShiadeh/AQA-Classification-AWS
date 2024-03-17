#!/usr/bin/env bash

#SBATCH --job-name=AQA
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=coms030144
#SBATCH --gres=gpu:2  # Requesting 2 GPUs
#SBATCH --time=2:00:00  # Requesting 2 hours
#SBATCH --mem=8GB  # Requesting 8GB of memory
#SBATCH --cpus-per-task=10

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python trainer.py

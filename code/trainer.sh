#!/usr/bin/env bash

#SBATCH --job-name=lab2
#SBATCH --partition=teach_gpu
#SBATCH --nodes=1
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=coms030144
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=10

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python trainer.py

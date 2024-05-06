#!/usr/bin/env bash

#SBATCH --job-name=ETE_AQA_c3d_classifier
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --partition=teach_gpu
#SBATCH --time=3:00:0
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=COMS030144
#SBATCH --mem=8gb

echo start time is "$(date)" for c3d_classifier c3d_classifier.py
echo Slurm job ID is "${SLURM_JOBID}"

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python -u c3d_classifier.py

echo end time is "$(date)"
hostname
#!/usr/bin/env bash

#SBATCH --job-name=ETE_AQA_c3d_classifier
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:0
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=COMS030144
#SBATCH --mem=10gb

echo start time is "$(date)" for c3d_classifier 128 size and 32 frames
echo Slurm job ID is "${SLURM_JOBID}"

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python -u less_size_trainer_128.py

echo end time is "$(date)"
hostname
#!/usr/bin/env bash

#SBATCH --job-name=Extended10layer
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:0
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=COMS030144
#SBATCH --mem=20gb

echo start time is "$(date)" for c3d_classifier 128 size and 32 frames for 1001 samples with 10 conv layers
echo Slurm job ID is "${SLURM_JOBID}"

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python -u trainer_c3d.py

echo end time is "$(date)"
hostname
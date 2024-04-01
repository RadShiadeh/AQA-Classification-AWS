#!/usr/bin/env bash

#SBATCH --job-name=ETE_AQA_c3d_classifier
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:0
#SBATCH -o ./log_%j.out # STDOUT out
#SBATCH -e ./log_%j.err # STDERR out
#SBATCH --account=COMS030144
#SBATCH --mem=20gb

echo start time is "$(date)" for resnet3D
echo Slurm job ID is "${SLURM_JOBID}"

module load "languages/anaconda3/2021-3.8.8-cuda-11.1-pytorch"

# run your Python script
python -u resNet18_classifier_trainer.py

echo end time is "$(date)"
hostname
#!/bin/bash
#SBATCH --partition=p5gpuA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=train_%j.out

MODEL_CONFIG=$1

source ~/.bashrc
conda activate openmmlab

cd ~/mmrotate

python tools/train.py $MODEL_CONFIG

#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=evaluate_%j.out

DOTA_VERSION=$1

source ~/.bashrc
conda activate openmmlab

cd ~/mmrotate

python tools/analysis_tools/evaluate_models.py --dota-version $DOTA_VERSION

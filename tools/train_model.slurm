#!/bin/bash
#SBATCH --partition=p5gpuA100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=3-00:00:00
#SBATCH --output=train_%j.out

MODEL_CONFIG=$1
export NUM_GPUS=${2:-1}

echo "CONFIG: $MODEL_CONFIG"

source ~/.bashrc
conda activate openmmlab
cd ~/mmrotate

# Find a free port to avoid communication conflicts
export PORT=$((29000 + RANDOM % 1000))
while ss -tuln | grep -q ":$PORT " ; do
    export PORT=$((29000 + RANDOM % 1000))
done
echo "Using port: $PORT"

bash tools/dist_train.sh $MODEL_CONFIG $NUM_GPUS --auto-resume

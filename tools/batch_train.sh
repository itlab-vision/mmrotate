#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_configs_directory>"
    exit 1
fi

CONFIG_DIR=$1
SLURM_SCRIPT="tools/train_model.slurm"

# Check if the specified directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Directory '$CONFIG_DIR' does not exist."
    exit 1
fi

# Check if the slurm script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: File '$SLURM_SCRIPT' not found."
    exit 1
fi

echo "Searching for configuration files in: $CONFIG_DIR"
echo "------------------------------------------------"

# Find all .py files and run sbatch for each
find "$CONFIG_DIR" -type f -name "*.py" | while read -r config_file; do
    echo "Submitting training job for: $config_file"

    # Extract the filename without the path and the .py extension
    job_name=$(basename "$config_file" .py)

    # Override the job name and output log file directly from the command line
    sbatch --job-name="$job_name" --output="train_${job_name}_%j.out" "$SLURM_SCRIPT" "$config_file"
done

echo "------------------------------------------------"
echo "All jobs successfully submitted to the queue!"

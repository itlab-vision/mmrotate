# Slurm Cluster Job Execution Guide

This document describes how to submit and manage automated batch evaluation jobs on HPC clusters using the Slurm Workload Manager.

---

## 1. Automated Model Evaluation Batch Script (`tools/evaluate_models.slurm`)

The `tools/evaluate_models.slurm` script allows submitting automated batch model evaluation jobs (`tools/analysis_tools/evaluate_models.py`) to a Slurm compute node via `sbatch`.

### Script Structure and Resource Allocation

```bash
#!/bin/bash
#SBATCH --partition=p5gpuA100
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
```

### Slurm Directives Breakdown

- `#SBATCH --partition=p5gpuA100`: Default GPU partition for task execution.
- `#SBATCH --nodes=1`: Allocates 1 compute node.
- `#SBATCH --ntasks=1`: Runs 1 task instance.
- `#SBATCH --cpus-per-task=6`: Allocates 6 CPU cores for data loading and processing.
- `#SBATCH --gres=gpu:1`: Requests 1 GPU.
- `#SBATCH --time=1-00:00:00`: Sets job time limit to 24 hours (1 day).
- `#SBATCH --output=evaluate_%j.out`: Writes output logs to `evaluate_<JOB_ID>.out` (`%j` resolves to Slurm Job ID).

### Submitting Evaluation Jobs

Submit job using default partition settings (`p5gpuA100`):

```bash
sbatch tools/evaluate_models.slurm 1.0
sbatch tools/evaluate_models.slurm 1.5
```

### Overriding Partition at Runtime

CLI arguments take precedence over script `#SBATCH` directives. To override the partition dynamically during job submission:

```bash
sbatch --partition=other_partition tools/evaluate_models.slurm 1.0
```

---

## 2. Job Monitoring and Management

### Check Job Queue Status

```bash
squeue -u $USER
```

### Monitor Real-Time Job Logs

```bash
tail -f evaluate_<JOB_ID>.out
```

### Cancel Job

```bash
scancel <JOB_ID>
```

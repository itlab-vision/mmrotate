# ITLab Model Benchmarks Directory

This directory stores benchmark evaluation results, raw performance statistics, and consolidated comparison tables for MMRotate models evaluated on DOTA datasets.

______________________________________________________________________

## Directory Structure

```text
itlab/benchmarks/
├── raw_results/             # Raw JSON log files exported by evaluate_models.py
├── results_a100.md          # Consolidated benchmark results on NVIDIA A100 GPU
└── README.md                # Benchmark storage documentation
```

______________________________________________________________________

## Workflow Integration

1. Execute batch evaluation using `evaluate_models.py` or Slurm:
2. Convert exported JSON log files into consolidated Markdown summary tables using `build_summary_table.py`:
3. Save generated Markdown tables in `itlab/benchmarks/` (e.g., `results_a100.md`).

# MMRotate - ITLab Documentation and Workflow

This directory contains technical guides, dataset preparation steps, and evaluation procedures tailored for ITLab GPU server infrastructure.

Information in these guides is extracted and structured from official MMRotate documentation and internal team experience.

---

## Documentation Index

- **[guides/get_started.md](guides/get_started.md)**: Recommended step-by-step onboarding roadmap for new developers.
- **[guides/environment_setup.md](guides/environment_setup.md)**: Verified environment setup, dependency installation (PyTorch, MMCV, MMDetection, MMClassification, MMRotate), and headless OpenCV fixes.
- **[guides/data_preparation.md](guides/data_preparation.md)**: DOTA dataset downloading (`download_dota.py`) and patch splitting (`run_dota_split.py` & `img_split.py`).
- **[guides/evaluation_and_benchmarking.md](guides/evaluation_and_benchmarking.md)**: Checkpoint downloads (`download_dota_weights.py`), single-model evaluation and visualization (`tools/test.py`), FPS benchmarking (`benchmark.py`), and automated multi-model batch evaluation (`evaluate_models.py` & `build_summary_table.py`).
- **[guides/slurm_execution.md](guides/slurm_execution.md)**: Slurm cluster job submission guide (`evaluate_models.slurm`, `slurm_test.sh`, `slurm_train.sh`), resource directives, and job control.
- **[guides/config_system.md](guides/config_system.md)**: Config inheritance resolution (`_base_`), dictionary and list merging rules, legacy vs modern dataloader configuration, and internal worker thread behavior.

---

## Quick Start

Follow the step-by-step onboarding roadmap in **[guides/get_started.md](guides/get_started.md)** to set up your environment, prepare datasets, and execute model evaluations.

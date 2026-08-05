# ITLab MMRotate Guides

This directory contains technical guides and step-by-step instructions for setup, data preparation, evaluation, and execution on ITLab GPU cluster infrastructure.

Information in these guides is extracted and structured from official MMRotate documentation and internal team experience.

---

## Documentation Index

- **[get_started.md](get_started.md)**: Recommended step-by-step onboarding roadmap for new developers.
- **[environment_setup.md](environment_setup.md)**: Verified environment setup, dependency installation (PyTorch, MMCV, MMDetection, MMClassification, MMRotate), and headless OpenCV fixes.
- **[data_preparation.md](data_preparation.md)**: DOTA dataset downloading (`download_dota.py`) and patch splitting (`run_dota_split.py` & `img_split.py`).
- **[evaluation_and_benchmarking.md](evaluation_and_benchmarking.md)**: Checkpoint downloads (`download_dota_weights.py`), single-model evaluation and visualization (`tools/test.py`), FPS benchmarking (`benchmark.py`), and automated multi-model batch evaluation (`evaluate_models.py` & `build_summary_table.py`).
- **[model_training.md](model_training.md)**: Model training workflow, dataset download (`train`/`val`), `trainval` patch splitting, interactive training (`tools/train.py`), Slurm training (`train_model.slurm`), and batch training generation (`batch_train.sh`).
- **[slurm_execution.md](slurm_execution.md)**: Slurm cluster job submission guide (`evaluate_models.slurm`), resource directives, partition overrides, and job control.
- **[config_system.md](config_system.md)**: Config inheritance resolution (`_base_`), dictionary and list merging rules, legacy vs modern dataloader configuration, and internal worker thread behavior.

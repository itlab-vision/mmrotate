# Config System Architecture and Rules

This guide explains how configuration files are parsed, inherited, merged, and configured in MMRotate.

______________________________________________________________________

## 1. Config Inheritance and Resolution Order

MMRotate uses hierarchical configuration files built on `mmcv.Config`.

Resolution sequence:

1. The base config listed in `_base_` is loaded first (e.g., `configs/_base_/datasets/dotav1.py`).
2. Model-specific configs inherit the base dictionary.
3. Explicit parameters defined in child configs override or extend base values.

______________________________________________________________________

## 2. Parameter Merging Rules

When combining base and child configs:

- **Dictionaries (`dict`)**: Merged recursively. Keys specified in the child config overwrite matching keys in the base config, while unmentioned keys are retained.
- **Lists (`list`)**: Replaced completely. A list defined in a child config completely replaces the corresponding list in the base config.
- **Primitive Values (int, float, str, bool)**: Replaced completely.

______________________________________________________________________

## 3. DataLoader Configuration Styles

In MMRotate v0.3.0+, a new dataloader configuration syntax was introduced alongside legacy syntax.

### Recommended (Modern) Syntax

```python
data = dict(
    train=dict(type='DOTADataset', ...),
    val=dict(type='DOTADataset', ...),
    test=dict(type='DOTADataset', ...),
    train_dataloader=dict(samples_per_gpu=2, workers_per_gpu=2),
    val_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
    test_dataloader=dict(samples_per_gpu=4, workers_per_gpu=4),
)
```

### Legacy Syntax

```python
data = dict(
    samples_per_gpu=2,  # Applied to train_dataloader
    workers_per_gpu=2,  # Broadcasted to all dataloaders
    train=dict(type='DOTADataset', ...),
    val=dict(type='DOTADataset', ...),
    test=dict(type='DOTADataset', ...),
    test_dataloader=dict(samples_per_gpu=4),
)
```

______________________________________________________________________

## 4. Internal Dataloader Worker Compatibility Behavior

As observed in `compat_config.py`, when `workers_per_gpu` is defined at the top-level `cfg.data` dictionary, MMRotate automatically pops and assigns `workers_per_gpu` across all dataloaders (`train_dataloader`, `val_dataloader`, `test_dataloader`):

```python
if 'workers_per_gpu' in cfg.data:
    workers_per_gpu = cfg.data.pop('workers_per_gpu')
    cfg.data.train_dataloader['workers_per_gpu'] = workers_per_gpu
    cfg.data.val_dataloader['workers_per_gpu'] = workers_per_gpu
    cfg.data.test_dataloader['workers_per_gpu'] = workers_per_gpu
```

`workers_per_gpu` is also passed into `setup_multi_processes`, which sets multiproc start methods and restricts intra-op CPU threads per worker process to 1, preventing CPU core contention during benchmark evaluation.

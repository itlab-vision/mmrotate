# Quantum Layer Module Guide

This document describes the configuration and usage of the hybrid quantum-classical [`QuantumLayer`](../../mmrotate/models/utils/quantum_layer.py#L68) module in MMRotate.

______________________________________________________________________

## 1. Module Overview

The `QuantumLayer` is a hybrid neural network module built with Qiskit and `qiskit-machine-learning`. It bridges PyTorch tensors with parameterized variational quantum circuits via `SamplerQNN` and `TorchConnector`.

In oriented object detection, box representations and transformation deltas are compact 5D vectors (`[x, y, w, h, a]` or `[dx, dy, dα, dβ, dγ]`). The `QuantumLayer` takes a 5D vector, passes it through an entangled quantum circuit where all 5 parameters interact simultaneously, and outputs a refined 5D vector.

______________________________________________________________________

## 2. Builder and Configuration

Quantum layers are instantiated using the [`build_quantum_layer`](../../mmrotate/models/utils/quantum_layer.py#L172) builder function, which accepts a configuration dictionary specifying `'type'` and module parameters:

```python
from mmrotate.models.utils import build_quantum_layer

layer = build_quantum_layer(dict(
    type='QuantumLayer', # Quantum layer class name
    num_qubits=5,        # Number of qubits (matches feature dimension)
    reps=2,              # Depth / repetitions of variational layers
    residual=True,       # Refinement mode: True = output is (input + delta); False = direct replacement
    scale_input=True,    # Normalizes inputs via tanh into [-pi/2, pi/2]
    chunk_size=128))     # Chunk size for batch splitting to optimize quantum simulation throughput
```

If `cfg` is `None`, `build_quantum_layer(None)` returns `None`. This allows components across different models to support optional quantum layers without boilerplate checks.

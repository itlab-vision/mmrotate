# Quantum Layer Module Guide

This document describes the configuration and usage of the hybrid quantum-classical [`QuantumLayer`](../../mmrotate/models/utils/quantum_layer.py#L10) module in MMRotate.

______________________________________________________________________

## 1. Module Overview

The `QuantumLayer` is a hybrid neural network module designed for quantum-classical workflows in MMRotate, implemented using [PennyLane](https://pennylane.ai/).

The module consists of three main processing stages:

1. **Input projection:** If `input_dim == num_qubits`, the input is passed directly (`nn.Identity()`). Otherwise, a linear layer transforms the features from `input_dim` to `num_qubits`.
2. **Quantum processing:** The projected features are encoded into qubits using angle encoding, processed by a parameterized variational quantum circuit, and measured in the computational basis.
3. **Classical output projection:** A linear layer maps the quantum measurement output of size `2^num_qubits` to the requested `output_dim`.

This design allows `input_dim`, `output_dim`, and `num_qubits` to be specified independently.

For example, in oriented object detection, box representations and transformation deltas are often compact 5D vectors (`[x, y, w, h, a]` or `[dx, dy, dα, dβ, dγ]`). The `QuantumLayer` can process these vectors using a configurable number of qubits and return an output with a configurable feature dimension.

The module also supports arbitrary leading tensor dimensions. For example, an input of shape `(batch_size, sequence_length, input_dim)` produces an output of shape `(batch_size, sequence_length, output_dim)`.

### Residual Connection

When `residual=True`, the original input is added to the output of the quantum processing branch.

If `input_dim` equals `output_dim`, the input is added directly using an identity mapping. Otherwise, a learnable linear projection transforms the input from `input_dim` to `output_dim` before addition.

The output is therefore computed as: `[y = Q(x) + R(x)]`,
where `Q(x)` denotes the quantum processing branch, including its classical input and output projections, and `R(x)` denotes the residual mapping.

When `residual=False`, only the quantum processing branch contributes to the output.

______________________________________________________________________

## 2. Builder and Configuration

Quantum layers are instantiated using the [`build_quantum_layer`](../../mmrotate/models/utils/quantum_layer.py#L152) builder function, which accepts a configuration dictionary specifying `'type'` and module parameters:

```python
from mmrotate.models.utils import build_quantum_layer

layer = build_quantum_layer(dict(
    type='QuantumLayer', # Quantum layer class name
    input_dim=128,       # Input feature dimension
    output_dim=64,       # Output feature dimension
    num_qubits=5,        # Number of qubits
    reps=2,              # Depth / repetitions of variational layers
    residual=True,       # Whether to add a residual connection
    scale_input=True))   # Scales encoded angles to [-pi/2, pi/2]
```

The input and output dimensions are independent of the number of qubits. The input projection maps `input_dim` to `num_qubits`, while the output projection maps the `2^num_qubits` measurement values to `output_dim`.

If `cfg` is `None`, `build_quantum_layer(None)` returns `None`. This allows components across different models to support optional quantum layers without additional boilerplate checks.

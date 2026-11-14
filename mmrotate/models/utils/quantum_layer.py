# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

# PennyLane 0.32.0 contains an artificial version guard (Torch>=1.8.1)
# in pennylane/devices/default_qubit_torch.py, though Torch 1.8.0 works
# identically. Temporarily adjust torch.__version__ during module import.
_orig_torch_version = torch.__version__
if _orig_torch_version.startswith('1.8.0'):
    torch.__version__ = '1.8.1'

import pennylane as qml  # isort:skip # noqa: E402
import pennylane.devices.default_qubit_torch  # isort:skip # noqa: E402, F401

torch.__version__ = _orig_torch_version


class QuantumLayer(BaseModule):
    """Hybrid Quantum-Classical Neural Network Layer.

    Applies classical input projection, angle encoding on qubits,
    parameterized variational rotations and circular entanglement, followed
    by measurement in the computational basis and classical output projection.

    Args:
        input_dim (int): Input feature dimension. Defaults to 5.
        output_dim (int): Output feature dimension. Defaults to 5.
        num_qubits (int): Number of qubits. Defaults to 5.
        reps (int): Repetitions of variational layers. Defaults to 2.
        residual (bool): Whether to use residual connection. Defaults to True.
        scale_input (bool): Whether to apply tanh(x) * (pi / 2) to scale inputs
            into [-pi/2, pi/2] rotation range. Defaults to True.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 input_dim=5,
                 output_dim=5,
                 num_qubits=5,
                 reps=2,
                 residual=True,
                 scale_input=True,
                 init_cfg=None):
        super(QuantumLayer, self).__init__(init_cfg=init_cfg)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_qubits = num_qubits
        self.reps = reps
        self.residual = residual
        self.scale_input = scale_input

        # Project input features to the number of qubits
        if input_dim == num_qubits:
            self.input_layer = nn.Identity()
        else:
            self.input_layer = nn.Linear(input_dim, num_qubits)

        # Variational quantum circuit weights
        self.weights = nn.Parameter(
            torch.empty(num_qubits * reps).uniform_(0, 2 * math.pi))
        self.quantum_circuit = self._build_quantum_circuit()

        # Post-processing linear layer: maps 2^num_qubits -> output_dim
        self.out = nn.Linear(2**num_qubits, output_dim)

        # Residual connection
        if self.residual:
            if input_dim == output_dim:
                self.residual_layer = nn.Identity()
            else:
                self.residual_layer = nn.Linear(input_dim, output_dim)

            # Initialize output layer near zero to preserve stability
            nn.init.normal_(self.out.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.out.bias)

    def _build_quantum_circuit(self):
        """Build a variational quantum circuit.

        Creates an angle-encoded parameterized quantum circuit with `reps`
        variational layers of RY rotations and circular CNOT entanglement,
        measured in the computational basis.

        Returns:
            qml.QNode: Quantum circuit node.
        """
        num_qubits = self.num_qubits
        reps = self.reps
        dev = qml.device('default.qubit', wires=num_qubits)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit(inputs, weights):
            # 1. Angle encoding layer: RY(x_i)
            for i in range(num_qubits):
                qml.RY(inputs[:, i], wires=i)

            # 2. Variational layers
            for r in range(reps):
                for i in range(num_qubits):
                    w_idx = i * reps + r
                    qml.RY(weights[w_idx], wires=i)

                if r != reps - 1:
                    # CX ladder: 0->1, 1->2, ..., (n-2)->(n-1)
                    for i in range(num_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    # Circular CX: (n-1)->0
                    if num_qubits > 2:
                        qml.CNOT(wires=[num_qubits - 1, 0])

            return qml.probs(wires=range(num_qubits))

        return circuit

    def forward(self, x):
        """Forward pass for QuantumLayer.

        Args:
            x (torch.Tensor): Input tensor with shape (..., input_dim).

        Returns:
            torch.Tensor: Transformed tensor with shape (..., output_dim).
        """
        if x.ndim < 1:
            raise ValueError('Input tensor must have at least one dimension.')
        if x.size(-1) != self.input_dim:
            raise ValueError(f'Expected input dimension {self.input_dim}, '
                             f'but got {x.size(-1)}.')

        if x.numel() == 0:
            return x.new_empty(*x.shape[:-1], self.output_dim)

        orig_shape = x.shape

        # Flatten leading dimensions to process each feature vector
        # independently in the quantum circuit
        x_2d = x.reshape(-1, self.input_dim)

        # Project input features to the number of qubits
        x_in = self.input_layer(x_2d)

        pi = getattr(torch, 'pi', math.pi)
        if self.scale_input:
            x_in = torch.mul(torch.tanh(x_in), pi / 2)

        # Evaluate quantum circuit
        q_out = self.quantum_circuit(x_in, self.weights).type_as(x_in)

        # Project quantum output to the required output dimension
        out_2d = self.out(q_out)

        if self.residual:
            out_2d = self.residual_layer(x_2d) + out_2d

        return out_2d.reshape(*orig_shape[:-1], self.output_dim)


def build_quantum_layer(cfg):
    """Build a quantum layer from config dictionary or return None.

    Args:
        cfg (dict, optional): Config dictionary with layer parameters.
            Must contain 'type' specifying the class name to instantiate.

    Returns:
        QuantumLayer or None: Instantiated quantum layer module, or None.
    """
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        raise TypeError(f'Cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError('The cfg dict must contain the key "type"')

    cfg_copy = cfg.copy()
    layer_type = cfg_copy.pop('type')

    if layer_type == 'QuantumLayer':
        return QuantumLayer(**cfg_copy)
    else:
        raise KeyError(f'Unknown quantum layer type: "{layer_type}". '
                       f'Available types: ["QuantumLayer"]')

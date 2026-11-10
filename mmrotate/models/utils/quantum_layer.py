# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit.primitives import Sampler
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


def create_encoding_circuit(num_qubits):
    """Create angle encoding quantum circuit.

    Args:
        num_qubits (int): Number of qubits.

    Returns:
        tuple[QuantumCircuit, ParameterVector]: Created circuit and input
            parameters.
    """
    qc = QuantumCircuit(num_qubits)
    input_params = ParameterVector('x', length=num_qubits)
    for i in range(num_qubits):
        qc.ry(input_params[i], i)
    return qc, input_params


def create_variational_circuit(num_qubits, reps=2):
    """Create parameterized variational quantum circuit with angle encoding.

    Args:
        num_qubits (int): Number of qubits.
        reps (int): Number of variational layer repetitions. Defaults to 2.

    Returns:
        tuple[QuantumCircuit, ParameterVector, list]: Complete circuit, input
            parameters, and weight parameters.
    """
    qc, input_params = create_encoding_circuit(num_qubits)
    ansatz = QuantumCircuit(num_qubits)
    params = ParameterVector('Ψ', num_qubits * reps)

    for r in range(reps):
        for i in range(num_qubits):
            ansatz.ry(params[i * reps + r], i)

        if r != reps - 1:
            for i in range(num_qubits - 1):
                ansatz.cx(i, i + 1)
            if num_qubits > 2:
                ansatz.cx(num_qubits - 1, 0)

    weight_params = ansatz.parameters
    qc.compose(ansatz, inplace=True)
    qc.measure_all()
    return qc, input_params, weight_params


class QuantumLayer(BaseModule):
    """Hybrid Quantum-Classical Neural Network Layer.

    Applies a classical input projection, angle encoding on qubits,
    parameterized variational rotations and entanglement, followed by
    measurement and a classical output projection.

    Args:
        input_dim (int): Input feature dimension. Defaults to 5.
        output_dim (int): Output feature dimension. Defaults to 5.
        num_qubits (int): Number of qubits. Defaults to 5.
        reps (int): Repetitions of variational layers. Defaults to 2.
        residual (bool): Whether to use residual connection. Defaults to True.
        scale_input (bool): Whether to apply tanh(x) * (pi / 2) to scale inputs
            into [-pi/2, pi/2] rotation range. Defaults to True.
        chunk_size (int): Max chunk size to evaluate in SamplerQNN to avoid
            memory bottlenecks. Defaults to 128.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 input_dim=5,
                 output_dim=5,
                 num_qubits=5,
                 reps=2,
                 residual=True,
                 scale_input=True,
                 chunk_size=128,
                 init_cfg=None):
        super(QuantumLayer, self).__init__(init_cfg=init_cfg)
        if not HAS_QISKIT:
            raise ImportError(
                'Qiskit and Qiskit-Machine-Learning are required to use '
                'QuantumLayer.')

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_qubits = num_qubits
        self.reps = reps
        self.residual = residual
        self.scale_input = scale_input
        self.chunk_size = chunk_size

        # Project input features to the number of qubits
        self.input_layer = nn.Linear(input_dim, num_qubits)

        # Create quantum module
        self._build_quantum_module()

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

    def _build_quantum_module(self):
        """Build quantum circuit and initialize PyTorch connector."""
        qc, in_params, w_params = create_variational_circuit(
            num_qubits=self.num_qubits, reps=self.reps)

        # Use V1 Sampler compatible with SamplerQNN
        sampler = Sampler()
        sampler_qnn = SamplerQNN(
            circuit=qc,
            input_params=in_params,
            weight_params=w_params,
            sampler=sampler)

        # PyTorch connector
        self.quantum_connector = TorchConnector(sampler_qnn)

    def _run_quantum(self, x):
        """Evaluate quantum layer, splitting the batch into chunks if needed.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, num_qubits).

        Returns:
            torch.Tensor: Quantum layer output.
        """
        if self.chunk_size and x.size(0) > self.chunk_size:
            chunks = torch.split(x, self.chunk_size)
            q_outs = [self.quantum_connector(chunk) for chunk in chunks]
            return torch.cat(q_outs, dim=0)

        return self.quantum_connector(x)

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

        # Evaluate quantum layer (chunked if batch size is large)
        q_out = self._run_quantum(x_in)

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

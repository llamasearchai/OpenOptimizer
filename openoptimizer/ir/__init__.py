"""IR (Intermediate Representation) module for OpenOptimizer.

This module defines the intermediate representation used by OpenOptimizer,
including tensor, operation, and graph components.
"""

from .graph import ComputationGraph, Node
from .operation import Operation
from .tensor import TensorShape, TensorDescriptor, DataType, Tensor

__all__ = [
    'ComputationGraph',
    'Node',
    'Operation',
    'TensorShape',
    'TensorDescriptor',
    'DataType',
    'Tensor',
] 
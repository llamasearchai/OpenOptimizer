"""Tensor module for OpenOptimizer IR.

This module defines tensor-related components of the IR, including data types,
tensor shapes, and tensor descriptors.
"""

from .DataType import DataType
from .TensorShape import TensorShape, Dimension, DYNAMIC_DIM
from .TensorDescriptor import TensorDescriptor
from .Tensor import Tensor

__all__ = [
    'DataType',
    'TensorShape',
    'Dimension',
    'DYNAMIC_DIM',
    'TensorDescriptor',
    'Tensor',
] 
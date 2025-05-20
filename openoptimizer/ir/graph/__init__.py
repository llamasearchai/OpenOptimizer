"""Graph module for OpenOptimizer IR.

This module defines the computation graph components of the IR, including
nodes and edges.
"""

from .ComputationGraph import ComputationGraph, Node

__all__ = [
    'ComputationGraph',
    'Node',
] 
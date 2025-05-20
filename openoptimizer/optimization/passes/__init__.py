"""Optimization passes for OpenOptimizer."""

from .base_pass import OptimizationPass
from .operator_fusion import OperatorFusionPass # Updated Python-based pass

# Example: If you add C++ wrapped passes later:
# from .conv_bn_fusion_cpp import ConvBnFusionPassCpp

__all__ = [
    "OptimizationPass",
    "OperatorFusionPass",
    # "ConvBnFusionPassCpp",
] 
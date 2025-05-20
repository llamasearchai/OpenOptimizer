"""Optimization module for OpenOptimizer."""

from .passes import OptimizationPass # Python base pass
# When C++ passes and PassManager are wrapped, they can be exposed here too.
# from .pass_manager import PassManager 
# from .passes import ConvBnFusionPassCpp # Example wrapped C++ pass

__all__ = [
    "OptimizationPass",
    # "PassManager",
] 
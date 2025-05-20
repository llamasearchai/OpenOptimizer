"""Frontend module for OpenOptimizer.

This module provides interfaces for importing models from different frameworks
into OpenOptimizer's IR (Intermediate Representation).
"""

# Import submodules
from . import python

# Import commonly used classes for convenience
from .python.optimizer import Optimizer

__all__ = [
    'Optimizer',
    'python',
] 
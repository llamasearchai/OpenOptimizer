"""
OpenOptimizer: Neural Network Optimization Framework
===================================================

OpenOptimizer is a comprehensive neural network optimization framework designed to optimize 
machine learning models for various target platforms, from high-performance servers to 
resource-constrained edge devices.

The framework is built with C++20 for performance-critical components and Python 3.11+ for 
high-level interfaces, providing a powerful set of tools for neural network optimization.

Features
--------
- Framework Compatibility: Import models from PyTorch, TensorFlow, and ONNX.
- Graph Optimization: Advanced techniques like operator fusion, constant folding, etc.
- Target-Specific Code Generation: Generate optimized code for CPUs, GPUs, and edge devices.
- Interactive Visualization: Visualize computation graphs and optimization transformations.
- Python API: High-level interfaces with full customization capabilities.
- C++ Core: High-performance core for critical components.
- TVM Integration: Leverage Apache TVM for tensor computations.
- Model Compression: Built-in support for quantization and pruning.

For more information, visit: https://github.com/openoptimizer/openoptimizer
"""

__version__ = '0.1.0'
__author__ = 'OpenOptimizer Team'

# Import top-level modules
from .frontend import *  # Expose frontend APIs
from .optimization import *  # Expose optimization passes

# Try to import C++ extension module (_cpp_extension)
try:
    from . import _cpp_extension
except ImportError:
    import logging
    logging.warning(
        "Failed to import C++ extension module. Some functionality may be unavailable. "
        "Please make sure the C++ components are properly built and installed."
    )

# Convenience imports for common use cases
from .frontend.python.optimizer import Optimizer

__all__ = [
    'Optimizer',
    '__version__',
] 
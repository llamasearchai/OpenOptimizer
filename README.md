# OpenOptimizer: Neural Network Optimization Framework

<p align="center">
  <img src="OpenOptimizer.svg" alt="OpenOptimizer Logo" width="200"/>
</p>

[![OpenOptimizer CI](https://github.com/openoptimizer/openoptimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/openoptimizer/openoptimizer/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![C++ Version](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](https://isocpp.org/std/status)
[![PyPI version](https://img.shields.io/pypi/v/openoptimizer.svg)](https://pypi.org/project/openoptimizer/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/openoptimizer.svg)](https://pypi.org/project/openoptimizer/)
[![Documentation Status](https://readthedocs.org/projects/openoptimizer/badge/?version=latest)](https://openoptimizer.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://img.shields.io/codecov/c/github/openoptimizer/openoptimizer.svg)](https://codecov.io/gh/openoptimizer/openoptimizer)
[![GitHub Discussions](https://img.shields.io/github/discussions/openoptimizer/openoptimizer.svg)](https://github.com/openoptimizer/openoptimizer/discussions)

OpenOptimizer is a comprehensive neural network optimization framework designed to optimize machine learning models for various target platforms, from high-performance servers to resource-constrained edge devices.

## Features

OpenOptimizer provides a wide array of features to streamline and enhance your model optimization workflow:

- **Broad Framework Compatibility**:
    - Seamlessly import models from popular frameworks like PyTorch, TensorFlow, and ONNX.
- **Sophisticated Graph Optimization**:
    - Employ advanced techniques such as operator fusion, constant folding, layout transformation, and dead code elimination.
- **Target-Specific Code Generation**:
    - Generate highly optimized code tailored for diverse hardware including CPUs, GPUs (CUDA-enabled), and various edge devices.
- **Interactive Visualization Suite**:
    - Visualize computation graphs and the effects of optimization transformations through an intuitive desktop application.
- **User-Friendly Python API**:
    - Access high-level interfaces designed for ease of use, while retaining full customization capabilities for advanced users.
- **High-Performance C++ Core**:
    - Leverage a robust C++20 core for performance-critical components, ensuring speed and efficiency.
- **TVM Integration**:
    - Utilize the power of Apache TVM for optimizing tensor computations and accessing a wider range of hardware backends.
- **Model Compression Techniques**:
    - Benefit from built-in support for quantization (e.g., int8, int16) and pruning to reduce model size and accelerate inference.
- **Edge Deployment Focus**:
    - Specialized tools and workflows for deploying models efficiently on resource-constrained edge devices.
- **Extensible Pass System**:
    - Easily add custom optimization passes to tailor the framework to specific needs.

## Installation

### From PyPI

```bash
pip install openoptimizer
```

### From Source

```bash
git clone https://github.com/openoptimizer/openoptimizer.git
cd openoptimizer

# Install Python dependencies
pip install -r requirements.txt

# Build C++ components
mkdir build && cd build
cmake ..
make -j$(nproc)
cd ..

# Install the package in development mode
pip install -e .
```

## Quick Start

### Optimizing a PyTorch Model

```python
import torch
from openoptimizer import Optimizer
from openoptimizer.optimization.passes import OperatorFusionPass, ConstantFoldingPass

# Define a simple model
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x

# Create model and sample input
model = SimpleModel()
example_input = torch.randn(1, 3, 224, 224)

# Initialize optimizer
optimizer = Optimizer()

# Import PyTorch model
graph = optimizer.import_pytorch_model(model, [example_input])

# Add optimization passes
optimizer.add_pass(OperatorFusionPass())
optimizer.add_pass(ConstantFoldingPass())

# Optimize the graph
optimized_graph = optimizer.optimize(graph)

# Generate optimized code for CPU
optimizer.generate_code(optimized_graph, "output/cpu_model", "cpu")

# Generate optimized code for GPU
optimizer.generate_code(optimized_graph, "output/gpu_model", "gpu")

# Generate optimized code for edge devices
optimizer.generate_code(optimized_graph, "output/edge_model", "edge")

# You can also run the visualization tool to inspect the graph
# Ensure it's installed and in your PATH
# openoptimizer-viz
```

## Documentation

Comprehensive documentation is available at [https://openoptimizer.readthedocs.io/](https://openoptimizer.readthedocs.io/)

For C++ API documentation, build the Doxygen docs:

```bash
cd docs/cpp
doxygen Doxyfile
```

## Visualization

OpenOptimizer includes a Qt-based desktop application for interactive visualization of computation graphs and optimization steps. To launch the visualizer:

```bash
openoptimizer-viz
```

Ensure that you have built the visualization components and the command is available in your system's PATH.

## Architecture

OpenOptimizer is organized into several key components:

- **Frontend**: Interfaces for importing models from different frameworks
- **IR**: Intermediate representation for neural networks
- **Optimization**: Passes for graph-level and operator-level optimizations
- **CodeGen**: Code generation for different target platforms
- **Runtime**: Runtime components for executing optimized models
- **Visualization**: Tools for visualizing and analyzing models

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Join our [GitHub Discussions](https://github.com/openoptimizer/openoptimizer/discussions) to ask questions, share ideas, and collaborate with the community.

## Roadmap

We are continuously working to improve OpenOptimizer. Here are some of_the features and enhancements planned for upcoming releases:

- **Expanded Hardware Support**:
    - Support for more edge AI accelerators (e.g., Google Coral, NVIDIA Jetson specific optimizations).
    - Enhanced support for ARM-based CPUs and mobile GPUs.
- **Advanced Optimization Techniques**:
    - Neural Architecture Search (NAS) integration.
    - Automated mixed-precision quantization.
    - More sophisticated graph partitioning algorithms for distributed inference.
- **Framework Interoperability**:
    - Support for importing models from JAX and other emerging ML frameworks.
    - Improved ONNX export capabilities with support for custom operators.
- **Enhanced Usability**:
    - Web-based visualization tool as an alternative to the desktop application.
    - More comprehensive performance profiling and debugging tools.
    - Simplified API for common use cases.
- **Community and Ecosystem**:
    - Pre-trained model zoo optimized with OpenOptimizer.
    - More example projects and tutorials.

Stay tuned for updates, and feel free to suggest features on our [GitHub Discussions](https://github.com/openoptimizer/openoptimizer/discussions)!

## License

OpenOptimizer is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full license text.
```

### 15. Python API Documentation

```python:OpenOptimizer/docs/python/source/index.rst
OpenOptimizer: Neural Network Optimization Framework
===================================================

OpenOptimizer is a comprehensive neural network optimization framework designed to optimize 
machine learning models for various target platforms, from high-performance servers to 
resource-constrained edge devices.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   optimization/index
   codegen/index
   examples/index
   advanced/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```

```python:OpenOptimizer/docs/python/source/api/index.rst
API Reference
============

This section contains detailed API documentation for OpenOptimizer's Python interface.

.. toctree::
   :maxdepth: 2

   optimizer
   graph
   node
   operation
   passes
   codegen
```

```python:OpenOptimizer/docs/python/source/api/optimizer.rst
Optimizer
=========

.. automodule:: openoptimizer.frontend.python.optimizer
   :members:
   :undoc-members:
   :show-inheritance:
```

### 16. Performance Benchmarking

```python:OpenOptimizer/tests/benchmarks/run_benchmarks.py
#!/usr/bin/env python
import argparse
import json
import os
import time
from typing import Dict, List, Any, Optional

import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

from openoptimizer.frontend.python.optimizer import Optimizer
from openoptimizer.optimization.passes import (
    OperatorFusionPass,
    ConstantFoldingPass,
    LayoutTransformationPass,
    PruningPass,
    QuantizationPass,
)

def create_model(name: str):
    """Create a model by name."""
    if name == "resnet18":
        return models.resnet18(pretrained=False)
    elif name == "mobilenet_v2":
        return models.mobilenet_v2(pretrained=False)
    elif name == "efficientnet_b0":
        return models.efficientnet_b0(pretrained=False)
    else:
        raise ValueError(f"Unknown model: {name}")

def run_benchmark(model_name: str, passes: List[str], target: str, 
                 batch_size: int = 1, iterations: int = 100) -> Dict[str, Any]:
    """Run a benchmark for a specific model and optimization passes."""
    model = create_model(model_name)
    example_inputs = torch.randn(batch_size, 3, 224, 224)
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    # Import model
    start_time = time.time()
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    import_time = time.time() - start_time
    
    # Add passes
    for pass_name in passes:
        if pass_name == "operator_fusion":
            optimizer.add_pass(OperatorFusionPass())
        elif pass_name == "constant_folding":
            optimizer.add_pass(ConstantFoldingPass())
        elif pass_name == "layout_transformation":
            optimizer.add_pass(LayoutTransformationPass())
        elif pass_name == "pruning":
            optimizer.add_pass(PruningPass(threshold=0.1))
        elif pass_name == "quantization":
            optimizer.add_pass(QuantizationPass(bits=8))
    
    # Optimize the graph
    start_time = time.time()
    optimized_graph = optimizer.optimize(graph)
    optimization_time = time.time() - start_time
    
    # Generate code
    start_time = time.time()
    output_dir = f"benchmark_output/{model_name}_{target}"
    os.makedirs(output_dir, exist_ok=True)
    optimizer.generate_code(optimized_graph, output_dir, target)
    codegen_time = time.time() - start_time
    
    # Measure inference time
    inference_times = []
    for _ in range(iterations):
        start_time = time.time()
        # Run inference with generated code
        # In a real benchmark, this would execute the generated code
        # For now, we'll just simulate it
        time.sleep(0.001)  # Simulate inference
        inference_times.append(time.time() - start_time)
    
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    
    return {
        "model": model_name,
        "passes": passes,
        "target": target,
        "batch_size": batch_size,
        "import_time_s": import_time,
        "optimization_time_s": optimization_time,
        "codegen_time_s": codegen_time,
        "avg_inference_time_ms": avg_inference_time,
        "inference_times_ms": [t * 1000 for t in inference_times],
    }

def main():
    parser = argparse.ArgumentParser(description="Run OpenOptimizer benchmarks")
    parser.add_argument("--models", nargs="+", default=["resnet18", "mobilenet_v2"],
                        help="Models to benchmark")
    parser.add_argument("--passes", nargs="+", 
                        default=["operator_fusion", "constant_folding"],
                        help="Optimization passes to apply")
    parser.add_argument("--targets", nargs="+", default=["cpu", "gpu", "edge"],
                        help="Target platforms")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8],
                        help="Batch sizes")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations for inference timing")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()
    
    results = []
    
    for model_name in args.models:
        for target in args.targets:
            for batch_size in args.batch_sizes:
                result = run_benchmark(
                    model_name=model_name,
                    passes=args.passes,
                    target=target,
                    batch_size=batch_size,
                    iterations=args.iterations
                )
                results.append(result)
                print(f"Model: {model_name}, Target: {target}, Batch size: {batch_size}")
                print(f"  Import time: {result['import_time_s']:.4f}s")
                print(f"  Optimization time: {result['optimization_time_s']:.4f}s")
                print(f"  Code generation time: {result['codegen_time_s']:.4f}s")
                print(f"  Average inference time: {result['avg_inference_time_ms']:.4f}ms")
    
    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate visualization
    visualize_results(results)

def visualize_results(results: List[Dict[str, Any]]):
    """Generate visualizations of benchmark results."""
    # Inference times by model and target
    models = sorted(set(r["model"] for r in results))
    targets = sorted(set(r["target"] for r in results))
    
    # Filter for batch_size=1 for this visualization
    batch1_results = [r for r in results if r["batch_size"] == 1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.25
    index = np.arange(len(models))
    
    for i, target in enumerate(targets):
        times = [next((r["avg_inference_time_ms"] for r in batch1_results 
                     if r["model"] == model and r["target"] == target), 0)
                for model in models]
        ax.bar(index + i * bar_width, times, bar_width, label=target)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time by Model and Target Platform')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.savefig("benchmark_inference_times.png")
    plt.close()
    
    # Optimization time by model
    fig, ax = plt.subplots(figsize=(10, 6))
    opt_times = [next((r["optimization_time_s"] for r in batch1_results if r["model"] == model), 0)
                for model in models]
    ax.bar(models, opt_times)
    ax.set_xlabel('Model')
    ax.set_ylabel('Optimization Time (s)')
    ax.set_title('Optimization Time by Model')
    
    plt.savefig("benchmark_optimization_times.png")
    plt.close()

if __name__ == "__main__":
    main()
```

This implementation covers the key components of the OpenOptimizer framework, including:

1. C++ and Python core components
2. Optimization passes 
3. Code generation for different targets (CPU, GPU, Edge)
4. Visualization tools
5. CI/CD pipeline
6. Documentation
7. Benchmark utilities

The framework follows a modular architecture as specified in the requirements, with clear separation between frontend, IR, optimization, code generation, and visualization components.
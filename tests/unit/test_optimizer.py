import pytest
import torch
import numpy as np
import tensorflow as tf
import os
import tempfile

from openoptimizer.frontend.python.optimizer import Optimizer
from openoptimizer.optimization.passes import OperatorFusionPass, ConstantFoldingPass

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

def test_optimizer_initialization():
    optimizer = Optimizer()
    assert optimizer is not None
    
def test_pytorch_model_import():
    optimizer = Optimizer()
    model = SimpleModel()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    assert graph is not None
    
    # Check that the graph has the correct structure
    nodes = graph.get_nodes()
    assert len(nodes) >= 3  # At least conv, relu, pool
    
    input_nodes = graph.get_input_nodes()
    output_nodes = graph.get_output_nodes()
    assert len(input_nodes) == 1
    assert len(output_nodes) == 1

def test_optimization_passes():
    optimizer = Optimizer()
    model = SimpleModel()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    
    # Add optimization passes
    optimizer.add_pass(OperatorFusionPass())
    optimizer.add_pass(ConstantFoldingPass())
    
    # Optimize the graph
    optimized_graph = optimizer.optimize(graph)
    
    # The optimized graph should have fewer nodes due to fusion
    assert len(optimized_graph.get_nodes()) < len(graph.get_nodes())

def test_code_generation():
    optimizer = Optimizer()
    model = SimpleModel()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    optimizer.add_pass(OperatorFusionPass())
    optimized_graph = optimizer.optimize(graph)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "generated_code")
        optimizer.generate_code(optimized_graph, output_path, "cpu")
        
        # Check that the code was generated
        assert os.path.exists(output_path)
        assert len(os.listdir(output_path)) > 0
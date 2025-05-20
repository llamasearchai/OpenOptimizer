import os
import tempfile
import shutil
import pytest
import torch
import numpy as np

from openoptimizer.frontend.python.optimizer import Optimizer
from openoptimizer.optimization.passes import OperatorFusionPass, QuantizationPass

class SimpleConvNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(32 * 56 * 56, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

@pytest.mark.integration
def test_pytorch_to_cpu():
    # Create a model
    model = SimpleConvNet()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    # Import model
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    
    # Add optimization passes
    optimizer.add_pass(OperatorFusionPass())
    
    # Optimize the graph
    optimized_graph = optimizer.optimize(graph)
    
    # Generate CPU code
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "cpu_model")
        optimizer.generate_code(optimized_graph, output_path, "cpu")
        
        # Verify generated files
        assert os.path.exists(os.path.join(output_path, "model.h"))
        assert os.path.exists(os.path.join(output_path, "model.cpp"))
        assert os.path.exists(os.path.join(output_path, "CMakeLists.txt"))

@pytest.mark.integration
def test_pytorch_to_edge():
    # Create a model
    model = SimpleConvNet()
    example_inputs = torch.randn(1, 3, 224, 224)
    
    # Initialize optimizer
    optimizer = Optimizer()
    
    # Import model
    graph = optimizer.import_pytorch_model(model, [example_inputs])
    
    # Add optimization passes
    optimizer.add_pass(OperatorFusionPass())
    optimizer.add_pass(QuantizationPass(bits=8))
    
    # Optimize the graph
    optimized_graph = optimizer.optimize(graph)
    
    # Generate Edge code
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "edge_model")
        optimizer.generate_code(optimized_graph, output_path, "edge")
        
        # Verify generated files
        assert os.path.exists(os.path.join(output_path, "edge_model.h"))
        assert os.path.exists(os.path.join(output_path, "edge_model.c"))
        assert os.path.exists(os.path.join(output_path, "model.tflite"))
        assert os.path.exists(os.path.join(output_path, "deployment.json"))
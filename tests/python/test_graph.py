"""Tests for the ComputationGraph class."""

import pytest
import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openoptimizer.ir.graph import ComputationGraph, Node
from openoptimizer.ir.operation import Operation

class MockOperation(Operation):
    """Mock operation for testing."""
    
    def __init__(self, name, type_str="MockOp"):
        super().__init__(name, type_str)
    
    def infer_shapes(self, input_shapes):
        return input_shapes

def test_computation_graph_creation():
    """Test creation of a ComputationGraph."""
    graph = ComputationGraph("TestGraph")
    assert graph.name == "TestGraph"
    assert len(graph.nodes) == 0

def test_add_node():
    """Test adding a node to a graph."""
    graph = ComputationGraph("TestGraph")
    op = MockOperation("op1")
    node = graph.add_node("node1", op)
    
    assert node.name == "node1"
    assert node.operation is op
    assert len(graph.nodes) == 1
    assert graph.get_node("node1") is node

def test_add_edge():
    """Test adding an edge between nodes."""
    graph = ComputationGraph("TestGraph")
    op1 = MockOperation("op1")
    op2 = MockOperation("op2")
    
    node1 = graph.add_node("node1", op1)
    node2 = graph.add_node("node2", op2)
    
    graph.add_edge(node1, node2)
    
    # node1 should be an input to node2
    node2_inputs = [input_node.lock() for input_node in node2.inputs]
    assert node1 in node2_inputs
    
    # node2 should be an output of node1
    node1_outputs = [output_node.lock() for output_node in node1.outputs]
    assert node2 in node1_outputs

def test_remove_edge():
    """Test removing an edge between nodes."""
    graph = ComputationGraph("TestGraph")
    op1 = MockOperation("op1")
    op2 = MockOperation("op2")
    
    node1 = graph.add_node("node1", op1)
    node2 = graph.add_node("node2", op2)
    
    graph.add_edge(node1, node2)
    graph.remove_edge(node1, node2)
    
    # node1 should no longer be an input to node2
    node2_inputs = [input_node.lock() for input_node in node2.inputs if input_node.lock()]
    assert node1 not in node2_inputs
    
    # node2 should no longer be an output of node1
    node1_outputs = [output_node.lock() for output_node in node1.outputs if output_node.lock()]
    assert node2 not in node1_outputs

def test_remove_node():
    """Test removing a node from a graph."""
    graph = ComputationGraph("TestGraph")
    op1 = MockOperation("op1")
    op2 = MockOperation("op2")
    op3 = MockOperation("op3")
    
    node1 = graph.add_node("node1", op1)
    node2 = graph.add_node("node2", op2)
    node3 = graph.add_node("node3", op3)
    
    # Connect nodes: node1 -> node2 -> node3
    graph.add_edge(node1, node2)
    graph.add_edge(node2, node3)
    
    # Remove the middle node
    result = graph.remove_node(node2)
    assert result is True
    
    # node2 should no longer be in the graph
    assert graph.get_node("node2") is None
    assert len(graph.nodes) == 2
    
    # Connections should be removed
    node1_outputs = [output_node.lock() for output_node in node1.outputs if output_node.lock()]
    assert node2 not in node1_outputs
    
    node3_inputs = [input_node.lock() for input_node in node3.inputs if input_node.lock()]
    assert node2 not in node3_inputs

def test_graph_io_nodes():
    """Test setting input and output nodes of a graph."""
    graph = ComputationGraph("TestGraph")
    op1 = MockOperation("op1")
    op2 = MockOperation("op2")
    
    node1 = graph.add_node("node1", op1)
    node2 = graph.add_node("node2", op2)
    
    # Set input and output nodes
    graph.set_input_nodes([node1])
    graph.set_output_nodes([node2])
    
    assert graph.input_nodes == [node1]
    assert graph.output_nodes == [node2]

if __name__ == "__main__":
    pytest.main(["-xvs", __file__]) 
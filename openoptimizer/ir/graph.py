"""Python wrappers for IR Graph and Node components."""

from typing import List, Optional, Any, Dict, Union
from openoptimizer import _cpp_extension
from .tensor import TensorShape, TensorDescriptor, Tensor # Python wrappers
from .operation import Operation # Python Operation wrapper

class Node:
    """Python wrapper for C++ ir::Node."""
    def __init__(self, name: str, operation: Operation, cpp_node: Optional[_cpp_extension.Node] = None):
        if cpp_node:
            self._cpp_node = cpp_node
            self._name = cpp_node.name # Sync name
            # Operation needs to be wrapped from C++ op if cpp_node is provided
            # This requires a way to know which Python Operation subclass to use.
            # For now, assuming operation is primarily set from Python side for new nodes.
            # If constructing from_cpp, operation wrapper would be created based on cpp_node.operation.type
            self._operation = Operation.from_cpp(cpp_node.operation) if cpp_node.operation else None 
        else:
            if not isinstance(operation, Operation):
                raise TypeError("operation must be an instance of ir.Operation")
            self._cpp_node = _cpp_extension.Node(name, operation.cpp_op)
            self._name = name
            self._operation = operation

    @property
    def cpp_node(self) -> Optional[_cpp_extension.Node]:
        return getattr(self, '_cpp_node', None)

    @property
    def name(self) -> str:
        # return self.cpp_node.name if self.cpp_node else self._name
        return self._name # Name is primarily Python managed after construction for consistency

    @property
    def operation(self) -> Optional[Operation]:
        # cpp_op = self.cpp_node.operation if self.cpp_node else None
        # if cpp_op and (self._operation is None or self._operation.cpp_op != cpp_op):
        #     self._operation = Operation.from_cpp(cpp_op) # Re-wrap if C++ changed
        return self._operation

    @property
    def inputs(self) -> List['Node']:
        if self.cpp_node:
            return [Node.from_cpp(cpp_n) for cpp_n in self.cpp_node.get_inputs_locked()]
        return []

    @property
    def outputs(self) -> List['Node']:
        if self.cpp_node:
            return [Node.from_cpp(cpp_n) for cpp_n in self.cpp_node.get_outputs_locked()]
        return []

    # add_input/add_output are typically graph-level operations via graph.add_edge
    # Direct node.add_input might bypass graph's understanding of edges.

    def set_metadata(self, key: str, value: Any) -> None:
        if self.cpp_node:
            self.cpp_node.set_metadata(key, value)

    def get_metadata(self, key: str) -> Any:
        if self.cpp_node:
            return self.cpp_node.get_metadata(key)
        return None # Or raise

    def has_metadata(self, key: str) -> bool:
        if self.cpp_node:
            return self.cpp_node.has_metadata(key)
        return False

    @property
    def metadata(self) -> Dict[str, Any]:
        if self.cpp_node:
            return self.cpp_node.metadata # pybind11 converts std::unordered_map<string, any>
        return {}

    def __repr__(self) -> str:
        op_type = self.operation.get_type() if self.operation else "None"
        return f"<Node name='{self.name}' op_type='{op_type}'>"

    @classmethod
    def from_cpp(cls, cpp_node: _cpp_extension.Node) -> 'Node':
        # When creating from C++ node, the operation also needs to be wrapped.
        # This requires a mapping from C++ op type string to Python Operation subclass.
        py_op_wrapper = Operation.from_cpp(cpp_node.operation) # Uses generic Operation.from_cpp
        instance = cls(cpp_node.name, py_op_wrapper, cpp_node=cpp_node)
        return instance

class ComputationGraph:
    """Python wrapper for C++ ir::ComputationGraph."""
    def __init__(self, name: str = "UnnamedGraph", cpp_graph: Optional[_cpp_extension.ComputationGraph] = None):
        if cpp_graph:
            self._cpp_graph = cpp_graph
        else:
            self._cpp_graph = _cpp_extension.ComputationGraph(name)
        self._node_cache: Dict[str, Node] = {} # Cache for Python Node wrappers

    @property
    def cpp_graph(self) -> Optional[_cpp_extension.ComputationGraph]:
        return getattr(self, '_cpp_graph', None)

    @property
    def name(self) -> str:
        return self.cpp_graph.name
    
    @name.setter
    def name(self, value: str) -> None:
        self.cpp_graph.name = value

    def _wrap_cpp_node(self, cpp_node: Optional[_cpp_extension.Node]) -> Optional[Node]:
        if not cpp_node: return None
        if cpp_node.name in self._node_cache:
            # TODO: Could verify if the cpp_node pointer is the same if concerned about stale cache
            return self._node_cache[cpp_node.name]
        py_node = Node.from_cpp(cpp_node)
        self._node_cache[cpp_node.name] = py_node
        return py_node

    def add_node(self, node_name: str, operation: Operation) -> Node:
        if not isinstance(operation, Operation):
            raise TypeError("operation must be an instance of ir.Operation")
        cpp_op = operation.cpp_op 
        if not cpp_op:
            raise ValueError("Provided Python Operation does not have a C++ counterpart.")
        
        cpp_node = self.cpp_graph.add_node(node_name, cpp_op)
        # Create Python wrapper and cache it
        py_node = Node(node_name, operation, cpp_node=cpp_node) # Pass Python op for consistency
        self._node_cache[node_name] = py_node
        return py_node

    def add_edge(self, from_node: Union[str, Node], to_node: Union[str, Node]) -> None:
        from_node_name = from_node if isinstance(from_node, str) else from_node.name
        to_node_name = to_node if isinstance(to_node, str) else to_node.name
        self.cpp_graph.add_edge(from_node_name, to_node_name)
        # Graph structure changes, node cache for inputs/outputs of affected nodes might be stale
        # but Node.inputs/outputs re-fetch and re-wrap, so it should be okay.

    def get_node(self, name: str) -> Optional[Node]:
        cpp_node = self.cpp_graph.get_node(name)
        if cpp_node:
            return self._wrap_cpp_node(cpp_node)
        return None

    @property
    def nodes(self) -> List[Node]:
        return [self._wrap_cpp_node(cpp_n) for cpp_n in self.cpp_graph.get_nodes()]
    
    def _resolve_nodes_from_names_or_nodes(self, nodes_or_names: List[Union[str, Node]]) -> List[_cpp_extension.Node]:
        cpp_nodes: List[_cpp_extension.Node] = []
        for item in nodes_or_names:
            if isinstance(item, str):
                cpp_node = self.cpp_graph.get_node(item)
                if not cpp_node: raise ValueError(f"Node '{item}' not found in graph.")
                cpp_nodes.append(cpp_node)
            elif isinstance(item, Node) and item.cpp_node:
                cpp_nodes.append(item.cpp_node)
            else:
                raise TypeError(f"Items must be node names (str) or Node objects, got {type(item)}.")
        return cpp_nodes

    def set_input_nodes(self, inputs: List[Union[str, Node]]) -> None:
        cpp_input_nodes = self._resolve_nodes_from_names_or_nodes(inputs)
        self.cpp_graph.set_input_nodes(cpp_input_nodes)

    def set_output_nodes(self, outputs: List[Union[str, Node]]) -> None:
        cpp_output_nodes = self._resolve_nodes_from_names_or_nodes(outputs)
        self.cpp_graph.set_output_nodes(cpp_output_nodes)

    @property
    def input_nodes(self) -> List[Node]:
        return [self._wrap_cpp_node(cpp_n) for cpp_n in self.cpp_graph.input_nodes]

    @property
    def output_nodes(self) -> List[Node]:
        return [self._wrap_cpp_node(cpp_n) for cpp_n in self.cpp_graph.output_nodes]
    
    def remove_node(self, node_or_name: Union[str, Node]) -> bool:
        if isinstance(node_or_name, str):
            # Clear cache if node is removed by name
            if node_or_name in self._node_cache: del self._node_cache[node_or_name]
            return self.cpp_graph.remove_node_by_name(node_or_name)
        elif isinstance(node_or_name, Node) and node_or_name.cpp_node:
            if node_or_name.name in self._node_cache: del self._node_cache[node_or_name.name]
            return self.cpp_graph.remove_node(node_or_name.cpp_node)
        return False

    def remove_edge(self, from_node: Union[str, Node], to_node: Union[str, Node]) -> None:
        from_node_name = from_node if isinstance(from_node, str) else from_node.name
        to_node_name = to_node if isinstance(to_node, str) else to_node.name
        self.cpp_graph.remove_edge_by_name(from_node_name, to_node_name)

    def dump(self) -> None:
        self.cpp_graph.dump()

    def __repr__(self) -> str:
        return f"<ComputationGraph name='{self.name}' nodes={len(self.nodes)} inputs={len(self.input_nodes)} outputs={len(self.output_nodes)} >"

    @classmethod
    def from_cpp(cls, cpp_graph: _cpp_extension.ComputationGraph) -> 'ComputationGraph':
        instance = cls(name=cpp_graph.name, cpp_graph=cpp_graph)
        # Optionally pre-populate cache, though _wrap_cpp_node handles it lazily
        # for cpp_n in cpp_graph.get_nodes():
        #     instance._wrap_cpp_node(cpp_n)
        return instance

__all__ = ["Node", "ComputationGraph"] 
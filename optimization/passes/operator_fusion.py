import structlog
from .base_pass import OptimizationPass
from openoptimizer.ir.graph import ComputationGraph, Node
# Attempt to import Operation, assuming it might be in ir.Operation or ir.operation
# This needs to be a valid import path for the Operation base class.
try:
    from openoptimizer.ir.Operation import Operation
except ImportError:
    from openoptimizer.ir.operation import Operation # Fallback path
from typing import List, Tuple, Set, Optional

logger = structlog.get_logger(__name__)

# Placeholder for FusedOperation, ideally this would be a proper IR Operation subclass
class FusedOperation(Operation):
    def __init__(self, op1: Operation, op2: Operation, name: str):
        super().__init__(name, "Fused")
        self.op1 = op1
        self.op2 = op2
        logger.debug("fused_operation.init", name=name, op1_type=op1.get_type(), op2_type=op2.get_type())

    def get_type(self) -> str:
        # Ensure op1 and op2 and their types are valid before calling get_type()
        op1_type_str = self.op1.get_type() if self.op1 else "UnknownOp1"
        op2_type_str = self.op2.get_type() if self.op2 else "UnknownOp2"
        return f"Fused({op1_type_str}+{op2_type_str})"

class FusedPythonOperation(Operation):
    """Python representation of a generic fused operation."""
    def __init__(self, name: str, original_op_types: List[str], **kwargs):
        fused_type_str = f"Fused[{','.join(original_op_types)}]"
        super().__init__(name=name, type_str=fused_type_str, **kwargs) # Pass any other attributes if Operation supports it
        self.original_op_types = original_op_types
        logger.debug("fused_python_operation.init", name=name, original_types=original_op_types)

    # infer_shapes would need to be implemented based on the specific fusion.
    # For a generic fused op, this is hard without knowing the constituent ops' details.
    # It might require access to the original ops or a predefined shape inference rule.
    def infer_shapes(self, input_shapes: List[object]) -> List[object]:
        logger.warn("fused_python_operation.infer_shapes.placeholder", name=self.name)
        # Placeholder: if it's an element-wise chain, output shape might be same as input
        if input_shapes: return [input_shapes[0]] 
        return []

class OperatorFusionPass(OptimizationPass):
    """Optimization pass that fuses compatible operators to reduce memory transfers"""
    
    def __init__(self, fusion_patterns: Optional[List[Tuple[str, str]]] = None, max_iterations: int = 10):
        super().__init__("OperatorFusion")
        self.fusion_patterns: List[Tuple[str, str]] = fusion_patterns or self._default_fusion_patterns()
        self.max_iterations = max_iterations
        logger.info("operator_fusion_pass.init", name=self.name, patterns_count=len(self.fusion_patterns), max_iter=self.max_iterations)
        
    def _default_fusion_patterns(self) -> List[Tuple[str, str]]:
        """Define default fusion patterns (operator type strings that can be fused)"""
        return [
            ("Conv2d", "ReLU"),
            ("Conv2d", "BatchNorm2d"),
            ("Linear", "ReLU"),
            ("BatchNorm2d", "ReLU"),
            ("Conv2d", "Add"),
            ("MatMul", "Add"),
        ]
    
    def run(self, graph: ComputationGraph) -> ComputationGraph:
        """Run operator fusion on the graph"""
        log = logger.bind(pass_name=self.name, graph_name=graph.getName())
        log.info("operator_fusion_pass.run.started")
        
        total_fusions_performed = 0
        for i in range(self.max_iterations):
            log.debug("run.iteration.started", iteration=i + 1)
            fusion_candidates = self._find_fusion_candidates(graph)

            if not fusion_candidates:
                log.info("run.iteration.no_candidates_found", iteration=i + 1)
                break # No more fusions possible in this iteration
            
            log.info("run.iteration.candidates_found", iteration=i + 1, count=len(fusion_candidates))
            
            fused_in_this_iteration = 0
            # Iterate on a copy for safety if _fuse_nodes modifies underlying node list used by _find_fusion_candidates indirectly
            for parent_node, child_node in list(fusion_candidates):
                # Fetch current nodes by name as graph structure might have changed
                current_parent = graph.getNode(parent_node.getName())
                current_child = graph.getNode(child_node.getName())

                if current_parent and current_child:
                    # Re-check fusibility with current graph state before fusing
                    if self._can_fuse(current_parent, current_child):
                        if self._fuse_nodes(graph, current_parent, current_child):
                            total_fusions_performed += 1
                            fused_in_this_iteration += 1
                            # Important: after a fusion, the graph structure changed.
                            # Break from inner loop and re-scan for candidates in a new iteration.
                            log.debug("run.iteration.fusion_successful_re_evaluating", 
                                      fused_pair=(current_parent.getName(), current_child.getName()))
                            break 
                    else:
                        log.debug("run.iteration.candidate_no_longer_fusable", 
                                   parent=current_parent.getName(), child=current_child.getName())
                else:
                    log.warn("run.iteration.stale_candidate_nodes_gone", 
                               parent_name=parent_node.getName(), child_name=child_node.getName())
            
            if fused_in_this_iteration == 0: # No fusions made in this full scan of candidates
                log.info("run.iteration.no_fusions_performed_this_iter", iteration=i + 1)
                break # Converged

            log.info("run.iteration.completed", iteration=i + 1, fusions_this_iter=fused_in_this_iteration)
            if i == self.max_iterations - 1 and fused_in_this_iteration > 0:
                log.warn("run.max_iterations_reached_with_ongoing_fusions", max_iter=self.max_iterations)

        log.info("run.completed", total_fused_pairs=total_fusions_performed, total_iterations=i + 1)
        return graph
    
    def _find_fusion_candidates(self, graph: ComputationGraph) -> List[Tuple[Node, Node]]:
        """Find pairs of nodes that can be fused"""
        candidates: List[Tuple[Node, Node]] = []
        all_nodes = graph.getNodes() 
        
        processed_children_names: Set[str] = set() 

        for current_node in all_nodes:
            if not current_node or not current_node.getOperation():
                continue

            output_nodes_shared: List[Node] = []
            for weak_output_node in current_node.getOutputs():
                if shared_output_node := weak_output_node.lock():
                    output_nodes_shared.append(shared_output_node)
            
            for child_node in output_nodes_shared:
                if not child_node or not child_node.getOperation():
                    continue
                
                if child_node.getName() in processed_children_names:
                    continue
                    
                if self._can_fuse(current_node, child_node):
                    candidates.append((current_node, child_node))
                    processed_children_names.add(child_node.getName())
        return candidates
    
    def _can_fuse(self, parent_node: Node, child_node: Node) -> bool:
        """Check if two nodes can be fused based on operation types and patterns"""
        # Ensure operations exist before getting their types
        parent_op = parent_node.getOperation()
        child_op = child_node.getOperation()
        if not parent_op or not child_op:
            return False
            
        parent_op_type = parent_op.get_type()
        child_op_type = child_op.get_type()

        if (parent_op_type, child_op_type) not in self.fusion_patterns:
            return False

        child_inputs_shared: List[Node] = []
        for weak_input_node in child_node.getInputs():
            if shared_input_node := weak_input_node.lock():
                child_inputs_shared.append(shared_input_node)
        
        if len(child_inputs_shared) != 1 or child_inputs_shared[0].getName() != parent_node.getName():
            return False

        parent_outputs_shared: List[Node] = []
        for weak_output_node in parent_node.getOutputs():
            if shared_output_node := weak_output_node.lock():
                parent_outputs_shared.append(shared_output_node)
        
        # If parent_node has only child_node as its consumer OR it's a special fusible case with multiple outputs
        if len(parent_outputs_shared) == 1 or self._can_fuse_with_multiple_outputs(parent_op_type, child_op_type):
            # Additional check: ensure child_node is indeed one of parent_node's outputs
            is_direct_output = any(node.getName() == child_node.getName() for node in parent_outputs_shared)
            if is_direct_output:
                 return True
        
        return False
    
    def _can_fuse_with_multiple_outputs(self, op1_type: str, op2_type: str) -> bool:
        """Some operations can be fused even if the first op has multiple outputs"""
        # For example, we can fuse BatchNorm into Conv even if Conv output is used elsewhere
        special_cases = [
            ("Conv2d", "BatchNorm2d"),
        ]
        return (op1_type, op2_type) in special_cases
    
    def _fuse_nodes(self, graph: ComputationGraph, parent: Node, child: Node) -> bool:
        """Fuse two nodes by creating a new fused operation"""
        log = logger.bind(pass_name=self.name, graph_name=graph.getName(), 
                          parent_name=parent.getName(), child_name=child.getName())
        log.info("operator_fusion_pass.fuse_nodes.attempting")

        parent_op = parent.getOperation()
        child_op = child.getOperation()
        if not parent_op or not child_op:
            log.error("operator_fusion_pass.fuse_nodes.null_operation")
            return False

        fused_op_name = f"{parent.getName()}_{child.getName()}_fused"
        fused_ir_op = FusedPythonOperation(name=fused_op_name, original_op_types=[parent_op.get_type(), child_op.get_type()])
        
        try:
            fused_node = graph.addNode(fused_op_name, fused_ir_op)
            log.info("operator_fusion_pass.fuse_nodes.created_fused_node", fused_node_name=fused_node.getName())

            # Rewire inputs of parent to fused_node
            for weak_input_node in parent.getInputs():
                if shared_input_node := weak_input_node.lock():
                    graph.addEdge(shared_input_node, fused_node)
            
            # Rewire outputs: consumers of child become consumers of fused_node
            child_consumers_shared: List[Node] = []
            for weak_consumer_node in child.getOutputs(): # Outputs of child are its consumers
                if shared_consumer_node := weak_consumer_node.lock():
                    child_consumers_shared.append(shared_consumer_node)

            for consumer_of_child in child_consumers_shared:
                graph.addEdge(fused_node, consumer_of_child)
                # Remove old edge from child to consumer_of_child.
                # This requires graph.removeEdge(child, consumer_of_child) or child.removeOutput(consumer_of_child)
                # and consumer_of_child.removeInput(child).
                # The current addEdge in C++ might implicitly handle some of this by overwriting,
                # but explicit removal is safer for graph integrity.
            
            # IMPORTANT: Remove original parent and child nodes from the graph.
            # This step is critical and requires a robust graph.removeNode() method.
            # If not done correctly, the graph will be in an inconsistent state.
            log.info("operator_fusion_pass.fuse_nodes.removing_original_nodes", 
                       parent_name=parent.getName(), child_name=child.getName())
            
            # Placeholder for node removal. These would call graph.removeNode(name_or_ptr)
            # graph.removeNode(parent.getName()) 
            # graph.removeNode(child.getName())
            # This functionality needs to be added to ComputationGraph class.
            # For now, this fusion is INCOMPLETE as old nodes are not removed.
            log.warn("operator_fusion_pass.fuse_nodes.node_removal_missing", 
                       message="Actual removal of original parent and child nodes is not implemented. Graph is not fully fused.")

            log.info("operator_fusion_pass.fuse_nodes.success", fused_node_name=fused_node.getName())
            return True
        except Exception as e:
            # If node creation/edge adding fails, graph might be in a partial state.
            # Robust fusion would require transactional graph edits or rollback.
            log.error("operator_fusion_pass.fuse_nodes.failed_during_rewiring", error=str(e), exc_info=True)
            # Attempt to remove the partially added fused_node if it exists
            if graph.getNode(fused_op_name):
                log.info("operator_fusion_pass.fuse_nodes.attempting_cleanup_fused_node", fused_node_name=fused_op_name)
                # graph.removeNode(fused_op_name) # Needs implementation
            return False
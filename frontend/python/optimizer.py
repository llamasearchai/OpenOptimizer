import structlog
from typing import List, Optional, Union, Dict, Any
import torch
import tensorflow as tf

from openoptimizer._cpp_extension import Optimizer as CppOptimizer
from openoptimizer.optimization.passes import OptimizationPass
from openoptimizer.ir.graph import ComputationGraph

logger = structlog.get_logger(__name__)

class OpenOptimizerError(Exception):
    """Base exception for errors raised by OpenOptimizer."""
    pass

class ModelImportError(OpenOptimizerError):
    """Raised when there is an error importing a model."""
    pass

class OptimizationError(OpenOptimizerError):
    """Raised when there is an error during the optimization process."""
    pass

class CodeGenerationError(OpenOptimizerError):
    """Raised when there is an error during code generation."""
    pass

class Optimizer:
    """High-level Python interface for the OpenOptimizer framework"""
    
    def __init__(self):
        try:
            self._cpp_optimizer = CppOptimizer()
            self._passes: List[OptimizationPass] = []
            logger.info("optimizer.init.success", framework="OpenOptimizer")
        except Exception as e:
            logger.error("optimizer.init.failed", error=str(e), exc_info=True)
            raise OpenOptimizerError(f"Failed to initialize C++ Optimizer core: {e}") from e
        
    def import_pytorch_model(self, 
                           model: Union[str, torch.nn.Module],
                           example_inputs: Optional[List[torch.Tensor]] = None) -> ComputationGraph:
        """Import a PyTorch model into OpenOptimizer's IR."""
        model_name = model if isinstance(model, str) else model.__class__.__name__
        log = logger.bind(model_name=model_name, framework="PyTorch")
        try:
            if isinstance(model, str):
                log.info("optimizer.import.pytorch.file.started")
                graph_cpp = self._cpp_optimizer.import_from_pytorch(model)
            else:
                log.info("optimizer.import.pytorch.module.started")
                raise NotImplementedError("Direct PyTorch module import requires TorchScript conversion first or direct C++ support.")
            
            log.info("optimizer.import.pytorch.success")
            return ComputationGraph()
        except NotImplementedError as e:
            log.warn("optimizer.import.pytorch.not_implemented", reason=str(e))
            raise
        except Exception as e:
            log.error("optimizer.import.pytorch.failed", error=str(e), exc_info=True)
            raise ModelImportError(f"Failed to import PyTorch model '{model_name}': {e}") from e
    
    def import_tensorflow_model(self,
                              model: Union[str, tf.keras.Model],
                              example_inputs: Optional[List[tf.Tensor]] = None) -> ComputationGraph:
        """Import a TensorFlow model into OpenOptimizer's IR."""
        model_name = model if isinstance(model, str) else model.name
        log = logger.bind(model_name=model_name, framework="TensorFlow")
        try:
            if isinstance(model, str):
                log.info("optimizer.import.tensorflow.file.started")
                graph_cpp = self._cpp_optimizer.import_from_tensorflow(model)
            else:
                log.info("optimizer.import.tensorflow.module.started")
                raise NotImplementedError("Direct TensorFlow Keras model import requires saving to SavedModel format first or direct C++ support.")
            
            log.info("optimizer.import.tensorflow.success")
            return ComputationGraph()
        except NotImplementedError as e:
            log.warn("optimizer.import.tensorflow.not_implemented", reason=str(e))
            raise
        except Exception as e:
            log.error("optimizer.import.tensorflow.failed", error=str(e), exc_info=True)
            raise ModelImportError(f"Failed to import TensorFlow model '{model_name}': {e}") from e

    def import_onnx_model(self, model_path: str) -> ComputationGraph:
        """Import an ONNX model into OpenOptimizer's IR."""
        log = logger.bind(model_path=model_path, framework="ONNX")
        log.info("optimizer.import.onnx.started")
        try:
            log.warn("optimizer.import.onnx.not_implemented", reason="ONNX import C++ backend not fully implemented yet.")
            raise NotImplementedError("ONNX import not fully implemented yet.")
        except Exception as e:
            log.error("optimizer.import.onnx.failed", error=str(e), exc_info=True)
            raise ModelImportError(f"Failed to import ONNX model from '{model_path}': {e}") from e
    
    def add_pass(self, optimization_pass: OptimizationPass):
        """Add an optimization pass to the pipeline."""
        if not isinstance(optimization_pass, OptimizationPass):
            logger.error("optimizer.add_pass.invalid_type", pass_type=type(optimization_pass).__name__)
            raise TypeError("optimization_pass must be an instance of OptimizationPass")
        
        try:
            self._passes.append(optimization_pass)
            self._cpp_optimizer.add_pass(optimization_pass._cpp_pass)
            logger.info("optimizer.add_pass.success", pass_name=optimization_pass.name, num_passes=len(self._passes))
        except Exception as e:
            logger.error("optimizer.add_pass.failed", pass_name=optimization_pass.name, error=str(e), exc_info=True)
            raise OpenOptimizerError(f"Failed to add optimization pass '{optimization_pass.name}': {e}") from e
    
    def optimize(self, graph: ComputationGraph) -> ComputationGraph:
        """Run optimization passes on the computation graph."""
        if not isinstance(graph, ComputationGraph):
            logger.error("optimizer.optimize.invalid_graph_type", graph_type=type(graph).__name__)
            raise TypeError("graph must be an instance of ComputationGraph")

        log = logger.bind(num_passes=len(self._passes), graph_name=graph.name if hasattr(graph, 'name') else 'UnnamedGraph')
        log.info("optimizer.optimize.started")
        if not self._passes:
            log.warn("optimizer.optimize.no_passes", message="No optimization passes added. Graph will not be changed.")
            return graph
        try:
            self._cpp_optimizer.optimize(graph._cpp_graph)
            log.info("optimizer.optimize.success")
            return graph
        except Exception as e:
            log.error("optimizer.optimize.failed", error=str(e), exc_info=True)
            raise OptimizationError(f"Optimization failed: {e}") from e
    
    def generate_code(self, 
                    graph: ComputationGraph,
                    output_path: str,
                    target: str,
                    options: Optional[Dict[str, str]] = None) -> None:
        """Generate optimized code for the target platform."""
        if not isinstance(graph, ComputationGraph):
            logger.error("optimizer.generate_code.invalid_graph_type", graph_type=type(graph).__name__)
            raise TypeError("graph must be an instance of ComputationGraph")
        
        log = logger.bind(target=target, output_path=output_path, graph_name=graph.name if hasattr(graph, 'name') else 'UnnamedGraph')
        log.info("optimizer.generate_code.started", options=options or {})
        
        try:
            self._cpp_optimizer.generate_code(graph._cpp_graph, output_path, target, options or {})
            log.info("optimizer.generate_code.success")
        except Exception as e:
            log.error("optimizer.generate_code.failed", error=str(e), exc_info=True)
            raise CodeGenerationError(f"Code generation for target '{target}' failed: {e}") from e
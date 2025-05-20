import structlog
from abc import ABC, abstractmethod

from openoptimizer.ir.graph import ComputationGraph

logger = structlog.get_logger(__name__)

class OptimizationPass(ABC):
    """Base class for all optimization passes (Python side)."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        # self._cpp_pass should be an instance of the C++ openoptimizer::optimization::OptimizationPass
        # It's the responsibility of derived Python classes that wrap C++ passes to initialize this.
        self._cpp_pass: 'openoptimizer._cpp_extension.OptimizationPass' = None # Type hint for clarity
        logger.debug("python_pass.init", pass_name=self.name, class_name=self.__class__.__name__)
    
    @abstractmethod
    def run(self, graph: ComputationGraph) -> ComputationGraph:
        """Run the optimization pass on the given computation graph.

        Args:
            graph: The Python ComputationGraph object to optimize.

        Returns:
            The optimized (potentially modified) ComputationGraph.
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} pass '{self.name}'>"
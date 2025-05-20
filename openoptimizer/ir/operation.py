"""Python wrapper for the base IR Operation class and concrete operations."""

from typing import List, Any, Dict, Optional, Type, TypeVar
from abc import ABC, abstractmethod
import structlog

logger = structlog.get_logger(__name__)

# Import Python tensor classes
from .tensor.TensorShape import TensorShape
from .tensor.TensorDescriptor import TensorDescriptor
from .tensor.Tensor import Tensor

# Will be imported if available
_cpp_extension = None
try:
    from openoptimizer import _cpp_extension
except ImportError:
    logger.warning("C++ extension module not available, using Python-only implementation")

# Generic type for Operation subclasses
_OpT = TypeVar("_OpT", bound="Operation")

class Operation(ABC):
    """Base Python wrapper for C++ ir::Operation."""

    # Map of operation type names to C++ operation classes (populated if _cpp_extension is available)
    _cpp_op_map: Dict[str, Type] = {}
    
    if _cpp_extension:
        # If C++ extension is available, populate the map
        try:
            _cpp_op_map = {
                "Conv2d": _cpp_extension.Conv2DOp,
                "ReLU": _cpp_extension.ReLUOp,
                "Add": _cpp_extension.AddOp,
                # Add other concrete C++ op types here as they are bound
            }
        except AttributeError as e:
            logger.warning(f"Error setting up C++ operation map: {e}")

    def __init__(self, name: str, type_str: str, cpp_op: Optional = None, **kwargs):
        """Initialize an Operation.
        
        Args:
            name: The name of the operation.
            type_str: The type of the operation (e.g., "Conv2d", "ReLU").
            cpp_op: Optional C++ operation to wrap.
            **kwargs: Additional attributes for the operation.
        """
        self.name = name
        self.type_str = type_str
        self._attributes: Dict[str, Any] = {}
        self._cpp_op = None
        
        if cpp_op:
            self._cpp_op = cpp_op
            # If C++ op exists, try to sync attributes (Python -> C++ is primary for creation for now)
            for key, value in kwargs.items():
                self.set_attribute(key, value) # This will set on C++ op via property
        else:
            # Create corresponding C++ operation if a mapping exists
            cpp_op_class = self._cpp_op_map.get(type_str)
            if cpp_op_class and _cpp_extension:
                # This requires concrete C++ ops to have constructors bindable from Python
                # For example, Conv2DOp has a specific constructor. Others might be simpler.
                # This part needs careful handling based on how C++ ops are constructed.
                # For now, assume a generic way or that kwargs are handled by concrete Python op wrappers.
                try:
                    # This is a simplification. Concrete Python ops should handle C++ instantiation.
                    logger.debug(f"Creating C++ operation of type {type_str}", name=name)
                    # self._cpp_op = cpp_op_class(name, **kwargs) # This won't work for all ops
                    # Defer C++ op creation to specific subclasses or factory methods
                except TypeError as e:
                    logger.warning(f"Could not directly instantiate C++ op {type_str} for {name}: {e}")
                    logger.info("Subclass should handle C++ op creation")
                    self._cpp_op = None # No C++ counterpart if direct instantiation fails
            else:
                if not _cpp_extension:
                    logger.debug(f"No C++ extension available, using Python-only operation", type=type_str, name=name)
                else:
                    logger.debug(f"No C++ mapping for operation type {type_str}", name=name)
                self._cpp_op = None # No C++ counterpart defined for this type
        
        # Store kwargs as Python attributes if not creating C++ op or for Python-only use
        for key, value in kwargs.items():
            self._attributes[key] = value

    @property
    def cpp_op(self):
        """The underlying C++ Operation object, if one exists."""
        return getattr(self, '_cpp_op', None)

    def get_name(self) -> str:
        """Get the name of the operation."""
        if self.cpp_op and hasattr(self.cpp_op, 'getName'):
            return self.cpp_op.getName()
        return self.name

    def get_type(self) -> str:
        """Get the type of the operation."""
        if self.cpp_op and hasattr(self.cpp_op, 'getType'):
            return self.cpp_op.getType()
        return self.type_str

    def set_attribute(self, attr_name: str, value: Any) -> None:
        """Set an attribute on the operation.
        
        Args:
            attr_name: The name of the attribute.
            value: The value of the attribute.
        """
        self._attributes[attr_name] = value
        if self.cpp_op and hasattr(self.cpp_op, 'setAttribute'):
            try:
                # Pybind11's std::any caster will handle basic types
                self.cpp_op.setAttribute(attr_name, value)
            except Exception as e:
                logger.warning(f"Could not set attribute '{attr_name}' on C++ op '{self.name}': {e}")

    def get_attribute(self, attr_name: str) -> Any:
        """Get an attribute from the operation.
        
        Args:
            attr_name: The name of the attribute.
            
        Returns:
            The value of the attribute.
            
        Raises:
            AttributeError: If the attribute is not found.
        """
        if self.cpp_op and hasattr(self.cpp_op, 'hasAttribute') and hasattr(self.cpp_op, 'getAttribute'):
            if self.cpp_op.hasAttribute(attr_name):
                # Attempt to retrieve from C++ op, need type-specific getters from bindings
                # This is simplified; real implementation might need to know the type
                # Or the Python binding for get_attribute on C++ op could return std::any directly
                try:
                    if isinstance(self._attributes.get(attr_name), str): # Guess type based on Python cache
                        return self.cpp_op.getAttribute_string(attr_name)
                    elif isinstance(self._attributes.get(attr_name), int):
                        return self.cpp_op.getAttribute_int(attr_name)
                    elif isinstance(self._attributes.get(attr_name), float):
                        return self.cpp_op.getAttribute_float(attr_name)
                    elif isinstance(self._attributes.get(attr_name), bool):
                        return self.cpp_op.getAttribute_bool(attr_name)
                    elif isinstance(self._attributes.get(attr_name), list): # Could be list of int
                        try: 
                            return self.cpp_op.getAttribute_int_vector(attr_name)
                        except Exception:
                            pass # Fall through or handle specific list types
                    # General getAttribute that returns std::any as a Python object if available
                    if hasattr(self.cpp_op, 'getAttribute'):
                        return self.cpp_op.getAttribute(attr_name)
                except Exception as e:
                    logger.warning(f"Error getting attribute '{attr_name}' from C++ op: {e}")
        
        if attr_name in self._attributes:
            return self._attributes[attr_name]
        
        raise AttributeError(f"Attribute '{attr_name}' not found in operation '{self.name}'")

    def has_attribute(self, attr_name: str) -> bool:
        """Check if the operation has an attribute.
        
        Args:
            attr_name: The name of the attribute.
            
        Returns:
            True if the attribute exists, False otherwise.
        """
        if self.cpp_op and hasattr(self.cpp_op, 'hasAttribute'):
            if self.cpp_op.hasAttribute(attr_name):
                return True
        return attr_name in self._attributes

    @property
    def attributes(self) -> Dict[str, Any]:
        """Get all attributes of the operation.
        
        Returns:
            A dictionary of attribute names to values.
        """
        if self.cpp_op and hasattr(self.cpp_op, 'getAttributes'):
            # Merge C++ attributes with Python ones (Python ones might take precedence or be primary cache)
            cpp_attrs = self.cpp_op.getAttributes() # This returns Dict[str, py::object] via std::any caster
            # Be careful with direct merge if types are not easily convertible or if Python cache is master
            # For simplicity, return python cache if it has values, else C++ ones.
            # A more robust way is needed if attributes can be set from C++ side too.
            return {**cpp_attrs, **self._attributes} # Python attributes override C++ ones on conflict
        return self._attributes.copy()

    @abstractmethod
    def infer_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """Abstract method for shape inference. 
        
        Subclasses must implement this or delegate to C++ op.
        
        Args:
            input_shapes: A list of input tensor shapes.
            
        Returns:
            A list of output tensor shapes.
        """
        if self.cpp_op and hasattr(self.cpp_op, 'inferShapes'):
            cpp_input_shapes = [ts._cpp_shape for ts in input_shapes if hasattr(ts, '_cpp_shape') and ts._cpp_shape]
            cpp_output_shapes = self.cpp_op.inferShapes(cpp_input_shapes)
            # Need to convert TensorShape from C++ to Python
            return [TensorShape.from_cpp(ts) for ts in cpp_output_shapes]
        raise NotImplementedError("infer_shapes must be implemented by concrete Operation subclasses.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.get_name()}', type='{self.get_type()}')"

    def __repr__(self) -> str:
        attrs_str = ", ".join(f"{k}={repr(v)}" for k, v in self._attributes.items())
        return f"{self.__class__.__name__}(name='{self.get_name()}', type='{self.get_type()}', {attrs_str})"

    @classmethod
    def from_cpp(cls: Type[_OpT], cpp_op) -> _OpT:
        """Creates a Python Operation wrapper from a C++ Operation object.
        
        Args:
            cpp_op: The C++ Operation object.
            
        Returns:
            A Python Operation wrapper.
        """
        if cpp_op is None:
            return None
            
        # This is a generic factory. Concrete Python Ops might have specific from_cpp.
        # It assumes constructor `Operation(name, type_str, cpp_op=cpp_op)`
        name = cpp_op.getName() if hasattr(cpp_op, 'getName') else "unknown"
        type_str = cpp_op.getType() if hasattr(cpp_op, 'getType') else "unknown"
        
        py_op = cls(name=name, type_str=type_str, cpp_op=cpp_op)
        
        # Sync attributes from C++ to Python cache if needed (though get_attribute handles it)
        if hasattr(cpp_op, 'getAttributes'):
            for k, v in cpp_op.getAttributes().items():
                py_op._attributes[k] = v # std::any should be converted by pybind11
                
        return py_op


# Example concrete Python operation wrappers 
class Conv2DOp(Operation):
    """Convolution 2D operation."""
    
    def __init__(self, name: str, out_channels: int, kernel_size: List[int],
                 stride: List[int] = [1,1], padding: List[int] = [0,0],
                 dilation: List[int] = [1,1], groups: int = 1, bias: bool = True):
        """Initialize a Conv2DOp.
        
        Args:
            name: The name of the operation.
            out_channels: The number of output channels.
            kernel_size: The kernel size.
            stride: The stride.
            padding: The padding.
            dilation: The dilation.
            groups: The number of groups.
            bias: Whether to use bias.
        """
        # Create C++ Conv2DOp instance if available
        cpp_conv_op = None
        if _cpp_extension and hasattr(_cpp_extension, 'Conv2DOp'):
            try:
                cpp_conv_op = _cpp_extension.Conv2DOp(name, out_channels, kernel_size, stride, 
                                                    padding, dilation, groups, bias)
            except Exception as e:
                logger.warning(f"Could not create C++ Conv2DOp: {e}")
                
        super().__init__(name, "Conv2d", cpp_op=cpp_conv_op, 
                         out_channels=out_channels, kernel_size=kernel_size, 
                         stride=stride, padding=padding, dilation=dilation, 
                         groups=groups, bias=bias) 

    def infer_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """Infer the output shapes given the input shapes.
        
        Args:
            input_shapes: A list of input tensor shapes.
            
        Returns:
            A list of output tensor shapes.
        """
        # Delegate to C++ op's inferShapes if available
        if self.cpp_op and hasattr(self.cpp_op, 'inferShapes'):
            cpp_input_shapes = [ts._cpp_shape for ts in input_shapes if hasattr(ts, '_cpp_shape') and ts._cpp_shape]
            try:
                cpp_output_shapes = self.cpp_op.inferShapes(cpp_input_shapes)
                return [TensorShape.from_cpp(ts) for ts in cpp_output_shapes]
            except Exception as e:
                logger.warning(f"Error in C++ inferShapes: {e}, falling back to Python implementation")
        
        # Fallback Python implementation if C++ op not available or fails
        if not input_shapes or len(input_shapes) < 1:
            raise ValueError("Conv2D requires at least one input shape")
        
        input_shape = input_shapes[0]
        if input_shape.rank != 4:  # N, C, H, W format
            raise ValueError(f"Conv2D expects 4D input shape, got {input_shape.rank}D")
        
        batch_size = input_shape[0].value  # N
        out_channels = self.get_attribute('out_channels')
        kernel_size = self.get_attribute('kernel_size')
        stride = self.get_attribute('stride')
        padding = self.get_attribute('padding')
        dilation = self.get_attribute('dilation')
        
        # Calculate output spatial dimensions
        in_h, in_w = input_shape[2].value, input_shape[3].value
        k_h, k_w = kernel_size
        s_h, s_w = stride
        p_h, p_w = padding
        d_h, d_w = dilation
        
        # Formula: out_dim = floor((in_dim + 2*padding - dilation * (kernel_size - 1) - 1) / stride) + 1
        out_h = (in_h + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_w = (in_w + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1
        
        return [TensorShape([batch_size, out_channels, out_h, out_w])]


class ReLUOp(Operation):
    """ReLU activation operation."""
    
    def __init__(self, name: str):
        """Initialize a ReLUOp.
        
        Args:
            name: The name of the operation.
        """
        # Create C++ ReLUOp instance if available
        cpp_relu_op = None
        if _cpp_extension and hasattr(_cpp_extension, 'ReLUOp'):
            try:
                cpp_relu_op = _cpp_extension.ReLUOp(name)
            except Exception as e:
                logger.warning(f"Could not create C++ ReLUOp: {e}")
                
        super().__init__(name, "ReLU", cpp_op=cpp_relu_op)

    def infer_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """Infer the output shapes given the input shapes.
        
        Args:
            input_shapes: A list of input tensor shapes.
            
        Returns:
            A list of output tensor shapes.
        """
        # Delegate to C++ op's inferShapes if available
        if self.cpp_op and hasattr(self.cpp_op, 'inferShapes'):
            cpp_input_shapes = [ts._cpp_shape for ts in input_shapes if hasattr(ts, '_cpp_shape') and ts._cpp_shape]
            try:
                cpp_output_shapes = self.cpp_op.inferShapes(cpp_input_shapes)
                return [TensorShape.from_cpp(ts) for ts in cpp_output_shapes]
            except Exception as e:
                logger.warning(f"Error in C++ inferShapes: {e}, falling back to Python implementation")
        
        # Fallback Python implementation
        if not input_shapes or len(input_shapes) < 1:
            raise ValueError("ReLU requires at least one input shape")
        
        # ReLU preserves input shape
        return [input_shapes[0]]


class AddOp(Operation):
    """Element-wise addition operation."""
    
    def __init__(self, name: str):
        """Initialize an AddOp.
        
        Args:
            name: The name of the operation.
        """
        # Create C++ AddOp instance if available
        cpp_add_op = None
        if _cpp_extension and hasattr(_cpp_extension, 'AddOp'):
            try:
                cpp_add_op = _cpp_extension.AddOp(name)
            except Exception as e:
                logger.warning(f"Could not create C++ AddOp: {e}")
                
        super().__init__(name, "Add", cpp_op=cpp_add_op)

    def infer_shapes(self, input_shapes: List[TensorShape]) -> List[TensorShape]:
        """Infer the output shapes given the input shapes.
        
        Args:
            input_shapes: A list of input tensor shapes.
            
        Returns:
            A list of output tensor shapes.
        """
        # Delegate to C++ op's inferShapes if available
        if self.cpp_op and hasattr(self.cpp_op, 'inferShapes'):
            cpp_input_shapes = [ts._cpp_shape for ts in input_shapes if hasattr(ts, '_cpp_shape') and ts._cpp_shape]
            try:
                cpp_output_shapes = self.cpp_op.inferShapes(cpp_input_shapes)
                return [TensorShape.from_cpp(ts) for ts in cpp_output_shapes]
            except Exception as e:
                logger.warning(f"Error in C++ inferShapes: {e}, falling back to Python implementation")
        
        # Fallback Python implementation
        if len(input_shapes) != 2:
            raise ValueError(f"Add requires exactly 2 input shapes, got {len(input_shapes)}")
        
        # Simple case: identical shapes
        if input_shapes[0] == input_shapes[1]:
            return [input_shapes[0]]
        
        # TODO: Implement broadcasting logic for different shapes
        logger.warning("Broadcasting not implemented yet, using shape of first input")
        return [input_shapes[0]]


# Export the classes
__all__ = ["Operation", "Conv2DOp", "ReLUOp", "AddOp"] 
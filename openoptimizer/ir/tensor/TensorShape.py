"""TensorShape class for representing tensor dimensions."""

from typing import List, Union, Optional, Iterator, Tuple, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

# Special value for dynamic dimensions
DYNAMIC_DIM = -1


class Dimension:
    """Represents a dimension of a tensor shape.
    
    A dimension can be static (fixed size) or dynamic (size determined at runtime).
    """
    
    def __init__(self, value: int):
        """Initialize a dimension.
        
        Args:
            value: The size of the dimension. Use DYNAMIC_DIM for dynamic dimensions.
        """
        self.value = value
    
    @property
    def is_dynamic(self) -> bool:
        """Check if this dimension is dynamic."""
        return self.value == DYNAMIC_DIM
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Dimension):
            return self.value == other.value
        elif isinstance(other, int):
            return self.value == other
        return NotImplemented
    
    def __int__(self) -> int:
        return self.value
    
    def __str__(self) -> str:
        return "?" if self.is_dynamic else str(self.value)
    
    def __repr__(self) -> str:
        return f"Dimension({self.value})"


class TensorShape:
    """Represents the shape of a tensor.
    
    A shape consists of dimensions, which can be static or dynamic.
    """
    
    def __init__(self, dims: Optional[List[Union[int, Dimension]]] = None):
        """Initialize a TensorShape.
        
        Args:
            dims: A list of dimensions. Each dimension can be an integer or a Dimension object.
        """
        self._dims: List[Dimension] = []
        self._cpp_shape = None  # Reference to C++ TensorShape, if available
        
        if dims is not None:
            for dim in dims:
                if isinstance(dim, Dimension):
                    self._dims.append(dim)
                elif isinstance(dim, int):
                    self._dims.append(Dimension(dim))
                else:
                    raise TypeError(f"Dimension must be int or Dimension, got {type(dim)}")
    
    @property
    def rank(self) -> int:
        """Get the rank (number of dimensions) of the shape."""
        return len(self._dims)
    
    @property
    def is_fully_static(self) -> bool:
        """Check if all dimensions are static."""
        return all(not dim.is_dynamic for dim in self._dims)
    
    @property
    def has_dynamic_dims(self) -> bool:
        """Check if any dimension is dynamic."""
        return any(dim.is_dynamic for dim in self._dims)
    
    @property
    def num_elements(self) -> int:
        """Get the total number of elements in the tensor.
        
        Returns:
            The product of all dimensions if all are static, or -1 if any dimension is dynamic.
        """
        if self.has_dynamic_dims:
            return -1
        
        if not self._dims:
            return 0
        
        result = 1
        for dim in self._dims:
            result *= dim.value
        return result
    
    def __getitem__(self, idx: int) -> Dimension:
        """Get a dimension by index."""
        return self._dims[idx]
    
    def __iter__(self) -> Iterator[Dimension]:
        """Iterate over dimensions."""
        return iter(self._dims)
    
    def __len__(self) -> int:
        """Get the number of dimensions."""
        return len(self._dims)
    
    def __eq__(self, other) -> bool:
        """Check if two shapes are equal."""
        if not isinstance(other, TensorShape):
            return NotImplemented
        
        if len(self._dims) != len(other._dims):
            return False
        
        return all(d1 == d2 for d1, d2 in zip(self._dims, other._dims))
    
    def __str__(self) -> str:
        """Get a string representation of the shape."""
        dims_str = ", ".join(str(dim) for dim in self._dims)
        return f"[{dims_str}]"
    
    def __repr__(self) -> str:
        """Get a debug representation of the shape."""
        dims_repr = ", ".join(repr(dim) for dim in self._dims)
        return f"TensorShape([{dims_repr}])"
    
    def as_list(self) -> List[int]:
        """Get the shape as a list of integers.
        
        Returns:
            A list of dimension values. Dynamic dimensions are represented as DYNAMIC_DIM.
        """
        return [dim.value for dim in self._dims]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            'dims': self.as_list(),
            'rank': self.rank,
            'is_fully_static': self.is_fully_static,
            'has_dynamic_dims': self.has_dynamic_dims,
            'num_elements': self.num_elements,
        }
    
    @staticmethod
    def from_cpp(cpp_shape) -> Optional['TensorShape']:
        """Create a Python TensorShape from a C++ TensorShape.
        
        Args:
            cpp_shape: The C++ TensorShape object.
            
        Returns:
            A new Python TensorShape, or None if the C++ shape is null.
        """
        if cpp_shape is None:
            return None
        
        # Get dimensions from C++ shape
        try:
            # This will depend on the exact C++ binding
            cpp_dims = cpp_shape.get_dims()
            shape = TensorShape([dim for dim in cpp_dims])
            shape._cpp_shape = cpp_shape  # Store reference to C++ shape
            return shape
        except Exception as e:
            logger.error("Failed to convert C++ TensorShape", error=str(e), exc_info=True)
            return None 
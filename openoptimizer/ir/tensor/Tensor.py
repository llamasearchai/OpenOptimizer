"""Tensor class for OpenOptimizer IR."""

from typing import Optional, List, Union, Any, Dict
import numpy as np
import structlog

from .TensorDescriptor import TensorDescriptor
from .TensorShape import TensorShape
from .DataType import DataType

logger = structlog.get_logger(__name__)


class Tensor:
    """Represents a tensor in the IR.
    
    A Tensor consists of a TensorDescriptor (shape, data type, etc.)
    and optional data.
    """
    
    def __init__(self, 
                descriptor: TensorDescriptor,
                data: Optional[np.ndarray] = None,
                name: str = ""):
        """Initialize a Tensor.
        
        Args:
            descriptor: The tensor descriptor.
            data: Optional data for the tensor.
            name: Optional name for the tensor.
        """
        self.descriptor = descriptor
        self.data = data
        
        # If a name is provided, use it (overrides the name in the descriptor)
        if name:
            self.descriptor.name = name
        
        self._cpp_tensor = None  # Reference to C++ Tensor, if available
        
        # Validate data against descriptor if provided
        if data is not None:
            self._validate_data()
    
    @property
    def name(self) -> str:
        """Get the name of the tensor."""
        return self.descriptor.name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set the name of the tensor."""
        self.descriptor.name = value
    
    @property
    def shape(self) -> TensorShape:
        """Get the shape of the tensor."""
        return self.descriptor.shape
    
    @property
    def data_type(self) -> DataType:
        """Get the data type of the tensor."""
        return self.descriptor.data_type
    
    @property
    def is_quantized(self) -> bool:
        """Check if the tensor is quantized."""
        return self.descriptor.is_quantized
    
    @property
    def has_data(self) -> bool:
        """Check if the tensor has data."""
        return self.data is not None
    
    def _validate_data(self) -> None:
        """Validate that the data matches the descriptor.
        
        Raises:
            ValueError: If the data shape doesn't match the descriptor shape.
            TypeError: If the data type doesn't match the descriptor data type.
        """
        if self.data is None:
            return
        
        # Check shape
        expected_shape = self.descriptor.shape.as_list()
        # Replace dynamic dimensions with actual dimensions from data
        for i, dim in enumerate(expected_shape):
            if dim == -1:  # Dynamic dimension
                if i < len(self.data.shape):
                    expected_shape[i] = self.data.shape[i]
        
        if list(self.data.shape) != expected_shape:
            logger.error("Data shape mismatch", 
                       expected=expected_shape, 
                       actual=self.data.shape)
            raise ValueError(f"Data shape {self.data.shape} doesn't match "
                           f"descriptor shape {expected_shape}")
        
        # TODO: Check data type (needs mapping between numpy dtypes and our DataType enum)
    
    def __str__(self) -> str:
        data_str = f", data={self.data.shape}" if self.has_data else ", no data"
        return f"<Tensor {self.name}: {self.shape}, {self.data_type}{data_str}>"
    
    def __repr__(self) -> str:
        data_repr = f", data=array(...)" if self.has_data else ", data=None"
        return f"Tensor({repr(self.descriptor)}{data_repr})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        result = {
            'descriptor': self.descriptor.to_dict(),
            'name': self.name,
            'has_data': self.has_data,
        }
        
        # Note: We don't include the actual data in the dict to avoid very large dictionaries
        if self.has_data:
            result['data_shape'] = list(self.data.shape)
        
        return result
    
    @staticmethod
    def from_numpy(array: np.ndarray, name: str = "") -> 'Tensor':
        """Create a Tensor from a numpy array.
        
        Args:
            array: The numpy array.
            name: Optional name for the tensor.
            
        Returns:
            A new Tensor with the given array as data.
        """
        # Map numpy dtype to DataType
        dtype_map = {
            np.int8: DataType.INT8,
            np.int16: DataType.INT16,
            np.int32: DataType.INT32,
            np.int64: DataType.INT64,
            np.uint8: DataType.UINT8,
            np.uint16: DataType.UINT16,
            np.uint32: DataType.UINT32,
            np.uint64: DataType.UINT64,
            np.float16: DataType.FLOAT16,
            np.float32: DataType.FLOAT32,
            np.float64: DataType.FLOAT64,
            np.bool_: DataType.BOOL,
            np.complex64: DataType.COMPLEX64,
            np.complex128: DataType.COMPLEX128,
        }
        
        data_type = dtype_map.get(array.dtype.type, DataType.FLOAT32)  # Default to FLOAT32
        
        # Create a TensorDescriptor
        shape = TensorShape(list(array.shape))
        descriptor = TensorDescriptor(shape, data_type, name)
        
        return Tensor(descriptor, array, name)
    
    @staticmethod
    def from_cpp(cpp_tensor) -> Optional['Tensor']:
        """Create a Python Tensor from a C++ Tensor.
        
        Args:
            cpp_tensor: The C++ Tensor object.
            
        Returns:
            A new Python Tensor, or None if the C++ tensor is null.
        """
        if cpp_tensor is None:
            return None
        
        try:
            # Assuming C++ binding exposes these attributes
            cpp_desc = cpp_tensor.descriptor
            descriptor = TensorDescriptor.from_cpp(cpp_desc)
            
            # Try to get data if available
            data = None
            if hasattr(cpp_tensor, 'data') and cpp_tensor.data is not None:
                # This depends on how data is exposed from C++
                # Might need special handling for different data types
                data = np.array(cpp_tensor.data)
            
            tensor = Tensor(descriptor, data)
            tensor._cpp_tensor = cpp_tensor  # Store reference to C++ tensor
            return tensor
        except Exception as e:
            logger.error("Failed to convert C++ Tensor", error=str(e), exc_info=True)
            return None 
"""TensorDescriptor class for describing tensor properties."""

from typing import Optional, Dict, Any, List, Union
import structlog

from .DataType import DataType
from .TensorShape import TensorShape, Dimension

logger = structlog.get_logger(__name__)


class QuantizationParams:
    """Parameters for quantized tensors."""
    
    def __init__(self, 
                scale: float = 1.0, 
                zero_point: int = 0,
                quant_min: int = -128,
                quant_max: int = 127):
        """Initialize quantization parameters.
        
        Args:
            scale: Scale factor for quantization.
            zero_point: Zero point for quantization.
            quant_min: Minimum quantized value.
            quant_max: Maximum quantized value.
        """
        self.scale = scale
        self.zero_point = zero_point
        self.quant_min = quant_min
        self.quant_max = quant_max
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, QuantizationParams):
            return NotImplemented
        
        return (self.scale == other.scale and 
                self.zero_point == other.zero_point and
                self.quant_min == other.quant_min and
                self.quant_max == other.quant_max)
    
    def __repr__(self) -> str:
        return (f"QuantizationParams(scale={self.scale}, zero_point={self.zero_point}, "
                f"quant_min={self.quant_min}, quant_max={self.quant_max})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            'scale': self.scale,
            'zero_point': self.zero_point,
            'quant_min': self.quant_min,
            'quant_max': self.quant_max,
        }
    
    @staticmethod
    def from_cpp(cpp_params) -> Optional['QuantizationParams']:
        """Create a Python QuantizationParams from a C++ QuantizationParams.
        
        Args:
            cpp_params: The C++ QuantizationParams object.
            
        Returns:
            A new Python QuantizationParams, or None if the C++ params is null.
        """
        if cpp_params is None:
            return None
        
        try:
            # Assuming C++ binding exposes these attributes
            return QuantizationParams(
                scale=cpp_params.scale,
                zero_point=cpp_params.zero_point,
                quant_min=cpp_params.quant_min,
                quant_max=cpp_params.quant_max
            )
        except Exception as e:
            logger.error("Failed to convert C++ QuantizationParams", error=str(e), exc_info=True)
            return None


class TensorDescriptor:
    """Describes the properties of a tensor.
    
    A tensor descriptor includes shape, data type, and optional quantization parameters.
    """
    
    def __init__(self, 
                shape: Union[TensorShape, List[int]],
                data_type: DataType,
                name: str = "", 
                quant_params: Optional[QuantizationParams] = None):
        """Initialize a TensorDescriptor.
        
        Args:
            shape: The shape of the tensor.
            data_type: The data type of the tensor.
            name: An optional name for the tensor.
            quant_params: Quantization parameters for quantized tensors.
        """
        if isinstance(shape, list):
            self.shape = TensorShape(shape)
        else:
            self.shape = shape
        
        self.data_type = data_type
        self.name = name
        self.quant_params = quant_params
        self._cpp_desc = None  # Reference to C++ TensorDescriptor, if available
        
        # Validate: if data_type is quantized, quant_params should be provided
        if data_type.is_quantized and quant_params is None:
            logger.warning("Quantized data type without quantization parameters", data_type=data_type)
    
    @property
    def is_quantized(self) -> bool:
        """Check if this tensor is quantized."""
        return self.data_type.is_quantized and self.quant_params is not None
    
    @property
    def byte_size(self) -> int:
        """Get the size of this tensor in bytes.
        
        Returns:
            The size in bytes if the shape is fully static, or -1 if the shape has dynamic dimensions.
        """
        num_elements = self.shape.num_elements
        if num_elements < 0:  # Dynamic shape
            return -1
        
        return num_elements * self.data_type.byte_size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorDescriptor):
            return NotImplemented
        
        return (self.shape == other.shape and 
                self.data_type == other.data_type and
                self.name == other.name and
                self.quant_params == other.quant_params)
    
    def __str__(self) -> str:
        quant_str = " [quantized]" if self.is_quantized else ""
        return f"<TensorDescriptor {self.name}: shape={self.shape}, type={self.data_type}{quant_str}>"
    
    def __repr__(self) -> str:
        return (f"TensorDescriptor(shape={repr(self.shape)}, data_type={repr(self.data_type)}, "
                f"name='{self.name}', quant_params={repr(self.quant_params)})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        result = {
            'shape': self.shape.to_dict(),
            'data_type': self.data_type.to_dict(),
            'name': self.name,
            'is_quantized': self.is_quantized,
            'byte_size': self.byte_size,
        }
        
        if self.quant_params:
            result['quant_params'] = self.quant_params.to_dict()
            
        return result
    
    @staticmethod
    def from_cpp(cpp_desc) -> Optional['TensorDescriptor']:
        """Create a Python TensorDescriptor from a C++ TensorDescriptor.
        
        Args:
            cpp_desc: The C++ TensorDescriptor object.
            
        Returns:
            A new Python TensorDescriptor, or None if the C++ descriptor is null.
        """
        if cpp_desc is None:
            return None
        
        try:
            # Assuming C++ binding exposes these attributes
            cpp_shape = cpp_desc.shape
            cpp_data_type = cpp_desc.data_type
            name = cpp_desc.name
            cpp_quant_params = getattr(cpp_desc, 'quant_params', None)
            
            shape = TensorShape.from_cpp(cpp_shape)
            data_type = DataType.from_cpp(cpp_data_type)
            quant_params = QuantizationParams.from_cpp(cpp_quant_params) if cpp_quant_params else None
            
            desc = TensorDescriptor(shape, data_type, name, quant_params)
            desc._cpp_desc = cpp_desc  # Store reference to C++ descriptor
            return desc
        except Exception as e:
            logger.error("Failed to convert C++ TensorDescriptor", error=str(e), exc_info=True)
            return None 
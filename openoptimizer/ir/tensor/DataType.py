"""Data type definitions for tensors."""

from enum import Enum, auto
from typing import Optional, Union, Dict, Any


class DataType(Enum):
    """Enumeration of supported data types for tensors."""
    
    # Integer types
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    
    # Unsigned integer types
    UINT8 = auto()
    UINT16 = auto()
    UINT32 = auto()
    UINT64 = auto()
    
    # Floating point types
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()
    
    # Boolean type
    BOOL = auto()
    
    # Quantized types
    QINT8 = auto()   # Quantized INT8
    QINT16 = auto()  # Quantized INT16
    QINT32 = auto()  # Quantized INT32
    
    # BF16 (Brain Floating Point)
    BF16 = auto()
    
    # Complex types
    COMPLEX64 = auto()
    COMPLEX128 = auto()
    
    @property
    def byte_size(self) -> int:
        """Get the size of this data type in bytes."""
        size_map = {
            DataType.INT8: 1,
            DataType.UINT8: 1,
            DataType.INT16: 2,
            DataType.UINT16: 2,
            DataType.INT32: 4,
            DataType.UINT32: 4,
            DataType.INT64: 8,
            DataType.UINT64: 8,
            DataType.FLOAT16: 2,
            DataType.FLOAT32: 4,
            DataType.FLOAT64: 8,
            DataType.BOOL: 1,
            DataType.QINT8: 1,
            DataType.QINT16: 2,
            DataType.QINT32: 4,
            DataType.BF16: 2,
            DataType.COMPLEX64: 8,
            DataType.COMPLEX128: 16,
        }
        return size_map.get(self, 0)
    
    @property
    def is_floating_point(self) -> bool:
        """Check if this data type is a floating point type."""
        return self in (DataType.FLOAT16, DataType.FLOAT32, DataType.FLOAT64, DataType.BF16)
    
    @property
    def is_integer(self) -> bool:
        """Check if this data type is an integer type."""
        return self in (
            DataType.INT8, DataType.INT16, DataType.INT32, DataType.INT64,
            DataType.UINT8, DataType.UINT16, DataType.UINT32, DataType.UINT64
        )
    
    @property
    def is_quantized(self) -> bool:
        """Check if this data type is a quantized type."""
        return self in (DataType.QINT8, DataType.QINT16, DataType.QINT32)
    
    @property
    def is_complex(self) -> bool:
        """Check if this data type is a complex type."""
        return self in (DataType.COMPLEX64, DataType.COMPLEX128)
    
    @staticmethod
    def from_string(type_str: str) -> 'DataType':
        """Convert a string to a DataType.
        
        Args:
            type_str: The string representation of the data type.
            
        Returns:
            The corresponding DataType enum value.
            
        Raises:
            ValueError: If the string does not correspond to a valid DataType.
        """
        try:
            return DataType[type_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown data type: {type_str}")
    
    @staticmethod
    def from_cpp(cpp_data_type) -> Optional['DataType']:
        """Convert a C++ DataType to a Python DataType.
        
        Args:
            cpp_data_type: The C++ DataType object.
            
        Returns:
            The corresponding Python DataType enum value, or None if no mapping exists.
        """
        if cpp_data_type is None:
            return None
            
        # This mapping should match the C++ DataType enum values
        cpp_to_py_map = {
            0: DataType.INT8,
            1: DataType.INT16,
            2: DataType.INT32,
            3: DataType.INT64,
            4: DataType.UINT8,
            5: DataType.UINT16,
            6: DataType.UINT32,
            7: DataType.UINT64,
            8: DataType.FLOAT16,
            9: DataType.FLOAT32,
            10: DataType.FLOAT64,
            11: DataType.BOOL,
            12: DataType.QINT8,
            13: DataType.QINT16,
            14: DataType.QINT32,
            15: DataType.BF16,
            16: DataType.COMPLEX64,
            17: DataType.COMPLEX128,
        }
        
        # Get the enum index or name, depending on how the C++ binding works
        cpp_value = getattr(cpp_data_type, 'value', cpp_data_type)
        return cpp_to_py_map.get(cpp_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            'name': self.name,
            'value': self.value,
            'byte_size': self.byte_size,
            'is_floating_point': self.is_floating_point,
            'is_integer': self.is_integer,
            'is_quantized': self.is_quantized,
            'is_complex': self.is_complex,
        }
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"DataType.{self.name}" 
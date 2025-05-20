"""Python wrappers for IR Tensor components."""

from typing import List, Optional, Any, Union, Sequence, Dict
from openoptimizer import _cpp_extension # Import the C++ bindings

# Re-export DataType enum and its utility functions
DataType = _cpp_extension.DataType
getDataTypeSizeBytes = _cpp_extension.getDataTypeSizeBytes
dataTypeToString = _cpp_extension.dataTypeToString
stringToDataType = _cpp_extension.stringToDataType

# Dimension type (std::optional<int64_t> in C++, int | None in Python)
Dimension = Optional[int]
DYNAMIC_DIM = None # Python equivalent of std::nullopt for dimensions
# The C++ DYNAMIC_DIM is also exposed via _cpp_extension.DYNAMIC_DIM, but Python None is more idiomatic

class TensorShape:
    """Python wrapper for C++ ir::TensorShape."""
    def __init__(self, dims: Optional[Sequence[Dimension]] = None):
        if dims is None:
            self._cpp_shape = _cpp_extension.TensorShape()
        else:
            # Convert Python None to C++ std::nullopt (which pybind11 handles for std::optional)
            cpp_dims: List[Optional[int]] = [] # Pybind11 will convert List[Optional[int]] to std::vector<std::optional<int64_t>>
            for d in dims:
                cpp_dims.append(d)
            self._cpp_shape = _cpp_extension.TensorShape(cpp_dims)

    @property
    def rank(self) -> int:
        return self._cpp_shape.rank

    @property
    def is_scalar(self) -> bool:
        return self._cpp_shape.is_scalar

    @property
    def has_dynamic_dimensions(self) -> bool:
        return self._cpp_shape.has_dynamic_dimensions

    @property
    def is_fully_static(self) -> bool:
        return self._cpp_shape.is_fully_static

    @property
    def dimensions(self) -> List[Dimension]:
        # Convert C++ std::vector<std::optional<int64_t>> back to List[Optional[int]]
        return [d for d in self._cpp_shape.dimensions] # pybind11 handles optional<T> to T | None

    def get_dimension(self, index: int) -> Dimension:
        return self._cpp_shape.get_dimension(index)

    def set_dimension(self, index: int, dim: Dimension) -> None:
        self._cpp_shape.set_dimension(index, dim)

    def append_dimension(self, dim: Dimension) -> None:
        self._cpp_shape.append_dimension(dim)

    def prepend_dimension(self, dim: Dimension) -> None:
        self._cpp_shape.prepend_dimension(dim)

    @property
    def num_elements(self) -> Optional[int]:
        return self._cpp_shape.num_elements

    def to_string(self) -> str:
        return self._cpp_shape.to_string()

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"<TensorShape '{self.to_string()}'>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TensorShape):
            return NotImplemented
        return self._cpp_shape == other._cpp_shape

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)
    
    @classmethod
    def from_cpp(cls, cpp_shape: _cpp_extension.TensorShape) -> 'TensorShape':
        instance = cls()
        instance._cpp_shape = cpp_shape
        return instance

class TensorDescriptor:
    """Python wrapper for C++ ir::TensorDescriptor."""
    def __init__(self, shape: TensorShape, dtype: DataType, name: str = "", 
                 quantization_params: Optional[Any] = None):
        self._cpp_tensor_desc = _cpp_extension.TensorDescriptor(shape._cpp_shape, dtype, name)
        if quantization_params is not None:
            self.quantization_params = quantization_params # Uses the property setter

    @property
    def shape(self) -> TensorShape:
        return TensorShape.from_cpp(self._cpp_tensor_desc.shape)

    @property
    def dtype(self) -> DataType:
        return self._cpp_tensor_desc.dtype

    @property
    def name(self) -> str:
        return self._cpp_tensor_desc.name

    @name.setter
    def name(self, value: str) -> None:
        self._cpp_tensor_desc.name = value

    @property
    def has_data(self) -> bool:
        return self._cpp_tensor_desc.has_data

    # Data pointer access is intentionally omitted from Python wrapper for now
    # for simplicity and to avoid direct memory management issues from Python.
    # Data would typically be handled via framework-specific tensors (NumPy, PyTorch, TF).

    @property
    def size_bytes(self) -> Optional[int]:
        return self._cpp_tensor_desc.size_bytes

    @property
    def quantization_params(self) -> Any:
        return self._cpp_tensor_desc.quantization_params

    @quantization_params.setter
    def quantization_params(self, params: Any) -> None:
        self._cpp_tensor_desc.quantization_params = params

    @property
    def has_quantization_params(self) -> bool:
        return self._cpp_tensor_desc.has_quantization_params

    def to_string(self) -> str:
        return self._cpp_tensor_desc.to_string()

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"<TensorDescriptor '{self.to_string()}'>"
    
    @classmethod
    def from_cpp(cls, cpp_tensor_desc: _cpp_extension.TensorDescriptor) -> 'TensorDescriptor':
        # This is a bit circular if constructor needs Python TensorShape.
        # Simpler if C++ constructor took C++ TensorShape directly.
        # Assuming _cpp_tensor_desc.shape returns a C++ TensorShape.
        py_shape = TensorShape.from_cpp(cpp_tensor_desc.shape)
        instance = cls(py_shape, cpp_tensor_desc.dtype, cpp_tensor_desc.name)
        instance._cpp_tensor_desc = cpp_tensor_desc # Ensure original C++ object is kept
        # Copy quantization params if any
        if cpp_tensor_desc.has_quantization_params:
            instance.quantization_params = cpp_tensor_desc.quantization_params
        return instance

# Alias Tensor to TensorDescriptor for convenience, matching C++ using statement
Tensor = TensorDescriptor

__all__ = [
    "DataType", 
    "getDataTypeSizeBytes", 
    "dataTypeToString", 
    "stringToDataType",
    "Dimension", 
    "DYNAMIC_DIM", 
    "TensorShape", 
    "TensorDescriptor", 
    "Tensor"
] 
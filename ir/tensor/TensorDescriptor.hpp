#pragma once

#include "ir/tensor/TensorShape.hpp"
#include "ir/tensor/DataType.hpp"
#include <string>
#include <memory> // For std::shared_ptr if it holds data buffer
#include <any>    // For quantization parameters or other metadata

namespace openoptimizer {
namespace ir {

// Describes the properties of a tensor (shape, data type, etc.)
// This can be used to represent tensors in the IR graph before actual
// memory allocation, or it can be extended to be a full Tensor class.
class TensorDescriptor {
public:
    TensorDescriptor(TensorShape shape, DataType dtype, std::string name = "")
        : shape_(std::move(shape)), dtype_(dtype), name_(std::move(name)), data_ptr_(nullptr) {}

    // Constructor for a tensor that might also hold data (acting as a full Tensor)
    // Data management (allocation/deallocation) is complex and depends on use case (e.g., external, managed)
    TensorDescriptor(TensorShape shape, DataType dtype, std::shared_ptr<void> data, std::string name = "")
        : shape_(std::move(shape)), dtype_(dtype), name_(std::move(name)), data_ptr_(std::move(data)) {}

    const TensorShape& getShape() const { return shape_; }
    DataType getDataType() const { return dtype_; }
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    // Check if this descriptor also has associated data buffer
    bool hasData() const { return data_ptr_ != nullptr; }
    std::shared_ptr<void> getData() const { return data_ptr_; } // Returns raw data pointer
    
    template<typename T>
    std::shared_ptr<T> getDataAs() const {
        if (!data_ptr_) return nullptr;
        // This assumes the stored void* was originally a T*
        // A more robust system might store type_index or use a variant for data_ptr_
        return std::static_pointer_cast<T>(data_ptr_); 
    }

    void setData(std::shared_ptr<void> data) { data_ptr_ = std::move(data); }

    // Size in bytes, if shape is static
    std::optional<size_t> getSizeBytes() const;

    std::string toString() const;

    // Quantization parameters (example)
    // These could be std::any or specific structs/classes.
    void setQuantizationParams(const std::any& params) { quantization_params_ = params; }
    std::any getQuantizationParams() const { return quantization_params_; }
    bool hasQuantizationParams() const { return quantization_params_.has_value(); }

private:
    TensorShape shape_;
    DataType dtype_;
    std::string name_; // Optional name for the tensor/value
    std::shared_ptr<void> data_ptr_; // Optional pointer to raw data buffer
    std::optional<std::any> quantization_params_; // Example of additional metadata
};


inline std::optional<size_t> TensorDescriptor::getSizeBytes() const {
    auto num_elements = shape_.getNumElements();
    if (!num_elements.has_value()) {
        return std::nullopt; // Dynamic shape, size unknown
    }
    size_t element_size_bytes = getDataTypeSizeBytes(dtype_);
    if (element_size_bytes == 0 && dtype_ != DataType::UNKNOWN) { // UNKNOWN might legitimately be 0 size if not an error
        // This case (known type but 0 size) should ideally not happen if getDataTypeSizeBytes is robust.
        // Potentially log a warning or throw if a known type has 0 size.
    }
    return num_elements.value() * element_size_bytes;
}

inline std::string TensorDescriptor::toString() const {
    std::string s = name_.empty() ? "Tensor" : name_;
    s += "(Shape: " + shape_.toString();
    s += ", DType: " + dataTypeToString(dtype_);
    if (hasData()) {
        s += ", HasData: Yes";
    }
    if (quantization_params_.has_value()) {
        s += ", HasQuantParams: Yes";
    }
    s += ")";
    return s;
}

// For convenience, if ir/tensor/Tensor.hpp was meant to be this file.
using Tensor = TensorDescriptor;


} // namespace ir
} // namespace openoptimizer 
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <any>
#include "ir/tensor/TensorShape.hpp" // Defines TensorShape, to be created next
#include "ir/tensor/DataType.hpp"    // Defines DataType, to be created next

// Forward declaration to avoid circular dependency with TensorDescriptor if it holds full Tensor info
// namespace openoptimizer { namespace ir { class TensorDescriptor; } }

namespace openoptimizer {
namespace ir {

// Abstract base class for all operations
class Operation {
public:
    explicit Operation(std::string name, std::string type)
        : name_(std::move(name)), type_(std::move(type)) {}
    
    virtual ~Operation() = default;

    const std::string& getName() const { return name_; }
    const std::string& getType() const { return type_; } // e.g., "Conv2D", "ReLU", "Add"

    // Attributes of the operation (e.g., stride, padding for Conv)
    void setAttribute(const std::string& attr_name, const std::any& value) {
        attributes_[attr_name] = value;
    }

    template<typename T>
    T getAttribute(const std::string& attr_name) const {
        auto it = attributes_.find(attr_name);
        if (it != attributes_.end()) {
            try {
                return std::any_cast<T>(it->second);
            } catch (const std::bad_any_cast& e) {
                // spdlog::error("Bad any_cast for attribute '{}' in op '{}': {}", attr_name, name_, e.what());
                throw;
            }
        }
        // spdlog::error("Attribute '{}' not found in op '{}'", attr_name, name_);
        throw std::out_of_range("Attribute not found: " + attr_name);
    }

    bool hasAttribute(const std::string& attr_name) const {
        return attributes_.count(attr_name);
    }

    const std::unordered_map<std::string, std::any>& getAttributes() const {
        return attributes_;
    }

    // Pure virtual function for shape inference
    // Takes a vector of input shapes and returns a vector of output shapes.
    // This is a simplified version. A more complete one might take TensorDescriptors
    // which include both shape and data type.
    virtual std::vector<TensorShape> inferShapes(const std::vector<TensorShape>& input_shapes) const = 0;

    // Optional: method to validate attributes or inputs
    // virtual bool validate() const = 0;

    // Optional: method to estimate cost or performance characteristics
    // virtual Cost getCostEstimate(const std::vector<TensorDescriptor>& inputs) const = 0;

protected:
    std::string name_; // Instance name of the operation, can be same as node name
    std::string type_; // Type of the operation (e.g., "Conv2D")
    std::unordered_map<std::string, std::any> attributes_;
};

} // namespace ir
} // namespace openoptimizer 
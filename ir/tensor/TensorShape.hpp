#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <sstream>
#include <numeric> // For std::accumulate
#include <optional>
#include <stdexcept> // For std::out_of_range

namespace openoptimizer {
namespace ir {

// Represents a single dimension, which can be static or dynamic
using Dimension = std::optional<int64_t>; // std::nullopt for dynamic, value for static
const Dimension DYNAMIC_DIM = std::nullopt;

class TensorShape {
public:
    // Constructors
    TensorShape() = default; // Creates a scalar (0-rank) or uninitialized shape
    explicit TensorShape(std::vector<Dimension> dims) : dimensions_(std::move(dims)) {}
    
    // Variadic template constructor for convenience e.g. TensorShape(1, DYNAMIC_DIM, 224, 224)
    template <typename... Args, 
              typename = std::enable_if_t<std::conjunction_v<std::is_convertible<Args, Dimension>...>>>
    TensorShape(Args... dims) : dimensions_{static_cast<Dimension>(dims)...} {}

    // Accessors
    size_t getRank() const { return dimensions_.size(); }
    bool isScalar() const { return dimensions_.empty(); }
    bool hasDynamicDimensions() const;
    bool isFullyStatic() const { return !hasDynamicDimensions(); }

    const std::vector<Dimension>& getDimensions() const { return dimensions_; }
    Dimension getDimension(size_t index) const;
    void setDimension(size_t index, Dimension dim);
    void appendDimension(Dimension dim) { dimensions_.push_back(dim); }
    void prependDimension(Dimension dim) { dimensions_.insert(dimensions_.begin(), dim); }

    // Calculate total number of elements if shape is static. Returns std::nullopt if dynamic.
    std::optional<int64_t> getNumElements() const;

    // String representation (e.g., "[1, ?, 224, 224]")
    std::string toString() const;

    // Comparison operators
    bool operator==(const TensorShape& other) const { return dimensions_ == other.dimensions_; }
    bool operator!=(const TensorShape& other) const { return !(*this == other); }

private:
    std::vector<Dimension> dimensions_;
};

// Implementation of methods that are a bit longer

inline bool TensorShape::hasDynamicDimensions() const {
    for (const auto& dim : dimensions_) {
        if (!dim.has_value()) {
            return true;
        }
    }
    return false;
}

inline Dimension TensorShape::getDimension(size_t index) const {
    if (index >= dimensions_.size()) {
        throw std::out_of_range("TensorShape::getDimension: index out of range.");
    }
    return dimensions_[index];
}

inline void TensorShape::setDimension(size_t index, Dimension dim) {
    if (index >= dimensions_.size()) {
        // Option 1: throw
        throw std::out_of_range("TensorShape::setDimension: index out of range. Use appendDimension if needed.");
        // Option 2: resize (might be unexpected if not careful)
        // dimensions_.resize(index + 1);
    }
    dimensions_[index] = dim;
}

inline std::optional<int64_t> TensorShape::getNumElements() const {
    if (hasDynamicDimensions()) {
        return std::nullopt;
    }
    if (dimensions_.empty()) { // Scalar
        return 1;
    }
    int64_t num_elements = 1;
    for (const auto& dim_opt : dimensions_) {
        // Already checked for dynamic, so dim_opt should have a value
        num_elements *= dim_opt.value();
    }
    return num_elements;
}

inline std::string TensorShape::toString() const {
    if (isScalar() && !hasDynamicDimensions()) { // Check for unranked (empty) vs scalar with 0 rank but known shape
        if (dimensions_.empty() && getRank() == 0) return "Scalar[]"; // or just "Scalar"
    }
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < dimensions_.size(); ++i) {
        if (dimensions_[i].has_value()) {
            ss << dimensions_[i].value();
        } else {
            ss << "?"; // Or "dyn", or "-1"
        }
        if (i < dimensions_.size() - 1) {
            ss << ", ";
        }
    }
    ss << "]";
    return ss.str();
}

} // namespace ir
} // namespace openoptimizer 
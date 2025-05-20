#pragma once

#include <string>
#include <cstdint>
#include <stdexcept> // For std::runtime_error

namespace openoptimizer {
namespace ir {

enum class DataType {
    FLOAT32,
    FLOAT16,
    INT32,
    INT16,
    INT8,
    UINT8,
    BOOL,
    // Add other types as needed, e.g., BFLOAT16, FLOAT64, UINT32, etc.
    UNKNOWN
};

// Utility function to get the size of a DataType in bytes
inline size_t getDataTypeSizeBytes(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return 4;
        case DataType::FLOAT16: return 2;
        case DataType::INT32:   return 4;
        case DataType::INT16:   return 2;
        case DataType::INT8:    return 1;
        case DataType::UINT8:   return 1;
        case DataType::BOOL:    return 1; // Often packed, but standalone bool can be 1 byte
        case DataType::UNKNOWN:
        default:
            // Or throw std::runtime_error("Cannot get size of UNKNOWN DataType");
            return 0; 
    }
}

// Utility function to convert DataType to string
inline std::string dataTypeToString(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32: return "float32";
        case DataType::FLOAT16: return "float16";
        case DataType::INT32:   return "int32";
        case DataType::INT16:   return "int16";
        case DataType::INT8:    return "int8";
        case DataType::UINT8:   return "uint8";
        case DataType::BOOL:    return "bool";
        case DataType::UNKNOWN: return "unknown";
        default:                return "invalid_datatype";
    }
}

// Utility function to convert string to DataType
inline DataType stringToDataType(const std::string& s) {
    if (s == "float32") return DataType::FLOAT32;
    if (s == "float16") return DataType::FLOAT16;
    if (s == "int32")   return DataType::INT32;
    if (s == "int16")   return DataType::INT16;
    if (s == "int8")    return DataType::INT8;
    if (s == "uint8")   return DataType::UINT8;
    if (s == "bool")    return DataType::BOOL;
    return DataType::UNKNOWN;
}

} // namespace ir
} // namespace openoptimizer 
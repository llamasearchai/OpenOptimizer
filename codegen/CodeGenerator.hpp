#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <any> // For more flexible options in the future, for now string-to-string

#include "ir/graph/ComputationGraph.hpp"

namespace openoptimizer {
namespace codegen {

// Using string for target names for more flexibility and user-friendliness externally
// The TargetType enum can remain for very specific internal optimized paths or legacy.
using TargetName = std::string;

// Example predefined target names (can be extended)
const TargetName CPU_TARGET = "cpu";
const TargetName GPU_TARGET = "gpu"; // Generic GPU, could be CUDA, ROCm, Metal etc.
const TargetName CUDA_TARGET = "cuda";
const TargetName METAL_TARGET = "metal";
const TargetName EDGE_GENERIC_TARGET = "edge";
// ... add more specific edge targets like 'hexagon_dsp', 'arm_ethos_u' etc.

// Options map for code generation
using CodeGenOptions = std::unordered_map<std::string, std::string>; 
// Example options: {"gpu_arch": "sm_86"}, {"quantization_mode": "int8"}

class TargetBackend {
public:
    virtual ~TargetBackend() = default;
    virtual void generate(std::shared_ptr<ir::ComputationGraph> graph, 
                        const std::string& outputPath,
                        const CodeGenOptions& options) = 0;
    virtual TargetName getName() const = 0; // Backend identifies itself with a TargetName
};

class CodeGenerator {
public:
    CodeGenerator();
    ~CodeGenerator();
    
    // Generate code using a registered backend for the given target name
    void generate(std::shared_ptr<ir::ComputationGraph> graph,
                 const std::string& outputPath,
                 const TargetName& target, // Use string target name
                 const CodeGenOptions& options = {});
    
    // Register a backend implementation for a specific target name
    void registerBackend(const TargetName& name, std::shared_ptr<TargetBackend> backend);

    // List available target names for which backends are registered
    std::vector<TargetName> listAvailableTargets() const;
    
private:
    // Internal enum TargetType could be deprecated or used for specific, known types if needed
    // For now, let's simplify and use string-based mapping directly.
    enum class InternalTargetType { // Kept for example, but string map is primary
        CPU = 0,
        GPU = 1,
        Edge = 2
    };
    // Helper to convert string to enum if ever needed, or to validate known target strings
    // InternalTargetType targetFromString(const TargetName& name);

    std::unordered_map<TargetName, std::shared_ptr<TargetBackend>> backends_;
    // If a mapping from old TargetType enum to TargetName is needed for compatibility:
    // std::unordered_map<InternalTargetType, TargetName> legacy_enum_to_name_map_;
};

} // namespace codegen
} // namespace openoptimizer
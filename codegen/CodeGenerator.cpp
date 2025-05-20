#include "codegen/CodeGenerator.hpp"
#include "codegen/cpu/CPUBackend.hpp"
#include "codegen/gpu/GPUBackend.hpp"
#include "codegen/edge/EdgeBackend.hpp"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <algorithm> // For std::transform in listAvailableTargets

namespace openoptimizer {
namespace codegen {

CodeGenerator::CodeGenerator() {
    // Register default backends
    registerBackend(CPU_TARGET, std::make_shared<cpu::CPUBackend>());
    registerBackend(CUDA_TARGET, std::make_shared<gpu::GPUBackend>()); // Assuming GPUBackend is CUDA for now
    registerBackend(EDGE_GENERIC_TARGET, std::make_shared<edge::EdgeBackend>());
    
    spdlog::info("CodeGenerator initialized with default backends: CPU, CUDA, Edge");
}

CodeGenerator::~CodeGenerator() {
    spdlog::info("CodeGenerator destroyed");
}

void CodeGenerator::generate(std::shared_ptr<ir::ComputationGraph> graph,
                           const std::string& outputPath,
                           const TargetName& target,
                           const CodeGenOptions& options) {
    auto it = backends_.find(target);
    if (it == backends_.end()) {
        spdlog::error("No backend registered for target: {}", target);
        // List available targets for better error message
        std::string available_targets_str = "Available targets are: ";
        for (const auto& pair : backends_) {
            available_targets_str += pair.first + " ";
        }
        if (backends_.empty()) {
            available_targets_str = "No backends registered.";
        }
        throw std::runtime_error("No backend registered for target: " + target + ". " + available_targets_str);
    }
    
    spdlog::info("Generating code for target \"{}\" using backend \"{}\"",
                target, it->second->getName());
    if (!options.empty()) {
        std::string opts_str;
        for(const auto& [key, val] : options) {
            opts_str += key + "=\'" + val + "\', ";
        }
        if (!opts_str.empty()) {
            opts_str.resize(opts_str.length() - 2); // Remove trailing comma and space
        }
        spdlog::info("With options: {}", opts_str);
    }
    
    it->second->generate(graph, outputPath, options);
    spdlog::info("Code generation completed for target \"{}\" to path \"{}\"", target, outputPath);
}

void CodeGenerator::registerBackend(const TargetName& name, 
                                   std::shared_ptr<TargetBackend> backend) {
    if (backends_.count(name)) {
        spdlog::warn("Overwriting existing backend registered for target: {}", name);
    }
    backends_[name] = backend;
    spdlog::info("Registered backend \"{}\" for target \"{}\"",
                backend->getName(), name);
}

std::vector<TargetName> CodeGenerator::listAvailableTargets() const {
    std::vector<TargetName> target_names;
    target_names.reserve(backends_.size());
    std::transform(backends_.begin(), backends_.end(), std::back_inserter(target_names),
                   [](const auto& pair) { return pair.first; });
    std::sort(target_names.begin(), target_names.end());
    return target_names;
}

} // namespace codegen
} // namespace openoptimizer
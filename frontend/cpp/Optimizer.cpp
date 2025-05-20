#include "frontend/cpp/Optimizer.hpp"
#include <spdlog/spdlog.h>

namespace openoptimizer {

Optimizer::Optimizer() {
    spdlog::info("C++ Optimizer instance created.");
    codeGenerator_ = std::make_shared<codegen::CodeGenerator>();
    spdlog::debug("CodeGenerator instance created and assigned in C++ Optimizer.");
}

Optimizer::~Optimizer() {
    spdlog::info("C++ Optimizer instance destroyed.");
}

std::shared_ptr<ir::ComputationGraph> Optimizer::importFromPyTorch(const std::string& modelPath) {
    spdlog::info("C++ Optimizer: Importing PyTorch model from {}", modelPath);
    // Actual implementation for PyTorch model import would involve:
    // 1. Loading the TorchScript model (if modelPath is a .pt file).
    // 2. Traversing the TorchScript graph.
    // 3. Converting TorchScript nodes and tensors to OpenOptimizer IR.
    // This is a complex task.
    spdlog::warn("C++ Optimizer: importFromPyTorch is a placeholder and does not perform actual import.");
    return std::make_shared<ir::ComputationGraph>("PyTorchModel_" + modelPath);
}

std::shared_ptr<ir::ComputationGraph> Optimizer::importFromTensorFlow(const std::string& modelPath) {
    spdlog::info("C++ Optimizer: Importing TensorFlow model from {}", modelPath);
    // Actual implementation for TensorFlow model import (e.g., from SavedModel):
    // 1. Using TensorFlow C API (or C++ API if suitable) to load the model.
    // 2. Extracting the graph definition (e.g., GraphDef).
    // 3. Converting TensorFlow operations and tensors to OpenOptimizer IR.
    // Also a complex task.
    spdlog::warn("C++ Optimizer: importFromTensorFlow is a placeholder and does not perform actual import.");
    return std::make_shared<ir::ComputationGraph>("TFModel_" + modelPath);
}

void Optimizer::addPass(std::shared_ptr<optimization::OptimizationPass> pass) {
    if (pass) {
        passes_.push_back(pass);
        spdlog::info("C++ Optimizer: Added optimization pass: {}", pass->getName());
    } else {
        spdlog::warn("C++ Optimizer: Attempted to add a null optimization pass.");
    }
}

void Optimizer::optimize(std::shared_ptr<ir::ComputationGraph> graph) {
    if (!graph) {
        spdlog::error("C++ Optimizer: Optimize called with a null graph.");
        return;
    }
    spdlog::info("C++ Optimizer: Starting optimization of graph '{}' with {} passes", graph->getName(), passes_.size());
    for (const auto& pass : passes_) {
        if (pass) {
            spdlog::info("C++ Optimizer: Running pass: {}", pass->getName());
            pass->run(graph); // Assuming OptimizationPass::run modifies the graph in-place or returns a new one
        } else {
            spdlog::warn("C++ Optimizer: Skipped a null optimization pass during optimize().");
        }
    }
    spdlog::info("C++ Optimizer: Optimization completed for graph '{}'", graph->getName());
}

// Updated implementation of generateCode
void Optimizer::generateCode(std::shared_ptr<ir::ComputationGraph> graph, 
                           const std::string& outputPath,
                           const codegen::TargetName& targetName,
                           const codegen::CodeGenOptions& options) {
    if (!graph) {
        spdlog::error("C++ Optimizer: generateCode called with a null graph.");
        return;
    }
    if (!codeGenerator_) {
        spdlog::error("C++ Optimizer: CodeGenerator is not initialized.");
        // Potentially throw an exception here
        return;
    }
    spdlog::info("C++ Optimizer: Generating code for graph '{}' targeting \"{}\" at path \"{}\"", 
                graph->getName(), targetName, outputPath);
    
    if (!options.empty()) {
        std::string opts_str;
        for(const auto& [key, val] : options) {
            opts_str += key + "=\'" + val + "\', ";
        }
        if (!opts_str.empty()) opts_str.resize(opts_str.length() - 2);
        spdlog::info("C++ Optimizer: Code generation options: {}", opts_str);
    }

    try {
        codeGenerator_->generate(graph, outputPath, targetName, options);
        spdlog::info("C++ Optimizer: Code generation finished successfully for target \"{}\"", targetName);
    } catch (const std::exception& e) {
        spdlog::error("C++ Optimizer: Code generation failed for target \"{}\". Error: {}", targetName, e.what());
        // Re-throw or handle as appropriate for the C++ API
        // If this is directly called by Python via Pybind11, Pybind11 can translate C++ exceptions
        // to Python exceptions. It might be good to use specific C++ exception types here.
        throw; // Re-throw to allow Pybind11 to convert it
    }
}

} // namespace openoptimizer
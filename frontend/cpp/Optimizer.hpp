#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map> // Required for CodeGenOptions

#include "ir/graph/ComputationGraph.hpp"
#include "optimization/OptimizationPass.hpp"
#include "codegen/CodeGenerator.hpp" // This defines TargetName and CodeGenOptions

namespace openoptimizer {

class Optimizer {
public:
    Optimizer();
    ~Optimizer();

    // Import from various frameworks
    std::shared_ptr<ir::ComputationGraph> importFromPyTorch(const std::string& modelPath);
    std::shared_ptr<ir::ComputationGraph> importFromTensorFlow(const std::string& modelPath);
    // Placeholder for ONNX import at C++ level, if desired directly here
    // std::shared_ptr<ir::ComputationGraph> importFromOnnx(const std::string& modelPath);
    
    // Add optimization passes
    void addPass(std::shared_ptr<optimization::OptimizationPass> pass);
    
    // Run optimization
    void optimize(std::shared_ptr<ir::ComputationGraph> graph);
    
    // Generate code - updated signature
    void generateCode(std::shared_ptr<ir::ComputationGraph> graph, 
                     const std::string& outputPath,
                     const codegen::TargetName& targetName,      // Now string
                     const codegen::CodeGenOptions& options); // Added options map

private:
    std::vector<std::shared_ptr<optimization::OptimizationPass>> passes_;
    std::shared_ptr<codegen::CodeGenerator> codeGenerator_;
};

} // namespace openoptimizer
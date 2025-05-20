#pragma once

#include "codegen/CodeGenerator.hpp"

namespace openoptimizer {
namespace codegen {
namespace edge {

class EdgeBackend : public TargetBackend {
public:
    EdgeBackend();
    ~EdgeBackend() override;
    
    void generate(std::shared_ptr<ir::ComputationGraph> graph, 
                const std::string& outputPath,
                const CodeGenOptions& options) override;
    
    TargetName getName() const override;
    
private:
    void generateOptimizedCCode(std::shared_ptr<ir::ComputationGraph> graph, 
                                const std::string& outputPath,
                                const CodeGenOptions& options);
    
    void generateTargetModelFile(std::shared_ptr<ir::ComputationGraph> graph, 
                                 const std::string& outputPath,
                                 const CodeGenOptions& options);
                              
    void generateDeploymentManifest(const std::string& outputPath,
                                    const CodeGenOptions& options);

    void packageForEdgeRuntime(const std::string& outputPath,
                               const CodeGenOptions& options);
};

} // namespace edge
} // namespace codegen
} // namespace openoptimizer
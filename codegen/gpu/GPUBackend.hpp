#pragma once

#include "codegen/CodeGenerator.hpp"

namespace openoptimizer {
namespace codegen {
namespace gpu {

class GPUBackend : public TargetBackend {
public:
    GPUBackend();
    ~GPUBackend() override;
    
    void generate(std::shared_ptr<ir::ComputationGraph> graph, 
                const std::string& outputPath,
                const CodeGenOptions& options) override;
    
    TargetName getName() const override;
    
private:
    void generateCudaCode(std::shared_ptr<ir::ComputationGraph> graph, 
                         const std::string& outputPath,
                         const CodeGenOptions& options);
    
    void generateCMakeLists(const std::string& outputPath,
                            const CodeGenOptions& options);
    
    void compileCudaCode(const std::string& outputPath,
                         const CodeGenOptions& options);
};

} // namespace gpu
} // namespace codegen
} // namespace openoptimizer
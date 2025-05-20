#pragma once

#include "codegen/CodeGenerator.hpp"

namespace openoptimizer {
namespace codegen {
namespace cpu {

class CPUBackend : public TargetBackend {
public:
    CPUBackend();
    ~CPUBackend() override;
    
    void generate(std::shared_ptr<ir::ComputationGraph> graph, 
                const std::string& outputPath,
                const CodeGenOptions& options) override;
    
    TargetName getName() const override;
    
private:
    void generateCppCode(std::shared_ptr<ir::ComputationGraph> graph, 
                        const std::string& outputPath,
                        const CodeGenOptions& options);
    
    void generateCMakeLists(const std::string& outputPath,
                            const CodeGenOptions& options);
    
    void compileCode(const std::string& outputPath,
                     const CodeGenOptions& options);
};

} // namespace cpu
} // namespace codegen
} // namespace openoptimizer
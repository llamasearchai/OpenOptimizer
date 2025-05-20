#pragma once

#include <unordered_set>
#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <utility>
#include "optimization/OptimizationPass.hpp"
#include "ir/graph/ComputationGraph.hpp"

namespace openoptimizer {
namespace optimization {
namespace passes {

class OperatorFusionPass : public OptimizationPass {
public:
    OperatorFusionPass();
    explicit OperatorFusionPass(const std::string& name);
    
    // Run the fusion pass on the given graph
    bool run(std::shared_ptr<ir::ComputationGraph> graph) override;
    
private:
    // Define fusion patterns as pairs of operator types
    using FusionPattern = std::pair<std::string, std::string>;
    std::vector<FusionPattern> fusion_patterns_;
    
    // Maximum number of fusion iterations
    int max_iterations_{10};
    
    // Initialize default fusion patterns
    void initDefaultPatterns();
    
    // Find nodes that can be fused according to patterns
    std::vector<std::pair<std::shared_ptr<ir::Node>, std::shared_ptr<ir::Node>>>
    findFusionCandidates(std::shared_ptr<ir::ComputationGraph> graph);
    
    // Check if two nodes can be fused
    bool canFuse(std::shared_ptr<ir::Node> parent, std::shared_ptr<ir::Node> child);
    
    // Special fusion case for nodes with multiple outputs
    bool canFuseWithMultipleOutputs(const std::string& op1_type, const std::string& op2_type);
    
    // Fuse two nodes together
    bool fuseNodes(std::shared_ptr<ir::ComputationGraph> graph,
                  std::shared_ptr<ir::Node> parent,
                  std::shared_ptr<ir::Node> child);
};

} // namespace passes
} // namespace optimization
} // namespace openoptimizer 
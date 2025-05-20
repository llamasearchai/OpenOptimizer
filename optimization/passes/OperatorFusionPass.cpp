#include "optimization/passes/OperatorFusionPass.hpp"
#include "ir/Operation.hpp"
#include "ir/graph/ComputationGraph.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <unordered_set>
#include <string>

namespace openoptimizer {
namespace optimization {
namespace passes {

// Implementation of a custom operation for the fused node
class FusedOperation : public ir::Operation {
public:
    FusedOperation(const std::string& name, 
                 std::shared_ptr<ir::Operation> op1, 
                 std::shared_ptr<ir::Operation> op2)
        : Operation(name, "Fused"), op1_(op1), op2_(op2) {
        // Set attributes to represent the constituent operations
        if (op1) setAttribute("op1_type", op1->getType());
        if (op2) setAttribute("op2_type", op2->getType());
        
        spdlog::debug("Created FusedOperation '{}' combining {} and {}", 
                      name, 
                      op1 ? op1->getType() : "null", 
                      op2 ? op2->getType() : "null");
    }
    
    // Override inferShapes to propagate input shapes through both operations
    std::vector<ir::TensorShape> inferShapes(const std::vector<ir::TensorShape>& input_shapes) const override {
        // For now, a simplified implementation that just returns the input shape
        // In a real implementation, this would simulate the effect of both ops
        if (op1_ && op2_) {
            auto intermediate = op1_->inferShapes(input_shapes);
            return op2_->inferShapes(intermediate);
        } else if (op1_) {
            return op1_->inferShapes(input_shapes);
        } else if (op2_) {
            return op2_->inferShapes(input_shapes);
        } else {
            spdlog::error("FusedOperation::inferShapes called with null operations");
            return input_shapes; // Fallback
        }
    }
    
private:
    std::shared_ptr<ir::Operation> op1_;
    std::shared_ptr<ir::Operation> op2_;
};

OperatorFusionPass::OperatorFusionPass() 
    : OptimizationPass("OperatorFusion") {
    initDefaultPatterns();
    spdlog::info("OperatorFusionPass created with {} default patterns", fusion_patterns_.size());
}

OperatorFusionPass::OperatorFusionPass(const std::string& name) 
    : OptimizationPass(name) {
    initDefaultPatterns();
    spdlog::info("OperatorFusionPass '{}' created with {} default patterns", 
                name, fusion_patterns_.size());
}

void OperatorFusionPass::initDefaultPatterns() {
    // Define common fusion patterns
    fusion_patterns_ = {
        {"Conv2d", "ReLU"},
        {"Conv2d", "BatchNorm2d"},
        {"Linear", "ReLU"},
        {"BatchNorm2d", "ReLU"},
        {"Conv2d", "Add"},
        {"MatMul", "Add"},
    };
}

bool OperatorFusionPass::run(std::shared_ptr<ir::ComputationGraph> graph) {
    if (!graph) {
        spdlog::error("OperatorFusionPass::run called with null graph");
        return false;
    }
    
    spdlog::info("Running OperatorFusionPass '{}' on graph '{}'", 
                getName(), graph->getName());
    
    int total_fusions = 0;
    
    // Run up to max_iterations_ rounds of fusion
    for (int iter = 0; iter < max_iterations_; ++iter) {
        spdlog::debug("OperatorFusionPass iteration {}", iter + 1);
        
        auto candidates = findFusionCandidates(graph);
        if (candidates.empty()) {
            spdlog::info("No more fusion candidates found after {} iterations", iter);
            break;
        }
        
        spdlog::info("Found {} fusion candidates in iteration {}", candidates.size(), iter + 1);
        
        int fused_in_iteration = 0;
        
        // Try to fuse each candidate pair
        for (const auto& [parent, child] : candidates) {
            if (canFuse(parent, child)) {
                if (fuseNodes(graph, parent, child)) {
                    total_fusions++;
                    fused_in_iteration++;
                    // Break and re-scan for candidates since the graph structure changed
                    break;
                }
            }
        }
        
        if (fused_in_iteration == 0) {
            spdlog::info("No fusions performed in iteration {}, stopping", iter + 1);
            break;
        }
        
        spdlog::info("Performed {} fusions in iteration {}", fused_in_iteration, iter + 1);
        
        // Warn if we hit the maximum iteration limit but are still finding candidates
        if (iter == max_iterations_ - 1 && fused_in_iteration > 0) {
            spdlog::warn("Reached maximum iterations ({}) with ongoing fusions", max_iterations_);
        }
    }
    
    spdlog::info("OperatorFusionPass completed with {} total fusions", total_fusions);
    return total_fusions > 0; // Return true if the graph was modified
}

std::vector<std::pair<std::shared_ptr<ir::Node>, std::shared_ptr<ir::Node>>>
OperatorFusionPass::findFusionCandidates(std::shared_ptr<ir::ComputationGraph> graph) {
    std::vector<std::pair<std::shared_ptr<ir::Node>, std::shared_ptr<ir::Node>>> candidates;
    std::unordered_set<std::string> processed_children;
    
    auto all_nodes = graph->getNodes();
    
    for (const auto& node : all_nodes) {
        if (!node || !node->getOperation()) continue;
        
        for (const auto& weak_output : node->getOutputs()) {
            auto child = weak_output.lock();
            if (!child || !child->getOperation()) continue;
            
            // Skip if we've already processed this child node
            if (processed_children.count(child->getName())) continue;
            
            if (canFuse(node, child)) {
                candidates.emplace_back(node, child);
                processed_children.insert(child->getName());
            }
        }
    }
    
    return candidates;
}

bool OperatorFusionPass::canFuse(std::shared_ptr<ir::Node> parent, std::shared_ptr<ir::Node> child) {
    if (!parent || !child || !parent->getOperation() || !child->getOperation()) {
        return false;
    }
    
    const auto& parent_op_type = parent->getOperation()->getType();
    const auto& child_op_type = child->getOperation()->getType();
    
    // Check if this pair matches any of our fusion patterns
    auto pattern_it = std::find(fusion_patterns_.begin(), fusion_patterns_.end(), 
                              std::make_pair(parent_op_type, child_op_type));
    if (pattern_it == fusion_patterns_.end()) {
        return false;
    }
    
    // Check if child has only one input and that input is parent
    auto child_inputs = child->getInputs();
    if (child_inputs.size() != 1 || child_inputs[0].lock() != parent) {
        return false;
    }
    
    // Check if parent has only one output (the child) or if this is a special fusible case
    auto parent_outputs = parent->getOutputs();
    if (parent_outputs.size() == 1 || canFuseWithMultipleOutputs(parent_op_type, child_op_type)) {
        // Ensure child is indeed one of parent's outputs
        for (const auto& weak_output : parent_outputs) {
            if (auto output = weak_output.lock()) {
                if (output == child) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

bool OperatorFusionPass::canFuseWithMultipleOutputs(const std::string& op1_type, const std::string& op2_type) {
    // Define special cases where fusion is allowed even when parent has multiple outputs
    static const std::vector<std::pair<std::string, std::string>> special_cases = {
        {"Conv2d", "BatchNorm2d"},
    };
    
    return std::find(special_cases.begin(), special_cases.end(), 
                   std::make_pair(op1_type, op2_type)) != special_cases.end();
}

bool OperatorFusionPass::fuseNodes(std::shared_ptr<ir::ComputationGraph> graph,
                                  std::shared_ptr<ir::Node> parent,
                                  std::shared_ptr<ir::Node> child) {
    spdlog::info("Attempting to fuse nodes: '{}' and '{}'", parent->getName(), child->getName());
    
    auto parent_op = parent->getOperation();
    auto child_op = child->getOperation();
    
    if (!parent_op || !child_op) {
        spdlog::error("Cannot fuse nodes with null operations");
        return false;
    }
    
    try {
        // Create a new fused operation
        std::string fused_op_name = parent->getName() + "_" + child->getName() + "_fused";
        auto fused_op = std::make_shared<FusedOperation>(fused_op_name, parent_op, child_op);
        
        // Add a new node with the fused operation
        auto fused_node = graph->addNode(fused_op_name, fused_op);
        spdlog::info("Created fused node '{}'", fused_node->getName());
        
        // Rewire inputs of parent to fused_node
        for (const auto& weak_input : parent->getInputs()) {
            if (auto input = weak_input.lock()) {
                graph->addEdge(input, fused_node);
            }
        }
        
        // Rewire outputs of child to fused_node
        for (const auto& weak_output : child->getOutputs()) {
            if (auto output = weak_output.lock()) {
                graph->addEdge(fused_node, output);
            }
        }
        
        // Remove original nodes
        // Note: Ensure graph has proper node removal support!
        // This is a critical step that needs to be implemented in ComputationGraph
        
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to fuse nodes: {}", e.what());
        return false;
    }
}

} // namespace passes
} // namespace optimization
} // namespace openoptimizer 
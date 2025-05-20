#pragma once

#include <string>
#include <memory>
#include "ir/graph/ComputationGraph.hpp" // Forward declare or include fully

namespace openoptimizer {
namespace optimization {

class OptimizationPass {
public:
    explicit OptimizationPass(std::string name)
        : name_(std::move(name)) {}
    
    virtual ~OptimizationPass() = default;

    // Pure virtual function to be implemented by derived passes
    virtual bool run(std::shared_ptr<ir::ComputationGraph> graph) = 0;

    // Get the name of the pass
    virtual std::string getName() const {
        return name_;
    }

protected:
    std::string name_;
};

} // namespace optimization
} // namespace openoptimizer 
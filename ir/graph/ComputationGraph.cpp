#include "ir/graph/ComputationGraph.hpp"
#include "ir/Operation.hpp" // Assuming Operation.hpp exists for op->getName()
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <iostream> // For dump()
#include <algorithm> // For std::remove_if, std::find_if
#include <vector>
#include <queue>
#include <unordered_set>

namespace openoptimizer {
namespace ir {

// --- Node Implementation ---
Node::Node(const std::string& name, std::shared_ptr<Operation> op)
    : name_(name), operation_(op) {
    if (!op) {
        spdlog::warn("Node '{}' created with null operation", name);
    } else {
        spdlog::debug("Node '{}' created with operation type '{}'", name, op->getType());
    }
}

void Node::addInput(std::shared_ptr<Node> inputNode, int input_port) {
    if (!inputNode) {
        spdlog::error("Attempted to add null input node to '{}'", name_);
        return;
    }
    
    // Check if the input already exists
    for (const auto& weak_input : inputs_) {
        if (auto existing = weak_input.lock()) {
            if (existing == inputNode) {
                spdlog::debug("Input '{}' already exists for node '{}'", 
                             inputNode->getName(), name_);
                return; // Already exists
            }
        }
    }
    
    inputs_.push_back(inputNode);
    spdlog::debug("Added input '{}' to node '{}'", inputNode->getName(), name_);
}

void Node::addOutput(std::shared_ptr<Node> outputNode, int output_port) {
    if (!outputNode) {
        spdlog::error("Attempted to add null output node to '{}'", name_);
        return;
    }
    
    // Check if the output already exists
    for (const auto& weak_output : outputs_) {
        if (auto existing = weak_output.lock()) {
            if (existing == outputNode) {
                spdlog::debug("Output '{}' already exists for node '{}'", 
                             outputNode->getName(), name_);
                return; // Already exists
            }
        }
    }
    
    outputs_.push_back(outputNode);
    spdlog::debug("Added output '{}' to node '{}'", outputNode->getName(), name_);
}

void Node::removeInput(std::shared_ptr<Node> inputNode) {
    if (!inputNode) {
        spdlog::error("Attempted to remove null input node from '{}'", name_);
        return;
    }
    
    // Find and remove the input
    auto it = std::find_if(inputs_.begin(), inputs_.end(), 
                         [&inputNode](const std::weak_ptr<Node>& weak_input) {
                             if (auto input = weak_input.lock()) {
                                 return input == inputNode;
                             }
                             return false;
                         });
    
    if (it != inputs_.end()) {
        inputs_.erase(it);
        spdlog::debug("Removed input '{}' from node '{}'", inputNode->getName(), name_);
    } else {
        spdlog::warn("Input '{}' not found in node '{}'", inputNode->getName(), name_);
    }
}

void Node::removeOutput(std::shared_ptr<Node> outputNode) {
    if (!outputNode) {
        spdlog::error("Attempted to remove null output node from '{}'", name_);
        return;
    }
    
    // Find and remove the output
    auto it = std::find_if(outputs_.begin(), outputs_.end(), 
                         [&outputNode](const std::weak_ptr<Node>& weak_output) {
                             if (auto output = weak_output.lock()) {
                                 return output == outputNode;
                             }
                             return false;
                         });
    
    if (it != outputs_.end()) {
        outputs_.erase(it);
        spdlog::debug("Removed output '{}' from node '{}'", outputNode->getName(), name_);
    } else {
        spdlog::warn("Output '{}' not found in node '{}'", outputNode->getName(), name_);
    }
}

void Node::clearConnections() {
    inputs_.clear();
    outputs_.clear();
    spdlog::trace("Node '{}': Cleared all input/output connections.", name_);
}

void Node::setMetadata(const MetadataKey& key, const std::any& value) {
    metadata_[key] = value;
}

std::any Node::getMetadata(const MetadataKey& key) const {
    auto it = metadata_.find(key);
    if (it != metadata_.end()) {
        return it->second;
    }
    throw std::out_of_range("Metadata key not found: " + key);
}

bool Node::hasMetadata(const MetadataKey& key) const {
    return metadata_.find(key) != metadata_.end();
}

// --- ComputationGraph Implementation ---
ComputationGraph::ComputationGraph(std::string name)
    : name_(std::move(name)) {
    spdlog::info("Created ComputationGraph '{}'", name_);
}

ComputationGraph::~ComputationGraph() {
    spdlog::debug("Destroying ComputationGraph '{}'", name_);
}

std::shared_ptr<Node> ComputationGraph::addNode(const std::string& nodeName, 
                                              std::shared_ptr<Operation> op) {
    if (nodes_.find(nodeName) != nodes_.end()) {
        spdlog::warn("Node '{}' already exists in graph '{}'", nodeName, name_);
        return nodes_[nodeName];
    }
    
    auto node = std::make_shared<Node>(nodeName, op);
    nodes_[nodeName] = node;
    spdlog::info("Added node '{}' to graph '{}'", nodeName, name_);
    return node;
}

void ComputationGraph::addEdge(const std::string& fromNodeName, const std::string& toNodeName) {
    auto fromNode = getNode(fromNodeName);
    auto toNode = getNode(toNodeName);
    
    if (!fromNode) {
        spdlog::error("Source node '{}' not found in graph '{}'", fromNodeName, name_);
        return;
    }
    
    if (!toNode) {
        spdlog::error("Target node '{}' not found in graph '{}'", toNodeName, name_);
        return;
    }
    
    addEdge(fromNode, toNode);
}

void ComputationGraph::addEdge(std::shared_ptr<Node> fromNode, std::shared_ptr<Node> toNode) {
    if (!fromNode || !toNode) {
        spdlog::error("Cannot add edge between null nodes in graph '{}'", name_);
        return;
    }
    
    // Update node connections in both directions
    fromNode->addOutput(toNode);
    toNode->addInput(fromNode);
    
    spdlog::info("Added edge from '{}' to '{}' in graph '{}'", 
                fromNode->getName(), toNode->getName(), name_);
}

void ComputationGraph::removeEdge(std::shared_ptr<Node> fromNode, std::shared_ptr<Node> toNode) {
    if (!fromNode || !toNode) {
        spdlog::error("Cannot remove edge between null nodes in graph '{}'", name_);
        return;
    }
    
    // Update node connections in both directions
    fromNode->removeOutput(toNode);
    toNode->removeInput(fromNode);
    
    spdlog::info("Removed edge from '{}' to '{}' in graph '{}'", 
                fromNode->getName(), toNode->getName(), name_);
}

void ComputationGraph::removeEdgeByName(const std::string& fromNodeName, const std::string& toNodeName) {
    auto fromNode = getNode(fromNodeName);
    auto toNode = getNode(toNodeName);
    
    if (!fromNode) {
        spdlog::error("Source node '{}' not found in graph '{}'", fromNodeName, name_);
        return;
    }
    
    if (!toNode) {
        spdlog::error("Target node '{}' not found in graph '{}'", toNodeName, name_);
        return;
    }
    
    removeEdge(fromNode, toNode);
}

bool ComputationGraph::removeNode(std::shared_ptr<Node> node) {
    if (!node) {
        spdlog::error("Cannot remove null node from graph '{}'", name_);
        return false;
    }
    
    return removeNodeByName(node->getName());
}

bool ComputationGraph::removeNodeByName(const std::string& nodeName) {
    auto node_it = nodes_.find(nodeName);
    if (node_it == nodes_.end()) {
        spdlog::warn("Node '{}' not found in graph '{}'", nodeName, name_);
        return false;
    }
    
    auto node = node_it->second;
    
    // Remove references to this node from input nodes
    for (const auto& weak_input : node->getInputs()) {
        if (auto input = weak_input.lock()) {
            input->removeOutput(node);
        }
    }
    
    // Remove references to this node from output nodes
    for (const auto& weak_output : node->getOutputs()) {
        if (auto output = weak_output.lock()) {
            output->removeInput(node);
        }
    }
    
    // Remove from input/output node lists if present
    auto input_it = std::find(inputNodes_.begin(), inputNodes_.end(), node);
    if (input_it != inputNodes_.end()) {
        inputNodes_.erase(input_it);
    }
    
    auto output_it = std::find(outputNodes_.begin(), outputNodes_.end(), node);
    if (output_it != outputNodes_.end()) {
        outputNodes_.erase(output_it);
    }
    
    // Remove the node itself from the graph
    nodes_.erase(node_it);
    
    spdlog::info("Removed node '{}' from graph '{}'", nodeName, name_);
    return true;
}

std::shared_ptr<Node> ComputationGraph::getNode(const std::string& name) const {
    auto it = nodes_.find(name);
    if (it != nodes_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<Node>> ComputationGraph::getNodes() const {
    std::vector<std::shared_ptr<Node>> result;
    result.reserve(nodes_.size());
    
    for (const auto& [name, node] : nodes_) {
        result.push_back(node);
    }
    
    return result;
}

void ComputationGraph::setInputNodes(const std::vector<std::string>& inputNodeNames) {
    inputNodes_.clear();
    
    for (const auto& name : inputNodeNames) {
        auto node = getNode(name);
        if (node) {
            inputNodes_.push_back(node);
        } else {
            spdlog::warn("Input node '{}' not found in graph '{}'", name, name_);
        }
    }
    
    spdlog::info("Set {} input nodes for graph '{}'", inputNodes_.size(), name_);
}

void ComputationGraph::setOutputNodes(const std::vector<std::string>& outputNodeNames) {
    outputNodes_.clear();
    
    for (const auto& name : outputNodeNames) {
        auto node = getNode(name);
        if (node) {
            outputNodes_.push_back(node);
        } else {
            spdlog::warn("Output node '{}' not found in graph '{}'", name, name_);
        }
    }
    
    spdlog::info("Set {} output nodes for graph '{}'", outputNodes_.size(), name_);
}

void ComputationGraph::setInputNodes(const std::vector<std::shared_ptr<Node>>& inputs) {
    inputNodes_ = inputs;
    spdlog::info("Set {} input nodes for graph '{}'", inputNodes_.size(), name_);
}

void ComputationGraph::setOutputNodes(const std::vector<std::shared_ptr<Node>>& outputs) {
    outputNodes_ = outputs;
    spdlog::info("Set {} output nodes for graph '{}'", outputNodes_.size(), name_);
}

void ComputationGraph::dump() const {
    std::cout << "=== ComputationGraph '" << name_ << "' ===" << std::endl;
    std::cout << "Nodes: " << nodes_.size() << std::endl;
    
    for (const auto& [name, node] : nodes_) {
        std::cout << "  Node: " << name;
        
        if (node->getOperation()) {
            std::cout << " (Op: " << node->getOperation()->getType() << ")";
        } else {
            std::cout << " (No operation)";
        }
        
        std::cout << std::endl;
        
        std::cout << "    Inputs: ";
        for (const auto& weak_input : node->getInputs()) {
            if (auto input = weak_input.lock()) {
                std::cout << input->getName() << " ";
            } else {
                std::cout << "<expired> ";
            }
        }
        std::cout << std::endl;
        
        std::cout << "    Outputs: ";
        for (const auto& weak_output : node->getOutputs()) {
            if (auto output = weak_output.lock()) {
                std::cout << output->getName() << " ";
            } else {
                std::cout << "<expired> ";
            }
        }
        std::cout << std::endl;
    }
    
    std::cout << "Input Nodes: ";
    for (const auto& node : inputNodes_) {
        std::cout << node->getName() << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Output Nodes: ";
    for (const auto& node : outputNodes_) {
        std::cout << node->getName() << " ";
    }
    std::cout << std::endl;
    
    std::cout << "===========================" << std::endl;
}

} // namespace ir
} // namespace openoptimizer 
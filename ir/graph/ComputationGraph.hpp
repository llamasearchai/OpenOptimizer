#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <any> // For metadata

#include "ir/tensor/Tensor.hpp"

namespace openoptimizer {
namespace ir {

class Operation; // Forward declaration

// Using a type alias for clarity if metadata keys become standardized
using MetadataKey = std::string;

class Node {
public:
    Node(const std::string& name, std::shared_ptr<Operation> op);
    
    std::string getName() const { return name_; }
    std::shared_ptr<Operation> getOperation() const { return operation_; }
    
    // Edge management - these imply a directed graph structure
    void addInput(std::shared_ptr<Node> inputNode, int input_port = 0); // Optional: port for multi-input ops
    void addOutput(std::shared_ptr<Node> outputNode, int output_port = 0); // Optional: port for multi-output ops
    
    // Remove connections to other nodes
    void removeInput(std::shared_ptr<Node> inputNode);
    void removeOutput(std::shared_ptr<Node> outputNode);
    
    const std::vector<std::weak_ptr<Node>>& getInputs() const { return inputs_; } // Use weak_ptr to avoid cycles if graph owns nodes
    const std::vector<std::weak_ptr<Node>>& getOutputs() const { return outputs_; } // Use weak_ptr

    // Direct access to input/output edges might be better if edges carry data (like Tensors or TensorDescriptors)
    // For now, keeping it node-centric.

    // Metadata
    void setMetadata(const MetadataKey& key, const std::any& value);
    std::any getMetadata(const MetadataKey& key) const;
    bool hasMetadata(const MetadataKey& key) const;
    const std::unordered_map<MetadataKey, std::any>& getAllMetadata() const { return metadata_; }

private:
    std::string name_;
    std::shared_ptr<Operation> operation_; // The operation this node performs
    
    // Adjacency list representation using weak_ptr to prevent ownership cycles
    // if the graph itself (ComputationGraph) owns the nodes primarily.
    // If nodes own their children, shared_ptr is fine but care is needed for cycles.
    std::vector<std::weak_ptr<Node>> inputs_; 
    std::vector<std::weak_ptr<Node>> outputs_;
    // To truly represent edges as first-class citizens carrying e.g. Tensor information,
    // you might have a list of Edge objects, where Edge contains source/target node and Tensor info.
    // For now, this is a common way.

    std::unordered_map<MetadataKey, std::any> metadata_;
};

class ComputationGraph {
public:
    explicit ComputationGraph(std::string name = "UnnamedGraph"); // Added constructor with name
    ~ComputationGraph();
    
    std::string getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    std::shared_ptr<Node> addNode(const std::string& nodeName, std::shared_ptr<Operation> op);
    // addEdge might be better if it creates an Edge object containing Tensor info
    // or if it just updates Node::inputs_/outputs_ directly.
    // The current Node::addInput/addOutput suggests nodes manage their own connections.
    void addEdge(const std::string& fromNodeName, const std::string& toNodeName);
    // Overload for direct node pointers if preferred internally
    void addEdge(std::shared_ptr<Node> fromNode, std::shared_ptr<Node> toNode);
    
    // Edge removal
    void removeEdge(std::shared_ptr<Node> fromNode, std::shared_ptr<Node> toNode);
    void removeEdgeByName(const std::string& fromNodeName, const std::string& toNodeName);
    
    // Node removal
    bool removeNode(std::shared_ptr<Node> node);
    bool removeNodeByName(const std::string& nodeName);

    std::shared_ptr<Node> getNode(const std::string& name) const;
    std::vector<std::shared_ptr<Node>> getNodes() const; // Returns all nodes in the graph
    
    // Manage explicit input/output nodes of the graph
    void setInputNodes(const std::vector<std::string>& inputNodeNames);
    void setOutputNodes(const std::vector<std::string>& outputNodeNames);
    // Or by direct node pointers
    void setInputNodes(const std::vector<std::shared_ptr<Node>>& inputs);
    void setOutputNodes(const std::vector<std::shared_ptr<Node>>& outputs);

    const std::vector<std::shared_ptr<Node>>& getInputNodes() const { return inputNodes_; }
    const std::vector<std::shared_ptr<Node>>& getOutputNodes() const { return outputNodes_; }
    
    void dump() const; // For debugging, print graph structure
    // void topologicalSort() const; // Useful utility
    // bool validate() const; // Check for cycles, disconnected parts etc.
    
private:
    std::string name_;
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_; // Primary ownership of nodes
    std::vector<std::shared_ptr<Node>> inputNodes_;  // References to nodes in nodes_
    std::vector<std::shared_ptr<Node>> outputNodes_; // References to nodes in nodes_
};

} // namespace ir
} // namespace openoptimizer
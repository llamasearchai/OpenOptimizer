#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
#include <pybind11/any.h>

#include "ir/graph/ComputationGraph.hpp"
#include "ir/Operation.hpp"
#include "ir/ops/StandardOps.hpp" // For concrete ops like Conv2DOp, ReLUOp
#include "ir/tensor/TensorDescriptor.hpp"
#include "ir/tensor/TensorShape.hpp"
#include "ir/tensor/DataType.hpp"
#include "frontend/cpp/Optimizer.hpp"
#include "codegen/CodeGenerator.hpp"
#include "optimization/OptimizationPass.hpp"

namespace py = pybind11;
using namespace openoptimizer;

// Helper to make std::any usable with pybind11 (basic string, int, double, bool for now)
// More sophisticated handling might be needed for other types in std::any.
namespace pybind11 { namespace detail {
    template <> struct type_caster<std::any> {
    public:
        PYBIND11_TYPE_CASTER(std::any, _("object"));

        bool load(handle src, bool convert) {
            if (py::isinstance<py::str>(src)) {
                value = py::cast<std::string>(src);
            } else if (py::isinstance<py::int_>(src)) {
                value = py::cast<int>(src);
            } else if (py::isinstance<py::float_>(src)) {
                value = py::cast<double>(src);
            } else if (py::isinstance<py::bool_>(src)) {
                value = py::cast<bool>(src);
            } else if (src.is_none()) {
                value = std::any(); // Or handle as specific type if None means something
            } else {
                // Could attempt other types or raise an error
                // For now, this is a simple conversion
                return false; 
            }
            return true;
        }

        static handle cast(const std::any& src, return_value_policy /* policy */, handle /* parent */) {
            if (!src.has_value()) {
                return py::none().release();
            }
            if (src.type() == typeid(std::string)) {
                return py::cast(std::any_cast<std::string>(src)).release();
            } else if (src.type() == typeid(int)) {
                return py::cast(std::any_cast<int>(src)).release();
            } else if (src.type() == typeid(double)) {
                return py::cast(std::any_cast<double>(src)).release();
            } else if (src.type() == typeid(bool)) {
                return py::cast(std::any_cast<bool>(src)).release();
            } else {
                // Add more types as needed or return a generic object/string representation
                // py::print("Warning: Casting std::any of unhandled type: ", src.type().name());
                return py::str("Unsupported std::any type: " + std::string(src.type().name())).release();
            }
        }
    };
}} // namespace pybind11::detail


PYBIND11_MODULE(_cpp_extension, m) {
    m.doc() = "OpenOptimizer C++ extension module";

    // Redirect C++ iostream to Python's sys.stdout, sys.stderr
    py::add_ostream_redirect(m, "ostream_redirect");

    // IR Submodule (optional, for organization)
    // auto ir_m = m.def_submodule("ir", "Intermediate Representation components");
    // For simplicity, binding directly to `_cpp_extension` module first.

    // ir::DataType
    py::enum_<ir::DataType>(m, "DataType")
        .value("FLOAT32", ir::DataType::FLOAT32)
        .value("FLOAT16", ir::DataType::FLOAT16)
        .value("INT32", ir::DataType::INT32)
        .value("INT16", ir::DataType::INT16)
        .value("INT8", ir::DataType::INT8)
        .value("UINT8", ir::DataType::UINT8)
        .value("BOOL", ir::DataType::BOOL)
        .value("UNKNOWN", ir::DataType::UNKNOWN)
        .export_values();
    m.def("getDataTypeSizeBytes", &ir::getDataTypeSizeBytes, "Get size of DataType in bytes");
    m.def("dataTypeToString", &ir::dataTypeToString, "Convert DataType to string");
    m.def("stringToDataType", &ir::stringToDataType, "Convert string to DataType");

    // ir::Dimension (std::optional<int64_t>)
    // Pybind11 handles std::optional automatically. We can alias it in Python if needed.
    // For DYNAMIC_DIM, we can expose it as a constant.
    m.attr("DYNAMIC_DIM") = py::cast(ir::DYNAMIC_DIM);
    // Python can use None for std::nullopt (DYNAMIC_DIM) when constructing TensorShape

    // ir::TensorShape
    py::class_<ir::TensorShape>(m, "TensorShape")
        .def(py::init<>())
        .def(py::init<std::vector<ir::Dimension>>())
        // Note: Variadic template constructor in C++ is not directly bindable like this.
        // Users would typically pass a list/tuple of dimensions from Python.
        // We can add a helper or make Python users pass std::vector<std::optional<int64_t>>.
        // For simplicity, only vector constructor for now from Python.
        .def_property_readonly("rank", &ir::TensorShape::getRank)
        .def_property_readonly("is_scalar", &ir::TensorShape::isScalar)
        .def_property_readonly("has_dynamic_dimensions", &ir::TensorShape::hasDynamicDimensions)
        .def_property_readonly("is_fully_static", &ir::TensorShape::isFullyStatic)
        .def_property_readonly("dimensions", &ir::TensorShape::getDimensions, py::return_value_policy::reference_internal)
        .def("get_dimension", &ir::TensorShape::getDimension)
        .def("set_dimension", &ir::TensorShape::setDimension)
        .def("append_dimension", &ir::TensorShape::appendDimension)
        .def("prepend_dimension", &ir::TensorShape::prependDimension)
        .def_property_readonly("num_elements", &ir::TensorShape::getNumElements)
        .def("to_string", &ir::TensorShape::toString)
        .def("__str__", &ir::TensorShape::toString)
        .def("__repr__", [](const ir::TensorShape& s) { return "<TensorShape '" + s.toString() + "'>"; })
        .def(py::self == py::self)
        .def(py::self != py::self);

    // ir::TensorDescriptor (aliased as ir::Tensor in C++ header)
    py::class_<ir::TensorDescriptor, std::shared_ptr<ir::TensorDescriptor>>(m, "TensorDescriptor")
        .def(py::init<ir::TensorShape, ir::DataType, std::string>(), 
             py::arg("shape"), py::arg("dtype"), py::arg("name") = "")
        // Constructor with data_ptr might be harder to expose safely with void*
        // It's better if data handling is done via Python (e.g., NumPy arrays converted at a higher level)
        .def_property_readonly("shape", &ir::TensorDescriptor::getShape, py::return_value_policy::reference_internal)
        .def_property_readonly("dtype", &ir::TensorDescriptor::getDataType)
        .def_property("name", &ir::TensorDescriptor::getName, &ir::TensorDescriptor::setName)
        .def_property_readonly("has_data", &ir::TensorDescriptor::hasData)
        // .def_property("data", &ir::TensorDescriptor::getData, &ir::TensorDescriptor::setData) // Careful with void*
        .def_property_readonly("size_bytes", &ir::TensorDescriptor::getSizeBytes)
        .def("to_string", &ir::TensorDescriptor::toString)
        .def("__str__", &ir::TensorDescriptor::toString)
        .def("__repr__", [](const ir::TensorDescriptor& td) { return "<TensorDescriptor '" + td.toString() + "'>"; })
        .def_property("quantization_params", &ir::TensorDescriptor::getQuantizationParams, &ir::TensorDescriptor::setQuantizationParams)
        .def_property_readonly("has_quantization_params", &ir::TensorDescriptor::hasQuantizationParams);

    // ir::Operation (Abstract base class)
    // We need a trampoline class if Python classes will derive from C++ Operation
    // For now, just exposing it as a non-derivable type, and binding concrete ops.
    py::class_<ir::Operation, std::shared_ptr<ir::Operation>>(m, "Operation")
        // .def(py::init<std::string, std::string>()) // Constructor is protected/abstract concept
        .def_property_readonly("name", &ir::Operation::getName)
        .def_property_readonly("type", &ir::Operation::getType)
        .def("set_attribute", &ir::Operation::setAttribute)
        .def("get_attribute_string", [](const ir::Operation& op, const std::string& key){ return op.getAttribute<std::string>(key); })
        .def("get_attribute_int", [](const ir::Operation& op, const std::string& key){ return op.getAttribute<int>(key); })
        .def("get_attribute_float", [](const ir::Operation& op, const std::string& key){ return op.getAttribute<double>(key); })
        .def("get_attribute_bool", [](const ir::Operation& op, const std::string& key){ return op.getAttribute<bool>(key); })
        .def("get_attribute_int_vector", [](const ir::Operation& op, const std::string& key){ return op.getAttribute<std::vector<int>>(key); })
        // Add more get_attribute_xxx for other common types if needed
        .def("has_attribute", &ir::Operation::hasAttribute)
        .def_property_readonly("attributes", &ir::Operation::getAttributes)
        .def("infer_shapes", &ir::Operation::inferShapes, py::arg("input_shapes"));
        // Python side Operation wrapper in openoptimizer.ir will need to match these.

    // Concrete Operations (ir::ops)
    py::class_<ir::Conv2DOp, ir::Operation, std::shared_ptr<ir::Conv2DOp>>(m, "Conv2DOp")
        .def(py::init<const std::string&, int, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>, int, bool>(),
             py::arg("name"), py::arg("out_channels"), py::arg("kernel_size"), 
             py::arg("stride") = std::vector<int>{1,1}, py::arg("padding") = std::vector<int>{0,0}, 
             py::arg("dilation") = std::vector<int>{1,1}, py::arg("groups") = 1, py::arg("bias") = true);

    py::class_<ir::ReLUOp, ir::Operation, std::shared_ptr<ir::ReLUOp>>(m, "ReLUOp")
        .def(py::init<const std::string&>(), py::arg("name"));

    py::class_<ir::AddOp, ir::Operation, std::shared_ptr<ir::AddOp>>(m, "AddOp")
        .def(py::init<const std::string&>(), py::arg("name"));

    // ir::Node
    // Need to handle weak_ptr for getInputs/getOutputs. Pybind11 might need help or custom casters for vector<weak_ptr>.
    // Often, one would expose methods that return vector<shared_ptr> by locking weak_ptrs internally.
    // For now, let's try to expose a method that returns locked shared_ptrs.
    py::class_<ir::Node, std::shared_ptr<ir::Node>>(m, "Node")
        .def(py::init<const std::string&, std::shared_ptr<ir::Operation>>(), py::arg("name"), py::arg("op"))
        .def_property_readonly("name", &ir::Node::getName)
        .def_property_readonly("operation", &ir::Node::getOperation)
        .def("add_input", [](ir::Node& self, std::shared_ptr<ir::Node> inputNode){ self.addInput(inputNode); }, "Adds an input connection (Node to self)")
        .def("add_output", [](ir::Node& self, std::shared_ptr<ir::Node> outputNode){ self.addOutput(outputNode); }, "Adds an output connection (self to Node)")
        .def("get_inputs_locked", [](const ir::Node& self) {
            std::vector<std::shared_ptr<ir::Node>> locked_inputs;
            for (const auto& wp : self.getInputs()) {
                if (auto sp = wp.lock()) locked_inputs.push_back(sp);
            }
            return locked_inputs;
        })
        .def("get_outputs_locked", [](const ir::Node& self) {
            std::vector<std::shared_ptr<ir::Node>> locked_outputs;
            for (const auto& wp : self.getOutputs()) {
                if (auto sp = wp.lock()) locked_outputs.push_back(sp);
            }
            return locked_outputs;
        })
        .def("set_metadata", &ir::Node::setMetadata)
        .def("get_metadata", &ir::Node::getMetadata)
        .def("has_metadata", &ir::Node::hasMetadata)
        .def_property_readonly("metadata", &ir::Node::getAllMetadata);
        // Python Node wrapper will use get_inputs_locked / get_outputs_locked for its .inputs / .outputs properties

    // ir::ComputationGraph
    py::class_<ir::ComputationGraph, std::shared_ptr<ir::ComputationGraph>>(m, "ComputationGraph")
        .def(py::init<std::string>(), py::arg("name") = "UnnamedGraph")
        .def_property("name", &ir::ComputationGraph::getName, &ir::ComputationGraph::setName)
        .def("add_node", &ir::ComputationGraph::addNode, py::arg("node_name"), py::arg("op"))
        .def("add_edge", static_cast<void (ir::ComputationGraph::*)(const std::string&, const std::string&)>(&ir::ComputationGraph::addEdge), py::arg("from_node_name"), py::arg("to_node_name"))
        .def("add_edge_nodes", static_cast<void (ir::ComputationGraph::*)(std::shared_ptr<ir::Node>, std::shared_ptr<ir::Node>)>(&ir::ComputationGraph::addEdge), py::arg("from_node"), py::arg("to_node"))
        .def("get_node", &ir::ComputationGraph::getNode, py::arg("name"), py::return_value_policy::reference_internal) // Or automatic_reference
        .def("get_nodes", &ir::ComputationGraph::getNodes)
        .def("set_input_nodes_by_name", static_cast<void (ir::ComputationGraph::*)(const std::vector<std::string>&)>(&ir::ComputationGraph::setInputNodes), py::arg("input_node_names"))
        .def("set_output_nodes_by_name", static_cast<void (ir::ComputationGraph::*)(const std::vector<std::string>&)>(&ir::ComputationGraph::setOutputNodes), py::arg("output_node_names"))
        .def("set_input_nodes", static_cast<void (ir::ComputationGraph::*)(const std::vector<std::shared_ptr<ir::Node>>&)>(&ir::ComputationGraph::setInputNodes), py::arg("inputs"))
        .def("set_output_nodes", static_cast<void (ir::ComputationGraph::*)(const std::vector<std::shared_ptr<ir::Node>>&)>(&ir::ComputationGraph::setOutputNodes), py::arg("outputs"))
        .def_property_readonly("input_nodes", &ir::ComputationGraph::getInputNodes)
        .def_property_readonly("output_nodes", &ir::ComputationGraph::getOutputNodes)
        .def("remove_node_by_name", static_cast<bool (ir::ComputationGraph::*)(const std::string&)>(&ir::ComputationGraph::removeNode))
        .def("remove_node", static_cast<bool (ir::ComputationGraph::*)(const std::shared_ptr<ir::Node>&)>(&ir::ComputationGraph::removeNode))
        .def("remove_edge_by_name", static_cast<void (ir::ComputationGraph::*)(const std::string&, const std::string&)>(&ir::ComputationGraph::removeEdge))
        .def("remove_edge", static_cast<void (ir::ComputationGraph::*)(const std::shared_ptr<ir::Node>&, const std::shared_ptr<ir::Node>&)>(&ir::ComputationGraph::removeEdge))
        .def("dump", &ir::ComputationGraph::dump, "Prints a textual representation of the graph to stdout");
        // The Python ComputationGraph wrapper will provide .name, .nodes, .input_nodes, .output_nodes properties
        // and methods like .add_node(name, op_wrapper), .add_edge(from, to), .remove_node(name_or_node)

    // optimization::OptimizationPass (Abstract)
    py::class_<optimization::OptimizationPass, std::shared_ptr<optimization::OptimizationPass>>(m, "OptimizationPassCpp") // Name it Cpp to distinguish from Python base
        // .def(py::init<std::string>()) // Constructor is protected
        .def("run", &optimization::OptimizationPass::run, py::arg("graph"))
        .def_property_readonly("name", &optimization::OptimizationPass::getName);
        // Python passes will need to instantiate their C++ counterparts if they are wrappers.

    // frontend::Optimizer (The main C++ Optimizer class used by Python frontend)
    py::class_<Optimizer, std::shared_ptr<Optimizer>>(m, "Optimizer")
        .def(py::init<>())
        .def("import_from_pytorch", &Optimizer::importFromPyTorch, py::arg("model_path"))
        .def("import_from_tensorflow", &Optimizer::importFromTensorFlow, py::arg("model_path"))
        // Add import_from_onnx if/when implemented in C++ Optimizer
        .def("add_pass", &Optimizer::addPass, py::arg("pass"))
        .def("optimize", &Optimizer::optimize, py::arg("graph"))
        .def("generate_code", &Optimizer::generateCode, 
             py::arg("graph"), py::arg("output_path"), py::arg("target_name"), py::arg("options"));

    // codegen::CodeGenerator (Might not need to be exposed if Optimizer handles it all)
    // py::class_<codegen::CodeGenerator, std::shared_ptr<codegen::CodeGenerator>>(m, "CodeGenerator")
    //     .def(py::init<>())
    //     .def("generate", &codegen::CodeGenerator::generate, 
    //          py::arg("graph"), py::arg("output_path"), py::arg("target_name"), py::arg("options") = codegen::CodeGenOptions{})
    //     .def("register_backend", &codegen::CodeGenerator::registerBackend, py::arg("name"), py::arg("backend"))
    //     .def("list_available_targets", &codegen::CodeGenerator::listAvailableTargets);

    // Expose target name constants from codegen::CodeGenerator.hpp
    m.attr("CPU_TARGET") = py::str(codegen::CPU_TARGET);
    m.attr("CUDA_TARGET") = py::str(codegen::CUDA_TARGET);
    m.attr("METAL_TARGET") = py::str(codegen::METAL_TARGET);
    m.attr("EDGE_GENERIC_TARGET") = py::str(codegen::EDGE_GENERIC_TARGET);

} 
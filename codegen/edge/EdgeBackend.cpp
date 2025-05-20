#include "codegen/edge/EdgeBackend.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <system_error>
#include <filesystem>
#include <cstdlib> // For std::system, if packaging step involves compilation

namespace openoptimizer {
namespace codegen {
namespace edge {

EdgeBackend::EdgeBackend() {
    spdlog::info("Edge device backend initialized. Call getName() to see its registered TargetName.");
}

EdgeBackend::~EdgeBackend() {
    spdlog::info("Edge device backend destroyed");
}

TargetName EdgeBackend::getName() const {
    return EDGE_GENERIC_TARGET; // Defined in CodeGenerator.hpp
}

void EdgeBackend::generate(std::shared_ptr<ir::ComputationGraph> graph, 
                        const std::string& outputPath,
                        const CodeGenOptions& options) {
    spdlog::info("Generating Edge code at \"{}\" with backend \"{}\"", outputPath, getName());
    if (!options.empty()) {
        std::string opts_str;
        for(const auto& [key, val] : options) {
            opts_str += key + "=\'" + val + "\', ";
        }
        if (!opts_str.empty()) opts_str.resize(opts_str.length() - 2);
        spdlog::info("EdgeBackend options: {}", opts_str);
    }

    try {
        std::filesystem::create_directories(outputPath);
        
        // Example: Choose generation path based on options like {"edge_target_runtime": "tflite_micro"} or {"edge_target_runtime": "tvm_micro"}
        std::string target_runtime = "generic_c"; // Default
        if (options.count("edge_target_runtime")) {
            target_runtime = options.at("edge_target_runtime");
        }
        spdlog::info("Targeting edge runtime: {}", target_runtime);

        if (target_runtime == "tflite_micro" || target_runtime == "generic_c" || options.count("generate_c_code")) {
            generateOptimizedCCode(graph, outputPath, options);
        }
        
        // This might produce a .tflite file, an ONNX model for ORT mobile, or a TVM relay model for MicroTVM.
        // The actual content and format depend heavily on the specific edge target and options.
        generateTargetModelFile(graph, outputPath, options);
        
        generateDeploymentManifest(outputPath, options);

        bool should_package = false;
        if (options.count("package_for_edge") && options.at("package_for_edge") == "true") {
            should_package = true;
        }
        if (should_package) {
            packageForEdgeRuntime(outputPath, options);
            spdlog::info("Edge packaging requested and attempted.");
        } else {
            spdlog::info("Packaging not requested for Edge. Skipping package step.");
        }
        
        spdlog::info("Edge device code generation completed successfully for path: {}", outputPath);
    } catch (const std::exception& e) {
        spdlog::error("Error during edge device code generation for path {}: {}", outputPath, e.what());
        throw;
    }
}

void EdgeBackend::generateOptimizedCCode(std::shared_ptr<ir::ComputationGraph> graph, 
                                     const std::string& outputPath,
                                     const CodeGenOptions& options) {
    spdlog::debug("generateOptimizedCCode for Edge target in {} called", outputPath);
    // ... (existing C code generation logic, potentially modified by options) ...
    // Example: Options might dictate header style, include guards, specific C features, or static/dynamic memory allocation strategies

    std::ofstream header(outputPath + "/edge_model.h");
    if (!header.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open file: edge_model.h");
    }
    header << "#pragma once\n#include <stdint.h>\n#include <stdlib.h>\n"
           << "// Generated with options: ";
    for(const auto& [key, val] : options) { header << key << "=\"" << val << "\"; "; }
    header << "\n"
           << "#ifdef __cplusplus\nextern \"C\" {\n#endif\n"
           << "typedef struct { void* data; int32_t dims[4]; size_t num_dims; int32_t element_size; } EdgeTensor;\n"
           << "int edge_model_init(void** model_context);\n"
           << "int edge_model_run(void* model_context, const EdgeTensor* inputs, int num_inputs, EdgeTensor* outputs, int num_outputs);\n"
           << "void edge_model_cleanup(void* model_context);\n"
           << "#ifdef __cplusplus\n}\n#endif\n";
    header.close();

    std::ofstream impl(outputPath + "/edge_model.c");
    if (!impl.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open file: edge_model.c");
    }
    impl << "#include \"edge_model.h\"\n#include <string.h>\n#include <stdio.h>\n"
         << "// Generated with options: ";
    for(const auto& [key, val] : options) { impl << key << "=\"" << val << "\"; "; }
    impl << "\n"
         << "// static const uint8_t model_weights[] = { /* quantized weights */ };\n"
         << "typedef struct { char name[32]; /* Other context data */ } ModelContext;\n"
         << "int edge_model_init(void** model_context) {\n"
         << "    printf(\"Initializing edge model (generic C)...\\n\");\n"
         << "    ModelContext* ctx = (ModelContext*)malloc(sizeof(ModelContext));\n"
         << "    if (!ctx) return -1; /* Allocation failed */ \n"
         << "    strncpy(ctx->name, \"MyEdgeModel\", 31); ctx->name[31] = '\\0';\n"
         << "    *model_context = ctx;\n"
         << "    return 0; \n}";
    impl << "int edge_model_run(void* model_context, const EdgeTensor* inputs, int num_inputs, EdgeTensor* outputs, int num_outputs) {\n"
         << "    if (!model_context) return -1;\n"
         << "    ModelContext* ctx = (ModelContext*)model_context;\n"
         << "    printf(\"Running edge model: %s\\n\", ctx->name);\n";
    if (graph) {
        impl << "    if (num_inputs != " << graph->getInputNodes().size() << " || num_outputs != " << graph->getOutputNodes().size() << ") return -1;\n";
    } else {
        impl << "    // graph pointer is null, input/output count check skipped.\n";
    }
    impl << "    // TODO: Implement actual inference using inputs, weights, and write to outputs\n"
         << "    return 0;\n}\n";
    impl << "void edge_model_cleanup(void* model_context) {\n"
         << "    if (!model_context) return;\n"
         << "    ModelContext* ctx = (ModelContext*)model_context;\n"
         << "    printf(\"Cleaning up edge model: %s\\n\", ctx->name);\n"
         << "    free(model_context);\n}\n";
    impl.close();
    spdlog::info("Generated C source files (edge_model.c, edge_model.h) in {}", outputPath);
}

void EdgeBackend::generateTargetModelFile(std::shared_ptr<ir::ComputationGraph> graph, 
                                      const std::string& outputPath,
                                      const CodeGenOptions& options) {
    std::string model_format = "tflite"; // Default
    if (options.count("model_format")) {
        model_format = options.at("model_format");
    }
    spdlog::info("Generating target model file in format: {} for Edge target in {}", model_format, outputPath);

    std::string model_filename = "model." + model_format;
    std::ofstream modelFile(outputPath + "/" + model_filename, std::ios::binary);
    if (!modelFile.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open file: " + model_filename);
    }
    
    // This would contain the actual model data (e.g., TFLite flatbuffer, ONNX protobuf)
    // Placeholder content depends on the format. Options could guide quantization (e.g. bits, calibration_data_path)
    modelFile << "PLACEHOLDER FOR " << model_format << " MODEL DATA. Graph name: " << (graph ? graph->getName() : "N/A");
    // Example for TFLite: invoke TensorFlow Lite Converter (either via Python script called by system() or C++ API if available)
    // Example for ONNX: use ONNX C++ library to serialize the graph
    // Example for TVM Relay: Compile Relay graph to a deployable module.
    modelFile.close();
    spdlog::info("Generated {} model file: {}", model_format, outputPath + "/" + model_filename);
}

void EdgeBackend::generateDeploymentManifest(const std::string& outputPath,
                                           const CodeGenOptions& options) {
    spdlog::debug("generateDeploymentManifest for Edge target in {} called", outputPath);
    std::ofstream manifest(outputPath + "/deployment.json");
    if (!manifest.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open file: deployment.json");
    }
    
    std::string model_format = options.count("model_format") ? options.at("model_format") : "tflite";
    std::string runtime_hint = options.count("edge_target_runtime") ? options.at("edge_target_runtime") : (model_format == "tflite" ? "tflite_micro" : "custom_c");

    manifest << "{\n"
             << "  \"name\": \"OpenOptimizer Edge Model\",\n"
             << "  \"version\": \"1.0.1\",\n"
             << "  \"description\": \"Edge-optimized neural network model (Options: ";
    for(const auto& [key, val] : options) { manifest << key << "=" << val << "; "; }
    manifest << ")\",\n"
             << "  \"files\": [\n"
             << "    {\"name\": \"model."<< model_format << "\", \"type\": \"model\", \"target\": \"" << model_format << "\"},
"
             << "    {\"name\": \"edge_model.c\", \"type\": \"source\", \"target\": \"c\"},
"
             << "    {\"name\": \"edge_model.h\", \"type\": \"header\", \"target\": \"c\"}\n"
             << "  ],\n"
             << "  \"runtime_hint\": \"" << runtime_hint << "\",\n"
             << "  \"generated_by\": \"OpenOptimizer EdgeBackend\"\n"
             << "}\n";
    manifest.close();
    spdlog::info("Generated deployment.json in {}", outputPath);
}

void EdgeBackend::packageForEdgeRuntime(const std::string& outputPath,
                                        const CodeGenOptions& options) {
    spdlog::info("Packaging for edge runtime in path: {}. This is a placeholder.", outputPath);
    // Implementation would depend heavily on the target edge runtime specified in options.
    // Examples:
    // 1. For TFLite Micro: Could involve generating a CMake project or Makefile that integrates
    //    the generated C code with the TFLM library, or zipping relevant files.
    // 2. For TVM Micro (MicroTVM): Would involve using TVM's Python scripts to create a 
    //    deployable module, possibly flashing to a device if specified and set up.
    // 3. For specific SDKs (e.g., for an NPU): Might involve invoking the SDK's packaging tools.

    std::string target_sdk = options.count("edge_sdk") ? options.at("edge_sdk") : "none";
    if (target_sdk != "none") {
        spdlog::info("Targeting SDK: {}. Packaging logic would go here.", target_sdk);
        // e.g., std::system(("some_sdk_packaging_tool -i " + outputPath + " -o " + outputPath + "/package --options_from_codegen").c_str());
    } else {
        spdlog::warn("No specific edge_sdk provided in options. Generic packaging (if any) or manual steps required.");
    }
    // Create a dummy package file
    std::ofstream package_info(outputPath + "/PACKAGED_FOR_EDGE.txt");
    package_info << "This directory would contain packaged artifacts for an edge runtime.\n";
    package_info << "Options used:\n";
    for(const auto& [key, val] : options) {
        package_info << key << " = " << val << "\n";
    }
    package_info.close();
}

} // namespace edge
} // namespace codegen
} // namespace openoptimizer
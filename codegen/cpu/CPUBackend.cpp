#include "codegen/cpu/CPUBackend.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <system_error>
#include <filesystem>
#include <cstdlib> // For std::system

namespace openoptimizer {
namespace codegen {
namespace cpu {

CPUBackend::CPUBackend() {
    spdlog::info("CPU backend initialized. Call getName() to see its registered TargetName.");
}

CPUBackend::~CPUBackend() {
    spdlog::info("CPU backend destroyed");
}

TargetName CPUBackend::getName() const {
    return CPU_TARGET; // Defined in CodeGenerator.hpp
}

void CPUBackend::generate(std::shared_ptr<ir::ComputationGraph> graph, 
                        const std::string& outputPath,
                        const CodeGenOptions& options) {
    spdlog::info("Generating CPU code at \"{}\" with backend \"{}\"", outputPath, getName());
    if (!options.empty()) {
        std::string opts_str;
        for(const auto& [key, val] : options) {
            opts_str += key + "=\'" + val + "\', ";
        }
        if (!opts_str.empty()) {
            opts_str.resize(opts_str.length() - 2); // Remove trailing comma and space
        }
        spdlog::info("CPUBackend options: {}", opts_str);
    }

    try {
        std::filesystem::create_directories(outputPath);
        generateCppCode(graph, outputPath, options);
        generateCMakeLists(outputPath, options);
        
        // Check for an option to trigger compilation, e.g., {"compile": "true"}
        bool should_compile = false;
        auto compile_it = options.find("compile");
        if (compile_it != options.end() && compile_it->second == "true") {
            should_compile = true;
        }

        if (should_compile) {
            compileCode(outputPath, options);
            spdlog::info("CPU code compilation requested and attempted.");
        } else {
            spdlog::info("Compilation not requested via options. Skipping compile step.");
        }
        
        spdlog::info("CPU code generation completed successfully for path: {}", outputPath);
    } catch (const std::exception& e) {
        spdlog::error("Error during CPU code generation for path {}: {}", outputPath, e.what());
        throw;
    }
}

void CPUBackend::generateCppCode(std::shared_ptr<ir::ComputationGraph> graph, 
                               const std::string& outputPath,
                               const CodeGenOptions& options) {
    spdlog::debug("generateCppCode for {} called.", outputPath);
    // Generate model.h header file
    std::ofstream header(outputPath + "/model.h");
    if (!header.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: model.h");
    }
    
    header << "#pragma once\n\n"
           << "#include <vector>\n"
           << "#include <cstdint>\n\n"
           << "// Generated with options: ";
    for(const auto& [key, val] : options) {
        header << key << "=\"" << val << "\"; ";
    }
    header << "\n"
           << "namespace generated_model {\n\n"
           << "// Forward declarations\n"
           << "class Tensor;\n\n"
           << "class Model {\n"
           << "public:\n"
           << "    Model();\n"
           << "    ~Model();\n\n"
           << "    // Run the entire model\n"
           << "    std::vector<Tensor> run(const std::vector<Tensor>& inputs);\n\n"
           << "private:\n";
    
    if (graph) { // Check if graph is valid
        for (const auto& node : graph->getNodes()) {
            if (!node) continue;
            header << "    // " << node->getName() << "\n"
                   << "    Tensor op_" << node->getName() << "("; // Prefix to avoid conflicts with keywords/member names
            
            bool first = true;
            for (const auto& input_edge : node->getInputs()) {
                if (!input_edge || !input_edge->getProducer()) continue;
                if (!first) {
                    header << ", ";
                }
                header << "const Tensor& " << input_edge->getProducer()->getName();
                first = false;
            }
            header << ");\n\n";
        }
    }
    header << "};\n\n"
           << "} // namespace generated_model\n";
    header.close();
    
    std::ofstream impl(outputPath + "/model.cpp");
    if (!impl.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: model.cpp");
    }
    impl << "#include \"model.h\"\n"
         << "#include <stdexcept>\n"
         << "#include <iostream>\n\n"
         << "// Generated with options: ";
    for(const auto& [key, val] : options) {
        impl << key << "=\"" << val << "\"; ";
    }
    impl << "\n"
         << "namespace generated_model {\n\n"
         << "class Tensor { public: Tensor() {} };\n\n"
         << "Model::Model() { std::cout << \"Generated Model Initialized (CPU)\" << std::endl; }\n"
         << "Model::~Model() { std::cout << \"Generated Model Destroyed (CPU)\" << std::endl; }\n\n"
         << "std::vector<Tensor> Model::run(const std::vector<Tensor>& inputs) {\n";
    if (graph && !graph->getInputNodes().empty()) {
        impl << "    if (inputs.size() != " << graph->getInputNodes().size() << ") {\n"
             << "        throw std::invalid_argument(\"Expected " << graph->getInputNodes().size() << " inputs\");\n"
             << "    }\n";
    } else {
        impl << "    // Graph has no defined input nodes or graph is null.\n";
    }
    impl << "    std::cout << \"Running generated model...\" << std::endl;\n"
         << "    // TODO: Actual graph execution logic with proper data flow\n"
         << "    std::vector<Tensor> outputs; return outputs;\n"
         << "}\n\n";
    if (graph) { // Check if graph is valid
        for (const auto& node : graph->getNodes()) {
            if (!node) continue;
            impl << "Tensor Model::op_" << node->getName() << "(";
            bool first = true;
            for (const auto& input_edge : node->getInputs()) {
                if (!input_edge || !input_edge->getProducer()) continue;
                if (!first) {
                    impl << ", ";
                }
                impl << "const Tensor& " << input_edge->getProducer()->getName();
                first = false;
            }
            impl << ") {\n"
                 << "    // Implementation for " << node->getOperation()->getType() << " operation\n"
                 << "    std::cout << \"Executing op: " << node->getName() << " (type: " << node->getOperation()->getType() << ")\" << std::endl;\n"
                 << "    // TODO: Generate actual operation implementation\n"
                 << "    return Tensor();\n"
                 << "}\n\n";
        }
    }
    impl << "} // namespace generated_model\n";
    impl.close();
    spdlog::info("Generated C++ source files (model.h, model.cpp) in {}", outputPath);
}

void CPUBackend::generateCMakeLists(const std::string& outputPath,
                                  const CodeGenOptions& options) {
    spdlog::debug("generateCMakeLists for {} called.", outputPath);
    std::ofstream cmake(outputPath + "/CMakeLists.txt");
    if (!cmake.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: CMakeLists.txt");
    }
    
    cmake << "cmake_minimum_required(VERSION 3.15)\n"
          << "project(GeneratedCpuModel CXX)\n\n"
          << "set(CMAKE_CXX_STANDARD 20)\n"
          << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
          << "set(CMAKE_CXX_EXTENSIONS OFF)\n\n"
          << "# Generated with options: ";
    for(const auto& [key, val] : options) {
        cmake << key << "=\"" << val << "\"; ";
    }
    cmake << "\n\n"
          << "add_library(generated_model SHARED model.cpp model.h)\n\n"
          << "# Example executable target\n"
          << "add_executable(model_runner main.cpp)\n"
          << "target_link_libraries(model_runner PRIVATE generated_model)\n\n"
          << "# Installation (optional)
"
          << "# install(TARGETS generated_model DESTINATION lib)
"
          << "# install(FILES model.h DESTINATION include)
";
    cmake.close();

    // Create a dummy main.cpp for the executable
    std::ofstream main_cpp(outputPath + "/main.cpp");
    if (!main_cpp.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: main.cpp");
    }
    main_cpp << "#include \"model.h\"\n"
             << "#include <iostream>\n"
             << "#include <vector>\n\n"
             << "int main() {\n"
             << "    try {\n"
             << "        generated_model::Model model;\n"
             << "        std::cout << \"Created model instance.\" << std::endl;\n"
             << "        // Create dummy inputs (adjust size based on your model's actual input nodes)\n"
             << "        std::vector<generated_model::Tensor> inputs;\n"
             // << "        inputs.emplace_back(); // Add dummy tensors as needed \n"
             << "        std::cout << \"Calling model.run()...\" << std::endl;\n"
             << "        std::vector<generated_model::Tensor> outputs = model.run(inputs);\n"
             << "        std::cout << \"Model execution finished. Outputs: \" << outputs.size() << std::endl;\n"
             << "    } catch (const std::exception& e) {\n"
             << "        std::cerr << \"Exception in model_runner: \" << e.what() << std::endl;\n"
             << "        return 1;\n"
             << "    }\n"
             << "    return 0;\n"
             << "}\n";
    main_cpp.close();

    spdlog::info("Generated CMakeLists.txt and main.cpp in {}", outputPath);
}

void CPUBackend::compileCode(const std::string& outputPath,
                             const CodeGenOptions& options) {
    spdlog::info("Attempting to compile generated code in {}", outputPath);
    
    std::string build_type = "Release";
    auto build_type_it = options.find("build_type");
    if (build_type_it != options.end()) {
        build_type = build_type_it->second;
    }

    std::string cmake_generator_cmd = "cmake ../"; // Assuming outputPath is 'build/output', so ../ is project root
    // For more robustness, cmake_path should be outputPath
    std::string cmake_path = outputPath;
    std::string build_path = outputPath + "/build_generated"; // Separate build dir for generated code
    std::filesystem::create_directories(build_path);

    std::string config_cmd = "cmake -S \"" + cmake_path + "\" -B \"" + build_path + "\"";
    std::string build_cmd = "cmake --build \"" + build_path + "\" --config " + build_type + " --parallel";
    // Add specific generator if needed, e.g. -G "Ninja" based on options or system check

    spdlog::info("Configure command: {}", config_cmd);
    int config_ret = std::system(config_cmd.c_str());
    if (config_ret != 0) {
        spdlog::error("CMake configuration failed with exit code: {}", config_ret);
        throw std::runtime_error("CMake configuration failed for generated code in " + outputPath);
    }
    
    spdlog::info("Build command: {}", build_cmd);
    int build_ret = std::system(build_cmd.c_str());
    if (build_ret != 0) {
        spdlog::error("CMake build failed with exit code: {}", build_ret);
        throw std::runtime_error("CMake build failed for generated code in " + outputPath);
    }
    
    spdlog::info("Compilation in {} completed (or attempted).", outputPath);
}

} // namespace cpu
} // namespace codegen
} // namespace openoptimizer

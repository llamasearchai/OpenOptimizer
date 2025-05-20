#include "codegen/gpu/GPUBackend.hpp"
#include <spdlog/spdlog.h>
#include <fstream>
#include <system_error>
#include <filesystem>
#include <cstdlib> // For std::system

namespace openoptimizer {
namespace codegen {
namespace gpu {

GPUBackend::GPUBackend() {
    spdlog::info("GPU (CUDA) backend initialized. Call getName() to see its registered TargetName.");
}

GPUBackend::~GPUBackend() {
    spdlog::info("GPU (CUDA) backend destroyed");
}

TargetName GPUBackend::getName() const {
    return CUDA_TARGET; // Defined in CodeGenerator.hpp
}

void GPUBackend::generate(std::shared_ptr<ir::ComputationGraph> graph, 
                        const std::string& outputPath,
                        const CodeGenOptions& options) {
    spdlog::info("Generating GPU (CUDA) code at \"{}\" with backend \"{}\"", outputPath, getName());
    if (!options.empty()) {
        std::string opts_str;
        for(const auto& [key, val] : options) {
            opts_str += key + "=\'" + val + "\', ";
        }
        if (!opts_str.empty()) {
            opts_str.resize(opts_str.length() - 2);
        }
        spdlog::info("GPUBackend options: {}", opts_str);
    }

    try {
        std::filesystem::create_directories(outputPath);
        generateCudaCode(graph, outputPath, options);
        generateCMakeLists(outputPath, options);

        bool should_compile = false;
        auto compile_it = options.find("compile");
        if (compile_it != options.end() && compile_it->second == "true") {
            should_compile = true;
        }

        if (should_compile) {
            compileCudaCode(outputPath, options);
            spdlog::info("GPU (CUDA) code compilation requested and attempted.");
        } else {
            spdlog::info("Compilation not requested for GPU. Skipping compile step.");
        }
        
        spdlog::info("GPU (CUDA) code generation completed successfully for path: {}", outputPath);
    } catch (const std::exception& e) {
        spdlog::error("Error during GPU (CUDA) code generation for path {}: {}", outputPath, e.what());
        throw;
    }
}

void GPUBackend::generateCudaCode(std::shared_ptr<ir::ComputationGraph> graph, 
                                const std::string& outputPath,
                                const CodeGenOptions& options) {
    spdlog::debug("generateCudaCode for {} called.", outputPath);
    // Example: Use an option to set CUDA architecture for code gen if TVM/other tool needs it
    // std::string cuda_arch = options.count("cuda_arch") ? options.at("cuda_arch") : "sm_70"; 

    std::ofstream header(outputPath + "/model.h");
    if (!header.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: model.h");
    }
    header << "#pragma once\n\n"
           << "#include <vector>\n#include <cstdint>\n#include <cuda_runtime.h>\n#include <stdexcept>\n#include <string>\n\n"
           << "// Generated with options: ";
    for(const auto& [key, val] : options) { header << key << "=\"" << val << "\"; "; }
    header << "\n"
           << "namespace generated_model {\n"
           << "class GpuTensor { public: float* data_ = nullptr; size_t size_ = 0; GpuTensor(){} \n"
           << "  GpuTensor(size_t elements) : size_(elements) { cudaMalloc(&data_, size_ * sizeof(float)); }\n"
           << "  ~GpuTensor() { if(data_) cudaFree(data_); } \n"
           << "  // Add copy constructors/assignment operators as needed \n"
           << "};\n\n"
           << "class Model {\npublic:\n    Model();\n    ~Model();\n    std::vector<GpuTensor> run(const std::vector<GpuTensor>& inputs);\nprivate:\n";
    if (graph) {
        for (const auto& node : graph->getNodes()) {
            if (!node) continue;
            header << "    GpuTensor op_" << node->getName() << "(";
            bool first = true;
            for (const auto& input_edge : node->getInputs()) {
                if (!input_edge || !input_edge->getProducer()) continue;
                if (!first) header << ", ";
                header << "const GpuTensor& " << input_edge->getProducer()->getName();
                first = false;
            }
            header << ");\n";
        }
    }
    header << "};\n} // namespace generated_model\n";
    header.close();

    std::ofstream impl(outputPath + "/model.cu");
    if (!impl.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: model.cu");
    }
    impl << "#include \"model.h\"\n#include <iostream>\n\n"
         << "// Generated with options: ";
    for(const auto& [key, val] : options) { impl << key << "=\"" << val << "\"; "; }
    impl << "\n"
         << "namespace generated_model {\n"
         << "#define CUDA_CHECK(call) do { \
                 cudaError_t err = call; \
                 if (err != cudaSuccess) { \
                     fprintf(stderr, \"CUDA Error in %s at line %d: %s\\n\", __FILE__, __LINE__, cudaGetErrorString(err)); \
                     throw std::runtime_error(std::string(\"CUDA error: \") + cudaGetErrorString(err)); \
                 } \
             } while(0)\n\n"
         << "Model::Model() { CUDA_CHECK(cudaSetDevice(0)); std::cout << \"Generated Model Initialized (GPU/CUDA)\" << std::endl; }\n"
         << "Model::~Model() { std::cout << \"Generated Model Destroyed (GPU/CUDA)\" << std::endl; /* cudaDeviceReset(); may be too aggressive here */ }\n\n"
         << "std::vector<GpuTensor> Model::run(const std::vector<GpuTensor>& inputs) {\n";
    if (graph && !graph->getInputNodes().empty()) {
        impl << "    if (inputs.size() != " << graph->getInputNodes().size() << ") { "
             << "throw std::invalid_argument(\"Expected " << graph->getInputNodes().size() << " inputs\"); }\n";
    } else {
        impl << "    // Graph has no defined input nodes or graph is null.\n";
    }
    impl << "    std::cout << \"Running generated GPU model...\" << std::endl;\n"
         << "    // TODO: Actual graph execution logic with proper data flow & kernel launches\n"
         << "    std::vector<GpuTensor> outputs; return outputs;\n"
         << "}\n\n";

    if (graph) {
        for (const auto& node : graph->getNodes()) {
            if (!node) continue;
            // Kernel definition
            impl << "__global__ void kernel_" << node->getName() << "(/* params */) {\n"
                 << "    // Kernel for " << node->getOperation()->getType() << " on node " << node->getName() << "\n"
                 << "    // printf(\"Executing kernel_" << node->getName() << " on block %u thread %u\\n\", blockIdx.x, threadIdx.x); \n"
                 << "}\n\n";
            // Host function calling the kernel
            impl << "GpuTensor Model::op_" << node->getName() << "(";
            bool first = true;
            for (const auto& input_edge : node->getInputs()) {
                if (!input_edge || !input_edge->getProducer()) continue;
                if (!first) impl << ", ";
                impl << "const GpuTensor& " << input_edge->getProducer()->getName();
                first = false;
            }
            impl << ") {\n"
                 << "    std::cout << \"Executing op: " << node->getName() << " (type: " << node->getOperation()->getType() << ") on GPU\" << std::endl;\n"
                 << "    dim3 numBlocks(1); dim3 threadsPerBlock(256);\n"
                 << "    kernel_" << node->getName() << "<<<numBlocks, threadsPerBlock>>>(/* params */);\n"
                 << "    CUDA_CHECK(cudaGetLastError());\n"
                 << "    CUDA_CHECK(cudaDeviceSynchronize());\n"
                 << "    GpuTensor result; /* TODO: allocate and manage result tensor */ return result;\n"
                 << "}\n\n";
        }
    }
    impl << "} // namespace generated_model\n";
    impl.close();
    spdlog::info("Generated CUDA source files (model.h, model.cu) in {}", outputPath);

    std::ofstream main_cpp(outputPath + "/main.cpp");
    if (!main_cpp.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open output file: main.cpp (GPU)");
    }
    main_cpp << "#include \"model.h\"\n#include <iostream>\n#include <vector>\n\n"
             << "int main() {\n    try {\n        generated_model::Model model;\n        std::cout << \"Created GPU model instance.\" << std::endl;\n"
             << "        std::vector<generated_model::GpuTensor> inputs; /* TODO: Init with actual data */ \n"
             << "        std::cout << \"Calling model.run()...\" << std::endl;\n"
             << "        std::vector<generated_model::GpuTensor> outputs = model.run(inputs);\n"
             << "        std::cout << \"GPU Model execution finished. Outputs: \" << outputs.size() << std::endl;\n"
             << "    } catch (const std::exception& e) { std::cerr << \"Exception: \" << e.what() << std::endl; return 1; }\n"
             << "    return 0;\n}";
    main_cpp.close();
    spdlog::info("Generated main.cpp for GPU model runner in {}", outputPath);
}

void GPUBackend::generateCMakeLists(const std::string& outputPath,
                                  const CodeGenOptions& options) {
    spdlog::debug("generateCMakeLists for GPU target in {} called.", outputPath);
    std::string cuda_arch_flag;
    if (options.count("cuda_arch")) {
        cuda_arch_flag = "-arch=compute_" + options.at("cuda_arch") + " -code=sm_" + options.at("cuda_arch");
        // For newer CMake, target_compile_options(tgt PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_XX>)
        // or set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86) etc.
        // This old way is simpler for direct injection if not using modern CMake CUDA features.
    }

    std::ofstream cmake(outputPath + "/CMakeLists.txt");
    if (!cmake.is_open()) {
        throw std::system_error(errno, std::system_category(), "Failed to open file: CMakeLists.txt (GPU)");
    }
    cmake << "cmake_minimum_required(VERSION 3.18) # Bump for CUDA language support improvements\n"
          << "project(GeneratedGpuModel LANGUAGES CXX CUDA)\n\n"
          << "set(CMAKE_CXX_STANDARD 20)\n"
          << "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
          << "set(CMAKE_CUDA_STANDARD 20) # Try C++20 for CUDA too if compiler supports
"
          << "set(CMAKE_CUDA_STANDARD_REQUIRED ON)\n\n"
          << "# Find CUDA package, should be found by CMake if installed\n"
          << "find_package(CUDA REQUIRED)\n
"
          << "# Generated with options: ";
    for(const auto& [key, val] : options) { cmake << key << "=\"" << val << "\"; "; }
    cmake << "\n\n"
          << "add_library(generated_gpu_model SHARED model.cu model.h)\n
"
          << "# Modern way to set CUDA arch if CMake > 3.18
"
          << "# set(CMAKE_CUDA_ARCHITECTURES 75) # Example: 70 75 80 86 etc. based on options or detection
"
          << "# Or, more traditionally for older CMakes or direct control:
"
          << "if(NOT CMAKE_CUDA_ARCHITECTURES AND EXISTS \"${CMAKE_CUDA_COMPILER}\")\n"
          << "    set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} " << (cuda_arch_flag.empty() ? "-arch=sm_70" : cuda_arch_flag) << "\")\n"
          << "    # For separated compilation (recommended for larger projects)
"
          << "    # set_target_properties(generated_gpu_model PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
"
          << "endif()\n\n"
          << "add_executable(gpu_model_runner main.cpp)\n"
          << "target_link_libraries(gpu_model_runner PRIVATE generated_gpu_model CUDA::cudart)\n\n"
          << "# Optional install
"
          << "# install(TARGETS generated_gpu_model DESTINATION lib)
"
          << "# install(FILES model.h DESTINATION include)
";
    cmake.close();
    spdlog::info("Generated CMakeLists.txt for GPU model in {}", outputPath);
}

void GPUBackend::compileCudaCode(const std::string& outputPath,
                                 const CodeGenOptions& options) {
    spdlog::info("Attempting to compile generated CUDA code in {}", outputPath);
    std::string build_type = "Release";
    if (options.count("build_type")) build_type = options.at("build_type");

    std::string cmake_path = outputPath;
    std::string build_path = outputPath + "/build_generated_gpu"; 
    std::filesystem::create_directories(build_path);

    std::string config_cmd = "cmake -S \"" + cmake_path + "\" -B \"" + build_path + "\"";
    // Add generator if needed, e.g., -G Ninja
    if (options.count("cmake_generator")) {
        config_cmd += " -G \"" + options.at("cmake_generator") + "\"";
    }
    std::string build_cmd = "cmake --build \"" + build_path + "\" --config " + build_type + " --parallel";

    spdlog::info("CUDA Configure command: {}", config_cmd);
    int config_ret = std::system(config_cmd.c_str());
    if (config_ret != 0) {
        spdlog::error("CUDA CMake configuration failed (code {}): {}", config_ret, config_cmd);
        throw std::runtime_error("CMake configuration failed for generated GPU code in " + outputPath);
    }
    
    spdlog::info("CUDA Build command: {}", build_cmd);
    int build_ret = std::system(build_cmd.c_str());
    if (build_ret != 0) {
        spdlog::error("CUDA CMake build failed (code {}): {}", build_ret, build_cmd);
        throw std::runtime_error("CMake build failed for generated GPU code in " + outputPath);
    }
    spdlog::info("CUDA code compilation in {} completed (or attempted).", outputPath);
}

} // namespace gpu
} // namespace codegen
} // namespace openoptimizer
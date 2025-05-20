cmake_minimum_required(VERSION 3.20)
project(OpenOptimizer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build type: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif()

# For development: where to put the final .so/_cpp_extension.pyd so Python can find it
# This should ideally match a location in your Python package structure or sys.path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/openoptimizer)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/openoptimizer) # For DLLs on Windows

# --- Options ---
option(OPENOPTIMIZER_WITH_CUDA "Enable CUDA support" ON)
option(OPENOPTIMIZER_WITH_QT_GUI "Build Qt based GUI Visualizer" ON)
option(OPENOPTIMIZER_BUILD_TESTS "Build tests" ON)
option(OPENOPTIMIZER_BUILD_DOCS "Build Doxygen documentation" ON)
option(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS "Build Python bindings" ON)

# --- Find Dependencies ---

# MLIR
find_package(MLIR REQUIRED)
message(STATUS "Found MLIR: ${MLIR_INCLUDE_DIRS} | ${MLIR_LIBRARIES}")

# TVM
# Note: TVM's CMake find script might need specific hints or be in CMAKE_PREFIX_PATH
find_package(TVM 0.13.0)
if(NOT TVM_FOUND)
    message(WARNING "TVM 0.13.0 not found. Some features might be unavailable. Searched TVM_DIR: ${TVM_DIR}")
else()
    message(STATUS "Found TVM: ${TVM_VERSION} | Includes: ${TVM_INCLUDE_DIRS} | Libs: ${TVM_LIBRARIES}")
endif()

# spdlog
find_package(spdlog 1.12.0 REQUIRED)
message(STATUS "Found spdlog: ${spdlog_INCLUDE_DIRS} | ${spdlog_LIBRARIES}")

# CUDA
if(OPENOPTIMIZER_WITH_CUDA)
    find_package(CUDA 12.2)
    if(CUDA_FOUND)
        message(STATUS "Found CUDA: ${CUDA_TOOLKIT_ROOT_DIR} (version ${CUDA_VERSION_STRING})")
        list(APPEND CMAKE_CUDA_FLAGS "--expt-extended-lambda") # Example flag
    else()
        message(WARNING "CUDA 12.2 not found, but OPENOPTIMIZER_WITH_CUDA is ON. Disabling CUDA support.")
        set(OPENOPTIMIZER_WITH_CUDA OFF)
    endif()
endif()

# Qt6 for Visualization
if(OPENOPTIMIZER_WITH_QT_GUI)
    find_package(Qt6 6.5.2 COMPONENTS Core Gui Widgets Qml Quick REQUIRED)
    if (Qt6_FOUND)
        message(STATUS "Found Qt6: ${Qt6_DIR}")
        set(CMAKE_AUTOMOC ON)
        set(CMAKE_AUTORCC ON)
        set(CMAKE_AUTOUIC ON)
    else()
        message(WARNING "Qt6 not found, OPENOPTIMIZER_WITH_QT_GUI is ON. Disabling Qt GUI support.")
        set(OPENOPTIMIZER_WITH_QT_GUI OFF)
    endif()
endif()

# Find Python and Pybind11 if Python bindings are enabled
if(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 REQUIRED)
    if (Python_FOUND AND pybind11_FOUND)
        message(STATUS "Python Interpreter: ${Python_EXECUTABLE}")
        message(STATUS "Python Includes: ${Python_INCLUDE_DIRS}")
        message(STATUS "Pybind11 Includes: ${pybind11_INCLUDE_DIRS}")
        # These variables (Pybind11_FOUND, PythonLibs_FOUND) will be available to subdirectories
        # Set Pybind11_FOUND and PythonLibs_FOUND for frontend/cpp/CMakeLists.txt
        set(Pybind11_FOUND TRUE)
        set(PythonLibs_FOUND TRUE) # Python_FOUND implies PythonLibs_FOUND for this purpose
    else()
        message(WARNING "Python or Pybind11 not found, though OPENOPTIMIZER_BUILD_PYTHON_BINDINGS is ON. Disabling Python bindings.")
        set(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS OFF)
        set(Pybind11_FOUND FALSE)
        set(PythonLibs_FOUND FALSE)
    endif()
else()
    message(STATUS "Python bindings disabled (OPENOPTIMIZER_BUILD_PYTHON_BINDINGS is OFF).")
    set(Pybind11_FOUND FALSE)
    set(PythonLibs_FOUND FALSE)
endif()

# --- Aggregate Sources for Core Library ---
# Initialize source lists that subdirectories will append to
set(OPENOPTIMIZER_CORE_LIB_SOURCES "") # This variable will be populated by subdirectories
# OPENOPTIMIZER_FRONTEND_CPP_SOURCES is also populated by its subdir and then added to CORE_LIB_SOURCES

# --- Add Subdirectories ---
# Each subdirectory's CMakeLists.txt should append its sources to OPENOPTIMIZER_CORE_LIB_SOURCES
# and its public headers to OPENOPTIMIZER_CORE_LIB_HEADERS (or just manage include dirs).

add_subdirectory(ir) 
add_subdirectory(optimization) 
add_subdirectory(codegen)    
add_subdirectory(frontend/cpp) # This will append OPENOPTIMIZER_FRONTEND_CPP_SOURCES to OPENOPTIMIZER_CORE_LIB_SOURCES
# add_subdirectory(runtime) # When added

# --- Core Library Target (openoptimizer_core) ---
add_library(openoptimizer_core SHARED 
    ${OPENOPTIMIZER_CORE_LIB_SOURCES}
)

target_include_directories(openoptimizer_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}> # For top-level includes if any
    $<INSTALL_INTERFACE:include> # For installed headers under <prefix>/include
    # Public include directories from components:
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ir> # Allows #include "graph/ComputationGraph.hpp"
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ir/graph>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ir/ops>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ir/tensor>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/optimization>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/codegen>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/codegen/cpu>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/codegen/gpu>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/codegen/edge>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/frontend/cpp>
    # External dependencies (already added but good to be explicit for the library interface)
    ${spdlog_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

if(TVM_FOUND)
    target_include_directories(openoptimizer_core PUBLIC ${TVM_INCLUDE_DIRS})
endif()

# Link against dependencies
target_link_libraries(openoptimizer_core PUBLIC
    spdlog::spdlog
    ${MLIR_LIBRARIES}
)

if(TVM_FOUND)
    target_link_libraries(openoptimizer_core PUBLIC ${TVM_LIBRARIES})
endif()

if(OPENOPTIMIZER_WITH_CUDA AND CUDA_FOUND)
    target_compile_definitions(openoptimizer_core PUBLIC WITH_CUDA)
    # If openoptimizer_core itself has .cu files, it needs CUDA as a language.
    # If it only links to CUDA-dependent libraries (like TVM with CUDA), this is enough.
    # For linking CUDA runtime: target_link_libraries(openoptimizer_core PUBLIC CUDA::cudart)
endif()

set_target_properties(openoptimizer_core PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON 
    POSITION_INDEPENDENT_CODE ON # Good for shared libraries
)

if(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS AND Pybind11_FOUND AND Python_FOUND)
    message(STATUS "Python bindings module (_cpp_extension) will be built by frontend/cpp/CMakeLists.txt")
    # Ensure _cpp_extension (built in frontend/cpp) can find openoptimizer_core
    # This is often handled by the linker if both are in CMAKE_LIBRARY_OUTPUT_DIRECTORY
    # or by setting RPATH for _cpp_extension on Linux/macOS.
    # Example for RPATH (might be needed in frontend/cpp/CMakeLists.txt for _cpp_extension):
    # set_target_properties(_cpp_extension PROPERTIES
    #    INSTALL_RPATH "$ORIGIN" # Or path to where openoptimizer_core.so is installed
    #    BUILD_WITH_INSTALL_RPATH TRUE
    # )
endif()

# --- Visualization Application (Optional) ---
if(OPENOPTIMIZER_WITH_QT_GUI AND Qt6_FOUND)
    add_subdirectory(visualization/desktop)
else()
    message(STATUS "Qt GUI Visualizer disabled or Qt6 not found.")
endif()

# --- Testing (Optional) ---
if(OPENOPTIMIZER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
else()
    message(STATUS "Tests disabled.")
endif()

# --- Documentation (Optional) ---
if(OPENOPTIMIZER_BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs/cpp)
        message(STATUS "Doxygen documentation build enabled.")
    else()
        message(WARNING "Doxygen not found. Cannot build C++ documentation.")
    endif()
else()
    message(STATUS "Doxygen documentation build disabled.")
endif()


# --- Installation ---
include(GNUInstallDirs) # For standard install paths like CMAKE_INSTALL_LIBDIR

# Install core library public headers from each component's include directory
# Example: if ir/graph/ComputationGraph.hpp is public, it should be in an include dir.
# Assuming a top-level 'include/openoptimizer' for installed public headers that mirror structure.
install(DIRECTORY ir/graph/ DESTINATION include/openoptimizer/ir/graph FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ir/ops/ DESTINATION include/openoptimizer/ir/ops FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ir/tensor/ DESTINATION include/openoptimizer/ir/tensor FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ir/ DESTINATION include/openoptimizer/ir FILES_MATCHING PATTERN "Operation.hpp") # Operation.hpp is directly in ir/

install(DIRECTORY optimization/ DESTINATION include/openoptimizer/optimization FILES_MATCHING PATTERN "*.hpp")
# Add optimization/passes/ if there are public C++ pass headers

install(DIRECTORY codegen/ DESTINATION include/openoptimizer/codegen FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY codegen/cpu/ DESTINATION include/openoptimizer/codegen/cpu FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY codegen/gpu/ DESTINATION include/openoptimizer/codegen/gpu FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY codegen/edge/ DESTINATION include/openoptimizer/codegen/edge FILES_MATCHING PATTERN "*.hpp")

install(DIRECTORY frontend/cpp/ DESTINATION include/openoptimizer/frontend/cpp FILES_MATCHING PATTERN "*.hpp")


install(TARGETS openoptimizer_core
    EXPORT OpenOptimizerTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} 
)


include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/OpenOptimizerConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/OpenOptimizerConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenOptimizer 
    PATH_VARS CMAKE_INSTALL_PREFIX CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/OpenOptimizerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/OpenOptimizerConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/OpenOptimizerConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenOptimizer
)

install(EXPORT OpenOptimizerTargets
    FILE OpenOptimizerTargets.cmake
    NAMESPACE OpenOptimizer::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OpenOptimizer
)


message(STATUS "OpenOptimizer Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tests: ${OPENOPTIMIZER_BUILD_TESTS}")
message(STATUS "  Build Docs: ${OPENOPTIMIZER_BUILD_DOCS}")
message(STATUS "  CUDA Support: ${OPENOPTIMIZER_WITH_CUDA}")
if(OPENOPTIMIZER_WITH_CUDA AND CUDA_FOUND)
    message(STATUS "    CUDA Version: ${CUDA_VERSION_STRING}")
endif()
message(STATUS "  Qt GUI Visualizer: ${OPENOPTIMIZER_WITH_QT_GUI}")
if(OPENOPTIMIZER_WITH_QT_GUI AND Qt6_FOUND)
    message(STATUS "    Qt Version: ${Qt6_VERSION_STRING}")
endif()
message(STATUS "  TVM Found: ${TVM_FOUND}")
if(TVM_FOUND)
    message(STATUS "    TVM Version: ${TVM_VERSION_STRING}") 
endif()
message(STATUS "  Build Python Bindings: ${OPENOPTIMIZER_BUILD_PYTHON_BINDINGS}")
if(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS AND Python_FOUND AND pybind11_FOUND)
    message(STATUS "    Python Executable: ${Python_EXECUTABLE}")
    message(STATUS "    Pybind11 Version: ${pybind11_VERSION}")
endif()


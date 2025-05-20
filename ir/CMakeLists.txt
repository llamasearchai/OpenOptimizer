# ir/CMakeLists.txt
# This file defines the C++ sources for the IR component.

set(OPENOPTIMIZER_IR_COMPONENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/graph/ComputationGraph.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/ops/StandardOps.cpp # Add if/when it has non-inline definitions
    # ${CMAKE_CURRENT_SOURCE_DIR}/tensor/TensorDescriptor.cpp # Add if/when it has non-inline definitions
)

# Append these component sources to the global list for the core library
set(OPENOPTIMIZER_CORE_LIB_SOURCES ${OPENOPTIMIZER_CORE_LIB_SOURCES} ${OPENOPTIMIZER_IR_COMPONENT_SOURCES} PARENT_SCOPE)

# Public headers for the IR component that openoptimizer_core and bindings might need
# These are typically added to target_include_directories of openoptimizer_core in the root CMakeLists.txt
# For example, the root CMake would have:
# target_include_directories(openoptimizer_core PUBLIC ${CMAKE_SOURCE_DIR}/ir)
# (or more granularly ${CMAKE_SOURCE_DIR}/ir/graph, ${CMAKE_SOURCE_DIR}/ir/ops, etc.) 
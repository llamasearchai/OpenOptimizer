# optimization/CMakeLists.txt
# This file defines the C++ sources for the Optimization component.

set(OPENOPTIMIZER_OPTIMIZATION_COMPONENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/OptimizationPass.cpp # Base class cpp
    # Add .cpp files for concrete C++ optimization passes here when created
    # e.g., ${CMAKE_CURRENT_SOURCE_DIR}/passes/OperatorFusionPassCpp.cpp
)

# Append to global list
set(OPENOPTIMIZER_CORE_LIB_SOURCES ${OPENOPTIMIZER_CORE_LIB_SOURCES} ${OPENOPTIMIZER_OPTIMIZATION_COMPONENT_SOURCES} PARENT_SCOPE)

# Public headers for optimization component:
# target_include_directories(openoptimizer_core PUBLIC ${CMAKE_SOURCE_DIR}/optimization) 
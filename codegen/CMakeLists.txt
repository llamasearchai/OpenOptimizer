# codegen/CMakeLists.txt
# Defines C++ sources for the CodeGen component.

set(OPENOPTIMIZER_CODEGEN_COMPONENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CodeGenerator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/cpu/CPUBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/GPUBackend.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/edge/EdgeBackend.cpp
    # Add other backend .cpp files here
)

# Append to global list
set(OPENOPTIMIZER_CORE_LIB_SOURCES ${OPENOPTIMIZER_CORE_LIB_SOURCES} ${OPENOPTIMIZER_CODEGEN_COMPONENT_SOURCES} PARENT_SCOPE)

# Public headers for codegen component:
# target_include_directories(openoptimizer_core PUBLIC ${CMAKE_SOURCE_DIR}/codegen) 
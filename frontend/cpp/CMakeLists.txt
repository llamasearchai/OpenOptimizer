cmake_minimum_required(VERSION 3.18)

# --- C++ Optimizer Library (part of openoptimizer_core) ---
# Sources for the C++ Optimizer library component (part of openoptimizer_core)
# These sources will be added to the openoptimizer_core target in the root CMakeLists.txt
set(OPENOPTIMIZER_FRONTEND_CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.cpp
    # Add other .cpp files for frontend::cpp if any
)

set(OPENOPTIMIZER_FRONTEND_CPP_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Optimizer.hpp
)

# The target_sources for openoptimizer_core in the root CMakeLists.txt
# should include ${OPENOPTIMIZER_FRONTEND_CPP_SOURCES}.
# Public headers are handled by target_include_directories for openoptimizer_core.

# Let's define a local static library for better modularity.
add_library(openoptimizer_frontend_cpp STATIC
    ${OPENOPTIMIZER_FRONTEND_CPP_SOURCES}
)

target_include_directories(openoptimizer_frontend_cpp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR} # For Optimizer.hpp
    ${CMAKE_SOURCE_DIR}         # For ir/, codegen/, optimization/ includes
)

# Link frontend_cpp library to the main core library if openoptimizer_core is defined in parent scope
# This dependency is typically handled by linking openoptimizer_core with this static library.
# The root CMakeLists.txt will link openoptimizer_core with openoptimizer_frontend_cpp.


# --- Python Bindings (pybind11) ---
# Check if Pybind11 is found (should be handled by root CMakeLists.txt or a FindPybind11.cmake module)
# For now, assume Pybind11_INCLUDE_DIRS and Pybind11_LIBRARIES are set/imported.

# Option to build Python bindings, can be controlled from root CMakeLists.txt
option(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS "Build Python bindings for OpenOptimizer" ON)

if(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS AND Pybind11_FOUND AND PythonLibs_FOUND)
    message(STATUS "Configuring Python bindings module _cpp_extension in frontend/cpp.")
    
    # The Python module itself is just the binding code.
    # It links against the main openoptimizer_core library where Optimizer.cpp is compiled.
    pybind11_add_module(_cpp_extension SHARED python_bindings.cpp)
    
    target_link_libraries(_cpp_extension PRIVATE 
        openoptimizer_core # Link against the main C++ library
        # Pybind11::module is handled by pybind11_add_module typically
        # PythonLibs_LIBRARIES might also be handled by pybind11_add_module or FindPython
        # If not, explicitly add: pybind11::embed (or pybind11::module) and Python::Python
    )
    
    target_include_directories(_cpp_extension PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR} # For python_bindings.cpp to find Optimizer.hpp etc.
        ${CMAKE_SOURCE_DIR}         # For ir/, codegen/, optimization/ includes from python_bindings.cpp
        # pybind11_INCLUDE_DIRS and PythonLibs_INCLUDE_DIRS are often handled by pybind11_add_module
        # or FindPython. If not, add them: ${pybind11_INCLUDE_DIR} ${Python_INCLUDE_DIRS}
    )
    
    set_target_properties(_cpp_extension PROPERTIES
        PREFIX "" 
        OUTPUT_NAME "_cpp_extension"
        # Ensure the output is placed where Python can find it, e.g., inside the openoptimizer package dir
        # This might require setting LIBRARY_OUTPUT_PATH or using install rules carefully.
        # For development, CMAKE_LIBRARY_OUTPUT_DIRECTORY can be set in the root CMakeLists.txt
        # to put all libraries (including this .so) in one place.
    )

    # Installation of the Python module
    # This path should align with your Python package structure.
    # For example, if your python package is `openoptimizer`, this might be `openoptimizer` directory.
    install(TARGETS _cpp_extension
            LIBRARY DESTINATION openoptimizer # Installs to <prefix>/lib/pythonX.Y/site-packages/openoptimizer usually
            RUNTIME DESTINATION openoptimizer # For Windows DLLs, if any, though less common for .pyd
    )

    message(STATUS "Python bindings module _cpp_extension configured in frontend/cpp.")

else()
    if(OPENOPTIMIZER_BUILD_PYTHON_BINDINGS)
        message(WARNING "Pybind11 or PythonLibs not found (or OPENOPTIMIZER_BUILD_PYTHON_BINDINGS is ON but they weren't found). Python bindings for frontend/cpp will NOT be built.")
    else()
        message(STATUS "Python bindings for frontend/cpp disabled (OPENOPTIMIZER_BUILD_PYTHON_BINDINGS is OFF).")
    endif()
endif() 
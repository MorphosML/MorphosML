cmake_minimum_required(VERSION 3.15)
project(morphosml VERSION 0.3.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # Compile anywhere 'PIC'

# Compiler options for make it more readable
set(CMAKE_CXX_COMPILER clang++)

add_compile_options(-fdiagnostics-color=always
        -ferror-limit=3
        -fdiagnostics-show-template-tree)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Adding modules
add_library(_core MODULE 
        cpp/src/vector.cpp
        cpp/src/matrix.cpp
        cpp/src/knn.cpp
        cpp/src/linear_regression.cpp
        cpp/src/logistic_regression.cpp
        cpp/src/data/sampler.cpp
        cpp/src/data/mmap_buffer.cpp
        cpp/src/calculus/differentiation.cpp
        cpp/src/calculus/integration.cpp
        cpp/src/calculus/limits.cpp
        cpp/src/bindings.cpp
)

target_include_directories(_core PRIVATE cpp/include cpp/include/morphosml)
target_link_libraries(_core PRIVATE pybind11::module)

set_target_properties(_core PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/morphosml
)

# Windows needs special handling
if(WIN32)
    set_target_properties(_core PROPERTIES SUFFIX ".pyd")
endif()

# Install 
install(TARGETS _core DESTINATION morphosml)

# C++ code → compiled → _core.so → imported in Python
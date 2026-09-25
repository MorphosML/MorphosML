cmake_minimum_required(VERSION 3.15)
project(morphosml VERSION 0.4.0 LANGUAGES CXX)

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

# Find OpenMP
find_package(OpenMP)

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

if(OpenMP_CXX_FOUND)
    target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compiler SIMD flags
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(_core PRIVATE -mavx2)
endif()
if(COMPILER_SUPPORTS_FMA)
    target_compile_options(_core PRIVATE -mfma)
endif()

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
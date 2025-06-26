cmake_minimum_required(VERSION 3.18)

# Project name and description
project(NBodySimulation LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA requirements
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architecture (RTX 3060 is Ampere - sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Compiler flags for different build types
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cu
    src/particle.cu
    src/nbody_simulation.cu
    src/initial_conditions.cu
)

# Phase 2 sources (Barnes-Hut)
set(PHASE2_SOURCES
    src/morton_code.cu
    src/octree.cu
    src/barnes_hut_force.cu
    src/particle_utils.cu
)

# Create executable
add_executable(nbody ${SOURCES})

# Set properties for CUDA compilation
set_property(TARGET nbody PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Create directories for build artifacts
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(nbody PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add compiler-specific options for our known working setup
if(WIN32)
    # Use the specific compiler path that we know works
    target_compile_options(nbody PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    >)
endif()

# Additional utilities for benchmarking
add_executable(benchmark src/benchmark.cu src/particle.cu src/nbody_simulation.cu src/initial_conditions.cu)
set_target_properties(benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_property(TARGET benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Main executable with Phase 2 integration
add_executable(nbody_barnes_hut ${SOURCES} ${PHASE2_SOURCES})
set_target_properties(nbody_barnes_hut PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_property(TARGET nbody_barnes_hut PROPERTY CUDA_SEPARABLE_COMPILATION ON)

if(WIN32)
    target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    >)
    target_compile_options(nbody_barnes_hut PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
    >)
endif() 
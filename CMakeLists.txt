cmake_minimum_required(VERSION 3.18)

# Project name and description
project(NBodySimulation LANGUAGES CXX CUDA C)

# Set C++ standard for CUDA files
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
include_directories(
    include
    deps/glad/include
    deps/glm
)

# Source files for the simulation library components
set(SIM_SOURCES
    src/particle.cu
    src/nbody_simulation.cu
    src/initial_conditions.cu
    src/morton_code.cu
    src/octree.cu
    src/barnes_hut_force.cu
    src/particle_utils.cu
)

# --- Visualization Dependencies ---
set(GLFW_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/deps/glfw-3.4.bin.WIN64/include")
set(GLFW_LIB_DIR "${CMAKE_SOURCE_DIR}/deps/glfw-3.4.bin.WIN64/lib-vc2022")
find_library(GLFW_LIBRARY NAMES glfw3 PATHS "${GLFW_LIB_DIR}")
find_package(OpenGL REQUIRED)

# --- GLAD Library ---
# Compile glad.c into its own static library to ensure it's handled correctly
add_library(glad_lib STATIC deps/glad/src/glad.c)
target_include_directories(glad_lib PUBLIC deps/glad/include)

# --- Main Simulation Executable ---
add_executable(nbody_simulation
  src/main.cu
  src/main_callbacks.cpp
  src/renderer.cpp
  src/render_kernel.cu
  ${SIM_SOURCES}
)

# Manually link graphics libraries
target_include_directories(nbody_simulation PRIVATE
  "${GLFW_INCLUDE_DIR}"
)
target_link_libraries(nbody_simulation PRIVATE 
  glad_lib
  "${GLFW_LIBRARY}"
  ${OPENGL_LIBRARIES}
)

set_target_properties(nbody_simulation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_property(TARGET nbody_simulation PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# --- Post-Build: Copy Shaders ---
add_custom_command(
    TARGET nbody_simulation POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/shaders ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders
    COMMENT "Copying shaders to output directory"
)

# Benchmark executable
add_executable(benchmark
  src/benchmark.cu
  ${SIM_SOURCES}
)
target_link_libraries(benchmark PRIVATE ${CUDA_LIBRARIES})
set_target_properties(benchmark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_property(TARGET benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Create directories for build artifacts
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Add compiler-specific options for our known working setup
if(WIN32)
    set(MSVC_CCBIN "C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64")
    # Use the specific compiler path that we know works
    target_compile_options(nbody_simulation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin ${MSVC_CCBIN}
    >)
    target_compile_options(benchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -ccbin ${MSVC_CCBIN}
    >)
endif()

target_link_libraries(nbody_simulation PRIVATE ${CUDA_LIBRARIES}) 
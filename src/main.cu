#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "errors.h"
#include "initial_conditions.h"
#include "nbody_simulation.h"
#include "renderer.h"

int main(int argc, char *argv[]) {
  std::cout << "===============================================" << std::endl;
  std::cout << "    GPU N-Body Simulation - Phase 4: Visualization"
            << std::endl;
  std::cout << "===============================================" << std::endl;

  // Check CUDA device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return 1;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Using GPU: " << deviceProp.name << std::endl;
  std::cout << "Compute Capability: " << deviceProp.major << "."
            << deviceProp.minor << std::endl;
  std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount
            << std::endl;
  std::cout << std::endl;

  int num_particles = 8192; // Default value

  // Parse command-line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--particles" && i + 1 < argc) {
      try {
        num_particles = std::stoi(argv[++i]);
      } catch (const std::invalid_argument &ia) {
        std::cerr << "Invalid number for --particles: " << argv[i] << std::endl;
        return 1;
      } catch (const std::out_of_range &oor) {
        std::cerr << "Number for --particles is out of range: " << argv[i]
                  << std::endl;
        return 1;
      }
    }
  }

  try {
    printf("\n--- Running Simple Sphere Demo with Visualization (N=%d) ---\n",
           num_particles);

    NBodySimulation simulation(num_particles, NBodySimulation::BARNES_HUT);
    InitialConditions::generateRandomSphere(*simulation.getParticleSystem(),
                                            5.0f, 0.0f, {500.0f, 2000.0f});

    simulation.getParticleSystem()->copyToDevice();

    Renderer renderer(1280, 720, "N-Body Simulation");

    while (!renderer.shouldClose()) {
      renderer.processInput(simulation);
      renderer.beginFrame();
      simulation.step();
      renderer.renderParticles(*simulation.getParticleSystem(),
                               simulation.getParticleCount());
      renderer.endFrame(simulation.getTimeScale());
    }
  } catch (const std::exception &e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n===============================================" << std::endl;
  std::cout << "    Simulation Complete" << std::endl;
  std::cout << "===============================================" << std::endl;

  return 0;
}
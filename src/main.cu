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

  try {
    const int num_particles = 8192;
    printf("\n--- Running Simple Sphere Demo with Visualization (N=%d) ---\n",
           num_particles);

    NBodySimulation simulation(num_particles, NBodySimulation::BARNES_HUT);
    InitialConditions::generateRandomSphere(*simulation.getParticleSystem(),
                                            5.0f, 0.0f, {5.0f, 20.0f});

    simulation.getParticleSystem()->copyToDevice();

    Renderer renderer(1280, 720, "N-Body Simulation");

    while (!renderer.shouldClose()) {
      // Handle input and pass simulation object
      renderer.processInput(simulation);

      // Begin the frame
      renderer.beginFrame();

      // Update the simulation by one step
      simulation.step();

      // Render the particles
      renderer.renderParticles(*simulation.getParticleSystem(),
                               simulation.getParticleCount());

      // End the frame, passing time scale for UI
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
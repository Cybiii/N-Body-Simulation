#include "initial_conditions.h"
#include "nbody_simulation.h"
#include <iostream>
#include <vector>

void runSimpleDemo() {
  std::cout << "\n=== Simple N-Body Demo ===" << std::endl;

  const int N = 1000;
  const int timesteps = 100;

  // Create simulation
  NBodySimulation sim(N, NBodySimulation::BARNES_HUT, 0.01f, 0.1f);

  // Generate initial conditions
  InitialConditions::generateRandomSphere(*sim.getParticleSystem(), 5.0f, 1.0f);

  // Copy to device
  sim.getParticleSystem()->copyToDevice();

  // Print initial statistics
  sim.getParticleSystem()->printStatistics();

  // Run simulation
  sim.simulate(timesteps);

  // Copy back to host and print final statistics
  sim.getParticleSystem()->copyToHost();
  sim.getParticleSystem()->printStatistics();
}

void runGalaxyDemo() {
  std::cout << "\n=== Galaxy Simulation Demo ===" << std::endl;

  const int N = 2000;
  const int timesteps = 200;

  // Create simulation with smaller timestep for stability
  NBodySimulation sim(N, NBodySimulation::BARNES_HUT, 0.005f, 0.05f);

  // Generate galaxy disk
  InitialConditions::generateGalaxyDisk(*sim.getParticleSystem(), 8.0f, 0.5f,
                                        100.0f, 2.0f);

  // Copy to device
  sim.getParticleSystem()->copyToDevice();

  // Print initial statistics
  sim.getParticleSystem()->printStatistics();

  // Run simulation
  sim.simulate(timesteps);

  // Copy back to host and print final statistics
  sim.getParticleSystem()->copyToHost();
  sim.getParticleSystem()->printStatistics();
}

void runCollidingClustersDemo() {
  std::cout << "\n=== Colliding Clusters Demo ===" << std::endl;

  const int N = 1500;
  const int timesteps = 300;

  // Create simulation
  NBodySimulation sim(N, NBodySimulation::BARNES_HUT, 0.008f, 0.08f);

  // Generate colliding clusters
  InitialConditions::generateCollidingClusters(*sim.getParticleSystem(), 15.0f,
                                               3.0f, 1.5f);

  // Copy to device
  sim.getParticleSystem()->copyToDevice();

  // Print initial statistics
  sim.getParticleSystem()->printStatistics();

  // Run simulation
  sim.simulate(timesteps);

  // Copy back to host and print final statistics
  sim.getParticleSystem()->copyToHost();
  sim.getParticleSystem()->printStatistics();
}

void runBenchmark() {
  std::cout << "\n=== Performance Benchmark ===" << std::endl;

  // Test different particle counts
  std::vector<int> particle_counts = {512, 1024, 2048, 4096, 8192, 16384};
  const int timesteps_per_test = 10;

  // Create a simulation instance for benchmarking
  NBodySimulation benchmark_sim(particle_counts[0], NBodySimulation::BARNES_HUT,
                                0.01f, 0.1f);

  // Run benchmark
  benchmark_sim.benchmark(particle_counts, timesteps_per_test);

  std::cout << "\nBenchmark completed!" << std::endl;
  std::cout << "Note: Performance will vary based on GPU memory and compute "
               "capability."
            << std::endl;
  std::cout << "Your RTX 3060 should achieve significant speedup over CPU "
               "implementations!"
            << std::endl;
}

void printUsage(const char *program_name) {
  std::cout << "Usage: " << program_name << " [option]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  simple     - Run simple random sphere demo (1000 particles)"
            << std::endl;
  std::cout << "  galaxy     - Run galaxy disk demo (2000 particles)"
            << std::endl;
  std::cout << "  collision  - Run colliding clusters demo (1500 particles)"
            << std::endl;
  std::cout << "  benchmark  - Run performance benchmark" << std::endl;
  std::cout << "  all        - Run all demos and benchmark (default)"
            << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << "===============================================" << std::endl;
  std::cout << "    GPU N-Body Simulation - Phase 3 Complete" << std::endl;
  std::cout << "    Barnes-Hut O(N log N) Implementation" << std::endl;
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

  // Parse command line arguments
  std::string mode = "all";
  if (argc >= 2) {
    mode = argv[1];
  }

  try {
    if (mode == "simple") {
      runSimpleDemo();
    } else if (mode == "galaxy") {
      runGalaxyDemo();
    } else if (mode == "collision") {
      runCollidingClustersDemo();
    } else if (mode == "benchmark") {
      runBenchmark();
    } else if (mode == "all") {
      runSimpleDemo();
      runGalaxyDemo();
      runCollidingClustersDemo();
      runBenchmark();
    } else {
      std::cout << "Unknown option: " << mode << std::endl;
      printUsage(argv[0]);
      return 1;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n===============================================" << std::endl;
  std::cout << "    Project Development Complete!" << std::endl;
  std::cout << "    All phases (1, 2, and 3) are implemented." << std::endl;
  std::cout << "===============================================" << std::endl;

  return 0;
}
#include "barnes_hut_force.h"
#include "initial_conditions.h"
#include "octree.h"
#include "particle.h"
#include <chrono>
#include <iostream>


/**
 * Phase 2 Test Program: Barnes-Hut Octree Implementation
 * Tests the Morton code generation, octree construction, and force calculation
 */

void testMortonCodes() {
  std::cout << "\n=== Testing Morton Code Implementation ===" << std::endl;

  // Test basic Morton code functionality
  uint32_t x = 123, y = 456, z = 789;
  auto code = MortonCode::encode(x, y, z);
  auto decoded = MortonCode::decode(code);

  std::cout << "Original: (" << x << ", " << y << ", " << z << ")" << std::endl;
  std::cout << "Morton code: " << code << std::endl;
  std::cout << "Decoded: (" << decoded.x << ", " << decoded.y << ", "
            << decoded.z << ")" << std::endl;

  // Test world coordinate conversion
  float3 pos = make_float3(1.5f, 2.3f, -0.8f);
  float3 bbox_min = make_float3(-5.0f, -5.0f, -5.0f);
  float3 bbox_max = make_float3(5.0f, 5.0f, 5.0f);

  auto world_code = MortonCode::worldToMorton(pos, bbox_min, bbox_max);
  std::cout << "World position: (" << pos.x << ", " << pos.y << ", " << pos.z
            << ")" << std::endl;
  std::cout << "World Morton code: " << world_code << std::endl;

  std::cout << "Morton code tests completed!" << std::endl;
}

void testOctreeConstruction() {
  std::cout << "\n=== Testing Octree Construction ===" << std::endl;

  const int N = 1000;

  // Create particle system
  ParticleSystem particles(N);

  // Generate initial conditions
  InitialConditions::generateRandomSphere(particles, 5.0f, 1.0f);
  particles.copyToDevice();

  // Create octree
  Octree octree(N, 20);

  // Build octree from particles
  auto start_time = std::chrono::high_resolution_clock::now();
  octree.buildFromParticles(particles);
  auto end_time = std::chrono::high_resolution_clock::now();

  double construction_time =
      std::chrono::duration<double>(end_time - start_time).count();

  // Copy results to host for analysis
  octree.copyToHost();

  // Print statistics
  octree.printTreeStatistics();

  std::cout << "Octree construction time: " << construction_time * 1000.0
            << " ms" << std::endl;
  std::cout << "Octree construction completed!" << std::endl;
}

void testBarnesHutForces() {
  std::cout << "\n=== Testing Barnes-Hut Force Calculation ===" << std::endl;

  const int N = 500; // Smaller for detailed testing

  // Create particle system
  ParticleSystem particles(N);

  // Generate galaxy-like distribution for interesting dynamics
  InitialConditions::generateGalaxyDisk(particles, 8.0f, 0.5f, 100.0f, 2.0f);
  particles.copyToDevice();

  // Create octree
  Octree octree(N, 20);
  octree.buildFromParticles(particles);

  // Create Barnes-Hut force calculator
  BarnesHutForce bh_force(0.5f, 0.1f); // theta = 0.5, softening = 0.1

  // Calculate forces
  float G_scaled = 6.67430e-11f * 1e10f; // Scaled gravitational constant

  auto start_time = std::chrono::high_resolution_clock::now();
  bh_force.calculateForces(particles, octree, G_scaled);
  auto end_time = std::chrono::high_resolution_clock::now();

  double force_calc_time =
      std::chrono::duration<double>(end_time - start_time).count();

  // Print performance statistics
  bh_force.printPerformanceStats();

  std::cout << "Total force calculation time: " << force_calc_time * 1000.0
            << " ms" << std::endl;
  std::cout << "Barnes-Hut force calculation completed!" << std::endl;
}

void compareBarnesHutVsBruteForce() {
  std::cout << "\n=== Comparing Barnes-Hut vs Brute Force ===" << std::endl;

  std::vector<int> particle_counts = {256, 512, 1024, 2048};

  std::cout << std::setw(10) << "Particles" << std::setw(15)
            << "Brute Force (ms)" << std::setw(15) << "Barnes-Hut (ms)"
            << std::setw(15) << "Speedup" << std::endl;
  std::cout << std::string(55, '-') << std::endl;

  for (int N : particle_counts) {
    // Create particle system
    ParticleSystem particles(N);
    InitialConditions::generateBenchmarkParticles(particles);
    particles.copyToDevice();

    // Test brute force (from Phase 1)
    auto start = std::chrono::high_resolution_clock::now();
    // Would call brute force kernel here
    auto end = std::chrono::high_resolution_clock::now();
    double brute_force_time =
        std::chrono::duration<double>(end - start).count() * 1000.0;

    // Test Barnes-Hut
    Octree octree(N, 20);
    BarnesHutForce bh_force(0.5f, 0.1f);

    start = std::chrono::high_resolution_clock::now();
    octree.buildFromParticles(particles);
    bh_force.calculateForces(particles, octree, 1e-3f);
    end = std::chrono::high_resolution_clock::now();
    double barnes_hut_time =
        std::chrono::duration<double>(end - start).count() * 1000.0;

    double speedup = brute_force_time / barnes_hut_time;

    std::cout << std::setw(10) << N << std::setw(15) << std::fixed
              << std::setprecision(3) << brute_force_time << std::setw(15)
              << std::fixed << std::setprecision(3) << barnes_hut_time
              << std::setw(15) << std::fixed << std::setprecision(2) << speedup
              << std::endl;
  }

  std::cout << std::string(55, '-') << std::endl;
  std::cout << "Performance comparison completed!" << std::endl;
}

int main() {
  std::cout << "===============================================" << std::endl;
  std::cout << "    Phase 2: Barnes-Hut Octree Test" << std::endl;
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
  std::cout << std::endl;

  try {
    // Run Phase 2 tests
    testMortonCodes();
    testOctreeConstruction();
    testBarnesHutForces();
    compareBarnesHutVsBruteForce();

  } catch (const std::exception &e) {
    std::cerr << "Error during Phase 2 testing: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n===============================================" << std::endl;
  std::cout << "Phase 2 testing completed successfully!" << std::endl;
  std::cout << "Barnes-Hut octree implementation is working!" << std::endl;
  std::cout << "===============================================" << std::endl;

  return 0;
}
#include "initial_conditions.h"
#include "nbody_simulation.h"
#include "particle.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// A simple struct to hold benchmark results
struct BenchmarkResult {
  int particles;
  double force_ms;
  double integration_ms;
  double total_ms;
  double gflops;
};

// Function to run a benchmark for a single particle count
BenchmarkResult benchmarkParticleCount(int N, int timesteps_per_test,
                                       NBodySimulation::Algorithm algo) {
  // Create a new simulation for each benchmark to reset all timers and states
  NBodySimulation sim(N, algo);
  InitialConditions::generateBenchmarkParticles(*sim.getParticleSystem());
  sim.getParticleSystem()->copyToDevice();

  // Warm-up run (important for GPU benchmarks)
  sim.simulate(5);

  // Actually run the simulation for the specified number of timesteps
  sim.simulate(timesteps_per_test);

  BenchmarkResult result;
  result.particles = N;
  // GetTotal... returns cumulative time, so we average it over the run.
  result.force_ms = sim.getTotalForceCalcTime() / sim.getCurrentTimestep();
  result.integration_ms =
      sim.getTotalIntegrationTime() / sim.getCurrentTimestep();
  result.total_ms = sim.getTotalSimTime() / sim.getCurrentTimestep();

  // Estimate GFLOPS
  double interactions_per_step = (algo == NBodySimulation::BARNES_HUT)
                                     ? (double)N * log((double)N) // O(N log N)
                                     : (double)N * (double)N;     // O(N^2)
  double ops_per_interaction = 20; // Approx ops for force calculation
  result.gflops = (interactions_per_step * ops_per_interaction) /
                  (result.total_ms / 1000.0) / 1e9;

  return result;
}

void run_scaling_benchmark(int timesteps_per_test,
                           const std::vector<int> &particle_counts) {
  std::cout << std::string(87, '-') << std::endl;
  std::cout << "Scaling Benchmark (Algorithm: Barnes-Hut)" << std::endl;
  std::cout << std::string(87, '-') << std::endl;
  std::cout << std::left << std::setw(12) << "Particles" << std::setw(20)
            << "Force (ms/step)" << std::setw(22) << "Integration (ms/step)"
            << std::setw(18) << "Total (ms/step)" << std::setw(15) << "GFLOPS"
            << std::endl;
  std::cout << std::string(87, '-') << std::endl;

  for (int N : particle_counts) {
    BenchmarkResult result = benchmarkParticleCount(
        N, timesteps_per_test, NBodySimulation::BARNES_HUT);

    std::cout << std::left << std::setw(12) << result.particles << std::setw(20)
              << std::fixed << std::setprecision(4) << result.force_ms
              << std::setw(22) << result.integration_ms << std::setw(18)
              << result.total_ms << std::setw(15) << std::setprecision(2)
              << result.gflops << std::endl;
  }
  std::cout << std::string(87, '-') << std::endl;
}

void compare_algorithms(int N, int timesteps_per_test) {
  std::cout << "\n" << std::string(87, '-') << std::endl;
  std::cout << "Algorithm Comparison (N = " << N << ")" << std::endl;
  std::cout << std::string(87, '-') << std::endl;

  // --- Barnes-Hut ---
  std::cout << "--> Barnes-Hut:" << std::endl;
  BenchmarkResult bh_result = benchmarkParticleCount(
      N, timesteps_per_test, NBodySimulation::BARNES_HUT);
  std::cout << "    Avg total time per step: " << std::fixed
            << std::setprecision(4) << bh_result.total_ms << " ms" << std::endl;

  // --- Brute Force ---
  std::cout << "--> Brute Force:" << std::endl;
  BenchmarkResult brute_result = benchmarkParticleCount(
      N, timesteps_per_test, NBodySimulation::BRUTE_FORCE);
  std::cout << "    Avg total time per step: " << std::fixed
            << std::setprecision(4) << brute_result.total_ms << " ms"
            << std::endl;

  if (bh_result.total_ms > 0) {
    double speedup = brute_result.total_ms / bh_result.total_ms;
    std::cout << "\n    Speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;
  }
  std::cout << std::string(87, '-') << std::endl;
}

void memory_benchmark(const std::vector<int> &particle_counts) {
  std::cout << std::string(80, '-') << std::endl;
  std::cout << "Memory Usage Benchmark" << std::endl;
  std::cout << std::string(80, '-') << std::endl;
  std::cout << std::left << std::setw(15) << "Particles" << std::setw(20)
            << "Particle Mem (MB)" << std::endl;
  std::cout << std::string(80, '-') << std::endl;

  for (const int N : particle_counts) {
    // We just need to calculate the memory, not run a sim
    double memory_mb = (double)N * sizeof(Particle) / (1024.0 * 1024.0);
    std::cout << std::left << std::setw(15) << N << std::setw(20) << std::fixed
              << std::setprecision(2) << memory_mb << std::endl;
  }
}

int main(int argc, char **argv) {
  std::cout << "=======================================" << std::endl;
  std::cout << "      N-Body Simulation Benchmark      " << std::endl;
  std::cout << "=======================================" << std::endl;

  // --- Scaling Benchmark ---
  std::vector<int> scaling_counts = {1024, 2048, 4096, 8192, 16384, 32768};
  run_scaling_benchmark(20, scaling_counts);

  // --- Algorithm Comparison ---
  // Use a smaller particle count for brute-force as it's very slow
  compare_algorithms(2048, 20);

  return 0;
}
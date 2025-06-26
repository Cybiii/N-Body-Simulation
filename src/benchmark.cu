#include "initial_conditions.h"
#include "nbody_simulation.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>


/**
 * Detailed benchmark for N-body simulation performance analysis
 * Outputs results to CSV file for analysis
 */
class DetailedBenchmark {
private:
  std::vector<int> particle_counts;
  int timesteps_per_test;
  std::string output_filename;

public:
  DetailedBenchmark(const std::vector<int> &counts, int timesteps,
                    const std::string &filename)
      : particle_counts(counts), timesteps_per_test(timesteps),
        output_filename(filename) {}

  void runBenchmark() {
    std::cout << "=== Detailed N-Body Benchmark ===" << std::endl;
    std::cout << "Particle counts: ";
    for (int n : particle_counts)
      std::cout << n << " ";
    std::cout << std::endl;
    std::cout << "Timesteps per test: " << timesteps_per_test << std::endl;
    std::cout << "Output file: " << output_filename << std::endl;

    // Open CSV file for output
    std::ofstream csv_file(output_filename);
    csv_file << "Particles,Force_Time_ms,Integration_Time_ms,Total_Time_ms,"
                "GFLOPS,Memory_MB,Throughput_particles_per_sec"
             << std::endl;

    // Print table header
    std::cout << std::endl;
    std::cout << std::setw(10) << "Particles" << std::setw(12) << "Force (ms)"
              << std::setw(12) << "Integrate (ms)" << std::setw(12)
              << "Total (ms)" << std::setw(12) << "GFLOPS" << std::setw(12)
              << "Memory (MB)" << std::setw(15) << "Throughput" << std::endl;
    std::cout << std::string(85, '-') << std::endl;

    for (int N : particle_counts) {
      auto results = benchmarkParticleCount(N);

      // Output to console
      std::cout << std::setw(10) << N << std::setw(12) << std::fixed
                << std::setprecision(3) << results.force_time_ms
                << std::setw(12) << std::fixed << std::setprecision(3)
                << results.integration_time_ms << std::setw(12) << std::fixed
                << std::setprecision(3) << results.total_time_ms
                << std::setw(12) << std::fixed << std::setprecision(2)
                << results.gflops << std::setw(12) << std::fixed
                << std::setprecision(1) << results.memory_mb << std::setw(15)
                << std::fixed << std::setprecision(0) << results.throughput
                << std::endl;

      // Output to CSV
      csv_file << N << "," << results.force_time_ms << ","
               << results.integration_time_ms << "," << results.total_time_ms
               << "," << results.gflops << "," << results.memory_mb << ","
               << results.throughput << std::endl;
    }

    csv_file.close();
    std::cout << std::string(85, '-') << std::endl;
    std::cout << "Benchmark results saved to: " << output_filename << std::endl;
  }

private:
  struct BenchmarkResult {
    double force_time_ms;
    double integration_time_ms;
    double total_time_ms;
    double gflops;
    double memory_mb;
    double throughput;
  };

  BenchmarkResult benchmarkParticleCount(int N) {
    // Create simulation
    NBodySimulation sim(N, 0.01f, 0.1f);

    // Generate benchmark particles (fast generation)
    InitialConditions::generateBenchmarkParticles(*sim.getParticleSystem());
    sim.getParticleSystem()->copyToDevice();

    // Warm-up run
    sim.step();

    // Reset timing counters
    sim.total_force_calc_time = 0.0;
    sim.total_integration_time = 0.0;

    // Run benchmark
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < timesteps_per_test; i++) {
      sim.step();
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_time =
        std::chrono::duration<double>(end_time - start_time).count();

    // Calculate results
    BenchmarkResult result;
    result.force_time_ms =
        (sim.total_force_calc_time / timesteps_per_test) * 1000.0;
    result.integration_time_ms =
        (sim.total_integration_time / timesteps_per_test) * 1000.0;
    result.total_time_ms = (total_time / timesteps_per_test) * 1000.0;

    // Calculate GFLOPS (approximate: 20 operations per particle pair +
    // integration)
    double force_operations = static_cast<double>(N) * N * 20.0;
    double integration_operations = static_cast<double>(N) * 10.0;
    double total_operations = force_operations + integration_operations;
    result.gflops = (total_operations / (result.total_time_ms / 1000.0)) / 1e9;

    // Memory usage
    result.memory_mb = (N * sizeof(Particle)) / (1024.0 * 1024.0);

    // Throughput (particles processed per second)
    result.throughput =
        (static_cast<double>(N) * timesteps_per_test) / total_time;

    return result;
  }
};

/**
 * Scaling analysis - how performance scales with particle count
 */
void analyzeScaling() {
  std::cout << "\n=== Scaling Analysis ===" << std::endl;

  std::vector<int> counts = {256, 512, 1024, 2048, 4096, 8192};

  std::cout << std::setw(10) << "N" << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Time/N²" << std::setw(15) << "Efficiency"
            << std::endl;
  std::cout << std::string(55, '-') << std::endl;

  double baseline_time_per_n2 = 0.0;

  for (size_t i = 0; i < counts.size(); i++) {
    int N = counts[i];

    // Quick benchmark
    NBodySimulation sim(N, 0.01f, 0.1f);
    InitialConditions::generateBenchmarkParticles(*sim.getParticleSystem());
    sim.getParticleSystem()->copyToDevice();

    auto start = std::chrono::high_resolution_clock::now();
    sim.step();
    auto end = std::chrono::high_resolution_clock::now();

    double time_ms =
        std::chrono::duration<double>(end - start).count() * 1000.0;
    double time_per_n2 = time_ms / (static_cast<double>(N) * N);

    if (i == 0)
      baseline_time_per_n2 = time_per_n2;
    double efficiency = baseline_time_per_n2 / time_per_n2;

    std::cout << std::setw(10) << N << std::setw(15) << std::fixed
              << std::setprecision(3) << time_ms << std::setw(15)
              << std::scientific << std::setprecision(2) << time_per_n2
              << std::setw(15) << std::fixed << std::setprecision(2)
              << efficiency << std::endl;
  }

  std::cout << std::string(55, '-') << std::endl;
  std::cout << "Note: Efficiency > 1.0 indicates better scaling (GPU "
               "utilization improvement)"
            << std::endl;
}

/**
 * Memory bandwidth analysis
 */
void analyzeMemoryBandwidth() {
  std::cout << "\n=== Memory Bandwidth Analysis ===" << std::endl;

  const int N = 8192; // Fixed particle count
  const int timesteps = 5;

  NBodySimulation sim(N, 0.01f, 0.1f);
  InitialConditions::generateBenchmarkParticles(*sim.getParticleSystem());
  sim.getParticleSystem()->copyToDevice();

  // Warm up
  sim.step();

  // Measure memory transfers
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < timesteps; i++) {
    sim.getParticleSystem()->copyToHost();
    sim.getParticleSystem()->copyToDevice();
  }

  auto end = std::chrono::high_resolution_clock::now();
  double transfer_time = std::chrono::duration<double>(end - start).count();

  // Calculate bandwidth
  double bytes_per_transfer =
      N * sizeof(Particle) * 2; // Host->Device + Device->Host
  double total_bytes = bytes_per_transfer * timesteps;
  double bandwidth_gb_s = (total_bytes / transfer_time) / 1e9;

  std::cout << "Particles: " << N << std::endl;
  std::cout << "Bytes per particle: " << sizeof(Particle) << std::endl;
  std::cout << "Total data transferred: " << total_bytes / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "Transfer time: " << transfer_time * 1000 << " ms" << std::endl;
  std::cout << "Memory bandwidth: " << bandwidth_gb_s << " GB/s" << std::endl;
}

int main(int argc, char *argv[]) {
  std::cout << "===============================================" << std::endl;
  std::cout << "    N-Body Simulation - Detailed Benchmark" << std::endl;
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
  std::cout << "GPU: " << deviceProp.name << std::endl;
  std::cout << "Compute Capability: " << deviceProp.major << "."
            << deviceProp.minor << std::endl;
  std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024)
            << " MB" << std::endl;
  std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock
            << std::endl;
  std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount
            << std::endl;
  std::cout << std::endl;

  try {
    // Main benchmark
    std::vector<int> particle_counts = {512, 1024, 2048, 4096, 8192, 16384};
    DetailedBenchmark benchmark(particle_counts, 10,
                                "nbody_benchmark_results.csv");
    benchmark.runBenchmark();

    // Additional analyses
    analyzeScaling();
    analyzeMemoryBandwidth();

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n===============================================" << std::endl;
  std::cout << "Benchmark completed successfully!" << std::endl;
  std::cout << "===============================================" << std::endl;

  return 0;
}
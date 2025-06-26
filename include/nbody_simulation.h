#pragma once

#include "particle.h"
#include <chrono>
#include <cuda_runtime.h>


/**
 * N-Body Simulation class - Phase 1: Brute Force O(N²) implementation
 */
class NBodySimulation {
public:
  NBodySimulation(int num_particles, float dt = 0.01f, float softening = 0.1f);
  ~NBodySimulation();

  // Main simulation functions
  void simulate(int num_timesteps);
  void step(); // Single timestep

  // Performance and analysis
  void benchmark(const std::vector<int> &particle_counts,
                 int timesteps_per_test = 10);
  void printPerformanceStats() const;

  // Getters
  ParticleSystem *getParticleSystem() { return particles; }
  float getDt() const { return dt; }
  float getSoftening() const { return softening; }

  // Setters
  void setDt(float new_dt) { dt = new_dt; }
  void setSoftening(float new_softening) { softening = new_softening; }

private:
  ParticleSystem *particles;
  float dt;        // Time step
  float softening; // Softening factor to prevent singularities
  int current_timestep;

  // Performance tracking
  std::chrono::high_resolution_clock::time_point last_time;
  double total_simulation_time;
  double total_force_calc_time;
  double total_integration_time;

  // CUDA events for timing
  cudaEvent_t start_event, stop_event;

  void updateTimers();
};

// Constants
const float G = 6.67430e-11f; // Gravitational constant (scaled for simulation)

// CUDA kernel declarations for force calculation and integration
extern "C" {
void launch_calculate_forces_brute_force(Particle *d_particles, int N,
                                         float softening_sq, float G_constant);

void launch_update_particles(Particle *d_particles, int N, float dt);
}
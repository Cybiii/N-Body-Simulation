#pragma once

#include "barnes_hut_force.h"
#include "octree.h"
#include "particle.h"
#include <chrono>
#include <cuda_runtime.h>
#include <string>

/**
 * N-Body Simulation class - Upgraded with Barnes-Hut O(N log N)
 */
class NBodySimulation {
public:
  enum Algorithm { BRUTE_FORCE, BARNES_HUT };

  NBodySimulation(int num_particles, Algorithm algo = BARNES_HUT,
                  float dt = 0.01f, float softening = 0.1f, float theta = 0.5f);
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
  Algorithm getAlgorithm() const { return algorithm; }
  double getTotalForceCalcTime() const { return total_force_calc_time; }
  double getTotalIntegrationTime() const { return total_integration_time; }
  double getTotalSimTime() const { return total_simulation_time; }
  int getCurrentTimestep() const { return current_timestep; }

  // Setters
  void setDt(float new_dt) { dt = new_dt; }
  void setSoftening(float new_softening);
  void setAlgorithm(Algorithm new_algo);

private:
  ParticleSystem *particles;
  Octree *octree;
  BarnesHutForce *barnes_hut_force;
  Algorithm algorithm;

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
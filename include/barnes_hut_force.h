#pragma once

#include "octree.h"
#include "particle.h"

/**
 * Barnes-Hut force calculation using octree traversal
 * This replaces the O(N²) brute force algorithm with O(N log N)
 */
class BarnesHutForce {
public:
  BarnesHutForce(float theta = 0.5f, float softening = 0.1f);

  /**
   * Calculate forces using Barnes-Hut algorithm
   * @param particles Particle system
   * @param octree Constructed octree
   * @param G_constant Gravitational constant
   */
  void calculateForces(ParticleSystem &particles, const Octree &octree,
                       float G_constant);

  // Parameter accessors
  float getTheta() const { return theta; }
  float getSoftening() const { return softening; }
  void setTheta(float new_theta) { theta = new_theta; }
  void setSoftening(float new_softening) { softening = new_softening; }

  // Performance analysis
  double getLastForceCalcTime() const { return last_force_calc_time; }
  void printPerformanceStats() const;

private:
  float theta;     // Barnes-Hut approximation parameter (θ)
  float softening; // Softening factor for force calculation

  // Performance tracking
  double last_force_calc_time;

  // CUDA events for timing
  cudaEvent_t start_event, stop_event;
};

/**
 * Stack-based tree traversal structure for GPU kernels
 * Avoids recursion which is problematic on GPU
 */
struct TraversalStack {
  static constexpr int MAX_STACK_SIZE = 64; // Should be enough for most trees
  int stack[MAX_STACK_SIZE];
  int top;

  __device__ TraversalStack() : top(-1) {}

  __device__ void push(int node_idx) {
    if (top < MAX_STACK_SIZE - 1) {
      stack[++top] = node_idx;
    }
  }

  __device__ int pop() { return (top >= 0) ? stack[top--] : -1; }

  __device__ bool isEmpty() const { return top < 0; }

  __device__ bool isFull() const { return top >= MAX_STACK_SIZE - 1; }
};

// CUDA kernel declarations for Barnes-Hut force calculation
extern "C" {
void launch_barnes_hut_force_calculation(
    const float3 *positions, const float *masses, float3 *accelerations, int N,
    const OctreeNode *nodes, int num_nodes, const int *particle_indices,
    float theta_sq, float softening_sq, float G_constant);

void launch_reset_accelerations_bh(float3 *accelerations, int N);
}
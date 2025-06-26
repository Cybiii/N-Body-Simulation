#include "particle.h"
#include <cuda_runtime.h>

/**
 * Utility kernels for particle data extraction and updates
 * These bridge the gap between ParticleSystem and Barnes-Hut algorithms
 */

// Extract positions from particle array
__global__ void extract_positions_kernel(const Particle *particles,
                                         float3 *positions, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    positions[idx] = particles[idx].position;
  }
}

// Extract masses from particle array
__global__ void extract_masses_kernel(const Particle *particles, float *masses,
                                      int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    masses[idx] = particles[idx].mass;
  }
}

// Update accelerations in particle array
__global__ void update_accelerations_kernel(Particle *particles,
                                            const float3 *accelerations,
                                            int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    particles[idx].acceleration = accelerations[idx];
  }
}

// Update accelerations with particle reordering (after Morton sort)
__global__ void
update_accelerations_reordered_kernel(Particle *particles,
                                      const float3 *accelerations,
                                      const int *particle_indices, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    int original_idx = particle_indices[idx];
    if (original_idx >= 0 && original_idx < N) {
      particles[original_idx].acceleration = accelerations[idx];
    }
  }
}

/**
 * Wrapper functions for kernel launches
 */
extern "C" void launch_extract_positions(const Particle *particles,
                                         float3 *positions, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  extract_positions_kernel<<<blocksPerGrid, threadsPerBlock>>>(particles,
                                                               positions, N);
  checkCudaError(cudaGetLastError(), "extract positions kernel launch");
}

extern "C" void launch_extract_masses(const Particle *particles, float *masses,
                                      int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  extract_masses_kernel<<<blocksPerGrid, threadsPerBlock>>>(particles, masses,
                                                            N);
  checkCudaError(cudaGetLastError(), "extract masses kernel launch");
}

extern "C" void launch_update_accelerations(Particle *particles,
                                            const float3 *accelerations,
                                            int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  update_accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      particles, accelerations, N);
  checkCudaError(cudaGetLastError(), "update accelerations kernel launch");
}

extern "C" void
launch_update_accelerations_reordered(Particle *particles,
                                      const float3 *accelerations,
                                      const int *particle_indices, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  update_accelerations_reordered_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      particles, accelerations, particle_indices, N);
  checkCudaError(cudaGetLastError(),
                 "update accelerations reordered kernel launch");
}
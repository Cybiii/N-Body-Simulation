#include "boundaries.h"
#include "errors.h"
#include <cuda_runtime.h>

/**
 * Kernel to enforce reflective boundary conditions
 */
__global__ void enforce_boundaries_kernel(Particle *particles, int N,
                                          float bound_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  Particle &p = particles[idx];
  float damping = 0.85f; // Lose some energy on bounce

  // Check X boundaries
  if (p.position.x > bound_size) {
    p.position.x = bound_size;
    p.velocity.x *= -damping;
  } else if (p.position.x < -bound_size) {
    p.position.x = -bound_size;
    p.velocity.x *= -damping;
  }

  // Check Y boundaries
  if (p.position.y > bound_size) {
    p.position.y = bound_size;
    p.velocity.y *= -damping;
  } else if (p.position.y < -bound_size) {
    p.position.y = -bound_size;
    p.velocity.y *= -damping;
  }

  // Check Z boundaries
  if (p.position.z > bound_size) {
    p.position.z = bound_size;
    p.velocity.z *= -damping;
  } else if (p.position.z < -bound_size) {
    p.position.z = -bound_size;
    p.velocity.z *= -damping;
  }
}

/**
 * Launcher for the enforce_boundaries_kernel
 */
void launch_enforce_boundaries(Particle *d_particles, int N, float bound_size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  enforce_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, N,
                                                                bound_size);
  checkCudaError(cudaGetLastError(), "enforce boundaries kernel launch");
}
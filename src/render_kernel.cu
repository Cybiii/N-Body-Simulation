#include "particle.h"
#include "render_kernel.h"
#include <cuda_runtime.h>
#include <vector_types.h>


__global__ void copy_positions_to_buffer(float2 *pbo_buffer,
                                         Particle *particles,
                                         int num_particles) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_particles) {
    return;
  }

  // Copy particle position to the PBO buffer
  // For now, we just copy x and y for a 2D view
  pbo_buffer[idx] =
      make_float2(particles[idx].position.x, particles[idx].position.y);
}

extern "C" void launch_copy_positions_to_buffer(float2 *pbo_buffer,
                                                Particle *particles,
                                                int num_particles) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

  copy_positions_to_buffer<<<blocksPerGrid, threadsPerBlock>>>(
      pbo_buffer, particles, num_particles);
}
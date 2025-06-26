#include "particle.h"
#include "render_kernel.h"
#include <cuda_runtime.h>
#include <math.h>
#include <vector_types.h>

__device__ float3 lerp(float3 a, float3 b, float t) {
  return make_float3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y),
                     a.z + t * (b.z - a.z));
}

__device__ float3 hsv2rgb(float3 c) { // c.x = H, c.y = S, c.z = V
  float4 K = make_float4(1.0f, 2.0f / 3.0f, 1.0f / 3.0f, 3.0f);

  float3 p_in = make_float3(c.x + K.x, c.x + K.y, c.x + K.z);
  p_in.x = fmodf(p_in.x, 1.0f);
  p_in.y = fmodf(p_in.y, 1.0f);
  p_in.z = fmodf(p_in.z, 1.0f);

  float3 p = make_float3(fabsf(p_in.x * 6.0f - K.w), fabsf(p_in.y * 6.0f - K.w),
                         fabsf(p_in.z * 6.0f - K.w));

  float3 p_minus_k = make_float3(p.x - K.x, p.y - K.x, p.z - K.x);
  float3 clamped = make_float3(fminf(fmaxf(p_minus_k.x, 0.0f), 1.0f),
                               fminf(fmaxf(p_minus_k.y, 0.0f), 1.0f),
                               fminf(fmaxf(p_minus_k.z, 0.0f), 1.0f));

  float3 k_xxx = make_float3(K.x, K.x, K.x);
  float3 result = lerp(k_xxx, clamped, c.y);

  return make_float3(c.z * result.x, c.z * result.y, c.z * result.z);
}

__global__ void compute_colors_and_interleave_kernel(float4 *d_vbo_buffer,
                                                     Particle *d_particles,
                                                     int particle_count,
                                                     float max_velocity_sq) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= particle_count)
    return;

  Particle p = d_particles[idx];
  float vel_sq = p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y +
                 p.velocity.z * p.velocity.z;
  float normalized_vel = fminf(vel_sq / max_velocity_sq, 1.0f);

  // Hue from sky blue (~210/360) to red (0)
  float hue = (1.0f - normalized_vel) * (210.0f / 360.0f);
  // Saturation from 0.7 to 1.0 for more vibrant reds
  float saturation = lerp(make_float3(0.7f, 0.7f, 0.7f),
                          make_float3(1.0f, 1.0f, 1.0f), normalized_vel)
                         .x;

  float3 rgb = hsv2rgb(make_float3(hue, saturation, 1.0f));

  d_vbo_buffer[idx * 2] =
      make_float4(p.position.x, p.position.y, p.position.z, 1.0f);
  d_vbo_buffer[idx * 2 + 1] = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
}

extern "C" void launch_compute_colors_and_interleave(float4 *d_vbo_buffer,
                                                     Particle *d_particles,
                                                     int particle_count,
                                                     float max_velocity_sq) {
  int threads_per_block = 256;
  int blocks_per_grid =
      (particle_count + threads_per_block - 1) / threads_per_block;
  compute_colors_and_interleave_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_vbo_buffer, d_particles, particle_count, max_velocity_sq);
}

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
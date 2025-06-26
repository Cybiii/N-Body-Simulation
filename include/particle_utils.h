#pragma once

#include "particle.h"

// --- Vector math operators for float3 ---
__host__ __device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3 &a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator*(float s, const float3 &a) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline void operator+=(float3 &a, const float3 &b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
}

__host__ __device__ inline float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length_sq(const float3 &a) {
  return dot(a, a);
}

/**
 * Utility functions for particle data extraction and updates
 * Bridge between ParticleSystem and Barnes-Hut algorithms
 */

extern "C" {
void launch_extract_positions(const Particle *particles, float3 *positions,
                              int N);

void launch_extract_masses(const Particle *particles, float *masses, int N);

void launch_update_accelerations(Particle *particles,
                                 const float3 *accelerations, int N);

void launch_update_accelerations_reordered(Particle *particles,
                                           const float3 *accelerations,
                                           const int *particle_indices, int N);
}
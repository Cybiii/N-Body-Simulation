#pragma once

#include "errors.h"
#include <cuda_runtime.h>
#include <vector>
#include <vector_types.h>

/**
 * Particle structure representing a single body in the N-body simulation
 * Designed for efficient GPU memory access and computation
 */
struct Particle {
  float3 position;     // Position (x, y, z)
  float3 velocity;     // Velocity (vx, vy, vz)
  float3 acceleration; // Acceleration (ax, ay, az)
  float mass;          // Mass of the particle

  // Default constructor
  __host__ __device__ Particle()
      : position(make_float3(0.0f, 0.0f, 0.0f)),
        velocity(make_float3(0.0f, 0.0f, 0.0f)),
        acceleration(make_float3(0.0f, 0.0f, 0.0f)), mass(1.0f) {}

  // Constructor with values
  __host__ __device__ Particle(float3 pos, float3 vel, float m)
      : position(pos), velocity(vel),
        acceleration(make_float3(0.0f, 0.0f, 0.0f)), mass(m) {}
};

/**
 * CUDA error checking utility
 */
void checkCudaError(cudaError_t error, const char *operation);

/**
 * Particle management functions
 */
class ParticleSystem {
public:
  ParticleSystem(int num_particles);
  ~ParticleSystem();

  // Host-device memory management
  void copyToDevice();
  void copyToHost();

  // Accessors
  Particle *getHostParticles() { return h_particles.data(); }
  Particle *getDeviceParticles() { return d_particles; }
  int getNumParticles() const { return num_particles; }

  // Utility functions
  void resetAccelerations();
  double getTotalEnergy() const;
  void printStatistics() const;

private:
  int num_particles;
  std::vector<Particle> h_particles; // Host particles
  Particle *d_particles;             // Device particles
  size_t particle_size;
};

// CUDA kernel declarations
extern "C" {
void launch_reset_accelerations(Particle *d_particles, int N);
}
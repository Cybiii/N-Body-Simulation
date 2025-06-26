#include "particle.h"
#include <cmath>
#include <iostream>


/**
 * CUDA error checking utility
 */
void checkCudaError(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error during " << operation << ": "
              << cudaGetErrorString(error) << std::endl;
    exit(1);
  }
}

/**
 * CUDA kernel to reset accelerations
 */
__global__ void reset_accelerations_kernel(Particle *particles, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    particles[idx].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }
}

/**
 * Wrapper function for reset accelerations kernel
 */
extern "C" void launch_reset_accelerations(Particle *d_particles, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reset_accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles,
                                                                 N);
  checkCudaError(cudaGetLastError(), "reset accelerations kernel launch");
}

/**
 * ParticleSystem implementation
 */
ParticleSystem::ParticleSystem(int num_particles)
    : num_particles(num_particles),
      particle_size(num_particles * sizeof(Particle)) {

  // Allocate host memory
  h_particles.resize(num_particles);

  // Allocate device memory
  checkCudaError(cudaMalloc(&d_particles, particle_size),
                 "device particle allocation");

  std::cout << "Initialized ParticleSystem with " << num_particles
            << " particles" << std::endl;
  std::cout << "Memory allocated: " << particle_size / (1024.0 * 1024.0)
            << " MB" << std::endl;
}

ParticleSystem::~ParticleSystem() {
  if (d_particles) {
    cudaFree(d_particles);
  }
}

void ParticleSystem::copyToDevice() {
  checkCudaError(cudaMemcpy(d_particles, h_particles.data(), particle_size,
                            cudaMemcpyHostToDevice),
                 "copy particles to device");
}

void ParticleSystem::copyToHost() {
  checkCudaError(cudaMemcpy(h_particles.data(), d_particles, particle_size,
                            cudaMemcpyDeviceToHost),
                 "copy particles to host");
}

void ParticleSystem::resetAccelerations() {
  launch_reset_accelerations(d_particles, num_particles);
  checkCudaError(cudaDeviceSynchronize(),
                 "reset accelerations synchronization");
}

double ParticleSystem::getTotalEnergy() const {
  double kinetic_energy = 0.0;
  double potential_energy = 0.0;

  // Calculate kinetic energy
  for (const auto &p : h_particles) {
    float v_sq = p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y +
                 p.velocity.z * p.velocity.z;
    kinetic_energy += 0.5 * p.mass * v_sq;
  }

  // Calculate potential energy (O(N²) - expensive!)
  const float G = 6.67430e-11f;
  for (int i = 0; i < num_particles; i++) {
    for (int j = i + 1; j < num_particles; j++) {
      const auto &p1 = h_particles[i];
      const auto &p2 = h_particles[j];

      float dx = p1.position.x - p2.position.x;
      float dy = p1.position.y - p2.position.y;
      float dz = p1.position.z - p2.position.z;
      float r = sqrtf(dx * dx + dy * dy + dz * dz);

      if (r > 0.0f) {
        potential_energy -= G * p1.mass * p2.mass / r;
      }
    }
  }

  return kinetic_energy + potential_energy;
}

void ParticleSystem::printStatistics() const {
  if (h_particles.empty()) {
    std::cout << "No particles to analyze" << std::endl;
    return;
  }

  // Calculate center of mass
  float3 center_of_mass = make_float3(0.0f, 0.0f, 0.0f);
  float total_mass = 0.0f;

  for (const auto &p : h_particles) {
    center_of_mass.x += p.position.x * p.mass;
    center_of_mass.y += p.position.y * p.mass;
    center_of_mass.z += p.position.z * p.mass;
    total_mass += p.mass;
  }

  if (total_mass > 0.0f) {
    center_of_mass.x /= total_mass;
    center_of_mass.y /= total_mass;
    center_of_mass.z /= total_mass;
  }

  // Calculate average velocity
  float3 avg_velocity = make_float3(0.0f, 0.0f, 0.0f);
  for (const auto &p : h_particles) {
    avg_velocity.x += p.velocity.x;
    avg_velocity.y += p.velocity.y;
    avg_velocity.z += p.velocity.z;
  }
  avg_velocity.x /= num_particles;
  avg_velocity.y /= num_particles;
  avg_velocity.z /= num_particles;

  double total_energy = getTotalEnergy();

  std::cout << "\n=== Particle System Statistics ===" << std::endl;
  std::cout << "Number of particles: " << num_particles << std::endl;
  std::cout << "Total mass: " << total_mass << std::endl;
  std::cout << "Center of mass: (" << center_of_mass.x << ", "
            << center_of_mass.y << ", " << center_of_mass.z << ")" << std::endl;
  std::cout << "Average velocity: (" << avg_velocity.x << ", " << avg_velocity.y
            << ", " << avg_velocity.z << ")" << std::endl;
  std::cout << "Total energy: " << total_energy << std::endl;
  std::cout << "==============================\n" << std::endl;
}
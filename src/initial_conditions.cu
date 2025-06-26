#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "initial_conditions.h"
#include <iostream>
#include <random>
#include <vector>

// Static member initialization
std::mt19937 InitialConditions::rng(42); // Fixed seed for reproducibility
std::uniform_real_distribution<float> InitialConditions::dist(-1.0f, 1.0f);

float3 InitialConditions::randomDirection() {
  float theta = dist(rng) * M_PI;      // 0 to π
  float phi = dist(rng) * 2.0f * M_PI; // 0 to 2π

  float sin_theta = sinf(theta);
  return make_float3(sin_theta * cosf(phi), sin_theta * sinf(phi), cosf(theta));
}

float3 InitialConditions::randomPositionInSphere(float radius) {
  // Use rejection sampling for uniform distribution in sphere
  float3 pos;
  do {
    pos =
        make_float3(dist(rng) * radius, dist(rng) * radius, dist(rng) * radius);
  } while (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z > radius * radius);

  return pos;
}

float3 InitialConditions::randomPositionInCube(float size) {
  float half_size = size * 0.5f;
  return make_float3(dist(rng) * half_size, dist(rng) * half_size,
                     dist(rng) * half_size);
}

void InitialConditions::generateRandomCube(ParticleSystem &particles,
                                           float cube_size, float max_velocity,
                                           std::pair<float, float> mass_range) {
  std::cout << "Generating random cube distribution..." << std::endl;
  std::cout << "  Cube size: " << cube_size << std::endl;
  std::cout << "  Max velocity: " << max_velocity << std::endl;
  std::cout << "  Mass range: [" << mass_range.first << ", "
            << mass_range.second << "]" << std::endl;

  std::uniform_real_distribution<float> mass_dist(mass_range.first,
                                                  mass_range.second);
  std::uniform_real_distribution<float> vel_dist(-max_velocity, max_velocity);

  Particle *host_particles = particles.getHostParticles();
  int N = particles.getNumParticles();

  for (int i = 0; i < N; i++) {
    // Random position in cube
    host_particles[i].position = randomPositionInCube(cube_size);

    // Random velocity
    host_particles[i].velocity =
        make_float3(vel_dist(rng), vel_dist(rng), vel_dist(rng));

    // Random mass
    host_particles[i].mass = mass_dist(rng);

    // Reset acceleration
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  std::cout << "Random cube generation completed!" << std::endl;
}

void InitialConditions::generateRandomSphere(
    ParticleSystem &particles, float radius, float max_velocity,
    std::pair<float, float> mass_range) {
  std::cout << "Generating random sphere distribution..." << std::endl;
  std::cout << "  Radius: " << radius << std::endl;
  std::cout << "  Max velocity: " << max_velocity << std::endl;
  std::cout << "  Mass range: [" << mass_range.first << ", "
            << mass_range.second << "]" << std::endl;

  std::uniform_real_distribution<float> mass_dist(mass_range.first,
                                                  mass_range.second);
  std::uniform_real_distribution<float> vel_dist(-max_velocity, max_velocity);

  Particle *host_particles = particles.getHostParticles();
  int N = particles.getNumParticles();

  for (int i = 0; i < N; i++) {
    // Random position in sphere
    host_particles[i].position = randomPositionInSphere(radius);

    // Random velocity
    host_particles[i].velocity =
        make_float3(vel_dist(rng), vel_dist(rng), vel_dist(rng));

    // Random mass
    host_particles[i].mass = mass_dist(rng);

    // Reset acceleration
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  std::cout << "Random sphere generation completed!" << std::endl;
}

void InitialConditions::generateGalaxyDisk(ParticleSystem &particles,
                                           float disk_radius,
                                           float disk_thickness,
                                           float central_mass,
                                           float rotation_velocity) {
  std::cout << "Generating galaxy disk distribution..." << std::endl;
  std::cout << "  Disk radius: " << disk_radius << std::endl;
  std::cout << "  Disk thickness: " << disk_thickness << std::endl;
  std::cout << "  Central mass: " << central_mass << std::endl;
  std::cout << "  Rotation velocity: " << rotation_velocity << std::endl;

  Particle *host_particles = particles.getHostParticles();
  int N = particles.getNumParticles();

  // First particle is the central massive object
  if (N > 0) {
    host_particles[0].position = make_float3(0.0f, 0.0f, 0.0f);
    host_particles[0].velocity = make_float3(0.0f, 0.0f, 0.0f);
    host_particles[0].mass = central_mass;
    host_particles[0].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  // Remaining particles form the disk
  std::uniform_real_distribution<float> radius_dist(0.1f, disk_radius);
  std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
  std::uniform_real_distribution<float> z_dist(-disk_thickness / 2.0f,
                                               disk_thickness / 2.0f);
  std::uniform_real_distribution<float> mass_dist(0.1f, 1.0f);

  for (int i = 1; i < N; i++) {
    // Random position in disk
    float r = radius_dist(rng);
    float theta = angle_dist(rng);
    float z = z_dist(rng);

    host_particles[i].position =
        make_float3(r * cosf(theta), r * sinf(theta), z);

    // Circular orbital velocity (approximately)
    float orbital_speed = rotation_velocity * sqrtf(central_mass / r);
    host_particles[i].velocity = make_float3(-orbital_speed * sinf(theta),
                                             orbital_speed * cosf(theta), 0.0f);

    // Random mass for disk particles
    host_particles[i].mass = mass_dist(rng);

    // Reset acceleration
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  std::cout << "Galaxy disk generation completed!" << std::endl;
}

void InitialConditions::generateCollidingClusters(ParticleSystem &particles,
                                                  float separation,
                                                  float cluster_radius,
                                                  float collision_velocity) {
  std::cout << "Generating colliding clusters..." << std::endl;
  std::cout << "  Separation: " << separation << std::endl;
  std::cout << "  Cluster radius: " << cluster_radius << std::endl;
  std::cout << "  Collision velocity: " << collision_velocity << std::endl;

  Particle *host_particles = particles.getHostParticles();
  int N = particles.getNumParticles();
  int half_N = N / 2;

  std::uniform_real_distribution<float> mass_dist(0.5f, 2.0f);
  std::uniform_real_distribution<float> vel_dist(-0.5f, 0.5f);

  // First cluster (left side)
  float3 cluster1_center = make_float3(-separation / 2.0f, 0.0f, 0.0f);
  float3 cluster1_velocity = make_float3(collision_velocity, 0.0f, 0.0f);

  for (int i = 0; i < half_N; i++) {
    // Random position in first cluster
    float3 random_offset = randomPositionInSphere(cluster_radius);
    host_particles[i].position =
        make_float3(cluster1_center.x + random_offset.x,
                    cluster1_center.y + random_offset.y,
                    cluster1_center.z + random_offset.z);

    // Cluster velocity plus small random component
    host_particles[i].velocity =
        make_float3(cluster1_velocity.x + vel_dist(rng),
                    cluster1_velocity.y + vel_dist(rng),
                    cluster1_velocity.z + vel_dist(rng));

    host_particles[i].mass = mass_dist(rng);
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  // Second cluster (right side)
  float3 cluster2_center = make_float3(separation / 2.0f, 0.0f, 0.0f);
  float3 cluster2_velocity = make_float3(-collision_velocity, 0.0f, 0.0f);

  for (int i = half_N; i < N; i++) {
    // Random position in second cluster
    float3 random_offset = randomPositionInSphere(cluster_radius);
    host_particles[i].position =
        make_float3(cluster2_center.x + random_offset.x,
                    cluster2_center.y + random_offset.y,
                    cluster2_center.z + random_offset.z);

    // Cluster velocity plus small random component
    host_particles[i].velocity =
        make_float3(cluster2_velocity.x + vel_dist(rng),
                    cluster2_velocity.y + vel_dist(rng),
                    cluster2_velocity.z + vel_dist(rng));

    host_particles[i].mass = mass_dist(rng);
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }

  std::cout << "Colliding clusters generation completed!" << std::endl;
}

void InitialConditions::generateBenchmarkParticles(ParticleSystem &particles) {
  // Simple, fast generation for benchmarking purposes
  Particle *host_particles = particles.getHostParticles();
  int N = particles.getNumParticles();

  for (int i = 0; i < N; i++) {
    float x = static_cast<float>(i % 100) / 10.0f - 5.0f;         // -5 to 5
    float y = static_cast<float>((i / 100) % 100) / 10.0f - 5.0f; // -5 to 5
    float z = static_cast<float>(i / 10000) / 10.0f - 5.0f;       // -5 to 5

    host_particles[i].position = make_float3(x, y, z);
    host_particles[i].velocity = make_float3(0.1f, 0.1f, 0.1f);
    host_particles[i].mass = 1.0f;
    host_particles[i].acceleration = make_float3(0.0f, 0.0f, 0.0f);
  }
}
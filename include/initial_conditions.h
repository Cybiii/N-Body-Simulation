#pragma once

#include "particle.h"
#include <random>

/**
 * Initial condition generators for N-body simulations
 */
class InitialConditions {
public:
  /**
   * Generate random particles in a cube
   * @param particles Particle system to initialize
   * @param cube_size Size of the cube (particles will be in [-cube_size/2,
   * cube_size/2])
   * @param max_velocity Maximum initial velocity magnitude
   * @param mass_range Pair of (min_mass, max_mass)
   */
  static void
  generateRandomCube(ParticleSystem &particles, float cube_size = 10.0f,
                     float max_velocity = 1.0f,
                     std::pair<float, float> mass_range = {0.5f, 2.0f});

  /**
   * Generate particles in a sphere
   * @param particles Particle system to initialize
   * @param radius Radius of the sphere
   * @param max_velocity Maximum initial velocity magnitude
   * @param mass_range Pair of (min_mass, max_mass)
   */
  static void
  generateRandomSphere(ParticleSystem &particles, float radius = 5.0f,
                       float max_velocity = 1.0f,
                       std::pair<float, float> mass_range = {0.5f, 2.0f});

  /**
   * Generate a simple galaxy disk
   * @param particles Particle system to initialize
   * @param disk_radius Radius of the disk
   * @param disk_thickness Thickness of the disk
   * @param central_mass Mass of the central body
   * @param rotation_velocity Base rotational velocity
   */
  static void generateGalaxyDisk(ParticleSystem &particles,
                                 float disk_radius = 8.0f,
                                 float disk_thickness = 0.5f,
                                 float central_mass = 100.0f,
                                 float rotation_velocity = 2.0f);

  /**
   * Generate two colliding galaxy clusters
   * @param particles Particle system to initialize
   * @param separation Initial separation between clusters
   * @param cluster_radius Radius of each cluster
   * @param collision_velocity Initial velocity towards each other
   */
  static void generateCollidingClusters(ParticleSystem &particles,
                                        float separation = 15.0f,
                                        float cluster_radius = 3.0f,
                                        float collision_velocity = 1.5f);

  /**
   * Generate particles for benchmarking (simple, fast generation)
   */
  static void generateBenchmarkParticles(ParticleSystem &particles);

private:
  static std::mt19937 rng;
  static std::uniform_real_distribution<float> dist;

  // Helper functions
  static float3 randomDirection();
  static float3 randomPositionInSphere(float radius);
  static float3 randomPositionInCube(float size);
};
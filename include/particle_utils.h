#pragma once

#include "particle.h"

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
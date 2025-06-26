#pragma once

#include "nbody_simulation.h"
#include "particle.h"
#include <cuda_runtime.h>
#include <vector_types.h>


extern "C" void launch_copy_positions_to_buffer(float2 *pbo_buffer,
                                                Particle *particles,
                                                int num_particles);

// Generates colors based on velocity and interleaves pos/color into a VBO
extern "C" void launch_compute_colors_and_interleave(float4 *d_vbo_buffer,
                                                     Particle *d_particles,
                                                     int particle_count,
                                                     float max_velocity_sq);
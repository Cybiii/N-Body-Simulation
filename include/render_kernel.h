#pragma once

#include "particle.h"
#include <vector_types.h>

extern "C" void launch_copy_positions_to_buffer(float2 *pbo_buffer,
                                                Particle *particles,
                                                int num_particles);
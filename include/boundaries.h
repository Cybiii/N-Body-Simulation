#pragma once

#include "particle.h"

// Forward declaration of the CUDA kernel launcher
void launch_enforce_boundaries(Particle *d_particles, int N, float bound_size);
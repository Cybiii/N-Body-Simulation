#include "barnes_hut_force.h"
#include "particle_utils.h"
#include <chrono>
#include <iostream>

/**
 * Barnes-Hut force calculation implementation
 * This replaces O(N²) with O(N log N) using octree approximation
 */

BarnesHutForce::BarnesHutForce(float theta, float softening)
    : theta(theta), softening(softening), last_force_calc_time(0.0) {

  // Create CUDA events for timing
  checkCudaError(cudaEventCreate(&start_event),
                 "create start event for Barnes-Hut");
  checkCudaError(cudaEventCreate(&stop_event),
                 "create stop event for Barnes-Hut");

  std::cout << "Barnes-Hut force calculator initialized:" << std::endl;
  std::cout << "  Theta (approximation parameter): " << theta << std::endl;
  std::cout << "  Softening factor: " << softening << std::endl;
}

void BarnesHutForce::calculateForces(ParticleSystem &particles,
                                     const Octree &octree, float G_constant) {
  int N = particles.getNumParticles();
  if (N == 0 || octree.getNumNodes() == 0) {
    std::cout << "Warning: No particles or nodes for force calculation"
              << std::endl;
    return;
  }

  // Record start time
  checkCudaError(cudaEventRecord(start_event), "record Barnes-Hut start event");

  // Extract particle data
  Particle *device_particles = particles.getDeviceParticles();

  // Allocate temporary arrays for positions, masses, and accelerations
  float3 *d_positions;
  float *d_masses;
  float3 *d_accelerations;

  checkCudaError(cudaMalloc(&d_positions, N * sizeof(float3)),
                 "positions for Barnes-Hut");
  checkCudaError(cudaMalloc(&d_masses, N * sizeof(float)),
                 "masses for Barnes-Hut");
  checkCudaError(cudaMalloc(&d_accelerations, N * sizeof(float3)),
                 "accelerations for Barnes-Hut");

  // Extract data from the main particle buffer into temporary flat arrays
  launch_extract_positions(device_particles, d_positions, N);
  launch_extract_masses(device_particles, d_masses, N);

  // Reset accelerations to zero before accumulation
  launch_reset_accelerations_bh(d_accelerations, N);

  // Calculate forces using Barnes-Hut algorithm
  launch_barnes_hut_force_calculation(
      d_positions, d_masses, d_accelerations, N, octree.getDeviceNodes(),
      octree.getNumNodes(), octree.getDeviceParticleIndices(),
      theta * theta,         // theta squared for efficiency
      softening * softening, // softening squared
      G_constant);

  // The new kernel computes accelerations in the original particle order.
  // Therefore, we use the simple update kernel, not the reordered one.
  launch_update_accelerations(device_particles, d_accelerations, N);

  // Record end time
  checkCudaError(cudaEventRecord(stop_event), "record Barnes-Hut stop event");
  checkCudaError(cudaEventSynchronize(stop_event),
                 "synchronize Barnes-Hut stop event");

  // Calculate elapsed time
  float elapsed_time_ms;
  checkCudaError(
      cudaEventElapsedTime(&elapsed_time_ms, start_event, stop_event),
      "get Barnes-Hut elapsed time");
  last_force_calc_time = elapsed_time_ms / 1000.0; // Convert to seconds

  // Clean up
  cudaFree(d_positions);
  cudaFree(d_masses);
  cudaFree(d_accelerations);
}

void BarnesHutForce::printPerformanceStats() const {
  std::cout << "\n=== Barnes-Hut Performance Statistics ===" << std::endl;
  std::cout << "Last force calculation time: " << last_force_calc_time * 1000.0
            << " ms" << std::endl;
  std::cout << "Theta parameter: " << theta << std::endl;
  std::cout << "Softening factor: " << softening << std::endl;
  std::cout << "======================================\n" << std::endl;
}

/**
 * CUDA kernels for Barnes-Hut force calculation
 */

// Kernel to reset accelerations
__global__ void reset_accelerations_kernel(float3 *accelerations, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    accelerations[idx] = make_float3(0.0f, 0.0f, 0.0f);
  }
}

// Main Barnes-Hut force calculation kernel
__global__ void
barnes_hut_force_kernel(const float3 *positions, const float *masses,
                        float3 *accelerations, int N, const OctreeNode *nodes,
                        int num_nodes, const int *particle_indices,
                        float theta_sq, float softening_sq, float G_constant) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;

  float3 my_pos = positions[i];
  float my_mass = masses[i];
  float3 acc = make_float3(0.0f, 0.0f, 0.0f);

  TraversalStack stack;
  if (num_nodes > 0) {
    stack.push(0); // Start traversal at the root (node 0)
  }

  while (!stack.isEmpty()) {
    int node_idx = stack.pop();
    if (node_idx < 0 || node_idx >= num_nodes)
      continue;

    const OctreeNode &node = nodes[node_idx];

    // Vector from particle i to node's center of mass
    float3 r_vec = node.center_of_mass - my_pos;
    float dist_sq = dot(r_vec, r_vec);

    if (node.isLeaf()) {
      // It's a leaf node. Iterate over the particles in this leaf.
      for (int k = 0; k < node.particle_count; ++k) {
        int sorted_particle_idx = node.particle_start_idx + k;
        if (sorted_particle_idx >= N)
          continue; // Boundary check

        int original_particle_idx = particle_indices[sorted_particle_idx];
        if (original_particle_idx == i)
          continue; // Skip self-interaction

        // Direct particle-particle interaction
        float3 p_pos = positions[original_particle_idx];
        float3 r_vec_p = p_pos - my_pos;
        float p_dist_sq = dot(r_vec_p, r_vec_p) + softening_sq;

        if (p_dist_sq > 1e-9f) {
          float inv_dist = rsqrtf(p_dist_sq);
          float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
          float p_mass = masses[original_particle_idx];
          float force_scalar = G_constant * my_mass * p_mass * inv_dist_cubed;
          acc += r_vec_p * force_scalar;
        }
      }
    } else {
      // It's an internal node. Check the Barnes-Hut criterion.
      float3 node_size_vec = node.getSize();
      float node_size_sq =
          dot(node_size_vec, node_size_vec); // Using squared size

      // s/d < theta  -> s^2 < d^2 * theta^2
      if (node_size_sq < dist_sq * theta_sq) {
        // Node is far enough away, approximate it as a single mass.
        float dist_with_softening_sq = dist_sq + softening_sq;
        if (dist_with_softening_sq > 1e-9f) {
          float inv_dist = rsqrtf(dist_with_softening_sq);
          float inv_dist_cubed = inv_dist * inv_dist * inv_dist;
          float force_mag =
              G_constant * my_mass * node.total_mass * inv_dist_cubed;
          acc += r_vec * force_mag;
        }
      } else {
        // Node is too close, traverse its children.
        if (node.first_child_idx >= 0) {
          for (int child = 0; child < 8; ++child) {
            int child_idx = node.first_child_idx + child;
            if (child_idx < num_nodes && nodes[child_idx].particle_count > 0) {
              if (!stack.isFull()) {
                stack.push(child_idx);
              }
            }
          }
        }
      }
    }
  }

  // Final acceleration is F/m
  if (my_mass > 0.0f) {
    acc = acc * (1.0f / my_mass);
  } else {
    acc = make_float3(0.0f, 0.0f, 0.0f);
  }
  accelerations[i] = acc;
}

/**
 * Wrapper functions for kernel launches
 */
extern "C" void launch_reset_accelerations_bh(float3 *accelerations, int N) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  reset_accelerations_kernel<<<blocksPerGrid, threadsPerBlock>>>(accelerations,
                                                                 N);
  checkCudaError(cudaGetLastError(), "reset accelerations kernel launch");
}

extern "C" void launch_barnes_hut_force_calculation(
    const float3 *positions, const float *masses, float3 *accelerations, int N,
    const OctreeNode *nodes, int num_nodes, const int *particle_indices,
    float theta_sq, float softening_sq, float G_constant) {

  // This kernel launch is simplified because we are not reordering inside.
  // The reordering is handled by the data extraction and final update steps.
  // Therefore, the kernel can operate on flat, unsorted arrays directly.
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  barnes_hut_force_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      positions, masses, accelerations, N, nodes, num_nodes, particle_indices,
      theta_sq, softening_sq, G_constant);
}
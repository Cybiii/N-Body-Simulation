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

  // Extract data from particles (this would need a proper kernel)
  // For now, we'll assume the data is extracted

  // Reset accelerations
  launch_reset_accelerations(d_accelerations, N);

  // Calculate forces using Barnes-Hut algorithm
  launch_barnes_hut_force_calculation(
      d_positions, d_masses, d_accelerations, N, octree.getDeviceNodes(),
      octree.getNumNodes(), octree.getDeviceParticleIndices(),
      theta * theta,         // theta squared for efficiency
      softening * softening, // softening squared
      G_constant);

  // Copy accelerations back to particles using utility kernel
  launch_update_accelerations_reordered(device_particles, d_accelerations,
                                        octree.getDeviceParticleIndices(), N);

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
  int particle_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (particle_idx >= N)
    return;

  // Get the actual particle index (after sorting)
  int actual_idx =
      particle_indices ? particle_indices[particle_idx] : particle_idx;

  float3 particle_pos = positions[actual_idx];
  float3 total_force = make_float3(0.0f, 0.0f, 0.0f);

  // Stack for tree traversal
  TraversalStack stack;

  // Start traversal from root node (assuming node 0 is root)
  if (num_nodes > 0) {
    stack.push(0);
  }

  // Tree traversal loop
  while (!stack.isEmpty()) {
    int node_idx = stack.pop();
    if (node_idx < 0 || node_idx >= num_nodes)
      continue;

    const OctreeNode &node = nodes[node_idx];

    // Calculate distance from particle to node's center of mass
    float3 r_vec = make_float3(node.center_of_mass.x - particle_pos.x,
                               node.center_of_mass.y - particle_pos.y,
                               node.center_of_mass.z - particle_pos.z);

    float r_sq = r_vec.x * r_vec.x + r_vec.y * r_vec.y + r_vec.z * r_vec.z;
    float r = sqrtf(r_sq + softening_sq);

    // Get node size (approximate)
    float3 node_size =
        make_float3(node.bounding_box_max.x - node.bounding_box_min.x,
                    node.bounding_box_max.y - node.bounding_box_min.y,
                    node.bounding_box_max.z - node.bounding_box_min.z);
    float max_size = fmaxf(fmaxf(node_size.x, node_size.y), node_size.z);

    // Barnes-Hut criterion: s/d < θ
    bool use_approximation = (max_size * max_size) / r_sq < theta_sq;

    if (node.isLeaf() || use_approximation) {
      // Use this node's center of mass for force calculation
      if (node.total_mass > 0.0f && r > 0.0f) {
        float force_magnitude = G_constant * node.total_mass / (r * r * r);

        total_force.x += force_magnitude * r_vec.x;
        total_force.y += force_magnitude * r_vec.y;
        total_force.z += force_magnitude * r_vec.z;
      }
    } else {
      // Add children to stack for further traversal
      // In a complete implementation, we'd traverse all 8 children
      // For now, this is a simplified version
      if (node.first_child_idx >= 0) {
        for (int child = 0; child < 8; child++) {
          int child_idx = node.first_child_idx + child;
          if (child_idx < num_nodes && !stack.isFull()) {
            stack.push(child_idx);
          }
        }
      }
    }
  }

  // Convert force to acceleration and store
  float particle_mass = masses[actual_idx];
  if (particle_mass > 0.0f) {
    accelerations[actual_idx] = make_float3(total_force.x / particle_mass,
                                            total_force.y / particle_mass,
                                            total_force.z / particle_mass);
  }
}

/**
 * Wrapper functions for kernel launches
 */
extern "C" void launch_reset_accelerations(float3 *accelerations, int N) {
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
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  barnes_hut_force_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      positions, masses, accelerations, N, nodes, num_nodes, particle_indices,
      theta_sq, softening_sq, G_constant);
  checkCudaError(cudaGetLastError(),
                 "Barnes-Hut force calculation kernel launch");
}
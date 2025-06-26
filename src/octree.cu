#include "octree.h"
#include "particle.h"
#include "particle_utils.h"
#include <algorithm>
#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

/**
 * Octree implementation using GPU-parallel Morton code construction
 * This is the core of the Barnes-Hut algorithm acceleration
 */

Octree::Octree(int max_particles, int max_depth)
    : max_particles(max_particles), max_depth(max_depth), num_particles(0),
      num_nodes(0), d_nodes(nullptr), d_morton_codes(nullptr),
      d_particle_indices(nullptr), d_temp_storage(nullptr) {

  // Estimate maximum number of nodes (worst case: 8 * N for complete octree)
  // In practice, much less, but we allocate conservatively
  max_nodes = max_particles * 8;

  allocateMemory();

  std::cout << "Octree initialized:" << std::endl;
  std::cout << "  Max particles: " << max_particles << std::endl;
  std::cout << "  Max depth: " << max_depth << std::endl;
  std::cout << "  Max nodes allocated: " << max_nodes << std::endl;
  std::cout << "  Memory allocated: "
            << (max_nodes * sizeof(OctreeNode) +
                max_particles * sizeof(MortonCode::Code) +
                max_particles * sizeof(int)) /
                   (1024.0 * 1024.0)
            << " MB" << std::endl;
}

Octree::~Octree() {
  if (d_nodes)
    cudaFree(d_nodes);
  if (d_morton_codes)
    cudaFree(d_morton_codes);
  if (d_particle_indices)
    cudaFree(d_particle_indices);
  if (d_temp_storage)
    cudaFree(d_temp_storage);
}

void Octree::allocateMemory() {
  // Allocate device memory
  checkCudaError(cudaMalloc(&d_nodes, max_nodes * sizeof(OctreeNode)),
                 "octree nodes allocation");

  checkCudaError(
      cudaMalloc(&d_morton_codes, max_particles * sizeof(MortonCode::Code)),
      "Morton codes allocation");

  checkCudaError(cudaMalloc(&d_particle_indices, max_particles * sizeof(int)),
                 "particle indices allocation");

  checkCudaError(cudaMalloc(&d_temp_storage, max_particles * sizeof(int) *
                                                 4), // Extra space for sorting
                 "temporary storage allocation");

  // Allocate host memory
  h_nodes.reserve(max_nodes);
  h_morton_codes.reserve(max_particles);
  h_particle_indices.reserve(max_particles);
}

void Octree::buildFromParticles(ParticleSystem &particles) {
  num_particles = particles.getNumParticles();

  if (num_particles == 0) {
    std::cout << "Warning: No particles to build octree from" << std::endl;
    return;
  }

  if (num_particles > max_particles) {
    std::cerr << "Error: Too many particles (" << num_particles
              << ") for octree capacity (" << max_particles << ")" << std::endl;
    return;
  }

  // Step 1: Compute global bounding box
  computeBoundingBox(particles);

  // Step 2: Generate Morton codes for all particles
  generateMortonCodes(particles);

  // Step 3: Sort particles by Morton code
  sortParticlesByMorton();

  // Step 4: Build tree structure from sorted Morton codes
  buildTreeStructure();

  // Step 5: Compute centers of mass for all nodes
  computeCentersOfMass(particles);
}

void Octree::computeBoundingBox(ParticleSystem &particles) {
  // Get particle positions
  Particle *device_particles = particles.getDeviceParticles();
  float3 *d_positions;
  checkCudaError(cudaMalloc(&d_positions, num_particles * sizeof(float3)),
                 "position array allocation");
  launch_extract_positions(device_particles, d_positions, num_particles);

  // Allocate device memory for bounding box
  float3 *d_bbox_min;
  float3 *d_bbox_max;
  checkCudaError(cudaMalloc(&d_bbox_min, sizeof(float3)),
                 "bbox min allocation");
  checkCudaError(cudaMalloc(&d_bbox_max, sizeof(float3)),
                 "bbox max allocation");

  // Initialize with opposite infinities
  float3 h_bbox_min_init = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 h_bbox_max_init = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  checkCudaError(cudaMemcpy(d_bbox_min, &h_bbox_min_init, sizeof(float3),
                            cudaMemcpyHostToDevice),
                 "init bbox_min");
  checkCudaError(cudaMemcpy(d_bbox_max, &h_bbox_max_init, sizeof(float3),
                            cudaMemcpyHostToDevice),
                 "init bbox_max");

  // Launch bounding box computation
  launch_compute_bounding_box(d_positions, num_particles, d_bbox_min,
                              d_bbox_max);

  // Copy results back to host
  checkCudaError(cudaMemcpy(&global_bbox_min, d_bbox_min, sizeof(float3),
                            cudaMemcpyDeviceToHost),
                 "copy bbox min to host");
  checkCudaError(cudaMemcpy(&global_bbox_max, d_bbox_max, sizeof(float3),
                            cudaMemcpyDeviceToHost),
                 "copy bbox max to host");

  // Expand bounding box slightly to avoid edge cases
  float3 expansion = make_float3(0.01f, 0.01f, 0.01f);
  global_bbox_min.x -= expansion.x;
  global_bbox_min.y -= expansion.y;
  global_bbox_min.z -= expansion.z;
  global_bbox_max.x += expansion.x;
  global_bbox_max.y += expansion.y;
  global_bbox_max.z += expansion.z;

  // Clean up
  cudaFree(d_positions);
  cudaFree(d_bbox_min);
  cudaFree(d_bbox_max);
}

void Octree::generateMortonCodes(ParticleSystem &particles) {
  Particle *device_particles = particles.getDeviceParticles();
  float3 *d_positions;
  checkCudaError(cudaMalloc(&d_positions, num_particles * sizeof(float3)),
                 "position array allocation for Morton codes");
  launch_extract_positions(device_particles, d_positions, num_particles);

  // Launch Morton code generation
  launch_generate_morton_codes(d_positions, num_particles, global_bbox_min,
                               global_bbox_max, d_morton_codes);

  cudaFree(d_positions);
}

void Octree::sortParticlesByMorton() {
  // Initialize particle indices (0, 1, 2, ..., N-1)
  thrust::device_ptr<int> indices_ptr(d_particle_indices);
  thrust::sequence(indices_ptr, indices_ptr + num_particles);

  // Sort particle indices by Morton code using Thrust
  thrust::device_ptr<MortonCode::Code> morton_ptr(d_morton_codes);

  try {
    thrust::sort_by_key(morton_ptr, morton_ptr + num_particles, indices_ptr);
  } catch (const thrust::system_error &e) {
    std::cerr << "Thrust error during sorting: " << e.what() << std::endl;
    throw;
  }
}

void Octree::buildTreeStructure() {
  int *d_num_nodes;
  checkCudaError(cudaMalloc(&d_num_nodes, sizeof(int)),
                 "node counter allocation");
  checkCudaError(cudaMemset(d_num_nodes, 0, sizeof(int)), "node counter reset");

  launch_build_tree_structure(d_morton_codes, num_particles, d_nodes,
                              d_num_nodes);

  // Copy node count back
  checkCudaError(
      cudaMemcpy(&num_nodes, d_num_nodes, sizeof(int), cudaMemcpyDeviceToHost),
      "copy node count to host");

  cudaFree(d_num_nodes);
}

void Octree::computeCentersOfMass(ParticleSystem &particles) {
  Particle *device_particles = particles.getDeviceParticles();

  float3 *d_positions;
  float *d_masses;
  checkCudaError(cudaMalloc(&d_positions, num_particles * sizeof(float3)),
                 "CoM position allocation");
  checkCudaError(cudaMalloc(&d_masses, num_particles * sizeof(float)),
                 "CoM mass allocation");

  launch_extract_positions(device_particles, d_positions, num_particles);
  launch_extract_masses(device_particles, d_masses, num_particles);

  launch_compute_centers_of_mass(d_positions, d_masses, d_particle_indices,
                                 num_particles, d_nodes, num_nodes);

  cudaFree(d_positions);
  cudaFree(d_masses);
}

void Octree::updateNodeProperties(ParticleSystem &particles) {
  // This would be called after particles move to update the tree
  // For now, just recompute centers of mass
  computeCentersOfMass(particles);
}

void Octree::copyToHost() {
  if (num_nodes > 0) {
    h_nodes.resize(num_nodes);
    checkCudaError(cudaMemcpy(h_nodes.data(), d_nodes,
                              num_nodes * sizeof(OctreeNode),
                              cudaMemcpyDeviceToHost),
                   "copy nodes to host");
  }

  if (num_particles > 0) {
    h_morton_codes.resize(num_particles);
    h_particle_indices.resize(num_particles);

    checkCudaError(cudaMemcpy(h_morton_codes.data(), d_morton_codes,
                              num_particles * sizeof(MortonCode::Code),
                              cudaMemcpyDeviceToHost),
                   "copy Morton codes to host");

    checkCudaError(cudaMemcpy(h_particle_indices.data(), d_particle_indices,
                              num_particles * sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "copy particle indices to host");
  }
}

void Octree::printTreeStatistics() const {
  std::cout << "\n=== Octree Statistics ===" << std::endl;
  std::cout << "Number of particles: " << num_particles << std::endl;
  std::cout << "Number of nodes: " << num_nodes << std::endl;
  std::cout << "Maximum depth: " << max_depth << std::endl;
  std::cout << "Memory usage: "
            << (num_nodes * sizeof(OctreeNode) +
                num_particles * sizeof(MortonCode::Code) +
                num_particles * sizeof(int)) /
                   (1024.0 * 1024.0)
            << " MB" << std::endl;

  if (!h_nodes.empty()) {
    int leaf_nodes = 0;
    int internal_nodes = 0;

    for (const auto &node : h_nodes) {
      if (node.isLeaf()) {
        leaf_nodes++;
      } else {
        internal_nodes++;
      }
    }

    std::cout << "Leaf nodes: " << leaf_nodes << std::endl;
    std::cout << "Internal nodes: " << internal_nodes << std::endl;

    if (num_particles > 0) {
      std::cout << "Average particles per leaf: "
                << static_cast<float>(num_particles) / leaf_nodes << std::endl;
    }
  }

  std::cout << "Bounding box: (" << global_bbox_min.x << ", "
            << global_bbox_min.y << ", " << global_bbox_min.z << ") to ("
            << global_bbox_max.x << ", " << global_bbox_max.y << ", "
            << global_bbox_max.z << ")" << std::endl;
  std::cout << "========================\n" << std::endl;
}
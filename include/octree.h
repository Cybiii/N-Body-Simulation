#pragma once

#include <cuda_runtime.h>
#include <vector>

/**
 * GPU-friendly octree node structure for Barnes-Hut algorithm
 * Uses flat array representation to avoid pointer chasing on GPU
 */
struct OctreeNode {
  float3 center_of_mass;   // Center of mass of all particles in this node
  float total_mass;        // Total mass of all particles in this node
  float3 bounding_box_min; // Minimum corner of bounding box
  float3 bounding_box_max; // Maximum corner of bounding box

  int first_child_idx;    // Index of first child in node array (-1 if leaf)
  int particle_start_idx; // Start index in sorted particle array (for leaves)
  int particle_count;     // Number of particles in this node
  int level;              // Tree level (0 = root)

  // Constructor
  __host__ __device__ OctreeNode()
      : center_of_mass(make_float3(0.0f, 0.0f, 0.0f)), total_mass(0.0f),
        bounding_box_min(make_float3(0.0f, 0.0f, 0.0f)),
        bounding_box_max(make_float3(0.0f, 0.0f, 0.0f)), first_child_idx(-1),
        particle_start_idx(-1), particle_count(0), level(0) {}

  // Check if this is a leaf node
  __host__ __device__ bool isLeaf() const { return first_child_idx == -1; }

  // Get the size of the bounding box
  __host__ __device__ float3 getSize() const {
    return make_float3(bounding_box_max.x - bounding_box_min.x,
                       bounding_box_max.y - bounding_box_min.y,
                       bounding_box_max.z - bounding_box_min.z);
  }

  // Get center of bounding box
  __host__ __device__ float3 getCenter() const {
    return make_float3((bounding_box_min.x + bounding_box_max.x) * 0.5f,
                       (bounding_box_min.y + bounding_box_max.y) * 0.5f,
                       (bounding_box_min.z + bounding_box_max.z) * 0.5f);
  }
};

/**
 * Morton code utilities for Z-order curve mapping
 * Essential for efficient parallel octree construction
 */
class MortonCode {
public:
  // 32-bit Morton code type
  using Code = uint32_t;

  // Maximum coordinate value that can be encoded (2^10 - 1 = 1023)
  static constexpr int MAX_COORD = 1023;

  /**
   * Encode 3D coordinates to Morton code
   * @param x, y, z Normalized coordinates [0, MAX_COORD]
   * @return 30-bit Morton code (10 bits per dimension)
   */
  __host__ __device__ static Code encode(uint32_t x, uint32_t y, uint32_t z);

  /**
   * Decode Morton code to 3D coordinates
   * @param code Morton code to decode
   * @return Decoded coordinates as uint3
   */
  __host__ __device__ static uint3 decode(Code code);

  /**
   * Convert world coordinates to Morton code
   * @param pos World position
   * @param bbox_min, bbox_max Bounding box of the simulation
   * @return Morton code
   */
  __host__ __device__ static Code worldToMorton(const float3 &pos,
                                                const float3 &bbox_min,
                                                const float3 &bbox_max);

  /**
   * Find the longest common prefix between two Morton codes
   * Used for determining parent-child relationships in tree construction
   */
  __host__ __device__ static int longestCommonPrefix(Code a, Code b);

private:
  // Utility functions for bit interleaving
  __host__ __device__ static uint32_t expandBits(uint32_t v);
  __host__ __device__ static uint32_t compactBits(uint32_t v);
};

/**
 * Octree construction and management class
 */
class Octree {
public:
  Octree(int max_particles, int max_depth = 20);
  ~Octree();

  /**
   * Build octree from particle positions using Morton codes
   * This is the core GPU-parallel algorithm
   */
  void buildFromParticles(class ParticleSystem &particles);

  /**
   * Update center of mass and total mass for all nodes
   * Bottom-up traversal after tree construction
   */
  void updateNodeProperties(class ParticleSystem &particles);

  // Accessors
  OctreeNode *getDeviceNodes() const { return d_nodes; }
  OctreeNode *getHostNodes() { return h_nodes.data(); }
  MortonCode::Code *getDeviceMortonCodes() const { return d_morton_codes; }
  int *getDeviceParticleIndices() const { return d_particle_indices; }
  int getNumNodes() const { return num_nodes; }
  int getNumParticles() const { return num_particles; }
  float3 getBoundingBoxMin() const { return global_bbox_min; }
  float3 getBoundingBoxMax() const { return global_bbox_max; }

  // Debug and analysis
  void printTreeStatistics() const;
  void copyToHost();

private:
  // Host data
  std::vector<OctreeNode> h_nodes;
  std::vector<MortonCode::Code> h_morton_codes;
  std::vector<int> h_particle_indices;

  // Device data
  OctreeNode *d_nodes;
  MortonCode::Code *d_morton_codes;
  int *d_particle_indices;
  int *d_temp_storage; // Temporary storage for various algorithms

  // Tree properties
  int max_particles;
  int max_depth;
  int num_particles;
  int num_nodes;
  int max_nodes; // Pre-allocated maximum

  // Global bounding box
  float3 global_bbox_min;
  float3 global_bbox_max;

  // Internal methods
  void allocateMemory();
  void computeBoundingBox(class ParticleSystem &particles);
  void generateMortonCodes(class ParticleSystem &particles);
  void sortParticlesByMorton();
  void buildTreeStructure();
  void computeCentersOfMass(class ParticleSystem &particles);
};

// CUDA kernel declarations
extern "C" {
void launch_compute_bounding_box(const float3 *positions, int N,
                                 float3 *bbox_min, float3 *bbox_max);

void launch_generate_morton_codes(const float3 *positions, int N,
                                  const float3 &bbox_min,
                                  const float3 &bbox_max,
                                  MortonCode::Code *morton_codes);

void launch_build_tree_structure(const MortonCode::Code *morton_codes, int N,
                                 OctreeNode *nodes, int *num_nodes);

void launch_compute_centers_of_mass(const float3 *positions,
                                    const float *masses,
                                    const int *particle_indices, int N,
                                    OctreeNode *nodes, int num_nodes);
}
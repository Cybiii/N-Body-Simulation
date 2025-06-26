#include "octree.h"
#include "particle.h"
#include <iostream>

/**
 * Morton code implementation for 3D Z-order curve
 * This is critical for efficient parallel octree construction
 */

// Expand a 10-bit integer into 30 bits by inserting 2 zeros after each bit
__host__ __device__ uint32_t MortonCode::expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Compact every 3rd bit to a 10-bit integer
__host__ __device__ uint32_t MortonCode::compactBits(uint32_t v) {
  v &= 0x49249249u;
  v = (v ^ (v >> 2)) & 0xC30C30C3u;
  v = (v ^ (v >> 4)) & 0x0F00F00Fu;
  v = (v ^ (v >> 8)) & 0xFF0000FFu;
  v = (v ^ (v >> 16)) & 0x000003FFu;
  return v;
}

// Encode 3D coordinates to Morton code
__host__ __device__ MortonCode::Code MortonCode::encode(uint32_t x, uint32_t y,
                                                        uint32_t z) {
  return (expandBits(z) << 2) + (expandBits(y) << 1) + expandBits(x);
}

// Decode Morton code to 3D coordinates
__host__ __device__ uint3 MortonCode::decode(Code code) {
  return make_uint3(compactBits(code), compactBits(code >> 1),
                    compactBits(code >> 2));
}

// Convert world coordinates to Morton code
__host__ __device__ MortonCode::Code
MortonCode::worldToMorton(const float3 &pos, const float3 &bbox_min,
                          const float3 &bbox_max) {
  // Normalize coordinates to [0, 1]
  float3 normalized =
      make_float3((pos.x - bbox_min.x) / (bbox_max.x - bbox_min.x),
                  (pos.y - bbox_min.y) / (bbox_max.y - bbox_min.y),
                  (pos.z - bbox_min.z) / (bbox_max.z - bbox_min.z));

  // Clamp to valid range and scale to integer coordinates
  uint32_t x =
      static_cast<uint32_t>(fminf(fmaxf(normalized.x, 0.0f), 1.0f) * MAX_COORD);
  uint32_t y =
      static_cast<uint32_t>(fminf(fmaxf(normalized.y, 0.0f), 1.0f) * MAX_COORD);
  uint32_t z =
      static_cast<uint32_t>(fminf(fmaxf(normalized.z, 0.0f), 1.0f) * MAX_COORD);

  return encode(x, y, z);
}

// Find longest common prefix between two Morton codes
__host__ __device__ int MortonCode::longestCommonPrefix(Code a, Code b) {
  if (a == b)
    return 32; // All bits match

  // Count leading zeros in XOR
  uint32_t xor_result = a ^ b;

#ifdef __CUDA_ARCH__
  return __clz(xor_result); // Device code: use CUDA intrinsic
#else
  // Host code: use builtin or manual implementation
  if (xor_result == 0)
    return 32;
  int count = 0;
  for (int i = 31; i >= 0; i--) {
    if (xor_result & (1u << i))
      break;
    count++;
  }
  return count;
#endif
}

/**
 * CUDA kernels for octree construction
 */

// Kernel to compute global bounding box using reduction
__global__ void compute_bounding_box_kernel(const float3 *positions, int N,
                                            float3 *bbox_min,
                                            float3 *bbox_max) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Shared memory for block-level reduction
  __shared__ float3 s_min[256];
  __shared__ float3 s_max[256];

  int tid = threadIdx.x;

  // Initialize shared memory
  if (idx < N) {
    s_min[tid] = positions[idx];
    s_max[tid] = positions[idx];
  } else {
    s_min[tid] = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    s_max[tid] = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
  }

  __syncthreads();

  // Block-level reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_min[tid] = make_float3(fminf(s_min[tid].x, s_min[tid + stride].x),
                               fminf(s_min[tid].y, s_min[tid + stride].y),
                               fminf(s_min[tid].z, s_min[tid + stride].z));
      s_max[tid] = make_float3(fmaxf(s_max[tid].x, s_max[tid + stride].x),
                               fmaxf(s_max[tid].y, s_max[tid + stride].y),
                               fmaxf(s_max[tid].z, s_max[tid + stride].z));
    }
    __syncthreads();
  }

  // Write block results to global memory
  if (tid == 0) {
    atomicMinFloat(&bbox_min->x, s_min[0].x);
    atomicMinFloat(&bbox_min->y, s_min[0].y);
    atomicMinFloat(&bbox_min->z, s_min[0].z);
    atomicMaxFloat(&bbox_max->x, s_max[0].x);
    atomicMaxFloat(&bbox_max->y, s_max[0].y);
    atomicMaxFloat(&bbox_max->z, s_max[0].z);
  }
}

// Atomic min/max for floats (if not available)
__device__ void atomicMinFloat(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fminf(val, __int_as_float(assumed))));
  } while (assumed != old);
}

__device__ void atomicMaxFloat(float *address, float val) {
  int *address_as_i = (int *)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
                    __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
}

// Kernel to generate Morton codes for all particles
__global__ void generate_morton_codes_kernel(const float3 *positions, int N,
                                             float3 bbox_min, float3 bbox_max,
                                             MortonCode::Code *morton_codes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    morton_codes[idx] =
        MortonCode::worldToMorton(positions[idx], bbox_min, bbox_max);
  }
}

// Kernel to build tree structure from sorted Morton codes
__global__ void
build_tree_structure_kernel(const MortonCode::Code *morton_codes, int N,
                            OctreeNode *nodes, int *num_nodes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= N - 1)
    return; // Need N-1 internal nodes for N leaves

  // Find the range of Morton codes that this internal node covers
  int left = idx;
  int right = idx + 1;

  // Extend the range to the right as long as codes have common prefix
  MortonCode::Code left_code = morton_codes[left];
  MortonCode::Code right_code = morton_codes[right];

  int common_prefix = MortonCode::longestCommonPrefix(left_code, right_code);

  // Find the actual range this node should cover
  int split = left;
  int step = 1;

  // Binary search to find the split point
  while (step < N) {
    int new_split = split + step;
    if (new_split < N &&
        MortonCode::longestCommonPrefix(left_code, morton_codes[new_split]) >
            common_prefix) {
      split = new_split;
      step <<= 1;
    } else {
      step >>= 1;
      if (step == 0)
        break;
    }
  }

  // This is a simplified version - full implementation would be more complex
  // For now, create a basic node structure
  int node_idx = atomicAdd(num_nodes, 1);
  if (node_idx < N * 2) { // Ensure we don't exceed allocated space
    nodes[node_idx].particle_start_idx = left;
    nodes[node_idx].particle_count = right - left + 1;
    nodes[node_idx].level =
        common_prefix / 3; // Approximate level based on common prefix
    nodes[node_idx].first_child_idx = -1; // Will be set later if needed
  }
}

// Kernel to compute centers of mass for all nodes
__global__ void compute_centers_of_mass_kernel(const float3 *positions,
                                               const float *masses,
                                               const int *particle_indices,
                                               int N, OctreeNode *nodes,
                                               int num_nodes) {
  int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (node_idx >= num_nodes)
    return;

  OctreeNode &node = nodes[node_idx];

  if (node.particle_count == 0)
    return;

  float3 center_of_mass = make_float3(0.0f, 0.0f, 0.0f);
  float total_mass = 0.0f;

  // Compute center of mass for particles in this node
  for (int i = node.particle_start_idx;
       i < node.particle_start_idx + node.particle_count; i++) {
    if (i < N) {
      int particle_idx = particle_indices ? particle_indices[i] : i;
      float mass = masses[particle_idx];
      float3 pos = positions[particle_idx];

      center_of_mass.x += pos.x * mass;
      center_of_mass.y += pos.y * mass;
      center_of_mass.z += pos.z * mass;
      total_mass += mass;
    }
  }

  if (total_mass > 0.0f) {
    node.center_of_mass = make_float3(center_of_mass.x / total_mass,
                                      center_of_mass.y / total_mass,
                                      center_of_mass.z / total_mass);
  }
  node.total_mass = total_mass;
}

/**
 * Wrapper functions for kernel launches
 */
extern "C" void launch_compute_bounding_box(const float3 *positions, int N,
                                            float3 *bbox_min,
                                            float3 *bbox_max) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize bounding box to extreme values
  float3 init_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  float3 init_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

  cudaMemcpy(bbox_min, &init_min, sizeof(float3), cudaMemcpyHostToDevice);
  cudaMemcpy(bbox_max, &init_max, sizeof(float3), cudaMemcpyHostToDevice);

  compute_bounding_box_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      positions, N, bbox_min, bbox_max);
  checkCudaError(cudaGetLastError(), "compute bounding box kernel launch");
}

extern "C" void launch_generate_morton_codes(const float3 *positions, int N,
                                             const float3 &bbox_min,
                                             const float3 &bbox_max,
                                             MortonCode::Code *morton_codes) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  generate_morton_codes_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      positions, N, bbox_min, bbox_max, morton_codes);
  checkCudaError(cudaGetLastError(), "generate Morton codes kernel launch");
}

extern "C" void
launch_build_tree_structure(const MortonCode::Code *morton_codes, int N,
                            OctreeNode *nodes, int *num_nodes) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Initialize node counter
  int zero = 0;
  cudaMemcpy(num_nodes, &zero, sizeof(int), cudaMemcpyHostToDevice);

  build_tree_structure_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      morton_codes, N, nodes, num_nodes);
  checkCudaError(cudaGetLastError(), "build tree structure kernel launch");
}

extern "C" void launch_compute_centers_of_mass(const float3 *positions,
                                               const float *masses,
                                               const int *particle_indices,
                                               int N, OctreeNode *nodes,
                                               int num_nodes) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

  compute_centers_of_mass_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      positions, masses, particle_indices, N, nodes, num_nodes);
  checkCudaError(cudaGetLastError(), "compute centers of mass kernel launch");
}
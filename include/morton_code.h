#ifndef MORTON_CODE_H
#define MORTON_CODE_H

#include <cstdint>
#include <cuda_runtime.h>

namespace MortonCode {

using Code = uint64_t;

// Maximum coordinate value for Morton encoding
const uint32_t MAX_COORD = (1 << 20) - 1;

__host__ __device__ uint32_t expandBits(uint32_t v);
__host__ __device__ uint32_t compactBits(uint32_t v);
__host__ __device__ Code encode(uint32_t x, uint32_t y, uint32_t z);
__host__ __device__ uint3 decode(Code code);
__host__ __device__ Code worldToMorton(const float3 &pos,
                                       const float3 &bbox_min,
                                       const float3 &bbox_max);
__host__ __device__ int longestCommonPrefix(Code a, Code b);

} // namespace MortonCode

#endif // MORTON_CODE_H
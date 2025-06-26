#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

// Error checking wrapper from the CUDA samples
inline void checkCudaError(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")"
              << std::endl;
    throw std::runtime_error(msg);
  }
}
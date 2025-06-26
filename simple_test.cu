#include <cuda_runtime.h>
#include <iostream>

__global__ void testKernel() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < 8) {
    printf("Hello from GPU thread %d!\n", idx);
  }
}

int main() {
  std::cout << "=== Simple CUDA Test for Phase 1 Validation ===" << std::endl;

  // Check CUDA device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return 1;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "GPU: " << deviceProp.name << std::endl;
  std::cout << "Compute Capability: " << deviceProp.major << "."
            << deviceProp.minor << std::endl;
  std::cout << "Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024)
            << " MB" << std::endl;

  // Test kernel launch
  std::cout << "\nLaunching test kernel..." << std::endl;
  testKernel<<<1, 8>>>();
  cudaDeviceSynchronize();

  std::cout << "\n=== CUDA Test Completed Successfully! ===" << std::endl;
  std::cout << "Ready to proceed with Phase 2: Barnes-Hut Octree" << std::endl;

  return 0;
}
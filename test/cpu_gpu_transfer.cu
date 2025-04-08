#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>

// kron_13: 17542263
// kron_15: 280025434
// kron_16: 1119440706
// kron_17: 4474931789

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_updates" << std::endl;
    exit(EXIT_FAILURE);
  }

  int device_id = cudaGetDevice(&device_id);
  int device_count = 0;

  cudaGetDeviceCount(&device_count);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device_id);
  std::cout << "-----CUDA Device Information-----\n";
  std::cout << "CUDA Device Count: " << device_count << "\n";
  std::cout << "CUDA Device ID: " << device_id << "\n";
  std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 
  std::cout << "CUDA Max. Shared memory per Block: " << (double)deviceProp.sharedMemPerBlockOptin / 1000 << "KB\n";

  size_t free_memory;
  size_t total_memory;

  cudaMemGetInfo(&free_memory, &total_memory);
  std::cout << "GPU Free (Available) Memory: " << (double)free_memory / 1000000000 << "GB\n";
  std::cout << "GPU Total Memory: " << (double)total_memory / 1000000000 << "GB\n";
  std::cout << "\n";

  size_t num_updates = std::stoull(argv[1]);
  double total_bytes = 2 * num_updates * sizeof(uint32_t);

  // Allocate buffers
  uint32_t *h_sketches, *d_sketches;
  uint32_t *h_edgeUpdates, *d_edgeUpdates;
  gpuErrchk(cudaMallocHost(&h_edgeUpdates, total_bytes));
  //gpuErrchk(cudaMallocHost(&h_sketches, total_bytes));
  gpuErrchk(cudaMalloc(&d_edgeUpdates, total_bytes));
  //gpuErrchk(cudaMalloc(&d_sketches, total_bytes));

  float time;
  cudaEvent_t start, stop;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  cudaStream_t stream0, stream1;

  // Initialize CudaStream
  cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

  // Data Transfer
  gpuErrchk(cudaEventRecord(start));

  gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice, stream0));
  //gpuErrchk(cudaMemcpyAsync(h_sketches, d_sketches, total_bytes, cudaMemcpyDeviceToHost, stream1));
  gpuErrchk(cudaMemcpyAsync(h_edgeUpdates, d_edgeUpdates, total_bytes, cudaMemcpyDeviceToHost, stream1));

  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  //gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice));
  //gpuErrchk(cudaMemcpy(h_sketches, d_sketches, total_bytes, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaEventRecord(stop));

  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&time, start, stop));

  std::cout << "Bytes Transferred (GB):            " << total_bytes / 1000000000 << std::endl;
  std::cout << "Data Transfer Time (s):            " << time * 0.001 << std::endl;
  std::cout << "Data Transfer Time (# of Edges/s): " << num_updates / (time * 0.001) << std::endl;
  std::cout << "Data Transfer Time (GB/s):         " << (total_bytes / 1000000000) / (time * 0.001) << std::endl;

  // Free Buffers
  cudaFree(d_edgeUpdates);
  //cudaFree(d_sketches);
}

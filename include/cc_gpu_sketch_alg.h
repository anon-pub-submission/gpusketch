#pragma once

#include <cmath>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"
#include "cuda_stream.h"

class CCGPUSketchAlg : public CCSketchAlg{
private:
  SketchParams sketchParams;
  CudaKernel cudaKernel;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads = 1024;

  // Number of CPU's graph workers
  int num_host_threads;

  // Number of batches in a buffer to accept until GPU kernel launch
  int num_batch_per_buffer = 540;
  CudaStream<CCGPUSketchAlg>** cudaStreams;

  size_t batch_size;
  double small_batch_perc = 0.0;
  size_t small_batch_size;

  std::vector<std::chrono::duration<double>> small_batch_time;

  // kron_13 = 54
  // kron_15 = 216
  // kron_16 = 540
  // kron_17 = 540
  // kron_18 = 540

public:
  CCGPUSketchAlg(node_id_t _num_nodes, size_t _num_updates, int num_threads, SketchParams _sketchParams, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(_num_nodes, _sketchParams.cudaUVM_enabled, _sketchParams.seed, _sketchParams.cudaUVM_buckets, config){ 
    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_host_threads = num_threads;
    sketchParams = _sketchParams;

    batch_size = get_desired_updates_per_batch();
    small_batch_size = batch_size * small_batch_perc;
    std::cout << "Batch Size: " << batch_size << "\n";
    std::cout << "  Small Batch Size (" << small_batch_perc << "): " << small_batch_size << "\n";

    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";
    std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 

    // Set maxBytes for GPU kernel's shared memory
    size_t maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize CUDA Streams
    cudaStreams = new CudaStream<CCGPUSketchAlg>*[num_host_threads];
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      cudaStreams[thr_id] = new CudaStream<CCGPUSketchAlg>(this, 0, _num_nodes, num_device_threads, num_batch_per_buffer, sketchParams);
      small_batch_time.push_back(std::chrono::nanoseconds::zero());
    }

    std::cout << "Num batches per buffer: " << num_batch_per_buffer << "\n";

    if (sketchParams.cudaUVM_enabled) {
      // Prefetch sketches to GPU
      gpuErrchk(cudaMemPrefetchAsync(sketchParams.cudaUVM_buckets, _num_nodes * sketchParams.num_buckets * sizeof(Bucket), device_id));
    }
  
    size_t free_memory;
    size_t total_memory;
    size_t size = 0.1;

    cudaMemGetInfo(&free_memory, &total_memory);
    std::cout << "Finished CCGPUSketchAlg's Initialization\n";
    std::cout << "Init - GPU Total Memory: " << (double)total_memory / 1000000000 << "GB\n";
    std::cout << "Init - GPU Free (Available) Memory: " << (double)free_memory / 1000000000 << "GB\n";
    std::cout << "Init - GPU Allocated Memory: " << (double)(total_memory - free_memory) / 1000000000 << "GB\n";

    //For limiting GPU Memory
    /* size_t size = 0.1;
    size_t sketch_bytes = _num_nodes * sketchParams.num_buckets * sizeof(Bucket);
    std::cout << "Allocating rest of the free memory in GPU: " << (double)(free_memory + (sketch_bytes * size)) / 1000000000 << "\n";
    void* dummy_pointer;
    gpuErrchk(cudaMalloc(&dummy_pointer, free_memory + (sketch_bytes * size)));
    cudaMemGetInfo(&free_memory, &total_memory);
    std::cout << "GPU Free (Available) Memory: " << (double)free_memory / 1000000000 << "GB\n";*/

    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "CCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  void flush_buffers();
  void display_time();

};
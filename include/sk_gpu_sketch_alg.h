#pragma once

#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include "cc_sketch_alg.h"
#include "cuda_kernel.cuh"
#include "cuda_stream.h"

class SKGPUSketchAlg : public CCSketchAlg{
private:
  SketchParams sketchParams;
  size_t maxBytes;

  CudaKernel cudaKernel;

  node_id_t num_nodes;
  size_t num_updates;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads = 1024;
  int num_device_blocks;

  // Number of CPU's graph workers
  int num_host_threads;

  // Maximum number of edge updates in one batch
  size_t batch_size;

  // Number of batches in a buffer to accept until GPU kernel launch
  int num_batch_per_buffer = 540;
  CudaStream<SKGPUSketchAlg>** cudaStreams;

  // List of edge ids that thread will be responsble for updating
  node_id_t *h_edgeUpdates, *d_edgeUpdates;

  vec_t *h_update_sizes, *d_update_sizes, *h_update_start_index, *d_update_start_index;
  node_id_t *h_update_src, *d_update_src;

  std::map<uint64_t, uint64_t> batch_sizes;
  std::map<uint64_t, uint64_t> batch_src;
  std::map<uint64_t, uint64_t> batch_start_index;
  std::mutex batch_mutex;

  // Atomic variables
  std::atomic<uint64_t> edgeUpdate_offset;
  std::atomic<uint64_t> batch_count;

  Sketch **delta_sketches;

  std::vector<std::chrono::duration<double>> batch_prepare_time;

public:
  SKGPUSketchAlg(bool using_gpu, node_id_t _num_nodes, size_t _num_updates, int num_threads, SketchParams _sketchParams, CCAlgConfiguration config = CCAlgConfiguration()) : CCSketchAlg(_num_nodes, _sketchParams.cudaUVM_enabled, _sketchParams.seed, _sketchParams.cudaUVM_buckets, config){ 
    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    num_nodes = _num_nodes;
    num_updates = _num_updates;

    edgeUpdate_offset = 0;
    batch_count = 0;
    
    num_host_threads = num_threads;
    sketchParams = _sketchParams;

    batch_size = get_desired_updates_per_batch();
    std::cout << "Batch Size: " << batch_size << "\n";

    gpuErrchk(cudaMallocHost(&h_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));
    // Initialize buffer with 0
    memset(h_edgeUpdates, 0, 2 * num_updates * sizeof(node_id_t));

    // Allocate memory for buffer that stores edge updates
    if (using_gpu) {
      int device_id = cudaGetDevice(&device_id);
      int device_count = 0;
      cudaGetDeviceCount(&device_count);
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device_id);
      std::cout << "CUDA Device Count: " << device_count << "\n";
      std::cout << "CUDA Device ID: " << device_id << "\n";
      std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n"; 

      // Set maxBytes for GPU kernel's shared memory
      maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t)); // Delta Sketch
      maxBytes += (batch_size * sizeof(node_id_t)); // Vertex-Based Batch
      cudaKernel.updateSharedMemory(maxBytes);
      std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

      // Initialize CUDA Streams
      cudaStreams = new CudaStream<SKGPUSketchAlg>*[num_host_threads];
      for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
        cudaStreams[thr_id] = new CudaStream<SKGPUSketchAlg>(this, 0, _num_nodes, num_device_threads, num_batch_per_buffer, sketchParams);
        batch_prepare_time.push_back(std::chrono::nanoseconds::zero());
      }

      gpuErrchk(cudaMalloc(&d_edgeUpdates, 2 * num_updates * sizeof(node_id_t)));

      if (sketchParams.cudaUVM_enabled) {
        // Prefetch sketches to GPU
        gpuErrchk(cudaMemPrefetchAsync(sketchParams.cudaUVM_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket), device_id));
      }
    }
    else {
      delta_sketches = new Sketch *[num_host_threads];
      for (size_t thr_id = 0; thr_id < num_host_threads; thr_id++) {
        delta_sketches[thr_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchParams.seed, Sketch::calc_cc_samples(num_nodes, 1));
      }
    }
    
    std::cout << "Bucket Memory Bytes: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";

    std::cout << "Finished SKGPUSketchAlg's Initialization\n";
    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "SKGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  };
  
  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  void launch_cpu_update();
  void launch_gpu_update();
  void flush_buffers();
  void launch_gpu_kernel();
  void buffer_transfer();
  void display_time();
  uint64_t get_batch_count() { 
    uint64_t temp = batch_count;
    return temp; }
};
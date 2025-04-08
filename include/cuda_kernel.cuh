#pragma once
#include <atomic>
#include <iostream>
#include "bucket.h"
#include "types.h"
#include "../src/cuda_library.cu"

typedef unsigned long long int uint64_cu;
typedef uint64_cu vec_t_cu;

// CUDA API Check
// Source: https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct SketchParams {
  // Sketch related variables
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;
  size_t seed;

  // Variables for CUDA UVM
  bool cudaUVM_enabled;
  Bucket* cudaUVM_buckets;

  // Variables for default
  Bucket* d_buckets;
};

class CudaKernel {
  public:
    /*
    *   
    *   Sketch's Update Functions
    *
    */
    void sketchUpdate(int num_threads, int num_blocks, cudaStream_t stream, node_id_t *edgeUpdates, node_id_t* update_src, vec_t* update_sizes, vec_t* update_start_index, SketchParams sketchParams);
    void single_sketchUpdate(int num_threads, int num_blocks, size_t num_batches, size_t batch_size, node_id_t* edgeUpdates, node_id_t* update_src, vec_t* update_sizes, vec_t* update_start_index, SketchParams sketchParams);

    void updateSharedMemory(size_t maxBytes);

};
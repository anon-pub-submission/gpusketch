#include <chrono>
#include <cmath>
#include <vector>

#include <sketch.h>
#include "../src/cuda_kernel.cu"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

__global__ void gpuSketchTest_kernel(int num_device_blocks, node_id_t num_nodes, size_t num_updates_per_block, node_id_t *edgeUpdates, size_t num_buckets, Bucket* buckets, size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  for (size_t i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  node_id_t node_id = blockIdx.x % num_nodes;
  for (size_t id = threadIdx.x; id < num_updates_per_block * num_columns; id += blockDim.x) {

    size_t column_id = id % num_columns;
    size_t update_id = id / num_columns;

    // Get random edge id based on current update_id
    vec_t edge_id = device_concat_pairing_fn(node_id, edgeUpdates[(blockIdx.x * num_updates_per_block) + update_id]);

    vec_hash_t checksum = bucket_get_index_hash(edge_id, sketchSeed);
    
    if ((blockIdx.x == num_device_blocks - 1)  && (column_id == 0)) {
      // Update depth 0 bucket
      bucket_update(bucket_a[num_buckets - 1], bucket_c[num_buckets - 1], edge_id, checksum);
    }

    // Update higher depth buckets
    col_hash_t depth = bucket_get_index_depth(edge_id, sketchSeed + ((column_id) * 5), bkt_per_col);
    size_t bucket_id = column_id * bkt_per_col + depth;
    if(depth < bkt_per_col)
      bucket_update(bucket_a[bucket_id], bucket_c[bucket_id], edge_id, checksum);
  }

  __syncthreads();

  for (size_t i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    atomicXor((vec_t_cu*)&buckets[(node_id * num_buckets) + i].alpha, bucket_a[i]);
    atomicXor((vec_t_cu*)&buckets[(node_id * num_buckets) + i].gamma, (vec_t_cu)bucket_c[i]);
  }
}


int main(int argc, char **argv) {
  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes" << std::endl;
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

  node_id_t num_nodes = std::atoi(argv[1]);

  // Single Sketch with size corresponding to num_nodes
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "\n";

  int num_device_threads = 1024;
  size_t num_updates_per_block = (sketchParams.num_buckets * sizeof(Bucket)) / sizeof(node_id_t);

  std::cout << "Batch Size: " << num_updates_per_block << "\n\n";

  size_t maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
  cudaFuncSetAttribute(gpuSketchTest_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);

  std::cout << "-----GPU Kernel Information-----\n";
  std::cout << "Number of threads per block: " << num_device_threads << "\n";
  std::cout << "Memory Size for buckets: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";
  std::cout << "  Allocated Shared Memory of: " << (double)maxBytes / 1000 << "KB\n";
  std::cout << "\n";

  Bucket* d_buckets;
  gpuErrchk(cudaMalloc(&d_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket)));

  node_id_t *h_edgeUpdates, *d_edgeUpdates;
  size_t max_num_updates = 4e9;
  size_t max_num_blocks = (2 * max_num_updates) / num_updates_per_block;
  size_t max_act_updates = max_num_blocks * num_updates_per_block;

  gpuErrchk(cudaMallocHost(&h_edgeUpdates, max_act_updates * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_edgeUpdates, max_act_updates * sizeof(node_id_t)));

  std::cout << "Allocating Host Memory: " << (max_act_updates * sizeof(node_id_t)) / 1e9 << "GB\n";
  std::cout << "Allocating GPU Memory: " << ((num_nodes * sketchParams.num_buckets * sizeof(Bucket)) + (max_act_updates * sizeof(node_id_t))) / 1e9 << "GB\n";

  for (size_t block_id = 0; block_id < max_num_blocks; block_id++) {
    for (size_t update_id = 0; update_id < num_updates_per_block; update_id++) {
      h_edgeUpdates[(block_id * num_updates_per_block) + update_id] = update_id;
    }
  }

  gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, max_act_updates * sizeof(node_id_t), cudaMemcpyHostToDevice));

  size_t sketchSeed = get_seed();

  float time;
  cudaEvent_t start, stop;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  std::vector<double> stream_updates = {10000, 20000, 30000, 40000, 50000,
                                        60000, 70000, 80000, 90000, 100000,
                                        200000, 300000, 400000, 500000,
                                        600000, 700000, 800000, 900000, 1000000,
                                        2000000, 3000000, 4000000, 5000000,
                                        6000000, 7000000, 8000000, 9000000, 10000000,
                                        20000000, 30000000, 40000000, 50000000,
                                        60000000, 70000000, 80000000, 90000000, 100000000,
                                        200000000, 300000000, 400000000, 500000000,
                                        600000000, 700000000, 800000000, 900000000, 1000000000,
                                        2000000000, 3000000000, 4000000000};

  for (auto& stream_update : stream_updates) {                                
    size_t num_device_blocks = (2 * stream_update) / num_updates_per_block;

    std::cout << "Number of stream updates: " << stream_update << "\n";
    std::cout << "  Number of device blocks: " << num_device_blocks << "\n";
    std::cout << "  Number of updates: " << num_updates_per_block * num_device_blocks << "\n";

    if (num_device_blocks == 0) {
      std::cout << "  Current number of stream updates too low, skipping\n";
      continue;
    }

    if (num_device_blocks > max_num_blocks) {
      std::cout << "  Current number of stream updates exceeds maximum, breaking out\n";
      break;
    }

    gpuErrchk(cudaEventRecord(start));
    gpuSketchTest_kernel<<<num_device_blocks, num_device_threads, maxBytes>>>(num_device_blocks, num_nodes, num_updates_per_block, d_edgeUpdates, sketchParams.num_buckets, d_buckets, sketchParams.num_columns, sketchParams.bkt_per_col, sketchSeed);
    gpuErrchk(cudaEventRecord(stop));

    cudaDeviceSynchronize();

    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    std::cout << "CUDA Event - Kernel Execution Time (s):           " << time * 0.001 << std::endl;
    std::cout << "CUDA Event - Rate (# of Edges / s):               " << ((num_updates_per_block * num_device_blocks) / 2) / (time * 0.001) << std::endl;
  } 

  cudaFree(d_buckets);
}

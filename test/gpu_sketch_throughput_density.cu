#include <chrono>
#include <cmath>
#include <vector>

#include <sketch.h>
#include "../src/cuda_kernel.cu"

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

__global__ void gpuSketchTest_kernel(int num_device_blocks, node_id_t num_nodes, size_t num_updates, node_id_t *edgeUpdates, size_t num_buckets, Bucket* buckets, size_t num_columns, size_t bkt_per_col, size_t sketchSeed) {

  extern __shared__ vec_t_cu sketches[];
  vec_t_cu* bucket_a = sketches;
  vec_hash_t* bucket_c = (vec_hash_t*)&bucket_a[num_buckets];

  for (size_t i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    bucket_a[i] = 0;
    bucket_c[i] = 0;
  }

  __syncthreads();

  node_id_t node_id = blockIdx.x / num_nodes;
  for (size_t id = threadIdx.x; id < num_updates * num_columns; id += blockDim.x) {

    size_t column_id = id % num_columns;
    size_t update_id = id / num_columns;

    // Get random edge id based on current update_id
    vec_t edge_id = device_concat_pairing_fn(node_id, edgeUpdates[update_id]);

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
    buckets[(node_id * num_buckets) + i].alpha = bucket_a[i];
    buckets[(node_id * num_buckets) + i].gamma = bucket_c[i];
  }
}


int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes start_density max_density density_inc" << std::endl;
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
  double start_density = std::stod(argv[2]);
  double max_density = std::stod(argv[3]);
  double density_inc = std::stod(argv[4]);
  size_t num_complete_edges = (((size_t)num_nodes * ((size_t)num_nodes - 1)) / 2);

  std::cout << "Max Density: " << max_density * 100 << "%\n";

  // Single Sketch with size corresponding to num_nodes
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "num_complete_edges: " << num_complete_edges << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "\n";

  int num_device_threads = 1024;
  size_t num_updates_per_blocks = (sketchParams.num_buckets * sizeof(Bucket)) / sizeof(node_id_t);
  size_t num_max_device_blocks = std::ceil(((double)num_complete_edges * 2) / num_updates_per_blocks);

  std::cout << "Batch Size: " << num_updates_per_blocks << "\n\n";

  size_t maxBytes = (sketchParams.num_buckets * sizeof(vec_t_cu)) + (sketchParams.num_buckets * sizeof(vec_hash_t));
  cudaFuncSetAttribute(gpuSketchTest_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxBytes);

  std::cout << "-----GPU Kernel Information-----\n";
  std::cout << "Number of max thread blocks: " << num_max_device_blocks << "\n";
  std::cout << "Number of threads per block: " << num_device_threads << "\n";
  std::cout << "Memory Size for buckets: " << (double)(num_nodes * sketchParams.num_buckets * sizeof(Bucket)) / 1000000000 << "GB\n";
  std::cout << "  Allocated Shared Memory of: " << (double)maxBytes / 1000 << "KB\n";
  std::cout << "\n";

  std::cout << "Allocating Host Memory: " << (num_updates_per_blocks * sizeof(node_id_t)) / 1e9 << "GB\n";
  std::cout << "Allocating GPU Memory: " << ((num_nodes * sketchParams.num_buckets * sizeof(Bucket)) + (num_updates_per_blocks * sizeof(node_id_t))) / 1e9 << "GB\n";

  Bucket* d_buckets;
  gpuErrchk(cudaMalloc(&d_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket)));

  node_id_t *h_edgeUpdates, *d_edgeUpdates;

  gpuErrchk(cudaMallocHost(&h_edgeUpdates, num_updates_per_blocks * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_edgeUpdates, num_updates_per_blocks * sizeof(node_id_t)));

  for (size_t update_id = 0; update_id < num_updates_per_blocks; update_id++) {
    h_edgeUpdates[update_id] = update_id;
  }
  
  gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, num_updates_per_blocks * sizeof(node_id_t), cudaMemcpyHostToDevice));

  size_t sketchSeed = get_seed();

  float time;
  cudaEvent_t start, stop;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  std::vector<double> densities = {0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005,
                                 0.0000006, 0.0000007, 0.0000008, 0.0000009, 0.000001, 
                                 0.000002, 0.000003, 0.000004, 0.000005,
                                 0.000006, 0.000007, 0.000008, 0.000009, 0.00001, 
                                 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 
                                 0.00007, 0.00008, 0.00009, 0.0001, 0.0002, 0.0003, 
                                 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
                                 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 
                                 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 
                                 0.07, 0.08, 0.09, 0.1, 0.15, 0.2};

  for (auto& density : densities) {                                
  //for (double density = start_density; density < (max_density + 0.000001); density += density_inc) {
    int num_device_blocks = density * num_max_device_blocks;

    std::cout << "Density: " << density * 100 << "%\n";
    std::cout << "  Number of device blocks: " << num_device_blocks << "\n";
    std::cout << "  Number of updates: " << num_updates_per_blocks * num_device_blocks << "\n";

    if (num_device_blocks == 0) {
      std::cout << "  Current Density too low, skipping\n";
      continue;
    }

    gpuErrchk(cudaEventRecord(start));
    gpuSketchTest_kernel<<<num_device_blocks, num_device_threads, maxBytes>>>(num_device_blocks, num_nodes, num_updates_per_blocks, d_edgeUpdates, sketchParams.num_buckets, d_buckets, sketchParams.num_columns, sketchParams.bkt_per_col, sketchSeed);
    gpuErrchk(cudaEventRecord(stop));

    cudaDeviceSynchronize();

    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&time, start, stop));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    std::cout << "CUDA Event - Kernel Execution Time (s):           " << time * 0.001 << std::endl;
    std::cout << "CUDA Event - Rate (# of Edges / s):               " << ((num_updates_per_blocks * num_device_blocks) / 2) / (time * 0.001) << std::endl;
  } 

  cudaFree(d_buckets);
}

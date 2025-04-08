#pragma once

#include <cmath>
#include <cstring>
#include <map>

#include "cuda_kernel.cuh"
#include "cuda_stream.h"
#include "edge_store.h"
#include "mc_sketch_alg.h"

class MCGPUSketchAlg;
class SketchSubgraph {
 private:
  struct Gutter {
    size_t elms = 0;
    std::vector<node_id_t> data;
  };

  std::atomic<size_t> num_updates;
  CudaStream<MCGPUSketchAlg>** cuda_streams = nullptr;
  SketchParams sketchParams;

  int num_streams;

  node_id_t num_nodes;
  size_t batch_size;

  std::vector<Gutter> subgraph_gutters;

  std::mutex *gutter_locks;

  void apply_update_batch(int thr_id, node_id_t src, std::vector<node_id_t> &dst_vertices) {
    if (cuda_streams == nullptr) 
      throw std::runtime_error("ERROR: Cannot call apply_update_batch() on uninit sketch subgraph");
    num_updates += dst_vertices.size();
    cuda_streams[thr_id]->process_batch(src, &dst_vertices[0], dst_vertices.size());
  }

 public:
  ~SketchSubgraph() {
    if (cuda_streams != nullptr) {
      for (int i = 0; i < num_streams; i++) {
        delete cuda_streams[i];
      }
      delete[] cuda_streams;
      delete[] gutter_locks;
    }
  }

  void initialize(MCGPUSketchAlg *sketching_alg, int graph_id, node_id_t _num_nodes,
                  int num_host_threads, int num_device_threads, int num_batch_per_buffer,
                  size_t _batch_size, SketchParams _sketchParams);

  // Insert an edge to the subgraph
  void batch_insert(int thr_id, const node_id_t src, const std::array<node_id_t, 16> dsts,
                    const size_t num_elms);

  void flush() {
    // flush subgraph gutters
    for (node_id_t v = 0; v < num_nodes; v++) {
      if (subgraph_gutters[v].elms > 0) {
        subgraph_gutters[v].data.resize(subgraph_gutters[v].elms);
        apply_update_batch(0, v, subgraph_gutters[v].data);
        subgraph_gutters[v].elms = 0;
        subgraph_gutters[v].data.resize(batch_size);
      }
    }

    // flush cuda streams
    for (int thr_id = 0; thr_id < num_streams; thr_id++) {
      cuda_streams[thr_id]->flush_buffers();
    }
  }

  size_t get_num_updates() {
    return num_updates;
  }

  const SketchParams get_skt_params() { return sketchParams; }

  void display_time() {
    int longest_thr_id = 0;
    double longest_process_time = 0;

    if (sketchParams.cudaUVM_enabled) {
      for (int thr_id = 0; thr_id < num_streams; thr_id++) {
        double total_process_time = cuda_streams[thr_id]->process_time.count();
        if (total_process_time > longest_process_time) {
          longest_process_time = total_process_time;
          longest_thr_id = thr_id;
        }
      }
      std::cout << "Longest Thread # " << longest_thr_id << ": " << cuda_streams[longest_thr_id]->process_time.count() << "\n";
      std::cout << "  Edge Fill Time: " << cuda_streams[longest_thr_id]->edge_fill_time.count() << "\n";
      std::cout << "  CUDA Stream Wait Time: " << cuda_streams[longest_thr_id]->wait_time.count() << "\n"; 
      std::cout << "  Sketch Prefetch Time: " << cuda_streams[longest_thr_id]->prefetch_time.count() << "\n";
      std::cout << "\n";
    }
    else {
      for (int thr_id = 0; thr_id < num_streams; thr_id++) {
        double total_process_time = cuda_streams[thr_id]->process_time.count();
        if (total_process_time > longest_process_time) {
          longest_process_time = total_process_time;
          longest_thr_id = thr_id;
        }
      }
      std::cout << "Longest Thread # " << longest_thr_id << ": " << cuda_streams[longest_thr_id]->process_time.count() << "\n";
      std::cout << "  Edge Fill Time: " << cuda_streams[longest_thr_id]->edge_fill_time.count() << "\n";
      std::cout << "  CUDA Stream Wait Time: " << cuda_streams[longest_thr_id]->wait_time.count() << "\n"; 
      std::cout << "  Delta Sketch Applying Time: " << cuda_streams[longest_thr_id]->apply_delta_time.count() << "\n";
      std::cout << "\n";
    }
  }

};

class MCGPUSketchAlg : public MCSketchAlg {
private:
  // holds general info about the sketches. Copied and populated with actual info for subgraphs.
  SketchParams default_skt_params;

  CudaKernel cudaKernel;

  node_id_t num_nodes;
  int k;
  int sketches_factor;

  // Maximum number of subgraphs
  size_t num_subgraphs;

  // Current number of initialized sketch subgraphs. Starts at 0.
  std::atomic<node_id_t> cur_subgraphs;

  // Number of threads and thread blocks for CUDA kernel
  int num_device_threads = 1024;

  // Number of CPU's graph workers
  int num_host_threads;

  // Number of CPU threads that read edge stream
  int num_reader_threads;

  // Maximum number of edge updates in one batch
  int num_batch_per_buffer = 1080;
  
  // Number of subgraphs in sketch representation
  int max_sketch_graphs; // Max. number of subgraphs that can be in sketch graphs

  // sketch subgraphs
  SketchSubgraph *subgraphs;

  // lossless edge storage
  EdgeStore edge_store;

  // Number of edge updates in single batch
  size_t batch_size;

  SubgraphTaggedUpdate **store_buffers;
  SubgraphTaggedUpdate **sketch_buffers;

  // helper functions for apply_update_batch()
  size_t get_and_apply_finished_stream(int thr_id);
  void complete_update_batch(int thr_id, const TaggedUpdateBatch &updates);

  std::mutex sketch_creation_lock;
public:
  MCGPUSketchAlg(node_id_t num_vertices, int num_threads, int _num_reader_threads,
                SketchParams _sketchParams, int _num_subgraphs,
                int _max_sketch_graphs, int _k, size_t _sketch_bytes, int _initial_sketch_graphs,
                CCAlgConfiguration config)
     : MCSketchAlg(num_vertices, _sketchParams.seed, _max_sketch_graphs, config),
       edge_store(_sketchParams.seed, num_vertices, _sketch_bytes, _num_subgraphs, _initial_sketch_graphs) {
    // Start timer for initializing
    auto init_start = std::chrono::steady_clock::now();

    default_skt_params = _sketchParams;
    num_nodes = num_vertices;
    k = _k;
    sketches_factor = config.get_sketches_factor();
    num_host_threads = num_threads;
    num_reader_threads = _num_reader_threads;

    if (_max_sketch_graphs < _initial_sketch_graphs) {
      std::cerr << "ERROR: Cannot have initial sketch graphs > max sketch graphs" << std::endl;
      exit(EXIT_FAILURE);
    }
    num_subgraphs = _num_subgraphs;
    cur_subgraphs = _initial_sketch_graphs;
    max_sketch_graphs = _max_sketch_graphs;
    subgraphs = new SketchSubgraph[max_sketch_graphs];

    // Create a bigger batch size to apply edge updates when subgraph is turning into sketch
    // representation
    batch_size = get_desired_updates_per_batch();
    std::cout << "Batch Size: " << batch_size << "\n";

    int device_id = cudaGetDevice(&device_id);
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);
    std::cout << "CUDA Device Count: " << device_count << "\n";
    std::cout << "CUDA Device ID: " << device_id << "\n";
    std::cout << "CUDA Device Number of SMs: " << deviceProp.multiProcessorCount << "\n";

    size_t maxBytes = (default_skt_params.num_buckets * sizeof(vec_t_cu)) +
                      (default_skt_params.num_buckets * sizeof(vec_hash_t));
    cudaKernel.updateSharedMemory(maxBytes);
    std::cout << "Allocated Shared Memory of: " << maxBytes << "\n";

    // Initialize Sketch Graphs
    for (int i = 0; i < cur_subgraphs; i++) {
      subgraphs[i].initialize(this, i, num_nodes, num_host_threads, num_device_threads,
                              num_batch_per_buffer, batch_size, default_skt_params);
      create_sketch_graph(i, subgraphs[i].get_skt_params());
    }
    
    store_buffers = new SubgraphTaggedUpdate*[num_host_threads];
    sketch_buffers = new SubgraphTaggedUpdate*[num_host_threads];

    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      store_buffers[thr_id] = new SubgraphTaggedUpdate[batch_size];
      sketch_buffers[thr_id] = new SubgraphTaggedUpdate[batch_size];
    }

    std::chrono::duration<double> init_time = std::chrono::steady_clock::now() - init_start;
    std::cout << "MCGPUSketchAlg's Initialization Duration: " << init_time.count() << std::endl;
  }

  ~MCGPUSketchAlg() {
    delete[] subgraphs;
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      delete[] store_buffers[thr_id];
      delete[] sketch_buffers[thr_id];
    }
    delete[] store_buffers;
    delete[] sketch_buffers;
  }

  /**
   * Update all the sketches for a node, given a batch of updates.
   * @param thr_id         The id of the thread performing the update [0, num_threads)
   * @param src_vertex     The vertex where the edges originate.
   * @param dst_vertices   A vector of destinations.
   */
  void apply_update_batch(int thr_id, node_id_t src_vertex,
                          const std::vector<node_id_t> &dst_vertices);

  // Update with the delta sketches that haven't been applied yet.
  void apply_flush_updates();

  void print_subgraph_edges() {
    std::cout << "Number of inserted updates for each subgraph:\n";
    for (int graph_id = 0; graph_id < cur_subgraphs.load(); graph_id++) {
    std::cout << "  Sub-Graph " << graph_id << "(Sketch): " << subgraphs[graph_id].get_num_updates()
                << std::endl;
    }
    std::cout << "  Adjacency list:      " << edge_store.get_num_edges() << std::endl;
  }

  std::vector<Edge> get_adjlist_spanning_forests();
  int get_num_sketch_graphs() { return cur_subgraphs; }

  size_t get_num_adjlist_edges() { return edge_store.get_num_edges(); }
  void display_time() {
    std::cout << "Displaying Overhead Information for each sketch subgraph...\n";
    for (int subgraph_id = 0; subgraph_id < cur_subgraphs; subgraph_id++) {
      std::cout << "Subgraph #" << subgraph_id << ":\n";
      subgraphs[subgraph_id].display_time();
    }
  }
};

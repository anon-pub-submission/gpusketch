#include <fstream>
#include <iomanip>
#include <vector>
#include <graph_sketch_driver.h>
#include <mc_gpu_sketch_alg.h>
#include <binary_file_stream.h>
#include <thread>
#include <sys/resource.h> // for rusage
#include <cuda_kernel.cuh>

static bool cert_clean_up = false;
static bool shutdown = false;
static bool cudaUVM_enabled = true;
constexpr double epsilon = 0.75;

static double get_max_mem_used() {
  struct rusage data;
  getrusage(RUSAGE_SELF, &data);
  return (double) data.ru_maxrss / 1024.0;
}

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

/*
 * Function which is run in a seperate thread and will query
 * the graph for the number of updates it has processed
 * @param total       the total number of edge updates
 * @param g           the graph object to query
 * @param start_time  the time that we started stream ingestion
 */
void track_insertions(uint64_t total, GraphSketchDriver<MCGPUSketchAlg> *driver,
                      std::chrono::steady_clock::time_point start_time) {
  total = total * 2; // we insert 2 edge updates per edge

  printf("Insertions\n");
  printf("Progress:                    | 0%%\r"); fflush(stdout);
  std::chrono::steady_clock::time_point start = start_time;
  std::chrono::steady_clock::time_point prev  = start_time;
  uint64_t prev_updates = 0;

  while(true) {
    sleep(1);
    uint64_t updates = driver->get_total_updates();
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_diff = now - start;
    std::chrono::duration<double> cur_diff   = now - prev;

    // calculate the insertion rate
    uint64_t upd_delta = updates - prev_updates;
    // divide insertions per second by 2 because each edge is split into two updates
    // we care about edges per second not about stream updates
    size_t ins_per_sec = (((double)(upd_delta)) / cur_diff.count()) / 2;

    if (updates >= total || shutdown)
      break;

    // display the progress
    int progress = updates / (total * .05);
    printf("Progress:%s%s", std::string(progress, '=').c_str(), std::string(20 - progress, ' ').c_str());
    printf("| %i%% -- %lu per second\r", progress * 5, ins_per_sec); fflush(stdout);
  }

  printf("Progress:====================| Done                             \n");
  return;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: stream_file, graph_workers, reader_threads, inital_sketch_graphs" << std::endl;
    exit(EXIT_FAILURE);
  }

  shutdown = false;
  std::string stream_file = argv[1];
  int num_threads = std::atoi(argv[2]);
  if (num_threads < 1) {
    std::cout << "ERROR: Invalid number of graph workers! Must be > 0." << std::endl;
    exit(EXIT_FAILURE);
  }
  int reader_threads = std::atoi(argv[3]);
  int initial_sketch_graphs = std::atoi(argv[4]);

  BinaryFileStream stream(stream_file);
  node_id_t num_nodes = stream.vertices();
  size_t num_updates  = stream.edges();
  std::cout << "Processing stream: " << stream_file << std::endl;
  std::cout << "nodes       = " << num_nodes << std::endl;
  std::cout << "num_updates = " << num_updates << std::endl;
  std::cout << std::endl;

  int k = ceil(log2(num_nodes) / (epsilon * epsilon));
  double reduced_k = (k / log2(num_nodes)) * 1.5;

  std::cout << "epsilon: " << epsilon << std::endl;
  std::cout << "k: " << k << std::endl;
  std::cout << "reduced_k: " << reduced_k << std::endl;

  int num_graphs = 1 + (int)(2 * log2(num_nodes));
  std::cout << "Total num_graphs: " << num_graphs << "\n";

  auto driver_config = DriverConfiguration().gutter_sys(CACHETREE).worker_threads(num_threads);
  auto mc_config = CCAlgConfiguration().batch_factor(1);

  // Get variables from sketch
  // (1) num_samples (2) num_columns (3) bkt_per_col (4) num_buckets
  SketchParams sketchParams;
  sketchParams.num_samples = Sketch::calc_cc_samples(num_nodes, reduced_k);
  sketchParams.num_columns = sketchParams.num_samples * Sketch::default_cols_per_sample;
  sketchParams.bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  sketchParams.num_buckets = sketchParams.num_columns * sketchParams.bkt_per_col + 1;

  std::cout << "num_samples: " << sketchParams.num_samples << "\n";
  std::cout << "num_buckets: " << sketchParams.num_buckets << "\n";
  std::cout << "num_columns: " << sketchParams.num_columns << "\n";
  std::cout << "bkt_per_col: " << sketchParams.bkt_per_col << "\n"; 

  // Total bytes of sketching datastructure of one subgraph
  int w = 4; // 4 bytes when num_nodes < 2^32
  double sketch_bytes = 4 * w * num_nodes * ((2 * log2(num_nodes)) + 2) * ((reduced_k * log2(num_nodes))/(1 - log2(1.2)));
  double adjlist_edge_bytes = 8;

  std::cout << "Total bytes of sketching data structure of one subgraph: " << sketch_bytes / 1000000000 << "GB\n";

  // Calculate number of minimum adj. list subgraph
  size_t num_edges_complete = (size_t(num_nodes) * (size_t(num_nodes) - 1)) / 2;
  int num_sketch_graphs = 0;
  int max_sketch_graphs = 0;

  for (int i = 0; i < num_graphs; i++) {
    // Calculate estimated memory for current subgraph
    size_t num_est_edges = num_edges_complete / (1 << i);
    double adjlist_bytes = adjlist_edge_bytes * num_est_edges;

    if (adjlist_bytes >= sketch_bytes) {
      max_sketch_graphs++;
    }
  }

  // Total number of estimated edges of minimum number of adj. list graphs
  size_t num_est_edges_adj_graphs = (2 * num_edges_complete) / (1 << (max_sketch_graphs));
  double total_adjlist_bytes = adjlist_edge_bytes * num_est_edges_adj_graphs;
  double total_sketch_bytes = sketch_bytes * max_sketch_graphs;

  std::cout << "Number of sketch graphs: " << num_sketch_graphs << "\n";
  std::cout << "  If complete graph with current num_nodes..." << "\n";
  std::cout << "    Maximum number of sketch graphs: " << max_sketch_graphs << "\n";
  std::cout << "    Total minimum memory required for minimum number of adj. list graphs: " << total_adjlist_bytes / 1000000000 << "GB\n";
  std::cout << "    Total minimum memory required for maximum number of sketch graphs: " << total_sketch_bytes / 1000000000 << "GB\n";
  std::cout << "    Total minimum memory required for current num_nodes: " << (total_adjlist_bytes + total_sketch_bytes) / 1000000000 << "GB\n";

  // Reconfigure sketches_factor based on reduced_k
  mc_config.sketches_factor(reduced_k);

  std::cout << "CUDA UVM Enabled: " << cudaUVM_enabled << "\n";
  sketchParams.cudaUVM_enabled = cudaUVM_enabled;
  /*if (cudaUVM_enabled) {
    // Allocate memory for buckets
    Bucket* cudaUVM_buckets;
    gpuErrchk(cudaMallocManaged(&cudaUVM_buckets, initial_sketch_graphs * num_nodes * sketchParams.num_buckets * sizeof(Bucket)));
    sketchParams.cudaUVM_buckets = cudaUVM_buckets;
  }*/

  // Getting sketch seed
  sketchParams.seed = get_seed();
  MCGPUSketchAlg mc_gpu_alg{num_nodes, num_threads, reader_threads, sketchParams, 
    num_graphs, max_sketch_graphs, reduced_k, sketch_bytes, initial_sketch_graphs, mc_config};

  GraphSketchDriver<MCGPUSketchAlg> driver{&mc_gpu_alg, &stream, driver_config, reader_threads};

  auto ins_start = std::chrono::steady_clock::now();
  std::thread querier(track_insertions, num_updates, &driver, ins_start);

  driver.process_stream_until(END_OF_STREAM);

  auto flush_start = std::chrono::steady_clock::now();
  driver.prep_query(KSPANNINGFORESTS);
  mc_gpu_alg.apply_flush_updates();

  // Re-measure flush_end to include time taken for applying delta sketches from flushing
  auto flush_end = std::chrono::steady_clock::now();

  // Display number of inserted updates to every subgraphs
  mc_gpu_alg.print_subgraph_edges();

  std::chrono::duration<double> sampling_forests_adj_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> sampling_forests_sketch_time = std::chrono::nanoseconds::zero();
  std::chrono::duration<double> viecut_time = std::chrono::nanoseconds::zero();

  std::cout << "After Insertion:\n";
  num_sketch_graphs = mc_gpu_alg.get_num_sketch_graphs();
  std::cout << "Number of sketch graphs: " << num_sketch_graphs << "\n";

  mc_gpu_alg.display_time();

  /********************************************************************\
  |                                                                    |
  |                       POST PROCESSING BEGINS                       |
  |                                                                    |
  \********************************************************************/

  // Get spanning forests then create a METIS format file
  std::cout << "Generating Certificates...\n";
  int num_sampled_zero_graphs = 0;
  for (int graph_id = 0; graph_id < num_graphs; graph_id++) {
    std::vector<Edge> SFs_edges;
    std::set<Edge> edges;

    if (graph_id >= num_sketch_graphs) { // Get Spanning forests from adj list
      std::cout << "S" << graph_id << " (Adj. list):\n";
      auto sampling_forests_start = std::chrono::steady_clock::now();
      SFs_edges = mc_gpu_alg.get_adjlist_spanning_forests();
      sampling_forests_adj_time += std::chrono::steady_clock::now() - sampling_forests_start;
    } 
    else { // Get Spanning forests from sketch subgraph
      std::cout << "S" << graph_id << " (Sketch):\n";
      auto sampling_forests_start = std::chrono::steady_clock::now();
      auto sfs = mc_gpu_alg.calc_disjoint_spanning_forests(graph_id, k);
      sampling_forests_sketch_time += std::chrono::steady_clock::now() - sampling_forests_start;

      std::cerr << "Query done" << std::endl;
      for (const auto &sf : sfs) {
        for (auto edge : sf.get_edges()) {
          SFs_edges.push_back(edge);
        }
      }
      std::cout << "  Number of edges in spanning forests: " << SFs_edges.size() << "\n";
    }

    // now perform minimum cut computation
    auto viecut_start = std::chrono::steady_clock::now();
    MinCut mc = mc_gpu_alg.calc_minimum_cut(SFs_edges);
    viecut_time += std::chrono::steady_clock::now() - viecut_start;
    if (graph_id >= num_sketch_graphs) {
      std::cout << "  S" << graph_id << " (Adj. list): " << mc.value << "\n";
    }
    else {
      std::cout << "  S" << graph_id << " (Sketch): " << mc.value << "\n";
    }

    if (graph_id >= num_sketch_graphs || mc.value < k) {
      // we return the adjacency answer regardless of its value
      // This works under the assumption that all the previous sketched min cuts were invalid
      // Otherwise, if a sketched subgraph returns a value < k, we use that
      std::cout << "Mincut found in graph: " << graph_id << " mincut: " << mc.value << std::endl;
      std::cout << "Final mincut value: " << (mc.value * (pow(2, graph_id))) << std::endl;
      break;
    }
  }
  auto query_end = std::chrono::steady_clock::now();


  shutdown = true;
  querier.join();

  std::chrono::duration<double> insert_time = flush_end - ins_start;
  std::chrono::duration<double> flush_time = flush_end - flush_start;
  std::chrono::duration<double> query_time = query_end - flush_start;

  double num_seconds = insert_time.count();
  std::cout << "Insertion time(sec): " << num_seconds << std::endl;
  std::cout << "  Updates per second: " << stream.edges() / num_seconds << std::endl;
  std::cout << "Total Query Latency(sec): " << query_time.count() << std::endl;
  std::cout << "  Flush Gutters + GPU Buffers(sec): " << flush_time.count() << std::endl;
  std::cout << "  K-Connectivity: (Sketch Subgraphs)" << std::endl;
  std::cout << "    Sampling Forests Time(sec): " << sampling_forests_sketch_time.count() + sampling_forests_adj_time.count() << std::endl;
  std::cout << "      From Sketch Subgraphs(sec): " << sampling_forests_sketch_time.count() << std::endl;
  std::cout << "      From Adj. list(sec): " << sampling_forests_adj_time.count() << std::endl;
  std::cout << "  VieCut Program Time(sec): " << viecut_time.count() << std::endl;
  std::cout << "Maximum Memory Usage(MiB): " << get_max_mem_used() << std::endl;

  std::ofstream out("runtime_results.txt", std::ios_base::out | std::ios_base::app);
  out << std::fixed;
  out << std::setprecision(3);
  out << num_threads << ", " << stream.edges() / num_seconds << std::endl;
}

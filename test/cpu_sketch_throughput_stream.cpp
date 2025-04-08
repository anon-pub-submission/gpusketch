#include <chrono>
#include <thread>
#include <vector>

#include <sketch.h>

struct SketchParams {
  // Sketch related variables
  size_t num_samples;
  size_t num_buckets;
  size_t num_columns;
  size_t bkt_per_col;
  size_t seed;
};

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes num_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  node_id_t num_nodes = std::atoi(argv[1]);
  int num_threads = std::atoi(argv[2]);

  size_t sketchSeed = get_seed();

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

  size_t num_updates_per_batch = (sketchParams.num_buckets * sizeof(Bucket)) / sizeof(node_id_t);

  std::cout << "Batch Size: " << num_updates_per_batch << "\n";

  size_t max_num_updates = 4e9;
  size_t max_num_batches = (2 * max_num_updates) / num_updates_per_batch;
  size_t max_act_updates = max_num_batches * num_updates_per_batch;

  node_id_t *edgeUpdates = new node_id_t[max_act_updates];

  for (size_t batch_id = 0; batch_id < max_num_batches; batch_id++) {
    for (size_t update_id = 0; update_id < num_updates_per_batch; update_id++) {
      edgeUpdates[(batch_id * num_updates_per_batch) + update_id] = update_id;
    }
  }

  Sketch **sketches = new Sketch *[num_nodes];
  Sketch **delta_sketches = new Sketch *[num_threads];

  for (node_id_t node_id = 0; node_id < num_nodes; node_id++) {
    sketches[node_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchSeed, Sketch::calc_cc_samples(num_nodes, 1));
  }

  for (int thr_id = 0; thr_id < num_threads; thr_id++) {
    delta_sketches[thr_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchSeed, Sketch::calc_cc_samples(num_nodes, 1));
  }

  std::vector<double> stream_updates = {
    10000, 20000, 30000, 40000, 50000,
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
    size_t num_batches = (2 * stream_update) / num_updates_per_batch;
    
    std::cout << "Number of stream updates: " << stream_update << "\n";
    std::cout << "  Number of batches: " << num_batches << "\n";
    std::cout << "  Number of updates: " << num_updates_per_batch * num_batches << "\n";

    if (num_batches == 0) {
      std::cout << "  Current number of stream updates too low, skipping\n";
      continue;
    }

    if (num_batches > max_num_batches) {
      std::cout << "  Current number of stream updates exceeds maximum, breaking out\n";
      break;
    }

    auto task = [&](int thr_id) {
      Sketch &delta_sketch = *delta_sketches[thr_id];

      for (size_t batch_id = thr_id; batch_id < num_batches; batch_id += num_threads) {
        // Reset delta sketch
        delta_sketch.zero_contents();

        node_id_t src_vertex = batch_id % num_nodes;

        for (node_id_t update_id = 0; update_id < num_updates_per_batch; update_id++) {
          delta_sketch.update(static_cast<vec_t>(concat_pairing_fn(src_vertex, edgeUpdates[(batch_id * num_updates_per_batch) + update_id])));
        }

        std::lock_guard<std::mutex> lk(sketches[src_vertex]->mutex);
        sketches[src_vertex]->merge(delta_sketch);
      }
    };

    std::vector<std::thread> threads;

    auto sketch_update_start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_threads; i++) threads.emplace_back(task, i);

    // wait for threads to finish
    for (int i = 0; i < num_threads; i++) threads[i].join();

    std::chrono::duration<double> sketch_update_duration = std::chrono::steady_clock::now() - sketch_update_start;
    std::cout << "Total insertion time(sec):    " << sketch_update_duration.count() << std::endl;
    std::cout << "Updates per second:           " << ((num_updates_per_batch * num_batches) / 2) / sketch_update_duration.count() << std::endl;

  }
  
}

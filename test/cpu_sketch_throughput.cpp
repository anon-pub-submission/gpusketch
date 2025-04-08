#pragma once
#include <chrono>
#include <thread>
#include <vector>

#include <sketch.h>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: num_nodes num_updates num_threads" << std::endl;
    exit(EXIT_FAILURE);
  }

  node_id_t num_nodes = std::atoi(argv[1]);
  size_t num_updates = std::stoull(argv[2]);
  int num_threads = std::atoi(argv[3]);

  size_t sketchSeed = get_seed();

  size_t num_samples = Sketch::calc_cc_samples(num_nodes, 1);
  size_t num_columns = num_samples * Sketch::default_cols_per_sample;
  size_t bkt_per_col = Sketch::calc_bkt_per_col(Sketch::calc_vector_length(num_nodes));
  size_t num_buckets = num_columns * bkt_per_col + 1;

  std::cout << "-----Sketch Information-----\n";
  std::cout << "num_nodes: " << num_nodes << "\n";
  std::cout << "num_updates: " << num_updates << "\n";
  std::cout << "bkt_per_col: " << bkt_per_col << "\n";
  std::cout << "num_columns: " << num_columns << "\n";
  std::cout << "num_buckets: " << num_buckets << "\n";
  std::cout << "\n";

  int num_updates_per_batch = (num_buckets * sizeof(Bucket)) / sizeof(node_id_t);
  int num_batches = std::ceil(((double)num_updates * 2) / num_updates_per_batch);

  std::cout << "Number of Batches: " << num_batches << "\n";
  std::cout << "Batch Size: " << num_updates_per_batch << "\n";

  Sketch **delta_sketches = new Sketch *[num_threads];
  for (size_t thr_id = 0; thr_id < num_threads; thr_id++) {
    delta_sketches[thr_id] = new Sketch(Sketch::calc_vector_length(num_nodes), sketchSeed, Sketch::calc_cc_samples(num_nodes, 1));
  }

  auto sketch_update_start = std::chrono::steady_clock::now();

  auto task = [&](int thr_id) {
    for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_threads) {
      // Reset delta sketch
      delta_sketches[thr_id]->zero_contents();

      node_id_t src_vertex = batch_id / num_nodes;

      for (int update_id = 0; update_id < num_updates_per_batch; update_id++) {
        delta_sketches[thr_id]->update(static_cast<vec_t>(concat_pairing_fn(src_vertex, update_id)));
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (size_t i = 0; i < num_threads; i++) threads[i].join();

  std::chrono::duration<double> sketch_update_duration = std::chrono::steady_clock::now() - sketch_update_start;
  std::cout << "Total insertion time(sec):    " << sketch_update_duration.count() << std::endl;
  std::cout << "Updates per second:           " << num_updates / sketch_update_duration.count() << std::endl;
}

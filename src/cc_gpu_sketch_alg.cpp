#include "cc_gpu_sketch_alg.h"

#include <iostream>
#include <thread>
#include <vector>

void CCGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();

  if (dst_vertices.size() <= small_batch_size) { // If batch if small, do sketch update in CPU
    auto small_batch_start = std::chrono::steady_clock::now();
    CCSketchAlg::apply_update_batch(thr_id, src_vertex, dst_vertices);
    small_batch_time[thr_id] += std::chrono::steady_clock::now() - small_batch_start;
  }
  else {
    cudaStreams[thr_id]->process_batch(src_vertex, &dst_vertices[0], dst_vertices.size());
  }
};

void CCGPUSketchAlg::flush_buffers() {
  auto task = [&](int thr_id) {
    cudaStreams[thr_id]->flush_buffers();
  };
  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_host_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (size_t i = 0; i < num_host_threads; i++) threads[i].join();
  cudaDeviceSynchronize();
}


void CCGPUSketchAlg::display_time() {
  int longest_thr_id = 0;
  double longest_process_time = 0;

  if (sketchParams.cudaUVM_enabled) {
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      double total_process_time = cudaStreams[thr_id]->process_time.count() + small_batch_time[thr_id].count();
      /*std::cout << "Thread # " << thr_id << ": " << total_process_time << "\n";
      std::cout << "  Edge Fill Time: " << cudaStreams[thr_id]->edge_fill_time.count() << "\n";
      std::cout << "  CUDA Stream Wait Time: " << cudaStreams[thr_id]->wait_time.count() << "\n";
      std::cout << "  Sketch Prefetch Time: " << cudaStreams[thr_id]->prefetch_time.count() << "\n";
      std::cout << "  Small Batch Update Time: " << small_batch_time[thr_id].count() << "\n";*/

      if (total_process_time > longest_process_time) {
        longest_process_time = total_process_time;
        longest_thr_id = thr_id;
      }
    }
    std::cout << "\n";
    std::cout << "Longest Thread # " << longest_thr_id << ": " << cudaStreams[longest_thr_id]->process_time.count() + small_batch_time[longest_thr_id].count() << "\n";
    std::cout << "  Edge Fill Time: " << cudaStreams[longest_thr_id]->edge_fill_time.count() << "\n";
    std::cout << "  CUDA Stream Wait Time: " << cudaStreams[longest_thr_id]->wait_time.count() << "\n"; 
    std::cout << "  Sketch Prefetch Time: " << cudaStreams[longest_thr_id]->prefetch_time.count() << "\n";
    std::cout << "  Small Batch Update Time: " << small_batch_time[longest_thr_id].count() << "\n";
  }
  else {
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      double total_process_time = cudaStreams[thr_id]->process_time.count() + small_batch_time[thr_id].count();
      /*std::cout << "Thread # " << thr_id << ": " << total_process_time << "\n";
      std::cout << "  Edge Fill Time: " << cudaStreams[thr_id]->edge_fill_time.count() << "\n";
      std::cout << "  CUDA Stream Wait Time: " << cudaStreams[thr_id]->wait_time.count() << "\n";
      std::cout << "  Delta Sketch Applying Time: " << cudaStreams[thr_id]->apply_delta_time.count() << "\n";
      std::cout << "  Small Batch Update Time: " << small_batch_time[thr_id].count() << "\n";*/

      if (total_process_time > longest_process_time) {
        longest_process_time = total_process_time;
        longest_thr_id = thr_id;
      }
    }
    std::cout << "\n";
    std::cout << "Longest Thread # " << longest_thr_id << ": " << cudaStreams[longest_thr_id]->process_time.count() + small_batch_time[longest_thr_id].count() << "\n";
    std::cout << "  Edge Fill Time: " << cudaStreams[longest_thr_id]->edge_fill_time.count() << "\n";
    std::cout << "  CUDA Stream Wait Time: " << cudaStreams[longest_thr_id]->wait_time.count() << "\n"; 
    std::cout << "  Delta Sketch Applying Time: " << cudaStreams[longest_thr_id]->apply_delta_time.count() << "\n";
    std::cout << "  Small Batch Update Time: " << small_batch_time[longest_thr_id].count() << "\n";
  }
}

#include "sk_gpu_sketch_alg.h"

#include <iostream>
#include <thread>
#include <vector>

void SKGPUSketchAlg::apply_update_batch(int thr_id, node_id_t src_vertex,
                                     const std::vector<node_id_t> &dst_vertices) {
  if (CCSketchAlg::get_update_locked()) throw UpdateLockedException();
  // Get offset
  size_t offset = edgeUpdate_offset.fetch_add(dst_vertices.size());

  // Fill in buffer
  size_t index = 0;
  for (auto dst : dst_vertices) {
    h_edgeUpdates[offset + index] = dst;
    index++;
  }

  size_t batch_id = batch_count.fetch_add(1);
  std::lock_guard<std::mutex> lk(batch_mutex);
  batch_sizes.insert({batch_id, dst_vertices.size()});
  batch_src.insert({batch_id, src_vertex});
  batch_start_index.insert({batch_id, offset});
};

void SKGPUSketchAlg::launch_cpu_update() {
  std::cout << "Performing sketch updates on CPU\n";
  size_t num_batches = batch_count;

  auto update_start = std::chrono::steady_clock::now();
  auto task = [&](int thr_id) {
    for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_host_threads) {
      // Reset delta sketch
      delta_sketches[thr_id]->zero_contents();

      node_id_t src_vertex = batch_src[batch_id];
      size_t update_offset = batch_start_index[batch_id];

      for (int update_id = 0; update_id < batch_sizes[batch_id]; update_id++) {
        delta_sketches[thr_id]->update(static_cast<vec_t>(concat_pairing_fn(src_vertex, h_edgeUpdates[update_offset + update_id])));
      }

      apply_raw_buckets_update(src_vertex, delta_sketches[thr_id]->get_bucket_ptr());
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_host_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (int i = 0; i < num_host_threads; i++) threads[i].join();
  std::chrono::duration<double> update_time = std::chrono::steady_clock::now() - update_start;

  std::cout << "  CPU Sketch Update Finished.\n";
  std::cout << "    Elapsed Time: " << update_time.count() << "\n";
  std::cout << "    Throughput: " << num_updates / update_time.count() << "\n";
}

void SKGPUSketchAlg::launch_gpu_update() {
  // Declare GPU block count and size
  std::cout << "Num GPU threads per block: " << num_device_threads << "\n";
  std::cout << "Number of batches: " << batch_count << "\n";

  size_t num_batches = batch_count;

  auto task = [&](int thr_id) {
    for (int batch_id = thr_id; batch_id < num_batches; batch_id += num_host_threads) {
      
      node_id_t src_vertex = batch_src[batch_id];
      std::vector<node_id_t> dst_vertices;

      size_t update_offset = batch_start_index[batch_id];

      cudaStreams[thr_id]->process_batch(src_vertex, &h_edgeUpdates[update_offset], batch_sizes[batch_id]);
    }
  };

  std::vector<std::thread> threads;
  for (int i = 0; i < num_host_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (int i = 0; i < num_host_threads; i++) threads[i].join();
  std::cout << "  GPU Sketch Update Finished.\n";
  cudaDeviceSynchronize();
}

void SKGPUSketchAlg::flush_buffers() {
  auto task = [&](int thr_id) {
    cudaStreams[thr_id]->flush_buffers();
  };
  std::vector<std::thread> threads;
  for (size_t i = 0; i < num_host_threads; i++) threads.emplace_back(task, i);

  // wait for threads to finish
  for (size_t i = 0; i < num_host_threads; i++) threads[i].join();
  cudaDeviceSynchronize();
}

void SKGPUSketchAlg::launch_gpu_kernel() {
  // Declare GPU block count and size
  std::cout << "Num GPU threads per block: " << num_device_threads << "\n";
  std::cout << "Number of batches: " << batch_count << "\n";

  //num_updates = (batch_count * batch_size) / 2;
  //std::cout << "New num_updates: "  << num_updates << "\n";

  std::cout << "Preparing update buffers for GPU...\n";
  gpuErrchk(cudaMallocHost(&h_update_sizes, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMallocHost(&h_update_src, batch_count * sizeof(node_id_t)));
  gpuErrchk(cudaMallocHost(&h_update_start_index, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_sizes, batch_count * sizeof(vec_t)));
  gpuErrchk(cudaMalloc(&d_update_src, batch_count * sizeof(node_id_t)));
  gpuErrchk(cudaMalloc(&d_update_start_index, batch_count * sizeof(vec_t)));
  // Fill in update_sizes and update_src
  for (auto it = batch_sizes.begin(); it != batch_sizes.end(); it++) {
    h_update_sizes[it->first] = it->second;
    h_update_src[it->first] = batch_src[it->first]; 
    h_update_start_index[it->first] = batch_start_index[it->first];
  }
  
  // Transfer buffers to GPU
  gpuErrchk(cudaMemcpy(d_edgeUpdates, h_edgeUpdates, 2 * num_updates * sizeof(node_id_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_sizes, h_update_sizes, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_src, h_update_src, batch_count * sizeof(node_id_t), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_update_start_index, h_update_start_index, batch_count * sizeof(vec_t), cudaMemcpyHostToDevice));

  int min_device_block = (batch_count / num_device_threads) + 1;

  /*for (int i = 1; i <= 100; i++) {
    memset(sketchParams.buckets, 0, num_nodes * num_buckets * sizeof(Bucket));
    // Prefetch sketches to GPU
    gpuErrchk(cudaMemPrefetchAsync(sketchParams.buckets, num_nodes * num_buckets * sizeof(Bucket), device_id));

    //num_device_blocks = i * min_device_block * 10;
    num_device_blocks = i * 10;
    // Launch GPU kernel
    std::cout << "Launching GPU Kernel... Iteration: " << i << " num_device_blocks = " << num_device_blocks;
    auto kernel_start = std::chrono::steady_clock::now();
    cudaKernel.single_sketchUpdate(num_device_threads, num_device_blocks, num_thread_blocks_per_batch, batch_count, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);
    cudaDeviceSynchronize();
    auto kernel_end = std::chrono::steady_clock::now();
    
    std::cout << "  GPU Kernel Finished.\n";
    std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
    std::cout << "    Elapsed Time: " << kernel_time.count() << "\n";
    std::cout << "    Throughput: " << num_updates / kernel_time.count() << "\n";
  }*/


  // Launch GPU kernel
  num_device_blocks = batch_count;
  std::cout << "Num GPU thread blocks: " << num_device_blocks << "\n";
  std::cout << "Launching GPU Kernel...\n";

  float time;
  cudaEvent_t start, stop;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  auto kernel_start = std::chrono::steady_clock::now();
  gpuErrchk(cudaEventRecord(start));
  cudaKernel.single_sketchUpdate(num_device_threads, num_device_blocks, batch_count, batch_size, d_edgeUpdates, d_update_src, d_update_sizes, d_update_start_index, sketchParams);
  gpuErrchk(cudaEventRecord(stop));
  cudaDeviceSynchronize();
  auto kernel_end = std::chrono::steady_clock::now();

  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&time, start, stop));
  
  std::cout << "  GPU Kernel Finished.\n";
  std::chrono::duration<double> kernel_time = kernel_end - kernel_start;
  /*std::cout << "    Elapsed Time: " << kernel_time.count() << "\n";
  std::cout << "    Throughput: " << num_updates / kernel_time.count() << "\n";*/
  std::cout << "Device Sync + CPU - Kernel Execution Time (s):    " << kernel_time.count() << std::endl;
  std::cout << "Device Sync + CPU - Rate (# of Edges / s):        " << num_updates / kernel_time.count() << std::endl;
  std::cout << "CUDA Event - Kernel Execution Time (s):           " << time * 0.001 << std::endl;
  std::cout << "CUDA Event - Rate (# of Edges / s):               " << num_updates / (time * 0.001) << std::endl;

  // Prefecth buffers back to CPU
  gpuErrchk(cudaMemPrefetchAsync(sketchParams.cudaUVM_buckets, num_nodes * sketchParams.num_buckets * sizeof(Bucket), cudaCpuDeviceId));
}

void SKGPUSketchAlg::buffer_transfer() {
  // Allocate buffers
  double total_bytes = 2 * num_updates * sizeof(node_id_t);

  node_id_t *h_sketches, *d_sketches;
  gpuErrchk(cudaMallocHost(&h_sketches, total_bytes));
  gpuErrchk(cudaMalloc(&d_sketches, total_bytes));

  float time;
  cudaEvent_t start, stop;

  gpuErrchk(cudaEventCreate(&start));
  gpuErrchk(cudaEventCreate(&stop));

  cudaStream_t stream0, stream1;

  // Initialize CudaStream
  cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);

  // Data Transfer
  gpuErrchk(cudaEventRecord(start));

  gpuErrchk(cudaMemcpyAsync(d_edgeUpdates, h_edgeUpdates, total_bytes, cudaMemcpyHostToDevice, stream0));
  gpuErrchk(cudaMemcpyAsync(h_sketches, d_sketches, total_bytes, cudaMemcpyDeviceToHost, stream1));

  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  gpuErrchk(cudaEventRecord(stop));

  gpuErrchk(cudaEventSynchronize(stop));
  gpuErrchk(cudaEventElapsedTime(&time, start, stop));

  std::cout << "Total Bytes Transferred between CPU-GPU (GB):      " << 2 * total_bytes / 1000000000 << std::endl;
  std::cout << "Data Transfer Time (s):                            " << time * 0.001 << std::endl;
  std::cout << "Data Transfer Time (# of Edges/s):                 " << num_updates / (time * 0.001) << std::endl;
  std::cout << "Data Transfer Time (GB/s):                         " << (2 * total_bytes / 1000000000) / (time * 0.001) << std::endl;
}

void SKGPUSketchAlg::display_time() {
  int longest_thr_id = 0;
  double longest_process_time = 0;

  if (sketchParams.cudaUVM_enabled) {
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      double total_process_time = cudaStreams[thr_id]->process_time.count();
      if (total_process_time > longest_process_time) {
        longest_process_time = total_process_time;
        longest_thr_id = thr_id;
      }
    }
    std::cout << "\n";
    std::cout << "Longest Thread # " << longest_thr_id << ": " << cudaStreams[longest_thr_id]->process_time.count()<< "\n";
    std::cout << "  Edge Fill Time: " << cudaStreams[longest_thr_id]->edge_fill_time.count()<< "\n";
    std::cout << "  CUDA Stream Wait Time: " << cudaStreams[longest_thr_id]->wait_time.count() << "\n"; 
    std::cout << "  Sketch Prefetch Time: " << cudaStreams[longest_thr_id]->prefetch_time.count() << "\n";
  }
  else {
    for (int thr_id = 0; thr_id < num_host_threads; thr_id++) {
      double total_process_time = cudaStreams[thr_id]->process_time.count();
      if (total_process_time > longest_process_time) {
        longest_process_time = total_process_time;
        longest_thr_id = thr_id;
      }
    }
    std::cout << "\n";
    std::cout << "Longest Thread # " << longest_thr_id << ": " << cudaStreams[longest_thr_id]->process_time.count() << "\n";
    std::cout << "  Edge Fill Time: " << cudaStreams[longest_thr_id]->edge_fill_time.count() << "\n";
    std::cout << "  CUDA Stream Wait Time: " << cudaStreams[longest_thr_id]->wait_time.count() << "\n"; 
    std::cout << "  Delta Sketch Applying Time: " << cudaStreams[longest_thr_id]->apply_delta_time.count() << "\n";
  }
}
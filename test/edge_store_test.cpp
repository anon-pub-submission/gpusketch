#include "edge_store.h"
#include "bucket.h"
#include "util.h"

#include <gtest/gtest.h>
#include <chrono>
#include <set>

static size_t get_seed() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

TEST(EdgeStoreTest, no_contract) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 25;  // artificially large
  EdgeStore edge_store(get_seed(), nodes, skt_bytes, 20);

  std::set<Edge> edges_added;

  for (size_t i = 0; i < nodes; i++) {
    std::vector<node_id_t> dsts;
    for (size_t j = 0; j < i; j++) {
      dsts.push_back(j);
      ASSERT_TRUE(edges_added.insert({std::min(i, j), std::max(i, j)}).second);
    }
    auto more_upds = edge_store.insert_adj_edges(i, dsts);
    ASSERT_EQ(more_upds.dsts_data.size(), 0);
  }

  ASSERT_EQ(edge_store.get_num_edges(), edges_added.size());
  ASSERT_EQ(edge_store.get_first_store_subgraph(), 0);

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), edges_added.size());

  for (auto edge : edges) {
    node_id_t src = std::min(edge.src, edge.dst);
    node_id_t dst = std::max(edge.src, edge.dst);
    ASSERT_NE(edges_added.find({src, dst}), edges_added.end());
  }
}

TEST(EdgeStoreTest, no_contract_dupl) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 30;  // artificially large
  EdgeStore edge_store(get_seed(), nodes, skt_bytes, 20);

  std::set<Edge> edges_added;

  for (size_t i = 0; i < nodes; i++) {
    std::vector<node_id_t> dsts;
    for (size_t j = 0; j < i; j++) {
      dsts.push_back(j);
      ASSERT_TRUE(edges_added.insert({i, j}).second);
    }
    auto more_upds = edge_store.insert_adj_edges(i, dsts);
    ASSERT_EQ(more_upds.dsts_data.size(), 0);
  }

  // remove num_nodes edges from graph
  for (size_t i = 0; i < nodes; i++) {
    auto it = edges_added.begin();
    edge_store.insert_adj_edges(it->src, std::vector<node_id_t>{it->dst});
    edges_added.erase(it);
  }

  ASSERT_EQ(edge_store.get_num_edges(), edges_added.size());
  ASSERT_EQ(edge_store.get_first_store_subgraph(), 0);

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), edges_added.size());

  for (auto edge : edges) {
    ASSERT_NE(edges_added.find(edge), edges_added.end());
  }
}

TEST(EdgeStoreTest, contract) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 19; // small enough to likely contract thrice
  size_t num_subgraphs = 20;
  size_t seed = get_seed();
  EdgeStore edge_store(seed, nodes, skt_bytes, num_subgraphs);

  std::set<Edge> edges_added;

  size_t num_returned[6] = {0, 0, 0, 0, 0, 0};
  size_t num_in_subgraphs[6] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < nodes; i++) {
    std::vector<SubgraphTaggedUpdate> dsts;
    for (size_t j = i + 1; j < nodes; j++) {
      node_id_t src = std::min(i, j);
      node_id_t dst = std::max(i, j);
      auto idx = concat_pairing_fn(src, dst);
      size_t depth = Bucket_Boruvka::get_index_depth(idx, seed, num_subgraphs);
      
      for (size_t i = 0; i <= std::min(size_t(5), depth); i++) {
        ++num_in_subgraphs[i];
      }
      for (size_t k = 0; k < std::min(depth + 1, edge_store.get_first_store_subgraph()); k++) {
        ++num_returned[k];
      }
      if (depth >= edge_store.get_first_store_subgraph()) {
        dsts.push_back({depth, j});
      }
      ASSERT_TRUE(edges_added.insert({src, dst}).second);
    }
    auto more_upds = edge_store.insert_adj_edges(i, edge_store.get_first_store_subgraph(), dsts.data(), dsts.size());
    node_id_t src = more_upds.src;
    for (auto dst_data : more_upds.dsts_data) {
      ++num_returned[edge_store.get_first_store_subgraph() - 1];
      node_id_t s = std::min(src, dst_data.dst);
      node_id_t d = std::max(src, dst_data.dst);
      ASSERT_NE(edges_added.find({s, d}), edges_added.end());
    }
  }

  while (edge_store.contract_in_progress()) {
    auto more_upds = edge_store.vertex_advance_subgraph(edge_store.get_first_store_subgraph());
    node_id_t src = more_upds.src;
    for (auto dst_data : more_upds.dsts_data) {
      ++num_returned[edge_store.get_first_store_subgraph() - 1];
      node_id_t s = std::min(src, dst_data.dst);
      node_id_t d = std::max(src, dst_data.dst);
      ASSERT_NE(edges_added.find({s, d}), edges_added.end());
    }
  }

  for (size_t i = 0; i < 6; i++) {
    std::cerr << num_returned[i] << " vs " << num_in_subgraphs[i] << std::endl;
  }

  ASSERT_EQ(num_in_subgraphs[0], edges_added.size());
  for (size_t i = 0; i < edge_store.get_first_store_subgraph(); i++) {
    ASSERT_EQ(num_returned[i], num_in_subgraphs[i]);
  }

  std::vector<Edge> edges = edge_store.get_edges();
  ASSERT_EQ(edges.size(), num_in_subgraphs[edge_store.get_first_store_subgraph()]);

  for (auto edge : edges) {
    node_id_t src = std::min(edge.src, edge.dst);
    node_id_t dst = std::max(edge.src, edge.dst);
    ASSERT_NE(edges_added.find({src, dst}), edges_added.end());
  }
}

TEST(EdgeStoreTest, contract_parallel) {
  size_t nodes = 1024;
  size_t skt_bytes = 1 << 19; // small enough to likely contract twice
  size_t num_subgraphs = 20;
  size_t seed = get_seed();
  EdgeStore edge_store(seed, nodes, skt_bytes, num_subgraphs);

  std::atomic<size_t> num_returned[6] = {0, 0, 0, 0, 0, 0};
  std::atomic<size_t> num_in_subgraphs[6] = {0, 0, 0, 0, 0, 0};

#pragma omp parallel for
  for (size_t i = 0; i < nodes; i++) {
    std::vector<SubgraphTaggedUpdate> dsts;
    size_t edge_store_subgraph = edge_store.get_first_store_subgraph();
    for (size_t j = i + 1; j < nodes; j++) {
      node_id_t src = std::min(i, j);
      node_id_t dst = std::max(i, j);
      auto idx = concat_pairing_fn(src, dst);
      size_t depth = Bucket_Boruvka::get_index_depth(idx, seed, num_subgraphs);
      
      for (size_t i = 0; i <= std::min(size_t(5), depth); i++) {
        ++num_in_subgraphs[i];
      }
      for (size_t k = 0; k < std::min(depth + 1, edge_store_subgraph); k++) {
        ++num_returned[k];
      }
      if (depth >= edge_store_subgraph) {
        dsts.push_back({depth, j});
      }
    }
    auto more_upds = edge_store.insert_adj_edges(i, edge_store_subgraph, dsts.data(), dsts.size());
    node_id_t src = more_upds.src;
    for (auto dst_data : more_upds.dsts_data) {
      ++num_returned[more_upds.min_subgraph];
      node_id_t s = std::min(src, dst_data.dst);
      node_id_t d = std::max(src, dst_data.dst);
    }
  }

#pragma omp parallel
  {
    while (edge_store.contract_in_progress()) {
      auto more_upds = edge_store.vertex_advance_subgraph(edge_store.get_first_store_subgraph());
      node_id_t src = more_upds.src;
      for (auto dst_data : more_upds.dsts_data) {
        ++num_returned[edge_store.get_first_store_subgraph() - 1];
        node_id_t s = std::min(src, dst_data.dst);
        node_id_t d = std::max(src, dst_data.dst);
      }
    }
  }

  for (size_t i = 0; i < 6; i++) {
    std::cerr << num_returned[i] << " vs " << num_in_subgraphs[i] << std::endl;
  }

  std::vector<Edge> edges = edge_store.get_edges();
  std::cerr << "edge store size = " << edges.size() << std::endl;

  for (size_t i = 0; i < edge_store.get_first_store_subgraph(); i++) {
    ASSERT_EQ(num_returned[i], num_in_subgraphs[i]);
  }

  
  ASSERT_EQ(edges.size(), num_in_subgraphs[edge_store.get_first_store_subgraph()]);
}



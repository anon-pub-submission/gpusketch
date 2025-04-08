#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <vector>

#include "types.h"
#include "util.h"
#include <binary_file_stream.h>

static constexpr size_t update_array_size = 10000;

int main(int argc, char** argv) {

  if (argc != 2) {
    std::cout << "ERROR: Incorrect number of arguments!" << std::endl;
    std::cout << "Arguments: graph_file" << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string stream_name = argv[1];
  BinaryFileStream stream(stream_name);
  std::string line;

  std::cout << "Processing stream: " << stream_name << std::endl;

  GraphStreamUpdate update_array[update_array_size];

  //std::string csv_name = stream_name + "_symm.csv";
  std::string csv_static_name = stream_name + "_static_symm.csv";
  //std::ofstream csv_file(csv_name);
  std::ofstream csv_static_file(csv_static_name);
  
  std::unordered_map<vec_t, int> edges;
  size_t total_read_updates = 0;
  bool read_complete = false;

  std::cout << "Reading edges...\n";
  while (!read_complete) {
    size_t updates = stream.get_update_buffer(update_array, update_array_size);

    total_read_updates += updates;
    if (total_read_updates % 100000000 == 0) {
      std::cout << "  Progress: " << total_read_updates << "\n";
    }

    for (size_t i = 0; i < updates; i++) {
      GraphUpdate upd;
      upd.edge = update_array[i].edge;
      upd.type = static_cast<UpdateType>(update_array[i].type);
      if (upd.type == BREAKPOINT) {
        total_read_updates -= (updates - i);
        read_complete = true;
        break;
      }
      else {
        vec_t edge_id = static_cast<vec_t>(concat_pairing_fn(upd.edge.src, upd.edge.dst));
        if (!edges.insert({edge_id, 1}).second) {
          // Current edge already exist, so delete
          if (edges.erase(edge_id) == 0) {
            std::cerr << "ERROR: We found a duplicate but couldn't remove???" << std::endl;
            exit(EXIT_FAILURE);
          }
        }
        //csv_file << upd.edge.src << " " << upd.edge.dst << "\n";
        //csv_file << upd.edge.dst << " " << upd.edge.src << "\n";
      }
    }
  }

  std::cout << "Finished Reading Input Graph File...\n";
  std::cout << "# of Total Updates: " << total_read_updates << "\n";
  std::cout << "# of Edges (without duplicate edges): " << edges.size() << "\n";
  std::cout << "Writing static graph\n";

  for (auto& edge : edges) {
    Edge e = inv_concat_pairing_fn(edge.first);
    csv_static_file << e.src << " " << e.dst << "\n";
    csv_static_file << e.dst << " " << e.src << "\n";
  }

  //csv_file.close();
  csv_static_file.close();
}
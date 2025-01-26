#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <chrono>

class Node {
public:
    int id;
    std::vector<int> timestamp_list;
    int occurrence;
    std::vector<int> neighbors;
    std::map<int, std::vector<int>> neighbors_timestamps;
    bool RNN_ready;
    int presence_1;
    int presence_2;

    Node(int node_id) : id(node_id), occurrence(0), RNN_ready(false), presence_1(0), presence_2(0) {}
};

int main() {
    std::vector<int> num_edges_list;
    std::vector<int> num_common_nodes_list;
    std::vector<std::map<int, Node>> group_adj_lists;
    std::vector<std::vector<int>> timestamp_range_list;

    int total_graphs = 219;
    int graphs_per_group = 2;
    int num_groups = total_graphs / graphs_per_group;
    std::cout << num_groups << std::endl;

    int buffer_capacity = 512 * 1024;
    int feature_vector_size = 128;
    int max_features_in_buffer = std::ceil(buffer_capacity / feature_vector_size);

    int spad_width = 64;
    int PE_array_dim = 256;
    int num_PEs_per_edge_pair = std::ceil(feature_vector_size / spad_width);
    int num_edges_processed_per_memory_access = std::floor(2 * (PE_array_dim / num_PEs_per_edge_pair));

    int RNN_input_dim = 128;
    int RNN_output_dim = 64;
    std::vector<Node> RNN_processing_queue;
    std::map<int, Node> ready_RNN_dict;

    std::vector<int> time_list;
    std::vector<int> internal_time;
    std::vector<int> z1_list;
    std::vector<int> z2_list;

    for (int group_num = 0; group_num <= num_groups; ++group_num) {
        int start_idx = group_num * graphs_per_group;
        int end_idx = std::min((group_num + 1) * graphs_per_group, total_graphs);
        timestamp_range_list.push_back(std::vector<int>(start_idx, end_idx));

        std::map<int, Node> combined_adj_list;

        for (int i = start_idx; i < end_idx; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            std::ifstream file("ca-cit-HepTh_sorted_adj_list_" + std::to_string(i) + ".txt");
            int z1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            z1_list.push_back(z1);

            int node_id;
            while (file >> node_id) {
    // Check if the node exists in combined_adj_list
    auto nodeIt = combined_adj_list.find(node_id);
    if (nodeIt == combined_adj_list.end()) {
        // Node doesn't exist, create a new one
        Node new_node(node_id);
        nodeIt = combined_adj_list.emplace(node_id, std::move(new_node)).first;
    }

    Node& current_node = nodeIt->second;  // Access the Node using the iterator
    current_node.timestamp_list.push_back(i);
    current_node.occurrence += 1;

    int neighbor_id;
    while (file >> neighbor_id) {
        current_node.neighbors.push_back(neighbor_id);
        current_node.neighbors_timestamps[i].push_back(neighbor_id);
    }
}
            auto end_time = std::chrono::high_resolution_clock::now();
            auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            z2_list.push_back(static_cast<int>(elapsed_time));
            internal_time.push_back(static_cast<int>(elapsed_time));
        }

        group_adj_lists.push_back(combined_adj_list);
    }
    int total_time = 0;
    for (int time : internal_time) {
        total_time += time;
    }
    double average_time = static_cast<double>(total_time) / total_graphs;

    std::cout << "Average Internal Time: " << average_time << " microseconds" << std::endl;

    return 0;
}


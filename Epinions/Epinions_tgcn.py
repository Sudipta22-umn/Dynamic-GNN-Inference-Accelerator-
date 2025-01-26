#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 23:30:00 2023

@author: monda089
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 21:23:26 2023

@author: monda089
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 19:50:48 2023

@author: monda089
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:29:20 2023

@author: sudiptamondal
"""

import os
import math

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.timestamp_list = []
        self.occurrence = 0
        self.neighbors = []
        self.neighbors_timestamps = {}
        self.unprocessed_edges=None
        self.RNN_ready=False

# Create a directory to store common graphs for groups if it doesn't exist
if not os.path.exists("common_graphs_for_groups_v2"):
    os.makedirs("common_graphs_for_groups_v2")
##initialize various lists
num_edges_list = []
num_common_nodes_list = []
group_adj_lists = []
timestamp_range_list=[]
## graph and group parameters
total_graphs = 500
graphs_per_group =4

num_groups = total_graphs // graphs_per_group
print(num_groups)
## buffer and feature parameters
buffer_capacity=512*1024
feature_vector_size= 32
max_features_in_buffer=math.ceil(buffer_capacity/feature_vector_size)
## PE parameters
spad_width=64
PE_array_dim=256
num_PEs_per_edge_pair=math.ceil(feature_vector_size/spad_width)
num_edges_processed_per_memory_access= math.floor(2*(PE_array_dim/num_PEs_per_edge_pair))
## cache replacement parameters

## RNN computation parameters
RNN_input_dim=128#128
RNN_output_dim =64#64 
RNN_processing_queue=[]
ready_RNN_dict={}


from resource import *
import time
#print(getrusage(RUSAGE_THREAD))
## for measuring time
time_list=[]


for group_num in range(num_groups+1):
    start_idx = group_num * graphs_per_group
    end_idx = min((group_num + 1) * graphs_per_group, total_graphs)  # Ensure the end index does not exceed the number of graphs
    timestamp_range_list.append(list(range(start_idx, end_idx)))
    # Initialize a dictionary to hold the combined adjacency list for the current group
    combined_adj_list = {}

    # Read and store the adjacency lists for the current group
    for i in range(start_idx, end_idx):
        with open(f'rec-epinions-user-ratings_adj_list_{i}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                node_id, *neighbors = map(int, line.strip().split())
                # Create or get the Node object
                if node_id not in combined_adj_list:
                    combined_adj_list[node_id] = Node(node_id)
                    
                    combined_adj_list[node_id].id=node_id
                    combined_adj_list[node_id].timestamp_list.append(i)
                    combined_adj_list[node_id].occurrence +=1
                    combined_adj_list[node_id].neighbors=neighbors
                    combined_adj_list[node_id].neighbors_timestamps[i]=neighbors
                
                else: 
                    combined_adj_list[node_id].id=node_id
                    combined_adj_list[node_id].timestamp_list.append(i)
                    combined_adj_list[node_id].occurrence +=1
                    combined_adj_list[node_id].neighbors.extend(neighbors)
                    combined_adj_list[node_id].neighbors_timestamps[i]=neighbors
               
    group_adj_lists.append(combined_adj_list)


import os
import matplotlib.pyplot as plt

def compute_common_exclusive (group_adj_lists, i):
    if (i<=num_groups-2):
        group1 = group_adj_lists[i]
        group2 = group_adj_lists[i + 1]
        timestamp_considered = timestamp_range_list[i]+timestamp_range_list[i+1]
    else:
        group1 = group_adj_lists[i]
        group2 = {}
        timestamp_considered = timestamp_range_list[i]

    # Find common nodes between the two groups
    common_node_ids = set(group1.keys()) & set(group2.keys())
    print(common_node_ids)

    # Initialize a dictionary to store the common graph
    common_adj_list = {}
    exclusive_adj_list_1={}
    exclusive_adj_list_2={}
    # Create the adjacency list for the common graph
    for node_id in common_node_ids:
        if node_id not in common_adj_list:
           common_adj_list[node_id] = Node(node_id)
       
        common_neighbors = group1[node_id].neighbors + group2[node_id].neighbors#set(group1[node_id].neighbors) & set(group2[node_id].neighbors)#
        common_adj_list[node_id].neighbors=list(common_neighbors)
        common_timestamps= group1[node_id].timestamp_list + group2[node_id].timestamp_list
        common_adj_list[node_id].timestamp_list=list(common_timestamps)
        common_adj_list[node_id].occurrence=len(common_timestamps)
    
        common_adj_list[node_id].id=node_id
        common_adj_list[node_id].neighbors_timestamps = {}
        for timestamp in common_timestamps:
            common_neighbors_timestamp = []
            
            if timestamp in group1[node_id].neighbors_timestamps:
                common_neighbors_timestamp += group1[node_id].neighbors_timestamps[timestamp]
            
            if timestamp in group2[node_id].neighbors_timestamps:
                common_neighbors_timestamp += group2[node_id].neighbors_timestamps[timestamp]
            
            common_adj_list[node_id].neighbors_timestamps[timestamp] = list(common_neighbors_timestamp)
    
    for node_id in group1:
        if node_id not in common_node_ids:
           if node_id not in exclusive_adj_list_1:
              exclusive_adj_list_1[node_id] = Node(node_id)
    
           exclusive_adj_list_1[node_id].id = node_id
           exclusive_adj_list_1[node_id].timestamp_list.extend(group1[node_id].timestamp_list)
           exclusive_adj_list_1[node_id].occurrence += group1[node_id].occurrence
           exclusive_adj_list_1[node_id].neighbors.extend(group1[node_id].neighbors)
           ## Compute neighbors_timestamps for exclusive_adj_list_1
           for timestamp, neighbors in group1[node_id].neighbors_timestamps.items():
               if timestamp not in exclusive_adj_list_1[node_id].neighbors_timestamps:
                  exclusive_adj_list_1[node_id].neighbors_timestamps[timestamp] = []
               exclusive_adj_list_1[node_id].neighbors_timestamps[timestamp].extend(neighbors)
               
    for node_id in group2:
         if node_id not in common_node_ids:
            if node_id not in exclusive_adj_list_2:
               exclusive_adj_list_2[node_id] = Node(node_id)
     
            exclusive_adj_list_2[node_id].id = node_id
            exclusive_adj_list_2[node_id].timestamp_list.extend(group2[node_id].timestamp_list)
            exclusive_adj_list_2[node_id].occurrence += group2[node_id].occurrence
            exclusive_adj_list_2[node_id].neighbors.extend(group2[node_id].neighbors)
            ## Compute neighbors_timestamps for exclusive_adj_list_1
            for timestamp, neighbors in group2[node_id].neighbors_timestamps.items():
                if timestamp not in exclusive_adj_list_2[node_id].neighbors_timestamps:
                   exclusive_adj_list_2[node_id].neighbors_timestamps[timestamp] = []
                exclusive_adj_list_2[node_id].neighbors_timestamps[timestamp].extend(neighbors)
      
    return common_adj_list, exclusive_adj_list_1, exclusive_adj_list_2, timestamp_considered

def compute_common_exclusive_2(group_adj_lists, i):
    if i <= num_groups - 2:
        group1 = group_adj_lists[i]
        group2 = group_adj_lists[i + 1]
        timestamp_considered = timestamp_range_list[i] + timestamp_range_list[i + 1]
    else:
        group1 = group_adj_lists[i]
        group2 = {}
        timestamp_considered = timestamp_range_list[i]

    # Find common nodes between the two groups
    common_node_ids = set(group1.keys()) & set(group2.keys())
    #print(common_node_ids)

    # Initialize a dictionary to store the common graph
    common_adj_list = {
        node_id: Node(node_id) for node_id in common_node_ids
    }

    for node_id in common_node_ids:
        common_neighbors = list(set(group1[node_id].neighbors) | set(group2[node_id].neighbors))
        common_adj_list[node_id].neighbors = common_neighbors
        common_timestamps = group1[node_id].timestamp_list + group2[node_id].timestamp_list
        common_adj_list[node_id].timestamp_list = common_timestamps
        common_adj_list[node_id].occurrence = len(common_timestamps)
        common_adj_list[node_id].id = node_id
        common_adj_list[node_id].neighbors_timestamps = {}

        for timestamp in common_timestamps:
            common_neighbors_timestamp = (
                group1[node_id].neighbors_timestamps.get(timestamp, []) +
                group2[node_id].neighbors_timestamps.get(timestamp, [])
            )
            common_adj_list[node_id].neighbors_timestamps[timestamp] = common_neighbors_timestamp

    # Merge exclusive_adj_list_1 and exclusive_adj_list_2 into a single exclusive_adj_list
    exclusive_adj_list = {}

    for group in [group1, group2]:
        for node_id in group:
            if node_id not in common_node_ids:
                if node_id not in exclusive_adj_list:
                    exclusive_adj_list[node_id] = Node(node_id)

                exclusive_adj_list[node_id].id = node_id
                exclusive_adj_list[node_id].timestamp_list.extend(group[node_id].timestamp_list)
                exclusive_adj_list[node_id].occurrence += group[node_id].occurrence
                exclusive_adj_list[node_id].neighbors.extend(group[node_id].neighbors)

                for timestamp, neighbors in exclusive_adj_list[node_id].neighbors_timestamps.items():
                    if timestamp not in exclusive_adj_list[node_id].neighbors_timestamps:
                        exclusive_adj_list[node_id].neighbors_timestamps[timestamp] = []
                    exclusive_adj_list[node_id].neighbors_timestamps[timestamp].extend(neighbors)

    return common_adj_list, exclusive_adj_list, timestamp_considered

def count_required_vertices (DRAM_dict, timestamp_range):
    # Initialize a variable to store the total count
    total_count = 0
    # Iterate through the nodes in common_adj_list
    for node_id, node in DRAM_dict.items():
        # Initialize a variable to store the count for this node
        node_count = 0
        # Iterate through the timestamps in the neighbors_timestamps dictionary
        for timestamp in timestamp_range:
            if timestamp in node.neighbors_timestamps:
            # Count the number of elements in the neighbors_list and add it to node_count
              node_count += len(node.neighbors_timestamps[timestamp])
        # Add node_count to the total_count
        total_count += node_count
    return total_count

def initalize_unprocessed_edges (timestamp_range, DRAM_dict):
    for node_id, node in DRAM_dict.items():
        unprocessed_edges=0
        for timestamp in timestamp_range:
            if timestamp in node.neighbors_timestamps:
                unprocessed_edges += len(node.neighbors_timestamps[timestamp])
            node.unprocessed_edges=unprocessed_edges
    return DRAM_dict


def RNN_current_feature_weighting(input_dim, output_dim,num_vertices,buffer_size):

    input_layer_dim=input_dim ## 16to match with GNNAdvsior   64 used originally## hidden layer dim
    output_dim=output_dim ## num of class/labels
    PE_array_dim=16 ## 16x16 PE array
    MACs_per_CPE=4
    num_vertices=num_vertices
    input_buffer_size= buffer_size

    num_pass= math.ceil(output_dim/PE_array_dim) # 1
    num_nodes_per_set=math.floor(input_buffer_size/input_layer_dim)#141
    #print(f'the num nodes per set :{num_nodes_per_set}')
    num_set=math.ceil(num_vertices/num_nodes_per_set)#24
    #print(f'num set:{num_set}')  
    sub_vector_len=math.ceil(input_layer_dim/PE_array_dim)
    #print(f'sub vec len:{sub_vector_len}')
    cycle_required_per_node=math.ceil(sub_vector_len/MACs_per_CPE) #58
    #print(f'cycles required per node {cycle_required_per_node}')
    
    if num_vertices < num_nodes_per_set:
        RNN_weight_cycles_required=math.ceil((cycle_required_per_node*num_vertices*num_pass*num_set))
    else:
        RNN_weight_cycles_required=math.ceil((cycle_required_per_node*num_nodes_per_set*num_pass*num_set))
    
    return RNN_weight_cycles_required

def process_RNN_weighting (ready_RNN_cnt, timestamp):
    #ready_RNN_cnt= len(ready_RNN_list)
   
    print(f'the number of nodes ready for RNN at timestamp {timestamp} is {ready_RNN_cnt}')   
    ## count the number of cycles required for RNN (WXand UH)
    RNN_weight_cycle_count = RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)
    print(f'the number of cycles required for RNN weighting at timestamp {timestamp} is {RNN_weight_cycle_count}') 
    return RNN_weight_cycle_count
    
def process_RNN_hidden (ready_RNN_cnt, timestamp):
    #ready_RNN_cnt= len(ready_RNN_list)
   
    print(f'the number of nodes ready for RNN at timestamp {timestamp} is {ready_RNN_cnt}')   
    ## count the number of cycles required for RNN (WXand UH)
    if timestamp >=1:
        RNN_hidden_DRAM_cycle_count=math.ceil((ready_RNN_cnt*feature_vector_size)/(256e9)/1e-9)
    else:
        RNN_hidden_DRAM_cycle_count=0
        
    RNN_sigmoid_cycle_cnt=3*1
    RNN_tanh_cycle_cnt=2*1
    RNN_addition_cycle_cnt=RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)/16
    RNN_hadammard_cycle_cnt=RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)*3
    
    RNN_hidden_cycle_count=RNN_sigmoid_cycle_cnt+RNN_tanh_cycle_cnt+RNN_addition_cycle_cnt+RNN_hadammard_cycle_cnt+RNN_hidden_DRAM_cycle_count
    #RNN_hidden_cycle_count = RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)
    total_RNN_hidden_cycle_count= RNN_hidden_DRAM_cycle_count + RNN_hidden_cycle_count
    print(f'the number of cycles required for RNN hidden at timestamp {timestamp} is {total_RNN_hidden_cycle_count}') 
    return total_RNN_hidden_cycle_count

   
def process_RNN_SFU (ready_RNN_cnt, timestamp):
    #ready_RNN_cnt= len(ready_RNN_list)
    RNN_sigmoid_cycle_cnt=3*1
    RNN_tanh_cycle_cnt=2*1
    RNN_addition_cycle_cnt=RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)/16
    RNN_hadammard_cycle_cnt=RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)*3
    RNN_SFU_DRAM_write_cycle_count = RNN_current_feature_weighting(RNN_input_dim,RNN_output_dim,ready_RNN_cnt,buffer_capacity)
    
    total_RNN_SFU_cycle_count=RNN_sigmoid_cycle_cnt+RNN_tanh_cycle_cnt+RNN_addition_cycle_cnt+RNN_hadammard_cycle_cnt+RNN_SFU_DRAM_write_cycle_count

    return total_RNN_SFU_cycle_count
      

def process_current_subgraph(subgraph, timestamp_range):
# Iterate over timestamps
    edge_processed_dict={}
    cache_miss_cnt_dict={}
    RNN_cycle_dict={}
    #ready_RNN_dict={}
    
    for timestamp in timestamp_range:
        total_edges_processed = 0
        cache_miss_cnt=0
        ready_RNN_list=[]
        # Iterate over nodes in the subgraph
        for node_id, node in subgraph.items():
            if timestamp in node.neighbors_timestamps:
                neighbors_list = node.neighbors_timestamps[timestamp]
                neighbors_to_remove = []
                ## if all nodes are processed then the  its reeady for RNN
                

                # Process neighbors for the current timestamp
                for nbr in neighbors_list:
                    # check every neighbor in the current neighbor list
                    if nbr in subgraph:
                        # if nbr already present in the subgraph process one more neighbors and increaase edges processed in the current timestamp
                        neighbors_to_remove.append(nbr)
                        total_edges_processed += 1
                        #print(total_edges_processed)
                    else:
                        ## if neighbor not in the current subgraph increase cache miss count
                        cache_miss_cnt +=1
                # Remove neighbors from the current timestamp's neighbors_list
                for nbr in neighbors_to_remove:
                    neighbors_list.remove(nbr)
    
                # Remove the node from the neighbor's neighbors_timestamps at the same timestamp
                for nbr in neighbors_to_remove:
                    if nbr in subgraph:
                        nbr_timestamps = subgraph[nbr].neighbors_timestamps
                        if timestamp in nbr_timestamps:
                            if node in nbr_timestamps[timestamp]:
                                nbr_timestamps[timestamp].remove(node)
                                
                if neighbors_list == [] and node.RNN_ready==False:
                    node.RNN_done=True
                    ready_RNN_list.append(node.id)
        # Store the total_edges_processed in edge_processed_dict under the current timestamp
        edge_processed_dict[timestamp] = total_edges_processed
        cache_miss_cnt_dict[timestamp]=cache_miss_cnt
        ## for storing the values ready for RNN
        ready_RNN_dict[timestamp]=len(ready_RNN_list)
        ## if current timestamp is greater or equal to 1 then both weighting and hidden feature muliplication can proceed
        if timestamp >=1:
             RNN_weight_cycle = process_RNN_weighting(ready_RNN_dict[timestamp],timestamp)#len(ready_RNN_list),timestamp)
             RNN_hidden_cycle = process_RNN_hidden(ready_RNN_dict[timestamp-1],timestamp-1)
             # if timestamp >=2:
             #     RNN_SFU_cycle=process_RNN_SFU(ready_RNN_dict[timestamp-1],timestamp-1)
             RNN_cycle_dict[timestamp]=RNN_hidden_cycle + RNN_weight_cycle
        ## else only hidden feature multiplication can proceed
        else:
            RNN_hidden_cycle = process_RNN_hidden(ready_RNN_dict[timestamp],timestamp)
            RNN_cycle_dict[timestamp]=RNN_hidden_cycle
    return edge_processed_dict, subgraph, cache_miss_cnt_dict, RNN_cycle_dict

def compute_cache_evictions (subgraph, timestamp_range, alpha):
    cache_evictions=[]
    for node_id, node in subgraph.items():
        unprocessed_edges=0
        for timestamp in timestamp_range:
            if timestamp in node.neighbors_timestamps:
                unprocessed_edges += len(node.neighbors_timestamps[timestamp])
            node.unprocessed_edges=unprocessed_edges
        #print(node.id, node.unprocessed_edges)
        if node.unprocessed_edges <=alpha:
            cache_evictions.append(node_id)
    return cache_evictions

def apply_evictions (subgraph, cache_evictions):
    for node in cache_evictions:
        del subgraph[node]
    return subgraph


def DRAM_to_cache_communication (DRAM_dict, tmp_key, available_buffer_space, timestamp_range, subgraph,):
    off_chip_access_cnt=0
    off_chip_cycle_cnt=0
    for node_id, node in DRAM_dict.items():
        #if available_buffer_space >0:
            if node_id not in subgraph and node.unprocessed_edges >0: ## if not already present in the subgraph and have unprocessed
                subgraph[node_id]=node
                tmp_key=node_id
                for timestamp in timestamp_range:
                    if timestamp in node.neighbors_timestamps and available_buffer_space >0:
                        available_buffer_space -= len(node.neighbors_timestamps[timestamp])
                        off_chip_access_cnt += len(node.neighbors_timestamps[timestamp])
                        print(f'available_buffer_space:{available_buffer_space}')
        # else:
        #     if (available_buffer_space <= 0):
        #         print("not enough buffer space")
        #     if (tmp_key == DRAM_index):
        #         print("completed one round")
        #     break
    off_chip_cycle_cnt += math.ceil((off_chip_access_cnt*feature_vector_size)/(256e9)/1e-9)
    return subgraph, off_chip_cycle_cnt 

total_edges_processed_per_group_top=[] # keeps tracks of total edges processed for each group
GNN_cycle_list_per_group_top=[]
off_chip_cycle_list_per_group_top=[]
RNN_cycle_list_per_group_top=[]

## for computing the indices i for the creation of common, exclusive_1 and exclusive_2
for i in range(0,len(group_adj_lists),2):
## for each i compute commong, ex_1, ex_2
    common_adj_list, exclusive_adj_list_1, exclusive_adj_list_2, timestamp_considered = compute_common_exclusive(group_adj_lists,i)  
    #get the current time stamps under consideration
    z1=getrusage(RUSAGE_THREAD)[0]
    common_adj_list, exclusive_adj_list, timestamp_considered = compute_common_exclusive_2(group_adj_lists,i)  
    z2=getrusage(RUSAGE_THREAD)[0]
    time_list.append(z2-z1)
    
    timestamp_range=timestamp_considered
    
    DRAM_dict={}
    for d in [common_adj_list, exclusive_adj_list_1, exclusive_adj_list_2]:
        ## keep on filling the DRAM dict, i.e.,the location we are going to consider
        DRAM_dict.update(d)
    ##initialize the unprocessed edges of DRAM elements (nodes)    
    DRAM_dict= initalize_unprocessed_edges(timestamp_range, DRAM_dict)
    ## compute the total DRAM to cache communication (in terms of features required)
    total_DRAM_access_required = count_required_vertices(DRAM_dict, timestamp_range)
    
    ## variables for measuring cycles/performance
    edges_processed_per_group=[]
    GNN_PE_cycle_per_group=[]
    cache_misses_per_group=[]
    off_chip_cycle_per_group=[]
    RNN_cycle_per_group=[]
    
    
    if (total_DRAM_access_required > max_features_in_buffer):
          print("cache replacement is required")
          #subgraph = {k: DRAM_dict[k] for i, (k, v) in enumerate(DRAM_dict.items()) if i < max_features_in_buffer}
          subgraph={}
          node_count=0
          for key, val in DRAM_dict.items():
              node_count += len(val.neighbors)
              if node_count >= max_features_in_buffer:
                    break 
              else:
                  subgraph[key]=val
          print(f'size of the subgraph before evictions: {count_required_vertices(subgraph, timestamp_range)}')    
          DRAM_index=list(subgraph.items())[-1][0]
          
          ## for counitng initial memory accesses
          initial_off_chip_access_cycles = math.ceil((max_features_in_buffer*feature_vector_size)/(256e9)/1e-9)
          off_chip_cycle_list_per_group_top.append(initial_off_chip_access_cycles)
          
          for iter in range(10):
              # Print the processed edge counts per timestamp
              edge_processed_dict, subgraph, cache_miss_cnt_dict, RNN_cycle_dict = process_current_subgraph(subgraph, timestamp_range)
              
              edges_processed_per_iteration=0
              ## for tracking edges processed per iteration
              GNN_PE_cycle_per_iteration=0
              ## for counitng PE cycles for GNN per iteration
              cache_miss_per_iteration=0
              ## for tracking cache miss per iteration
              
              for timestamp, count in edge_processed_dict.items():
                  ## for counitng the number of edges processed per timestamp
                  print(f"Timestamp {timestamp}: {count} edges processed")
                  edges_processed_per_iteration += count
              edges_processed_per_group.append(edges_processed_per_iteration)
              GNN_PE_cycle_per_iteration = math.ceil(edges_processed_per_iteration/num_edges_processed_per_memory_access)
              GNN_PE_cycle_per_group.append(GNN_PE_cycle_per_iteration)
              
              RNN_cycle_per_iteration=0
              for timestamp, RNN_cycle_cnt in RNN_cycle_dict.items():
                  RNN_cycle_per_iteration += RNN_cycle_cnt
              RNN_cycle_per_group.append(RNN_cycle_per_iteration)
              
              for timestamp, count in cache_miss_cnt_dict.items():
                  ## for counitng the number of edges processed per timestamp
                  print(f"Timestamp {timestamp}: {count} edges processed")
                  cache_miss_per_iteration += count
              cache_misses_per_group.append(cache_miss_per_iteration )
              
              
              if iter >=1:
                  ## for checking if for two consecutive iterations there is no/slow progress
                  if edges_processed_per_group [iter] == 0 and edges_processed_per_group [iter-1] == 0:
                      print("increasing alpha due to slow progress")
                      ## for bumpiong the value of alpha
                      alpha = 10
                  else:
                      # else reset the alpha to initial value
                      alpha=5
              else:
                  ## setting the initital alpha
                  alpha = 5
                      
              ## for computing the cache evictions (candidates for cahce replacement)
              cache_evictions = compute_cache_evictions(subgraph, timestamp_range, alpha)
              ## after updating the unprocessed edges check if there are still there are edges to unprocessed if so stop the computation
              remaining_unprocessed_edges=0
              for node_id, node in DRAM_dict.items():
                  remaining_unprocessed_edges += node.unprocessed_edges
              if (remaining_unprocessed_edges ==0):
                  print(f'All the edges are processes for the group {i} and now Breaking!!')
                  break
               
              ## for implementing the cache replacements
              subgraph = apply_evictions(subgraph, cache_evictions)
              
              ## for compuitng the available buffer space for bringing the next set of vertices
              available_buffer_space=max_features_in_buffer-count_required_vertices(subgraph,timestamp_range)
              ## assinging the last key of the DRAM dict as the tmp_key
              tmp_key= DRAM_index
              print(f'tmp_key is: {tmp_key}')
              ## for implementing the cache replacement in the cuurent subgraph
              subgraph, off_chip_cycle_cnt_per_iter = DRAM_to_cache_communication(DRAM_dict, tmp_key, available_buffer_space, timestamp_range, subgraph)  
              iter +=1
              off_chip_cycle_per_group.append(off_chip_cycle_cnt_per_iter)
              
          ## for updatig the total edge list at top     
          total_edges_processed_per_group_top.append(sum(edges_processed_per_group))
          ## for updating the off chip cycle count list at top
          off_chip_cycle_list_per_group_top.append(sum(off_chip_cycle_per_group))
          ## for updating the GNN cycle count list at top
          GNN_cycle_list_per_group_top.append(sum(GNN_PE_cycle_per_group))
          ## for updating th RNN cycle count list at top
          RNN_cycle_list_per_group_top.append(sum(RNN_cycle_per_group))
          
          
          
    else:
        print("no cache replacement is required")
        subgraph=DRAM_dict
       # DRAM_index=list(subgraph.items())[-1][0]
        edge_processed_dict,subgraph,cache_miss_cnt_dict, RNN_cycle_dict = process_current_subgraph(subgraph, timestamp_range)
        
        ## for computing the initial number of DRAM access cycles
        initial_off_chip_access_cycles = math.ceil((total_DRAM_access_required*feature_vector_size)/(256e9)/1e-9)
        off_chip_cycle_list_per_group_top.append(initial_off_chip_access_cycles)
        
        edges_processed_per_iteration=0
        ## for tracking edges processed per iteration
        for timestamp, count in edge_processed_dict.items():
            ## for counitng the number of edges processed per timestamp
            print(f"Timestamp {timestamp}: {count} edges processed")
            edges_processed_per_iteration += count
        edges_processed_per_group.append(edges_processed_per_group)        
        ## for computing the cache evictions (candidates for cahce replacement) 
        total_edges_processed_per_group_top.append(edges_processed_per_iteration)
        GNN_PE_cycle_per_iteration = math.ceil(edges_processed_per_iteration/num_edges_processed_per_memory_access)
        GNN_cycle_list_per_group_top.append(GNN_PE_cycle_per_iteration)
        ## for compuitng the RNN cycles
        RNN_cycle_per_iteration=0
        for timestamp, RNN_cycle_cnt in RNN_cycle_dict.items():
            RNN_cycle_per_iteration += RNN_cycle_cnt
        RNN_cycle_list_per_group_top.append(RNN_cycle_per_iteration)
        
        
total_processing_cycle=sum(GNN_cycle_list_per_group_top)+sum(off_chip_cycle_list_per_group_top)+sum(RNN_cycle_list_per_group_top)

total_processing_time=total_processing_cycle*1e-9
print(f'total processing time for the dataset:{total_processing_time}')
print(f'avearge inference time for a graph:{total_processing_time/total_graphs}')
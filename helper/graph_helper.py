# import numpy as np
# import random

# from helper.debug import *

# """
# Graph generation
#     Adding home button is good
#     Now need to make sure that every node is reachable from the home page
#     Two ways to represent the graph (adjacency matrix and adjacency list)
#     Creating skeleton of graph is a good approximation
# Creating graphs that are sparse and testing on that
#     Exclude the home page in sparse calculation
#     Have some constant scaled on the number of states as an upper bound
#     Make sure that the graph is still connected at its lowest sparsity level

# Parameters
# - number of nodes --> num_screens
# - max number of edges going from a node --> num_buttons
# - sparse constant --> NEED TO SPECIFY NEW ENV VARIABLE
# - number of edges --> sparse constant * number of nodes

# - number of chains
# - maximum chain length
# """

# # creates main frame of graph to ensure it is connected
# def create_tree(num_branches, num_tiers, ):
#     skeleton = []
    
#     length_chains = [max_chain_length]
#     added_nodes = max_chain_length + (num_chains - 1) + 1

#     debugc("", 0)

#     for chain_id in range(1, num_chains):
#         min_nodes = max(0, num_nodes - added_nodes - (max_chain_length - 1) * (num_chains - chain_id - 1))
#         max_nodes = min(max_chain_length-1, num_nodes - added_nodes)

#         debugc("chain ID: {}".format(chain_id), 0)
#         debugc("-------------------------", 0)
#         debugc("added nodes: {}".format(added_nodes), 0)
#         debugc("num chains remaining: {}".format(num_chains - chain_id - 1), 0)
#         debugc("min num nodes: {}".format(min_nodes), 0)
#         debugc("max num nodes: {}".format(max_nodes), 0)
#         debugc("", 0)

#         chain_body_len = random.randint(min_nodes, max_nodes)
#         length_chains.append(chain_body_len + 1)
#         added_nodes += chain_body_len
    
#     debugc(length_chains, 0)
#     debugc("", 0)

#     ind = 1

#     for i in range(num_chains):
#         skeleton.append([])
#         for j in range(length_chains[i]):
#             skeleton[i].append(ind)
#             ind += 1
    
#     for i in range(num_chains):
#         debugc(skeleton[i], 0)
    
#     assert(sum(length_chains) == num_nodes - 1)
#     assert(max(length_chains) == max_chain_length)
#     assert(min(length_chains) > 0)
#     assert(len(length_chains) == num_chains)
    
#     return skeleton, length_chains

# def create_skeleton_dict(num_chains, length_chains, skeleton):
#     # dict value is tuple with (chain_id, tier)
#     dict = {}

#     for i in range(num_chains):
#         for j in range(length_chains[i]):
#             dict[skeleton[i][j]] = (i, j)
    
#     return dict

# def create_skeleton_tiers(num_chains, length_chains, skeleton):
#     tier_list = []
#     max_tier = max(length_chains)

#     for tier in range(max_tier):
#         tier_list.append([])
#         for chain_id in range(num_chains):
#             if tier < length_chains[chain_id]:
#                 tier_list[tier].append(skeleton[chain_id][tier])
    
#     return tier_list

# def multi_chaining(num_chains, length_chains, skeleton, num_nodes, num_edges):
#     # num_edges doesn't include directed edges from nodes to home node
#     edges = []

#     curr_num_edges = sum(length_chains)
#     dict = create_skeleton_dict(num_chains, length_chains, skeleton)
#     tier_list = create_skeleton_tiers(num_chains, length_chains, skeleton)

#     max_tier = len(tier_list) - 1

#     # adding edges from skeleton
#     for chain_id in range(num_chains):
#         for tier in range(length_chains[chain_id]):
#             if tier == 0:
#                 edges.append([0, skeleton[chain_id][tier]])
#             else:
#                 edges.append([skeleton[chain_id][tier-1], skeleton[chain_id][tier]])

#     assert(len(edges) == sum(length_chains))

#     # adding new edges on top of skeleton
#     for iter in range(curr_num_edges, num_edges):
#         node1 = random.randint(1, num_nodes-1)

#         tier1 = dict[node1][1]
#         tier2 = random.randint(0, max_tier-1)
#         if tier2 >= tier1:
#             tier2 += 1
        
#         node2 = tier_list[tier2][random.randint(0, len(tier_list[tier2]) - 1)]

#         if tier1 > tier2:
#             edges.append([node2, node1])
#         else:
#             edges.append([node1, node2])
    
#     debugc(len(edges), 1)
#     assert(len(edges) == num_edges)

#     # adding edges from every other node to home node
#     for node in range(1, num_nodes):
#         edges.append([node, 0])
    
#     debugc(len(edges), 1)
    
#     return edges

# def generate_adjacency_matrix(edges, num_nodes):
#     adj_mat = np.zeros((num_nodes, num_nodes)).astype(int)

#     for edge in edges:
#         adj_mat[edge[0]][edge[1]] += 1
    
#     return adj_mat

# """ def main():
#     num_chains = 1
#     num_nodes = 4
#     max_chain_length = 3
#     num_edges = 4

#     skeleton, length_chains = create_skeleton(num_chains, max_chain_length, num_nodes)
#     edges = multi_chaining(num_chains, length_chains, skeleton, num_nodes, num_edges)

#     debugc("\nskeleton:", 1)
#     for chain in skeleton:
#         debugc(chain, 1)

#     debugc("\nedges:", 2)
#     for edge in edges:
#         debugc(edge, 2)
    
#     adj_mat = generate_adjacency_matrix(edges, num_nodes)
    
#     debugc("\nadjacency matrix:", 2)
#     for row in adj_mat:
#         debugc(row, 2)

# if __name__ == '__main__':
#     main() """

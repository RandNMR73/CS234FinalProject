import numpy as np

from helper.debug import *

"""
Graph generation
    Adding home button is good
    Now need to make sure that every node is reachable from the home page
    Two ways to represent the graph (adjacency matrix and adjacency list)

    Adjacency list representation likely much better for tree structure

Create graphs as trees
    Tree structure can be varied with number of branches within a subtree
    Can increase complexity by adding edges jumping between nodes on different trees

Parameters
- number of tiers/depth of tree
- number of branches
- amount of jumping within tier

Optional parameters???
- max depth of tree
- min depth of tree
- max branching
- min branching

- max number of nodes?
- max number of branches?
"""

# creates tree structure of graph
# output is as an adjacency list (dictionary)
def create_tree(num_tiers, num_branches):
    last_tier = [0]
    edges = []
    adj_list = {}
    adj_list[0] = []

    cur_node = 1

    for tier in range(num_tiers):
        new_tier = []

        for node in last_tier:
            for branch in range(num_branches[tier]):
                edges.append([node, cur_node])
                edges.append([cur_node, 0])

                adj_list[cur_node] = [0]
                adj_list[node].append(cur_node)
                new_tier.append(cur_node)

                cur_node += 1
        
        last_tier = new_tier
    
    num_nodes = cur_node
    
    return edges, adj_list, num_nodes

def generate_adjacency_matrix(edges, num_nodes):
    adj_mat = np.zeros((num_nodes, num_nodes)).astype(int)

    for edge in edges:
        adj_mat[edge[0]][edge[1]] += 1
    
    return adj_mat

""" def main():
    num_tiers = 3
    num_branches = [1, 2, 2]

    edges, adj_list, num_nodes = create_tree(num_tiers, num_branches)

    debugc("\nedges:", 4)
    for edge in edges:
        debugc(edge, 4)
    
    debugc("\nadjacency list:", 4)
    for i in range(num_nodes):
        debugc(str(i) + ": " + str(adj_list[i]), 4)
    
    adj_mat = generate_adjacency_matrix(edges, num_nodes)
    
    debugc("\nadjacency matrix:", 4)
    for row in adj_mat:
        debugc(row, 4)

if __name__ == '__main__':
    main() """

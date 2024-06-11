import numpy as np
import random

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
def create_tree(num_tiers, num_branches_per_tier):
    last_tier = [0]
    edges = []

def generate_adjacency_matrix(edges, num_nodes):
    adj_mat = np.zeros((num_nodes, num_nodes)).astype(int)

    for edge in edges:
        adj_mat[edge[0]][edge[1]] += 1
    
    return adj_mat

""" def main():
    num_chains = 1
    num_nodes = 4
    max_chain_length = 3
    num_edges = 4

    skeleton, length_chains = create_skeleton(num_chains, max_chain_length, num_nodes)
    edges = multi_chaining(num_chains, length_chains, skeleton, num_nodes, num_edges)

    debugc("\nskeleton:", 1)
    for chain in skeleton:
        debugc(chain, 1)

    debugc("\nedges:", 2)
    for edge in edges:
        debugc(edge, 2)
    
    adj_mat = generate_adjacency_matrix(edges, num_nodes)
    
    debugc("\nadjacency matrix:", 2)
    for row in adj_mat:
        debugc(row, 2)

if __name__ == '__main__':
    main() """

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def main():
    train_folder = "output/train/05-06-2024-08-54/"
    valid_folder = "output/predict/05-06-2024-08-54/"

    env_folder = train_folder + "env/"
    env_valid_folder = valid_folder + "env/"

    adj_mat_train = np.load(env_folder + "adjacency_matrix.npy")
    states_train = np.load(env_folder + "states.npy")
    transition_train = np.load(env_folder + "transition_matrix.npy")
    target_train = np.load(env_folder + "target.npy")

    adj_mat_valid = np.load(env_valid_folder + "adjacency_matrix.npy")
    states_valid = np.load(env_valid_folder + "states.npy")
    transition_valid = np.load(env_valid_folder + "transition_matrix.npy")
    target_valid = np.load(env_valid_folder + "target.npy")

    assert((adj_mat_train == adj_mat_valid).all())
    assert((states_train == states_valid).all())
    assert((transition_train == transition_valid).all())
    assert((target_train == target_valid).all())

    '''for i in range(adj_mat_train.shape[0]):
        print(adj_mat_train[i])
    print()
    for i in range(transition_train.shape[0]):
        print(transition_train[i])'''

    adj_mat = adj_mat_train
    graph = nx.from_numpy_array(adj_mat)
    nx.draw(graph)
    plt.imsave("output/graph.png")

if __name__ == '__main__':
    main()

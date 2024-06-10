import sys

import math
import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

from gymnasium import Env, spaces

from helper.screen_helper import *
from helper.graph_helper import *
from helper.utils import *

class ScreenNavDiscEnv(Env):
    def __init__(self, config, adj_mat=None, transition=None, states=None, target=-1):
        super(ScreenNavDiscEnv, self).__init__()
        
        # storing parameters from config dictionary
        self.seed = config['agent_seed']
        random.seed(self.seed) # setting random seed
        
        self.device = config['device']
        self.gamma = config['gamma']
        
        self.width = config['screen_width']
        self.height = config['screen_height']

        self.num_screens = config['num_screens']
        self.num_chains = config['num_chains']
        self.max_chain_length = config['max_chain_length']
        self.num_edges = config['num_edges']
        self.sparsity_const = config['sparsity_constant'] # E / V

        if (self.sparsity_const > 0.0):
            self.num_edges = math.ceil(self.num_screens * self.sparsity_const)

        self.max_ep_len = config['max_episode_length']      
        # self.num_buttons = config['num_buttons'] # likely do not need this parameter

        # setting up graph structure of environment
        skeleton, length_chains = create_skeleton(self.num_chains, self.max_chain_length, self.num_screens)
        edges = multi_chaining(self.num_chains, length_chains, skeleton, self.num_screens, self.num_edges)

        self.adj_mat = None
        if (adj_mat is None):
            self.adj_mat = generate_adjacency_matrix(edges, self.num_screens)
        else:
            self.adj_mat = adj_mat

        self.num_buttons_all = find_num_buttons_per_state(self.adj_mat)
        self.max_num_buttons = find_max_num_buttons(self.num_buttons_all)
        
        self.transition = None
        if (transition is None):
            self.transition = generate_transition_matrix(self.adj_mat, self.num_screens, self.num_buttons_all, self.max_num_buttons)
        else:
            self.transition = transition
        
        # setting up physical images for the environment
        button_cmap = mpl.colormaps['autumn']
        screen_cmap = mpl.colormaps['winter']

        self.button_colors = button_cmap(np.linspace(0, 1, self.max_num_buttons)) * 255
        self.screen_colors = screen_cmap(np.linspace(0, 1, self.num_screens)) * 255

        self.button_colors = self.button_colors[:,:3].astype(int)
        self.screen_colors = self.screen_colors[:,:3].astype(int)

        np.random.shuffle(self.button_colors)
        np.random.shuffle(self.screen_colors)
        
        self.num_cols = math.ceil(math.sqrt(self.max_num_buttons))
        self.button_width = math.floor(4.0 * self.width / (5 * self.num_cols + 1))
        self.button_height = self.button_width
        self.gap_x = math.floor(self.button_width / 4)
        self.gap_y = self.gap_x

        if (states is None):
            self.states = set_states(
                self.height,
                self.width,
                self.button_height,
                self.button_width,
                self.gap_x,
                self.gap_y,
                self.button_colors,
                self.num_cols,
                self.screen_colors,
                self.num_buttons_all,
                self.num_screens
            )
            self.states = self.states.astype(np.uint8)
        else:
            self.states = states
        
        self.state = random.randint(0, self.num_screens-1)

        # define action space
        self.action_space = spaces.Discrete(self.max_num_buttons)

        # define observation space
        self.output_shape = (self.height, self.width, 1) # choose dimensions of image
        self.output_full_shape = (self.height, self.width, 3) # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)

        # initialize reward
        self.total_reward = 0
        self.agent_stats = None
        self.trajectories = {}
        self.episode_num = 0
        self.traj_log_req = config['traj_log_freq']

        self.target = None
        if (target == -1):
            self.target = skeleton[0][-1] # node at the end of the longest chain
        else:
            self.target = target
        
        self.timesteps = 0

    def render(self):
        return self.states[self.state]

    def reset(self, seed=None):
        self.total_reward = 0
        self.timesteps = 0

        if ((self.episode_num + 1) % self.traj_log_req == 0):
            self.trajectories[self.episode_num] = self.agent_stats
        
        self.agent_stats = None
        self.episode_num += 1

        self.state = random.randint(0, self.num_screens-1)
        obs = self.render()

        return obs, {}
    
    def step(self, action):
        old_state = self.state
        self.state = self.transition[self.state, action]

        # store the new screen image (i.e. new observation) and reward    
        obs = self.render()

        # update reward
        new_reward = int(self.state == self.target) - 1
        self.total_reward += new_reward

        # store trajectory
        sarst = np.array([old_state, action, new_reward, self.state, self.total_reward]).reshape((1, 5))
        if self.agent_stats is None:
            self.agent_stats = sarst
        else:
            np.append(self.agent_stats, sarst, axis=0)

        # update number of time steps
        self.timesteps += 1
        
        # decide termination/truncation conditions
        truncated = (self.timesteps >= self.max_ep_len) # truncate environment when you exceed max time steps
        terminated = (not truncated) and (self.state == self.target) # terminate environment when target state is reached

        return obs, new_reward, terminated, truncated, {}

    def close(self):
        super().close() # call close function of parent's class
    
    def _save_env(self, path):
        # need to save adjacency matrix, transition matrix, screen images
        np.save(path + "adjacency_matrix.npy", self.adj_mat)
        np.save(path + "transition_matrix.npy", self.transition)
        np.save(path + "states.npy", self.states)
        np.save(path + "target.npy", np.array([self.target]))

        image_path = path + "image/"

        for i in range(self.num_screens):
            screen = plt.imsave(image_path + 'screen' + str(i) + '.png', self.states[i])
    
    def _save_trajs(self, path):
        np.save(path + 'trajectories.npy', self.trajectories)

    def _reset_trajs(self):
        self.episode_num = 0
        self.trajectories = {}
        self.agent_stats = None
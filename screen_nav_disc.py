"""TODO: 
    - run DQN
""" 
import sys

import math
import random
import numpy as np
import matplotlib.pyplot as plt

from gymnasium import Env, spaces

from helper.screen_helper import *
from helper.graph_helper import *

class ScreenNavDiscEnv(Env):
    def __init__(self, config):
        super(ScreenNavDiscEnv, self).__init__()
        
        # storing parameters from config dictionary
        self.seed = config['seed']
        random.seed(self.seed) # setting random seed
        
        self.device = config['device']
        self.discount = config['discount']
        self.max_ep_len = config['max_episode_length']
        self.sparsity_const = config['sparsity_constant'] # E / V

        self.width = config['screen_width']
        self.height = config['screen_height']

        self.num_screens = config['num_screens']
        self.num_chains = config['num_chains']
        self.max_chain_length = config['max_chain_length']
        self.num_edges = config['num_edges'] 

        if (self.sparsity_const > 0):
            self.num_edges = math.ceil(self.num_screens * self.sparsity_const)        
        # self.num_buttons = config['num_buttons'] # likely do not need this parameter

        # setting up graph structure of environment
        skeleton, length_chains = create_skeleton(self.num_chains, self.max_chain_length, self.num_screens)
        edges = multi_chaining(self.num_chains, length_chains, skeleton, self.num_screens, self.num_edges)

        self.adj_mat = generate_adjacency_matrix(edges, self.num_screens)

        self.num_buttons_all = find_num_buttons_per_state(self.adj_mat)
        self.max_num_buttons = find_max_num_buttons(self.num_buttons_all)
        self.transition = generate_transition_matrix(self.adj_mat, self.num_screens, self.num_buttons_all, self.max_num_buttons)
        
        # setting up physical images for the environment
        self.button_colors = [
            [220, 20, 60],
            [255, 99, 71],
            [255, 0, 0],
            [200, 70, 0],
            [255, 100, 0],
            [255, 150, 0],
            [255, 200, 0],
            [234, 229, 140],
            [255, 255, 0]
        ]
        self.screen_colors = [
            [50, 252, 0],
            [0, 128, 0],
            [50, 205, 50],
            [32, 178, 170],
            [0, 139, 139],
            [0, 206, 209],
            [138, 43, 224],
            [149, 0, 211],
            [228, 161, 228]
        ]
        random.shuffle(self.button_colors)
        random.shuffle(self.screen_colors)
        
        self.num_cols = math.ceil(math.sqrt(self.max_num_buttons))
        self.button_width = math.floor(4.0 * self.width / (5 * self.num_cols + 1))
        self.button_height = self.button_width
        self.gap_x = math.floor(self.button_width / 4)
        self.gap_y = self.gap_x
                
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
        self.state = random.randint(0, self.num_screens-1)

        # define action space
        self.action_space = spaces.Discrete(self.max_num_buttons)

        # define observation space
        self.output_shape = (self.height, self.width, 1) # choose dimensions of image
        self.output_full_shape = (self.height, self.width, 3) # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)

        # initialize reward
        self.total_reward = 0

        self.agent_stats = []

        self.target = skeleton[0][-1] # node at the end of the longest chain
        self.timesteps = 0

    def render(self):
        return self.states[self.state]

    def reset(self):
        self.total_reward = 0
        self.timesteps = 0
        self.state = random.randint(0, self.num_screens-1)
        self.agent_stats = []
    
    def step(self, action):
        # # store the new agent state obtained from the corresponding memory address
        # # memory addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        # X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
        # LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]    
        # x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        # y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        # levels = [self.pyboy.get_memory_value(a) for a in LEVELS_ADDRESSES]
        # self.agent_stats.append({
        #     'x': x_pos, 'y': y_pos, 'levels': levels
        # })

        old_state = self.state
        self.state = self.transition[self.state, action]

        # store the new screen image (i.e. new observation) and reward    
        obs = self.render()

        # update reward
        new_reward = int(self.state == self.target) - 1
        self.total_reward += new_reward

        self.agent_stats.append(
            (old_state, action, new_reward, self.state, self.total_reward)
        )

        # update number of timesteps
        self.timesteps+=1
        
        # for simplicity, don't handle terminate or truncated conditions here
        terminated = (self.state == self.target) # no max number of step
        truncated = (self.timesteps > self.max_ep_len) # no max number of step

        return obs, new_reward, terminated, truncated, {}

    def close(self):
        super().close() # call close function of parent's class
            
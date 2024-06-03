"""TODO: 
    - add number of epochs and max length of episode as arg params
    - set the seed of the environment in init
    - create dynamics model
    - run DQN
    - investigate termination and truncation in step function
""" 
import sys
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env, spaces
import math
import random
class ScreenNavDiscEnv(Env):
    def __init__(self, config):
        super(ScreenNavDiscEnv, self).__init__()
        self.agent_stats = []
        self.total_reward = 0

        self.width = config['screen-width']
        self.height = config['screen-height']
        self.seed = config['seed']
        self.device = config['device']
        self.discount = config['discount']
        self.num_screens = config['num-screens']
        self.num_buttons = config['num-buttons']
        self.action_space = spaces.Discrete(self.num_buttons)
        self.button_colors = random.shuffle([
            [220, 20, 60],
            [255, 99, 71],
            [255, 0, 0],
            [255, 69, 0],
            [255, 140, 0],
            [255, 180, 0],
            [255, 215, 0],
            [234, 229, 140],
            [255, 255, 0]
        ])
        self.screen_colors = random.shuffle([
            [124, 252, 0],
            [0, 128, 0],
            [50, 205, 50],
            [32, 178, 170],
            [0, 139, 139],
            [0, 206, 209],
            [138, 43, 224],
            [149, 0, 211],
            [228, 161, 228]
        ])
        self.num_cols = math.floor(math.sqrt(self.num_buttons)) 
        self.button_width = math.floor(4 * self.width / (5 * self.num_cols + 1))
        self.button_height = self.button_width
        self.gap_x = math.floor(self.button_width / 4)
        self.gap_y = self.gap_x
        
        self.states = self.set_states(
            self.height,
            self.width,
            self.button_height,
            self.button_width,
            self.gap_x,
            self.gap_y,
            self.button_colors,
            self.num_cols,
            self.screen_colors
        )
        self.state = 0

        # Define observation space
        self.output_shape = (self.height, self.width, 1) # choose dimensions of image
        self.output_full_shape = (self.height, self.width, 3) # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)
    
    def create_background(self, height, width, background_color):
        color_grid = np.ones((height, width, 3))
        background_color = np.array(background_color).reshape((1,1,3))
        color_grid = color_grid * background_color 
        return color_grid        

    def add_button(self, button_height, button_width, corner_x, corner_y, color, color_grid):
        color_grid[corner_y : corner_y + button_height, corner_x : corner_x + button_width] = np.array(color).reshape((1,1,3))
        return color_grid
    
    def get_grid(self, height, width, background_color, button_height, button_width, gap_x, gap_y, colors, num_cols):
        grid = self.create_background(height, width, background_color)
        num_buttons = len(colors)
        for button_id in range(num_buttons):
            col_idx = button_id % num_cols
            row_idx = button_id // num_cols
            corner_x = gap_x + col_idx * (gap_x + button_width)
            corner_y = gap_y + row_idx * (gap_y + button_height)
            grid = self.add_button(button_height, button_width, corner_x, corner_y, colors[button_id], grid)

        return grid
    
    def set_states(self, height, width, button_height, button_width, gap_x, gap_y, colors, num_cols, screen_colors):
        states = np.array([])
        num_screens = len(screen_colors)
        for screen_id in range(num_screens):
            grid = self.get_grid(height, width, screen_colors[screen_id], button_height, button_width, gap_x, gap_y, colors, num_cols)
            states.append(grid)
        return states
        

    def render(self):
        return self.states[self.state]

    def reset(self):
        self.total_reward = 0
        self.state = random.randint(0, self.num_screens-1)
    
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

        self.state = self.transition[self.state, action]


        # store the new screen image (i.e. new observation) and reward    
        obs_memory = self.render()
        new_reward = int(self.state == self.target) - 1
        self.total_reward += new_reward
        
        # for simplicity, don't handle terminate or truncated conditions here
        terminated = False # no max number of step
        truncated = False # no max number of step

        return obs_memory, new_reward, terminated, truncated, {}

    def close(self):
        super().close() # call close function of parent's class
            
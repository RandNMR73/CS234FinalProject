import sys
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Env, spaces

class ScreenNavContEnv(Env):
    def __init__(self, config):
        super(ScreenNavContEnv, self).__init__()
        self.agent_stats = []
        self.total_reward = 0
        # Define action space
        self.valid_actions = [

        ]
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Define observation space
        self.output_shape = (144, 160, 1) # choose dimensions of image
        self.output_full_shape = (144, 160, 3) # 3: RGB
        self.observation_space = spaces.Box(low=0, high=255, shape=self.output_full_shape, dtype=np.uint8)

        # Define action frequency
        #self.act_freq = config['action_freq']

        
    def render(self):
        # game_pixels_render = self.screen.screen_ndarray() # (144, 160, 3)
        # return game_pixels_render
        pass

    def reset(self):
        # restart game, skipping credits
        # with open(self.init_state, "rb") as f:
        #     self.pyboy.load_state(f)  
        
        # reset reward value
        self.total_reward = 0
        return self.render(), {}
    
    def step(self, action):
        # store the new agent state obtained from the corresponding memory address
        # memory addresses from https://datacrystal.romhacking.net/wiki/Pok%C3%A9mon_Red/Blue:RAM_map
        X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
        LEVELS_ADDRESSES = [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]    
        x_pos = self.pyboy.get_memory_value(X_POS_ADDRESS)
        y_pos = self.pyboy.get_memory_value(Y_POS_ADDRESS)
        levels = [self.pyboy.get_memory_value(a) for a in LEVELS_ADDRESSES]
        self.agent_stats.append({
            'x': x_pos, 'y': y_pos, 'levels': levels
        })

        # store the new screen image (i.e. new observation) and reward    
        obs_memory = self.render()
        new_reward = levels
        
        # for simplicity, don't handle terminate or truncated conditions here
        terminated = False # no max number of step
        truncated = False # no max number of step

        return obs_memory, new_reward, terminated, truncated, {}

    def close(self):
        super().close() # call close function of parent's class
            
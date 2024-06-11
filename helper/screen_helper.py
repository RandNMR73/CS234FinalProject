import numpy as np
import random

from bisect import bisect_left

from helper.graph_helper import *
from helper.debug import *

# functions for creating images for each state
def create_background(height, width, background_color):
    color_grid = np.ones((height, width, 3))
    background_color = np.array(background_color).reshape((1,1,3))
    color_grid = color_grid * background_color 
    return color_grid        

def add_button(button_height, button_width, corner_x, corner_y, color, color_grid):
    color_grid[corner_y : corner_y + button_height, corner_x : corner_x + button_width] = np.array(color).reshape((1,1,3))
    return color_grid

def get_grid(height, width, background_color, button_height, button_width, gap_x, gap_y, colors, num_cols, num_buttons):
    grid = create_background(height, width, background_color)
    np.random.shuffle(colors)
    
    for button_id in range(num_buttons):
        col_idx = button_id % num_cols
        row_idx = button_id // num_cols
        corner_x = gap_x + col_idx * (gap_x + button_width)
        corner_y = gap_y + row_idx * (gap_y + button_height)
        
        grid = add_button(button_height, button_width, corner_x, corner_y, colors[button_id], grid)

    return grid

def set_states(height, width, button_height, button_width, gap_x, gap_y, colors, num_cols, screen_colors, num_buttons_all, num_screens):
    states = np.empty((num_screens, height, width, 3))
    
    for screen_id in range(num_screens):
        grid = get_grid(height, width, screen_colors[screen_id], button_height, button_width, gap_x, gap_y, colors, num_cols, num_buttons_all[screen_id])
        states[screen_id] = grid
    return states

# function for creating transition matrix for the environment
"""
Parameters:
    num_chains
    num_nodes
    max_chain_length
    num_edges
"""

def find_num_buttons_per_state(adj_matrix):
    return np.sum(adj_matrix, axis=1)

def find_max_num_buttons(num_buttons_all):
    return np.max(num_buttons_all)

def generate_transition_matrix(adj_matrix, num_screens, num_buttons_all, max_num_buttons):
    trans_mat = np.full((num_screens, max_num_buttons), -1, dtype=int)

    for screen_id in range(num_screens):
        num_buttons = num_buttons_all[screen_id]

        button_to_screen = np.zeros((num_buttons), dtype=int)
        
        shuffle_buttons = list(range(num_buttons))
        random.shuffle(shuffle_buttons)

        button_ind = 0
        for screen_id2 in range(num_screens):
            for i in range(adj_matrix[screen_id][screen_id2]):
                button_to_screen[button_ind] = screen_id2
                button_ind += 1

        for button_id in range(num_buttons):
            trans_mat[screen_id][button_id] = button_to_screen[shuffle_buttons[button_id]]
    
    return trans_mat

# function for getting button index from selected pixel coordinates
def get_button(act_x, act_y, width, height, gap_x, gap_y, button_width, button_height, num_cols, num_buttons):
    # defining button pixel ranges in each dimension
    x_coords = [0]
    y_coords = [0]

    for col_idx in range(num_cols):
        b_start_x = gap_x + col_idx * (gap_x + button_width)
        b_start_y = gap_y + col_idx * (gap_y + button_height)

        b_end_x = b_start_x + button_width
        b_end_y = b_start_y + button_height

        x_coords.append(b_start_x)
        x_coords.append(b_end_x)

        y_coords.append(b_start_y)
        y_coords.append(b_end_y)

    x_coords.append(width)
    y_coords.append(height)

    act_x = int(act_x)
    act_y = int(act_y)

    # check that chosen action meets dimension restrictions
    assert(act_x >= 0 and act_x < width)
    assert(act_y >= 0 and act_y < height)

    # check if given coordinates match up with button ranges
    x_ind = bisect_left(x_coords, act_x)
    y_ind = bisect_left(y_coords, act_y)

    if (x_ind % 2 == 0) or (y_ind % 2 == 0):
        return -1
    else:
        x_id = (x_ind - 1) // 2
        y_id = (y_ind - 1) // 2

        button_id = x_id + y_id * num_cols

        if (button_id >= 0 and button_id < num_buttons):
            return button_id
        else:
            return -1 # no valid button selected

""" def main():
    num_chains = 1
    num_screens = 4
    max_chain_length = 3
    num_edges = 4

    skeleton, length_chains = create_skeleton(num_chains, max_chain_length, num_screens)
    edges = multi_chaining(num_chains, length_chains, skeleton, num_screens, num_edges)
    
    adj_mat = generate_adjacency_matrix(edges, num_screens)
    
    debugc("\nadjacency matrix:", 3)
    for row in adj_mat:
        debugc(row, 3)
    
    num_buttons_all = find_num_buttons_per_state(adj_mat)
    max_num_buttons = find_max_num_buttons(num_buttons_all)
    trans_mat1 = generate_transition_matrix(adj_mat, num_screens, num_buttons_all, max_num_buttons)
    trans_mat2 = generate_transition_matrix(adj_mat, num_screens, num_buttons_all, max_num_buttons)
    trans_mat3 = generate_transition_matrix(adj_mat, num_screens, num_buttons_all, max_num_buttons)

    debugc("\ntransition matrix 1:", 3)
    for row in trans_mat1:
        debugc(row, 3)
    
    debugc("\ntransition matrix 2:", 3)
    for row in trans_mat2:
        debugc(row, 3)
    
    debugc("\ntransition matrix 3:", 3)
    for row in trans_mat3:
        debugc(row, 3)

if __name__ == '__main__':
    main() """
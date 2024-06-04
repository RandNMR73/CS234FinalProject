import numpy as np

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
    
    for button_id in range(num_buttons):
        col_idx = button_id % num_cols
        row_idx = button_id // num_cols
        corner_x = gap_x + col_idx * (gap_x + button_width)
        corner_y = gap_y + row_idx * (gap_y + button_height)
        grid = add_button(button_height, button_width, corner_x, corner_y, colors[button_id], grid)

    return grid

def set_states(height, width, button_height, button_width, gap_x, gap_y, colors, num_cols, screen_colors, num_buttons, num_screens):
    states = np.empty((num_screens, height, width, 3))
    
    for screen_id in range(num_screens):
        grid = get_grid(height, width, screen_colors[screen_id], button_height, button_width, gap_x, gap_y, colors, num_cols, num_buttons)
        states[screen_id] = grid
    return states
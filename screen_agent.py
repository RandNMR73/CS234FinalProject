from screen_nav_disc import ScreenNavDiscEnv
from screen_nav_cont import ScreenNavContEnv
import argparse

import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('RL Screen Agent', add_help=False)

    # general arguments
    parser.add_argument('--env-type', choices=['discrete', 'continuous'], default='discrete', type=str)
    parser.add_argument('--algorithm', choices=['DQN', 'VPG'], default='DQN', type=str)

    parser.add_argument('--screen-width', default=256, type=int)
    parser.add_argument('--screen-height', default=512, type=int)

    parser.add_argument('--num-screens', default=4, type=int)
    parser.add_argument('--num-chains', default=2, type=int)
    parser.add_argument('--max-chain-length', default=2, type=int)
    parser.add_argument('--num-edges', default=3, type=int)
    # parser.add_argument('--num-buttons', default=3, type=int) # likely do not need this parameter
    
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str)

    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--lr-value', default=0.001, type=float)
    parser.add_argument('--discount', default=1.0, type=float)

    # VPG-specific arguments
    parser.add_argument('--lr-policy', default=0.001, type=float)

    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    config = vars(args)
    if (args.env_type == "discrete"):
        env = ScreenNavDiscEnv(config)
        
        for i in range(env.num_screens):
            screen = plt.imsave('output/screen' + str(i) + '.png', env.states[i])
    
    if (args.env_type == "continuous"):
        env = ScreenNavContEnv(config)

if __name__ == '__main__':
    main()

import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from datetime import datetime

from environments.screen_nav_disc import ScreenNavDiscEnv
from environments.screen_nav_cont import ScreenNavContEnv

from algorithms.dqn import train_dqn, test_dqn
from algorithms.ppo import train_ppo, test_ppo
from algorithms.ddpg import train_ddpg, test_ddpg
from algorithms.sac import train_sac, test_sac

def get_args():
    parser = argparse.ArgumentParser('RL Screen Agent', add_help=False)

    # testing arguments
    parser.add_argument('--mode', choices=['test', 'train', 'eval'], default='test', type=str)

    # environment arguments
    parser.add_argument('--env-type', choices=['discrete', 'continuous'], default='discrete', type=str)
    parser.add_argument('--algorithm', choices=['DQN', 'PPO', 'DDPG', 'SAC'], default='DQN', type=str)

    parser.add_argument('--screen-width', default=6, type=int)
    parser.add_argument('--screen-height', default=12, type=int)

    parser.add_argument('--num-screens', default=4, type=int)
    parser.add_argument('--num-chains', default=2, type=int)
    parser.add_argument('--max-chain-length', default=2, type=int)
    parser.add_argument('--num-edges', default=3, type=int)
    parser.add_argument('--sparsity-constant', default=0.0, type=float)

    parser.add_argument('--num-tiers', default=3, type=int)
    parser.add_argument('--num-branches', default=[1, 2, 2], type=int, nargs='+')
    # parser.add_argument('--num-buttons', default=3, type=int) # likely do not need this parameter

    parser.add_argument('--max-episode-length', default=20, type=int) # tune this later
    parser.add_argument('--env-seed', default=1, type=int)
    parser.add_argument('--save-freq', default=1000, type=int)
    
    # policy training arguments (DQN)
    parser.add_argument('--policy', choices=["MlpPolicy", "CnnPolicy", "MultiInputPolicy"], default="MlpPolicy", type=str)
    parser.add_argument('--lr-rate', default=1e-4, type=float)
    parser.add_argument('--buffer-size', default=10000, type=int)
    parser.add_argument('--learning-starts', default=100, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--tau', default=1.0, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--train-freq', default=4, type=int)
    parser.add_argument('--gradient-steps', default=1, type=int)
    parser.add_argument('--optimize-memory-usage', default=False, type=bool)
    parser.add_argument('--target-update-interval', default=1e4, type=int)
    parser.add_argument('--exploration-fraction', default=0.1, type=float)
    parser.add_argument('--exploration-initial-eps', default=1.0, type=float)
    parser.add_argument('--exploration-final-eps', default=0.05, type=float)
    parser.add_argument('--max-grad-norm', default=10.0, type=float)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--agent-seed', default=1, type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', type=str)
    
    parser.add_argument('--total-timesteps', default=1e4, type=int)
    parser.add_argument('--log-interval', default=4, type=int)

    # policy validation (prediction) arguments (DQN)
    parser.add_argument('--model-name', default="", type=str)
    parser.add_argument('--model-dir', default="", type=str)
    parser.add_argument('--traj-log-freq', default=100, type=int)

    # additional policy training arguments (PPO)
    parser.add_argument('--n-epochs', default=10, type=int)
    parser.add_argument('--n-steps', default=256, type=int)

    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    config = vars(args)

    dt = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_path = "output/{}/{}/{}/{}/".format(args.env_type, args.algorithm, args.mode, dt)
    output_checkpoint_path = output_path + "checkpoints/"
    output_env_path = output_path + "env/"
    output_image_path = output_env_path + "image/"

    os.mkdir(output_path)
    os.mkdir(output_env_path)
    os.mkdir(output_image_path)

    with open(output_path + "config.json", "w") as file:
        json.dump(config, file)
        
    if args.env_type == 'discrete':
        if args.mode == 'test':
            env = ScreenNavDiscEnv(config)
            env._save_env(output_env_path)

        elif args.mode == 'train':
            # setting up environment
            env = ScreenNavDiscEnv(config)
            env._save_env(output_env_path)

            # using custom logger
            new_logger = configure(output_path, ["log", "tensorboard", "json"])

            if (args.algorithm == "DQN"):
                train_dqn(env, args, output_path, new_logger, output_checkpoint_path)

        elif args.mode == 'eval':
            if (args.algorithm == "DQN"):
                test_dqn(args, output_path, output_env_path)
    
    elif args.env_type == 'continuous':
        if args.mode == 'test':
            env = ScreenNavContEnv(config)
            env._save_env(output_env_path)
        
        elif args.mode == 'train':
            # setting up environment
            env = ScreenNavContEnv(config)
            env._save_env(output_env_path)

            # using custom logger
            new_logger = configure(output_path, ["log", "tensorboard", "json"])

            if (args.algorithm == "PPO"):
                train_ppo(env, args, output_path, new_logger, output_checkpoint_path)
            if (args.algorithm == "DDPG"):
                train_ddpg(env, args, output_path, new_logger, output_checkpoint_path)
            if (args.algorithm == "SAC"):
                train_sac(env, args, output_path, new_logger, output_checkpoint_path)

        elif args.mode == 'eval':
            if (args.algorithm == "PPO"):
                test_ppo(args, output_path, output_env_path)
            if (args.algorithm == "DDPG"):
                test_ddpg(args, output_path, output_env_path)
            if (args.algorithm == "SAC"):
                test_sac(args, output_path, output_env_path)


if __name__ == '__main__':
    main()

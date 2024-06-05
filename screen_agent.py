import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from datetime import datetime

from screen_nav_disc import ScreenNavDiscEnv
from screen_nav_cont import ScreenNavContEnv

def get_args():
    parser = argparse.ArgumentParser('RL Screen Agent', add_help=False)

    # testing arguments
    parser.add_argument('--mode', choices=['test', 'train', 'predict'], default='test', type=str)

    # environment arguments
    parser.add_argument('--env-type', choices=['discrete', 'continuous'], default='discrete', type=str)
    parser.add_argument('--algorithm', choices=['DQN', 'VPG'], default='DQN', type=str)

    parser.add_argument('--screen-width', default=256, type=int)
    parser.add_argument('--screen-height', default=512, type=int)

    parser.add_argument('--num-screens', default=4, type=int)
    parser.add_argument('--num-chains', default=2, type=int)
    parser.add_argument('--max-chain-length', default=2, type=int)
    parser.add_argument('--num-edges', default=3, type=int)
    parser.add_argument('--sparsity-constant', default=0.0, type=float)
    # parser.add_argument('--num-buttons', default=3, type=int) # likely do not need this parameter

    parser.add_argument('--max-episode-length', default=20, type=int) # tune this later
    parser.add_argument('--env-seed', default=1, type=int)
    
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
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--agent-seed', default=1, type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto', type=str)
    
    parser.add_argument('--total-timesteps', default=1e4, type=int)
    parser.add_argument('--log-interval', default=4, type=int)

    # policy validation (prediction) arguments (DQN)
    parser.add_argument('--model-name', default="", type=str)
    parser.add_argument('--model-path', default="", type=str)

    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    args = parser.parse_args()
    return args
    
def main():
    args = get_args()
    config = vars(args)

    dt = datetime.now().strftime("%d-%m-%Y-%H-%M")
    output_path = "output/{}/{}/".format(args.mode, dt)
    output_env_path = output_path + "env/"
    output_image_path = output_env_path + "image/"

    os.mkdir(output_path)
    os.mkdir(output_env_path)
    os.mkdir(output_image_path)

    with open(output_path + "config.json", "w") as file:
        json.dump(config, file)
    
    # using custom logger
    new_logger = configure(output_path, ["log", "tensorboard", "json"])

    if args.env_type == 'discrete':
        env = ScreenNavDiscEnv(config)
        env._save_env(output_env_path)

        if args.mode == 'test':
            pass

        elif args.mode == 'train':
            if (args.algorithm == "DQN"):
                # change parameters using args from argument parser
                model = DQN(
                    policy=args.policy,
                    env=env,
                    learning_rate=args.lr_rate,
                    buffer_size=args.buffer_size,
                    learning_starts=args.learning_starts,
                    batch_size=args.batch_size,
                    tau=args.tau,
                    gamma=args.gamma,
                    train_freq=args.train_freq,
                    gradient_steps=args.gradient_steps,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=args.optimize_memory_usage,
                    target_update_interval=args.target_update_interval,
                    exploration_fraction=args.exploration_fraction,
                    exploration_initial_eps=args.exploration_initial_eps,
                    exploration_final_eps=args.exploration_final_eps,
                    max_grad_norm=args.max_grad_norm,
                    stats_window_size=100,
                    tensorboard_log=None,
                    policy_kwargs=None,
                    verbose=args.verbose,
                    seed=args.agent_seed,
                    device=args.device,
                    _init_setup_model=True,
                )
                model.set_logger(new_logger)

                model.learn(
                    total_timesteps=args.total_timesteps,
                    log_interval=args.log_interval,
                    progress_bar=True
                )

                model.save(output_path + args.model_name)

        elif args.mode == 'predict':
            if (args.env_type == "DQN"):
                model = DQN.load(args.model_path)

                obs, info = env.reset()
                while True:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        obs, info = env.reset()
    
    elif args.env_type == 'continuous':
        pass

if __name__ == '__main__':
    main()

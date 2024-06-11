import os

import matplotlib.pyplot as plt
import matplotlib as mpl

import json
import numpy as np

from stable_baselines3 import PPO

from environments.screen_nav_cont import ScreenNavContEnv

def main():
    args, output_path, output_env_path = None, None, None

    train_path = "output/continuous/PPO/train/11-06-2024-09-54-12/"
    train_checkpoint_path = train_path + "checkpoints/"
    train_env_path = train_path + "env/"
    train_image_path = train_env_path + "image/"

    train_config_path = train_path + "config.json"

    eval_path = "output/continuous/PPO/eval/11-06-2024-09-54-12/"
    eval_trajectories_path = eval_path + "trajectories/"
    eval_env_path = eval_path + "env/"
    eval_image_path = eval_env_path + "image/"

    # os.mkdir(eval_path)
    # os.mkdir(eval_trajectories_path)
    # os.mkdir(eval_env_path)
    # os.mkdir(eval_image_path)

    # initialize new config dictionary from previous run
    train_config = None
    with open(train_config_path) as json_file:
        train_config = json.load(json_file)
    
    # initialize remaining environment parameters from previous run
    adj_mat = np.load(train_env_path + 'adjacency_matrix.npy')
    transition = np.load(train_env_path + 'transition_matrix.npy')
    states = np.load(train_env_path + 'states.npy')
    target = np.load(train_env_path + 'target.npy')[0]

    env = ScreenNavContEnv(
        config=train_config,
        adj_mat=adj_mat,
        transition=transition,
        states=states,
        target=target
    )
    env._save_env(eval_env_path)

    with open(eval_path + "train_config.json", "w") as file:
        json.dump(train_config, file)

    assert(1 == 0)


    for checkpoint in range(train_config["save_freq"], train_config["total_timesteps"]+1, train_config["save_freq"]):
        model_name = "rl_model_" + str(checkpoint) + "_steps.zip"
        model = PPO.load(train_checkpoint_path + model_name)
        trajectory = []

        obs, info = env.reset()
        for i in range(args.max_episode_length):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(info)
            if terminated or truncated:
                obs, info = env.reset()
                break
    
    """ actions = [
        [1, 1],
        [1, 1],
        [1, 7]
    ]

    obs, info = env.reset()
    for i in range(len(actions)):
        # action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(actions[i])
        print(info)
        if terminated or truncated:
            obs, info = env.reset()
            break """

if __name__ == '__main__':
    main()
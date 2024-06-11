import os

import json
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from environments.screen_nav_cont import ScreenNavContEnv

# function to train DQN algorithm given parameters
def train_ppo(env, args, output_path, new_logger, output_checkpoint_path):
    # change parameters using args from argument parser
    # there are still more parameters which can be changed
    model = PPO(
        policy=args.policy,
        env=env,
        learning_rate=args.lr_rate, # 3e-4
        n_steps=args.ppo_n_steps, # 2048
        batch_size=args.batch_size, # 64
        n_epochs=args.ppo_n_epochs, # 10
        gamma=args.gamma, # 0.99
        gae_lambda=args.ppo_gae_lambda, # 0.95
        clip_range=args.ppo_clip_range, # 0.2
        clip_range_vf=None,
        normalize_advantage=args.ppo_normalize_advantage, # True
        ent_coef=args.ppo_ent_coef, # 0.0
        vf_coef=args.ppo_vf_coef, # 0.5
        max_grad_norm=args.max_grad_norm, # 0.5
        use_sde=False, # False
        sde_sample_freq=-1, # -1
        rollout_buffer_class=None,
        rollout_buffer_kwargs=None,
        target_kl=args.ppo_target_kl, # 0.01
        stats_window_size=100,
        tensorboard_log=output_path,
        policy_kwargs=None,
        verbose=args.verbose, # 0
        seed=args.agent_seed,
        device=args.device,
        _init_setup_model=True,
    )

    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=output_checkpoint_path)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
        log_interval=args.log_interval,
        progress_bar=True
    )

    model.save(output_path + args.model_name)
    env._save_trajs(output_path)
    env._reset_trajs()

# function to test DQN algorithm
def test_ppo(args):
    # store various paths to respective training run
    train_path = "output/{}/{}/{}/{}/".format(args.env_type, args.algorithm, "train", args.datetime)
    train_checkpoint_path = train_path + "checkpoints/"
    train_env_path = train_path + "env/"
    train_image_path = train_env_path + "image/"

    train_config_path = train_path + "config.json"

    eval_path = "output/{}/{}/{}/{}/".format(args.env_type, args.algorithm, "eval", args.datetime)
    eval_trajectories_path = eval_path + "trajectories/"
    eval_env_path = eval_path + "env/"
    eval_image_path = eval_env_path + "image/"

    # create directories to store eval output
    os.mkdir(eval_path)
    os.mkdir(eval_trajectories_path)
    os.mkdir(eval_env_path)
    os.mkdir(eval_image_path)

    # load config of the training run being evaluated and eval run
    train_config = None
    with open(train_config_path) as json_file:
        train_config = json.load(json_file)

    eval_config = vars(args)

    # initialize environment parameters from previous training run
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
    
    # store config of training run and eval run in eval folder
    with open(eval_path + "train_config.json", "w") as file:
        json.dump(train_config, file)
    
    with open(eval_path + "eval_config.json", "w") as file:
        json.dump(eval_config, file)
    
    # recording and storing trajectories from checkpoints/model
    do_checkpoints = eval_config["do_checkpoints"]
    num_trajs = eval_config["num_trajectories"]
    traj_len = None

    is_discrete = train_config["env_type"] == 'discrete'

    if (is_discrete):
        traj_len = 5
    else:
        traj_len = 7

    if (do_checkpoints):
        save_freq = train_config["save_freq"]
        total_timesteps = train_config["total_timesteps"]

        for checkpoint in range(save_freq, total_timesteps+1, save_freq):
            model_name = "rl_model_" + str(checkpoint) + "_steps.zip"
            model = PPO.load(train_checkpoint_path + model_name)

            traj_dir = eval_trajectories_path + "checkpoint_" + str(checkpoint) + "/"
            os.mkdir(traj_dir)
                        
            for traj in range(num_trajs):
                trajectory = None
                start = True

                obs, info = env.reset()
                for i in range(args.max_episode_length):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    sarst = None

                    if is_discrete:
                        sarst = [
                            info["s"],
                            info["a"],
                            info["r"],
                            info["s'"],
                            info["t_r"]
                        ]
                    else:
                        sarst = [
                            info["s"],
                            info["ax"],
                            info["ay"],
                            info["button"],
                            info["r"],
                            info["s'"],
                            info["t_r"]
                        ]
                    
                    sarst = np.array(sarst)
                    sarst.astype(np.int8)
                    sarst = np.reshape(sarst, (1, traj_len))

                    if start:
                        trajectory = sarst
                        start = False
                    else:
                        trajectory = np.append(trajectory, sarst, axis=0)
                    
                    if terminated or truncated:
                        obs, info = env.reset()
                        break
                
                np.save(traj_dir + "trajectory_" + str(traj) + ".npy", trajectory)

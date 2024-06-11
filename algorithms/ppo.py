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
        policy=args.ppo_policy,
        env=env,
        learning_rate=args.ppo_lr_rate,
        n_steps=args.ppo_n_steps,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_n_epochs,
        gamma=args.ppo_gamma,
        max_grad_norm=args.ppo_max_grad_norm,
        stats_window_size=100,
        tensorboard_log=output_path,
        policy_kwargs=None,
        verbose=args.ppo_verbose,
        seed=args.ppo_agent_seed,
        device=args.ppo_device,
        _init_setup_model=True,
    )

    model.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=args.save_freq, save_path=output_checkpoint_path)

    model.learn(
        total_timesteps=args.ppo_total_timesteps,
        callback=checkpoint_callback,
        log_interval=args.ppo_log_interval,
        progress_bar=True
    )

    model.save(output_path + args.model_name)
    env._save_trajs(output_path)
    env._reset_trajs()

# function to test DQN algorithm
def test_ppo(args, output_path, output_env_path):
    # initialize new config dictionary from previous run
    test_config = None
    with open(args.model_dir + 'config.json') as json_file:
        test_config = json.load(json_file)
    
    # initialize remaining environment parameters from previous run
    env_path = args.model_dir + 'env/'
    adj_mat = np.load(env_path + 'adjacency_matrix.npy')
    transition = np.load(env_path + 'transition_matrix.npy')
    states = np.load(env_path + 'states.npy')
    target = np.load(env_path + 'target.npy')[0]

    with open(output_path + "test_config.json", "w") as file:
        json.dump(test_config, file)

    env = ScreenNavContEnv(
        config=test_config,
        adj_mat=adj_mat,
        transition=transition,
        states=states,
        target=target
    )
    env._save_env(output_env_path)
    
    model = PPO.load(args.model_dir + args.model_name)

    obs, info = env.reset()
    for i in range(args.total_timesteps):
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
                break
    
    env._save_trajs(output_path)
    env._reset_trajs()

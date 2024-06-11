import json
import numpy as np

from stable_baselines3 import DQN

from environments.screen_nav_disc import ScreenNavDiscEnv

# function to train DQN algorithm given parameters
def train_dqn(env, args, output_path, new_logger):
    # change parameters using args from argument parser
    model = DQN(
        policy=args.dqn_policy,
        env=env,
        learning_rate=args.dqn_lr_rate,
        buffer_size=args.dqn_buffer_size,
        learning_starts=args.dqn_learning_starts,
        batch_size=args.dqn_batch_size,
        tau=args.dqn_tau,
        gamma=args.dqn_gamma,
        train_freq=args.dqn_train_freq,
        gradient_steps=args.dqn_gradient_steps,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=args.dqn_optimize_memory_usage,
        target_update_interval=args.dqn_target_update_interval,
        exploration_fraction=args.dqn_exploration_fraction,
        exploration_initial_eps=args.dqn_exploration_initial_eps,
        exploration_final_eps=args.dqn_exploration_final_eps,
        max_grad_norm=args.dqn_max_grad_norm,
        stats_window_size=100,
        tensorboard_log=output_path,
        policy_kwargs=None,
        verbose=args.dqn_verbose,
        seed=args.dqn_agent_seed,
        device=args.dqn_device,
        _init_setup_model=True
    )

    model.set_logger(new_logger)

    model.learn(
        total_timesteps=args.ddpg_total_timesteps,
        log_interval=args.ddpg_log_interval,
        progress_bar=True
    )

    model.save(output_path + args.model_name)
    env._save_trajs(output_path)
    env._reset_trajs()

# function to test DQN algorithm
def test_dqn(args, output_path, output_env_path):
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

    env = ScreenNavDiscEnv(
        config=test_config,
        adj_mat=adj_mat,
        transition=transition,
        states=states,
        target=target
    )
    env._save_env(output_env_path)
    
    model = DQN.load(args.model_dir + args.model_name)

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

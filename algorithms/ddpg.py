import json
import numpy as np

from stable_baselines3 import DDPG

from environments.screen_nav_cont import ScreenNavContEnv

# function to train DQN algorithm given parameters
def train_ddpg(env, args, output_path, new_logger, output_checkpoint_path):
    # change parameters using args from argument parser
    model = DDPG(
        policy=args.policy,
        env=env,
        learning_rate=args.lr_rate,
        buffer_size=args.ddpg_buffer_size,
        learning_starts=args.ddpg_learning_starts,
        batch_size=args.ddpg_batch_size,
        tau=args.ddpg_tau,
        gamma=args.ddpg_gamma,
        train_freq=args.ddpg_train_freq,
        gradient_steps=args.ddpg_gradient_steps,
        action_noise=None,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=args.ddpg_optimize_memory_usage,
        tensorboard_log=output_path,
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
    env._save_trajs(output_path)
    env._reset_trajs()

# function to test DQN algorithm
def test_ddpg(args, output_path, output_env_path):
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
    
    model = DDPG.load(args.model_dir + args.model_name)

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
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

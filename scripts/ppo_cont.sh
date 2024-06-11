env_type='continuous'
algorithm='PPO'
mode='train'
model_name='ppo-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

total_timesteps='1000000'
addtl_params=("--total-timesteps" "${total_timesteps}")

python screen_agent.py ${base_params[@]} ${addtl_params[@]}
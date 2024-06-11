env_type='discrete'
algorithm='PPO'
mode='test'
model_name='ppo-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

python screen_agent.py ${base_params[@]}
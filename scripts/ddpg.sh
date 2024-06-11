env_type='continuous'
algorithm='DDPG'
mode='test'
model_name='ddpg-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

python screen_agent.py ${base_params[@]}
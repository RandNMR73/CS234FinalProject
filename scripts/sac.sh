env_type='continuous'
algorithm='SAC'
mode='test'
model_name='sac-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

python screen_agent.py ${base_params[@]}
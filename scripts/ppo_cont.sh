env_type='continuous'
algorithm='PPO'
mode='train'
model_name='ppo-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

total_timesteps='10000'
save_freq='1000'
train_params=("--total-timesteps" "${total_timesteps}" "--save-freq" "${save_freq}")

width='16'
height='32'
num_tiers='3'
num_branches=(1 2 2)
env_params=("--num-tiers" "${num_tiers}" "--num-branches" ${num_branches[@]} "--screen-width" "${width}" "--screen-height" "${height}")

python screen_agent.py ${base_params[@]} ${train_params[@]} ${env_params[@]}
env_type='discrete'
algorithm='PPO'
mode='test'
model_name='ppo-test'
base_params=("--env-type" "${env_type}" "--algorithm" "${algorithm}" "--mode" "${mode}" "--model-name" "${model_name}")

total_timesteps='10000'
norm_adv=True
target_kl=0.01
save_freq='1000'
train_params=("--total-timesteps" "${total_timesteps}" "--save-freq" "${save_freq}" "--ppo-normalize-advantage" "${norm_adv}" "--ppo-target-kl" "${target_kl}")

width='16'
height='32'
num_tiers='3'
num_branches=(1 2 2)
env_params=("--num-tiers" "${num_tiers}" "--num-branches" ${num_branches[@]} "--screen-width" "${width}" "--screen-height" "${height}")

python screen_agent.py ${base_params[@]} ${train_params[@]} ${env_params[@]}
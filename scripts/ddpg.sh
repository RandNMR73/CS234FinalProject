# general arguments
env_type="continuous"
algorithm="DDPG"
mode="train"

# environment arguments
width="16"
height="32"
num_tiers="3"
num_branches=(1 2 2)
max_ep_len="20"
env_seed="1"

# general policy arguments
policy="MlpPolicy"
lr_rate="1e-4"
batch_size="32"
gamma="1.0"
max_grad_norm="10.0"
verbose="1"
agent_seed="1"
device="cpu"

total_timesteps='25000'
log_interval="4"
model_name="ddpg-test"
save_freq='1000'

# ddpg policy arguments
ddpg_buffer_size="1000"
ddpg_learn_starts="100"
ddpg_tau="1.0"
ddpg_train_freq="4"
ddpg_grad_steps="1"
ddpg_opt_mem_us="False"

python screen_agent.py \
    --env-type $env_type \
    --algorithm $algorithm \
    --mode $mode \
    --screen-width $width \
    --screen-height $height \
    --num-tiers $num_tiers \
    --num-branches ${num_branches[@]} \
    --max-episode-length $max_ep_len \
    --env-seed $env_seed \
    --policy $policy \
    --lr-rate $lr_rate \
    --batch-size $batch_size \
    --gamma $gamma \
    --max-grad-norm $max_grad_norm \
    --verbose $verbose \
    --agent-seed $agent_seed \
    --device $device \
    --total-timesteps $total_timesteps \
    --log-interval $log_interval \
    --model-name $model_name \
    --save-freq $save_freq \
    --ddpg-buffer-size $ddpg_buffer_size \
    --ddpg-learning-starts $ddpg_learn_starts \
    --ddpg-tau $ddpg_tau \
    --ddpg-train-freq $ddpg_train_freq \
    --ddpg-gradient-steps $ddpg_grad_steps \
    --ddpg-optimize-memory-usage $ddpg_opt_mem_us
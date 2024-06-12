# general arguments
env_type="discrete"
algorithm="DQN"
mode="test"

# environment arguments
width="64"
height="128"
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

total_timesteps='50000'
log_interval="4"
model_name="dqn-branch-run-5"
save_freq='5000'

# dqn policy arguments
dqn_buffer_size="1000"
dqn_learn_starts="100"
dqn_tau="1.0"
dqn_train_freq="4"
dqn_grad_steps="1"
dqn_opt_mem_us="False"
dqn_target_updt_intrvl="10000"
dqn_exp_frac="0.1"
dqn_exp_init_eps="1.0"
dqn_exp_final_eps="0.05"

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
    --dqn-buffer-size $dqn_buffer_size \
    --dqn-learning-starts $dqn_learn_starts \
    --dqn-tau $dqn_tau \
    --dqn-train-freq $dqn_train_freq \
    --dqn-gradient-steps $dqn_grad_steps \
    --dqn-optimize-memory-usage $dqn_opt_mem_us \
    --dqn-target-update-interval $dqn_target_updt_intrvl \
    --dqn-exploration-fraction $dqn_exp_frac \
    --dqn-exploration-initial-eps $dqn_exp_init_eps \
    --dqn-exploration-final-eps $dqn_exp_final_eps
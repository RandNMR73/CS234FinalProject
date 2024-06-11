# general arguments
env_type="discrete"
algorithm="PPO"
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

total_timesteps='100000'
log_interval="4"
model_name="ppo-disc-test"
save_freq='1000'

# ppo policy arguments
ppo_n_steps="256"
ppo_n_epochs="10"
ppo_gae_lambda="0.95"
ppo_clip_range="0.2"
ppo_norm_adv="True"
ppo_ent_coef="0.0"
ppo_vf_coef="0.5"
ppo_target_kl="0.01"

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
    --ppo-n-steps $ppo_n_steps \
    --ppo-n-epochs $ppo_n_epochs \
    --ppo-gae-lambda $ppo_gae_lambda \
    --ppo-clip-range $ppo_clip_range \
    --ppo-normalize-advantage $ppo_norm_adv \
    --ppo-ent-coef $ppo_ent_coef \
    --ppo-vf-coef $ppo_vf_coef \
    --ppo-target-kl $ppo_target_kl
# general arguments
env_type="continuous"
algorithm="PPO"
mode="eval"
dt="11-06-2024-11-16-13"

num_trajectories="10"
do_checkpoints="True"
# model_name=""
# model_dir=""

python screen_agent.py \
    --env-type $env_type \
    --algorithm $algorithm \
    --mode $mode \
    --datetime $dt \
    --num-trajectories $num_trajectories \
    --do-checkpoints $do_checkpoints
    # --model-name $model_name \
    # --model-dir $model_dir
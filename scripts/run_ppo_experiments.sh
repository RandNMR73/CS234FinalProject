#!/bin/bash

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$(pwd):$PYTHONPATH

# Define hyperparameter combinations
declare -a learning_rates=("3e-4" "1e-4")
declare -a n_steps=("256" "512")
declare -a batch_sizes=("64" "128")
declare -a n_epochs=("10" "20")
declare -a gamma=("0.99" "0.95")

# Create output directory if it doesn't exist
output_dir="CS234FinalProject/output/continuous/PPO"
mkdir -p $output_dir

# Loop through each combination of hyperparameters
for lr in "${learning_rates[@]}"; do
  for steps in "${n_steps[@]}"; do
    for batch in "${batch_sizes[@]}"; do
      for epochs in "${n_epochs[@]}"; do
        for g in "${gamma[@]}"; do
          echo "Running experiment with lr=$lr, steps=$steps, batch=$batch, epochs=$epochs, gamma=$g"

          # Define the output path for this combination
          exp_output_dir="${output_dir}/lr_${lr}_steps_${steps}_batch_${batch}_epochs_${epochs}_gamma_${g}"
          mkdir -p $exp_output_dir

          echo "Output directory: $exp_output_dir"

          # Run the training script with the current hyperparameters
          echo "Starting training..."
          python algorithms/ppo.py \
            --ppo-lr-rate $lr \
            --ppo-n-steps $steps \
            --ppo-batch-size $batch \
            --ppo-n-epochs $epochs \
            --ppo-gamma $g \
            --output-path $exp_output_dir
          echo "Finished training for lr=$lr, steps=$steps, batch=$batch, epochs=$epochs, gamma=$g"
        done
      done
    done
  done
done

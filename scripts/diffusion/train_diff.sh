#!/bin/bash
# Sweep template for training the diffusion planner on a dataset written by create_dataset.py
# (datasets/<Env>_*.h5). Edit the dataset name, wandb team/project and the grids below.
# Usage: bash scripts/diffusion/train_diff.sh

diffusion_dataset="DubinsCar_N10000.h5"
diffusion_args=("--diffusion_trajectory_mode sg --sample_goal_change" "--diffusion_trajectory_mode sg")

num_features=("128" "256")
trajectory_lengths=("15" "30")

for args in "${diffusion_args[@]}"; do
  for num_feature in "${num_features[@]}"; do
    for trajectory_length in "${trajectory_lengths[@]}"; do
      python train_diffusion.py --log --wandb_project my-project --wandb_team my-team --dataset_name $diffusion_dataset --num_features $num_feature --num_blocks 3 --val_ratio 0.05 --trajectory_length $trajectory_length --stl_train_traj_len $trajectory_length --batch_size 32 $args
    done
  done
done

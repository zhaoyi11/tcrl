#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --constraint='volta'

#SBATCH --output=/scratch/work/%u/pets-pytorch/triton_log/%A_%a.%j.out
#SBATCH --array=0-7

case $SLURM_ARRAY_TASK_ID in
    0) ENV="acrobot_swingup";;
    1) ENV="cheetah_run" ;;
    2) ENV="fish_swim";;
    3) ENV="dog_run" ;;
    4) ENV="quadruped_walk" ;;
    5) ENV="walker_walk" ;;
    6) ENV="humanoid_walk" ;;
    7) ENV="dog_walk" ;;
esac

source /home/zhaoy13/.bashrc

conda activate minlps

export MUJOCO_GL="egl"
export LC_ALL=en_US.UTF-8

srun python3 main_dreamer.py task=$ENV exp_name="default" save_video=true
# srun python3 main_dreamer.py task=$ENV use_wandb=true use_mppi=false exp_name="mb-repr"
# srun python3 main_dreamer.py task=$ENV use_wandb=true use_mppi=true exp_name="mb-repr-data"
# srun python3 main_dreamer.py task=$ENV use_wandb=true use_mppi=false exp_name="mb-repr-3td"
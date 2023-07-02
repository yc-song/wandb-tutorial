#!/bin/bash
#SBATCH --job-name=wandb
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=<directory you want>/%j.out                # Adjustement required: Write directory you want
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh

srun python wandb_tutorial.py
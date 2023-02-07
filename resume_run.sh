#!/bin/bash
#SBATCH --job-name=wandb-resume
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/wandb_tutorial/slurm/%j.out
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wandb-tutorial

lr=`grep -o '"lr": [^,]*' ./training_params/$1/training_params.json | grep -o '[^ ]*$'`
batch_size=`grep -o '"batch_size": [^,]*' ./training_params/$1/training_params.json | grep -o '[^ ]*$'`
python_output=$(python /home/jongsong/wandb_tutorial/wandb_tutorial.py --resume --lr=${lr} --batch_size=${batch_size} --run_id="$1")
timeout=$(echo "$python_output" | grep -E "timeout")
if [ "$timeout" = "timeout" ]; then
  sbatch resume_run.sh $1
else
  # Python script ran successfully
  sbatch endless_train.sh
fi

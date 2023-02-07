#!/bin/bash
#SBATCH --job-name=wandb-tutorial
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/jongsong/wandb_tutorial/slurm/%j.out
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=8
source /home/${USER}/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda activate wandb-tutorial
python_output=$(wandb agent jongsong/wandb-tutorial/fpl9ck3c --count 1)
echo $python_output
timeout=$(echo "$python_output" | grep -E "timeout")
run_id=${python_output: 0:8}
if [ "$timeout" = "timeout" ]; then
  sbatch resume_run.sh $run_id
else
  # Python script ran successfully
  sbatch endless_train.sh
fi

# if [ $? -eq 124 ]; then
#   # Script exceeded the set timeout of $timeout_in_seconds second
#   sbatch resume_run.sh run_id
# else
#   # Script completed successfully within the set timeout
# #   sbatch endless_train.sh
# fi

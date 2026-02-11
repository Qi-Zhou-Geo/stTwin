#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=opt             # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-1                # job array id
#SBATCH --cpus-per-task=4         # reuest 12 cpu to run in parallel

#SBATCH --mem-per-cpu=32G		     # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/step1_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/step1_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

# Print the current combination
echo "Starting Optuna worker on $SLURM_PROCID"

num_trials=8
num_worker=4
srun python opt_main.py --num_trials "$num_trials" --num_worker "$num_worker"
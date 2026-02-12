#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=opt             # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-1                # job array id
#SBATCH --cpus-per-task=16         # reuest 12 cpu to run in parallel

#SBATCH --mem-per-cpu=32G		     # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/step1_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/step1_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

# delete the existing file
rm sedcas_calibration.db

# Print the current combination
echo "Starting Optuna worker on $SLURM_PROCID"

num_worker=16
trials_per_worker=100 # around 10 minutes to finish one trail
num_trials=$((num_worker * trials_per_worker))

srun python opt_main.py --num_trials "$num_trials" --num_worker "$num_worker"
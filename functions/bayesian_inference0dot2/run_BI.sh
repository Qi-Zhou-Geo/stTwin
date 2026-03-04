#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS) 
#SBATCH --job-name=BI2             # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-1                # job array id
#SBATCH --cpus-per-task=48         # reuest 12 cpu to run in parallel

#SBATCH --mem-per-cpu=8G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/step1_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/step1_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs 

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

# Print the current combination
echo "Starting Bayesian Infernce worker on $SLURM_PROCID"


num_steps1=20
num_steps2=500
num_worker=48

srun python main_BI.py --num_steps1 "$num_steps1" --num_steps2 "$num_steps2" --num_worker "$num_worker"
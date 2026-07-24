#!/bin/bash
#SBATCH -t 00:30:00                # time limit: (D-HH:MM:SS)
#SBATCH --job-name=s01             # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-1                 # job array id, 324
#SBATCH --cpus-per-task=1           # reuest 1 cpu to run in parallel

#SBATCH --mem-per-cpu=8G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

srun python s01_exceedance_pro.py

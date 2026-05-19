#!/bin/bash
#SBATCH -t 24:00:00                # time limit: (D-HH:MM:SS)
#SBATCH --job-name=whatif          # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-540               # job array id
#SBATCH --cpus-per-task=1           # reuest 1 cpu to run in parallel

#SBATCH --mem-per-cpu=8G		    # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin


scenario_idx=$((SLURM_ARRAY_TASK_ID - 1)) # from 0
srun python s02_run_scenario.py --scenario_idx "${scenario_idx}"

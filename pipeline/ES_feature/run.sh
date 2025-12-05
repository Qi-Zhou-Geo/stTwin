#!/bin/bash
#SBATCH -t 4-00:00:00              # time limit: (D-HH:MM:SS)
#SBATCH --job-name=ES              # job name, "Qi_run"

#SBATCH --ntasks=1                 # each individual task in the job array will have a single task associated with it
#SBATCH --array=1-106              # job array id

#SBATCH --mem-per-cpu=4G		   # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/step1_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/step1_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate seismic


# Define arrays for parameters1, parameters2, and parameters3
year=2018
parameters1=($(seq 145 250)) # 106 = 250 - 145 + 1
window_size=3600 # unit is second
window_overlap=0


# SLURM_ARRAY_TASK_ID goes from 1 .. 106
idx=$((SLURM_ARRAY_TASK_ID - 1))
julday=${parameters1[$idx]}


echo "Running: year=$year, julday=$julday"
srun python main.py \
    --year "$year" \
    --julday "$julday" \
    --window_size "$window_size" \
    --window_overlap "$window_overlap"
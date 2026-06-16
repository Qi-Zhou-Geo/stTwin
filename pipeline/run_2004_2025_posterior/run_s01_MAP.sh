#!/bin/bash
#SBATCH -t 00:30:00                # time limit: (D-HH:MM:SS)
#SBATCH --job-name=MAP_2004-2025       # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-1                 # job array id
#SBATCH --cpus-per-task=1           # reuest 1 cpu to run in parallel

#SBATCH --mem-per-cpu=24G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

start_time=$(date)
echo "Job started at: $start_time"


model_version="v0dot4"
theta_draw_idx=$((SLURM_ARRAY_TASK_ID - 1)) # from 0

srun python run_2004_2005_data.py \
    --model_version "${model_version}"\
    --theta_draw_idx "${theta_draw_idx}"


end_time=$(date)
echo "Job finished at: $end_time"

elapsed=$(( ($(date -d "$end_time" +%s) - $(date -d "$start_time" +%s)) ))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))

echo "Total running time: ${hours}h ${minutes}m ${seconds}s"

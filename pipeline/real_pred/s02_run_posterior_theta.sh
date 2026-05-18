#!/bin/bash
#SBATCH -t 00:15:00              # time limit: (D-HH:MM:SS)
#SBATCH --job-name=posterior       # job name, "Qi_run"

#SBATCH --ntasks=1
#SBATCH --array=1-100               # job array id
#SBATCH --cpus-per-task=1           # reuest 1 cpu to run in parallel

#SBATCH --mem-per-cpu=16G		       # Memory Request (per CPU; can use on GLIC)

#SBATCH --output=logs/Posterior_out_%A_%a_%x.txt  # Standard Output Log File
#SBATCH --error=logs/Posterior_err_%A_%a_%x.txt   # Standard Error Log File

# create the “log” folder in case it doesn't exist
mkdir -p logs

source /home/qizhou/miniforge3/bin/activate
conda activate stTwin

model_version="bayesian_inference0dot4"
#project_root="/Users/qizhou/#python/stTwin"
project_root="/home/qizhou/3paper/stTwin"
output_dir="pipeline/real_pred/${model_version}" # under the project path
posterior_h5_dir="${project_root}/functions/${model_version}/sedcas_mcmc_results.h5"
burn_in_step=100
theta_draw_idx=$((SLURM_ARRAY_TASK_ID - 1)) # from 0
model_params="SedCas_input_params_10min_bo_calibrated.yaml"

python_script="${project_root}/functions/post_bayesian_inference/run_posterior_theta.py"

srun python "${python_script}" \
            --project_root "${project_root}" \
            --output_dir "${output_dir}" \
            --posterior_h5_dir "${posterior_h5_dir}" \
            --burn_in_step "${burn_in_step}" \
            --theta_draw_idx "${theta_draw_idx}" \
            --model_params "${model_params}"

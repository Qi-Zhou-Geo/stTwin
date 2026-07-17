#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-19T16:26:47
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse

import numpy as np
import xarray as xr
import pandas as pd

from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the func. from the same folder
from func.SedCas.mass_balance_checker import mass_balance_checker
from func.generator.main_generator import s2d_workflow
from func.bayesian_inference.params_boundary import custom_boundary
from func.bayesian_inference.sample_posterior import get_posterior_theta, get_MAP_theta
from func.SedCas_pred.run_posterior import run_posterior_sedcas


def creat_input(scenario_idx):

    scenario_bound = Path(current_dir) / "scenario_bound.txt"
    df = pd.read_csv(scenario_bound, header=0)

    scenario = df.iloc[scenario_idx, :].values
    idx, cycle_period, storm2drought_ratio, storm_onset_month, storm_onset_day = scenario

    file_format, data = s2d_workflow(
        year_list=(2023, 2024, 2025),
        cycle_period=int(cycle_period),
        storm2drought_ratio=float(storm2drought_ratio),
        storm_onset_month=int(storm_onset_month),
        storm_onset_day=storm_onset_day,
        sigma_scale=3,
        plot=False,
        seed=None,
        ref_history=True,
    )

    print(f"scenario_idx={scenario_idx}, {file_format}")

    return file_format 


def config_whatif_params(climate_frocing_input, 
                         select_t1="2023-01-01T00:00:00", 
                         select_t2="2026-01-01T00:00:00",
                         resolution=600):


    data0_path = Path(project_root) / f"data/SedCas_input/climate_2004_2025_t.txt"
    data0 = pd.read_csv(data0_path, header=0)
    time_str = data0.iloc[:, 1].values
    id1 = np.where(time_str==select_t1)[0][0]
    data0 = data0.iloc[:id1, :]
    
    data1_path = Path(project_root) / f"data/SedCas_whatif_input/{climate_frocing_input}"
    data1 = pd.read_csv(data1_path, header=0)
    data = pd.concat([data0, data1], axis=0)
    
    time_float = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
    time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 1]]

    # Extract variables
    precipitation = data.iloc[:, 2].values
    temperature = data.iloc[:, 3].values
    sun_radiation = data.iloc[:, 4].values
    
    climate_forcing = xr.Dataset(
        coords={
            "time": ("time", np.array(time_float)),  # numeric UTC+0 time
            "time_str": ("time", np.array(time_str)),  # string UTC+0 time
        },
        data_vars={
            "precipitation": ("time", precipitation,
                              {"units": f"mm per {resolution} s", "description": "Total precipitation"}),

            "temperature": ("time", temperature,
                            {"units": f"°C per {resolution} s", "description": "Air temperature"}),

            "sun_radiation": ("time", sun_radiation,
                              {"units": "W/m^2", "description": "Incoming solar radiation"})
        },
        attrs={
            "data_source": climate_frocing_input,
            "station": climate_frocing_input,
            "resolution": resolution,
            "resolution_unit": f"seconds",
            "create_time": UTCDateTime().isoformat()
        }
    )
    
    return climate_forcing


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--scenario_idx", type=int, default=0)
    parser.add_argument("--model_version", type=str, default="v0dot4")
    parser.add_argument("--num_draw", type=int, default=1)
    args = parser.parse_args()

    scenario_idx = args.scenario_idx
    model_version = args.model_version
    num_draw = args.num_draw
    
    
    posterior_h5_dir = Path(project_root) / "func/bayesian_inference/sedcas_mcmc_results.h5"
    burn_in_step = 100
    

    # (1) create the input based defined what-if scenario for model
    file_format = creat_input(scenario_idx)
    climate_frocing_input = f"climate_2023_2026_t_whatif_{file_format}.txt"
    climate_forcing = config_whatif_params(climate_frocing_input=climate_frocing_input)
    
    
    # (2) loop MAP + 50 randowm draw posteriors
    for theta_draw_idx in range(num_draw):
        
        # (3) pack the params
        params_trial = {}
        params_trial['data_type'] = "10min_what_if"
        params_trial["climate_forcing"] = climate_forcing
        params_trial["model_input_params"] =  "SedCas_input_params_10min_QZ.yaml"
        params_trial["project_root"] = Path(project_root)
        params_trial["output_dir"] = f"{current_dir}/{model_version}/{file_format}/theta_{str(theta_draw_idx + 1).zfill(3)}"
        output_ls_dir = f"{current_dir}/{model_version}/{file_format}/SedCas_ls"
        
        
        # (4) load the posterior for model
        if theta_draw_idx == 0:
            theta = get_MAP_theta(posterior_h5_dir, burn_in_step=burn_in_step)

            fix_ls = False
            save_ls = True
            plot_output = True
            save_climate_forcing = True
        else:
            # retuen shape (num_draw, num_paramsters)
            sampled_theta = get_posterior_theta(posterior_h5_dir, num_draw=num_draw, burn_in_step=burn_in_step, fix_seed=True)
            theta = sampled_theta[theta_draw_idx, :] # select theta
            
            fix_ls = True
            save_ls = False
            plot_output = False
            save_climate_forcing = False
            
            
        # normalize it back to real scale
        theta_names, lower_bounds, upper_bounds = custom_boundary()
        params_trial["theta_names"] = theta_names
        # in nature (non-log) sapce
        params_trial["lower_bounds"] = lower_bounds
        params_trial["upper_bounds"] = upper_bounds
        theta = theta * (upper_bounds - lower_bounds) + lower_bounds
        
        
        
        # (5) pack the params
        params_trial["fix_ls"] = fix_ls
        params_trial["save_ls"] = save_ls

        
        # (6) uppdate the params_trial for current process / thrend
        current_params_trial = params_trial.copy()

        current_theta = {}
        for theta_name, theta_value in zip(params_trial["theta_names"], theta):
            current_params_trial[theta_name] = theta_value
            current_theta[theta_name] = theta_value
        print(f"Current theta is\n: {current_theta} \n")
        
        
        # (7) run the model, this is most expensive time-consuming part
        model = run_posterior_sedcas(current_params_trial, num_iteration=100, output_ls_dir=output_ls_dir,
                                    
                                    progress_bars=True, save_output=True, 
                                    save_sed_container=True, save_climate_forcing=save_climate_forcing,
                                    
                                    fix_ls=params_trial["fix_ls"], save_ls=params_trial["save_ls"],
                                    
                                    plot_output=plot_output, show_plot=False,
                                    
                                    select_t1="2004-01-01T00:00:00", select_t2="2026-01-01T00:00:00")
        mass_balance_checker(sed_container=model.sed_container, residual=1.0, iteration=None, silence=True)

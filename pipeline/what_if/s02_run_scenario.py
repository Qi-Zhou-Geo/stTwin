#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-19
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import yaml
import argparse

import numpy as np
import xarray as xr
import pandas as pd

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the func. from the same folder
from func.generator.main_generator import workflow
from func.SedCas_pred.thin_posterior import sample_posterior
from func.SedCas_pred.run_model_with_theta import run_sedcas_once
from func.visulize.plotly_visualize import plotly_multi_time_series_xr


def creat_input(scenario_idx):

    scenario_bound = Path(current_dir) / "scenario_bound.txt"
    df = pd.read_csv(scenario_bound, header=0)

    scenario = df.iloc[scenario_idx, :].values
    idx, cycle_period, storm2drought_ratio, storm_onset_month, storm_onset_day = scenario

    file_format = workflow(
        year_list=(2023, 2024, 2025),
        cycle_period=cycle_period,  # every 60 day
        storm2drought_ratio=storm2drought_ratio,  # duration of storm / drought is 0.1
        storm_onset_month=storm_onset_month,  # start from 1st of April
        storm_onset_day=storm_onset_day,
        plot=True
    )

    print(f"scenario_idx={scenario_idx}, {file_format}")
    
    return file_format 


def load_posterior(posterior_h5_dir="func/bayesian_inference/sedcas_mcmc_results.h5",
                   num_draw=100,
                   burn_in_step=100):
    
    posterior_h5_dir = Path(project_root) / posterior_h5_dir
    
    # return as shape (num_draw, num_theta)
    sampled_theta = sample_posterior(posterior_h5_dir, num_draw=num_draw, burn_in_step=burn_in_step, fix_seed=True)
    
    # load YAML file
    yaml_file = Path(project_root) / "config" / "SedCas_params" / "SedCas_mcmc_params.yaml"
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    theta_name = data["mcmc_theta_meta"]["theta_names"]
    lower = np.array(data["mcmc_theta_meta"]["lower_bounds"], dtype=float)
    upper =  np.array(data["mcmc_theta_meta"]["upper_bounds"], dtype=float)
    theta = sampled_theta * (upper - lower) + lower # normalize it back to real scale
    
    return theta, theta_name


def config_whatif_params(climate_frocing_input):

    resolution = 600  # unit is second
    data = pd.read_csv(f"{project_root}/data/SedCas_input/{climate_frocing_input}", header=0)

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


def save_result(params_trial, model, select_t1, select_t2, show_plot=False):
    
    # prepare the output
    output_dir = f"{params_trial['project_root']}/{params_trial['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)

    # save the results
    model.hydro_output.to_netcdf(f"{output_dir}/hydro_output.nc")
    model.sed_output.to_netcdf(f"{output_dir}/sed_output.nc")

    # plot it
    # update the attrs if the xr is 2024 version
    template_sed_container = model._create_sed_dataset(num_iteration=1)
    for var in model.sed_container.data_vars:
        model.sed_container[var].attrs = template_sed_container[var].attrs.copy()
        model.sed_output[f"{var}_Q1"].attrs = template_sed_container[var].attrs.copy()
        model.sed_output[f"{var}_Q50"].attrs = template_sed_container[var].attrs.copy()
        model.sed_output[f"{var}_Q99"].attrs = template_sed_container[var].attrs.copy()


    time_coord = "time_str"
    t1 = select_t1 # model.climate_forcing.coords["time_str"].values[0]
    t2 = select_t2 # model.climate_forcing.coords["time_str"].values[-1]

    ## sed
    mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
    sed_output_2017 = model.sed_output.isel(time=mask)
    # sed
    list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                            (time_coord, "channel_storage_Q50"),
                            (time_coord, "sed_transport_real_Q50")]
    fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                        list_of_col_names=list_of_col_names,
                                        show_plot=show_plot)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_sediments.html")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--scenario_idx", type=int)
    parser.add_argument("--model_version", type=str, default="v0dot4")
    args = parser.parse_args()

    scenario_idx = args.scenario_idx
    model_version = args.model_version
    
    # (1) create the input based defined what-if scenario for model
    file_format = creat_input(scenario_idx)
    climate_forcing = config_whatif_params(climate_frocing_input=f"climate_2023_2026_t_whatif_{file_format}.txt")
    
    # (2) load the posterior for model
    theta_arr, theta_name = load_posterior() # return as shape (num_draw, num_theta)
    
    # (3) loop all posteriors
    for theta_draw_idx in range(50):#range(len(theta_arr)):
        
        # (4) pack the params
        params_trial = {}
        
        params_trial['data_type'] = "10min_what_if"
        params_trial["climate_forcing"] = climate_forcing
        params_trial["model_params"] = "SedCas_input_params_10min_after_mcmc.yaml"
        params_trial["project_root"] = Path(project_root)
        params_trial["output_dir"] = f'pipeline/what_if/{model_version}/{file_format}/theta_{str(theta_draw_idx + 1).zfill(3)}'
        
        theta = theta_arr[theta_draw_idx, :]
        for name, value in zip(theta_name, theta):
            params_trial[name] = value

        # (5) run the model
        model = run_sedcas_once(params_trial, num_iteration=100,
                            progress_bars=True, save_output=False,
                            fix_ls=True, save_ls=None,
                            plot_output=False, show_plot=False,
                            select_t1="2023-01-01T00:00:00", select_t2="2026-01-01T00:00:00")
        
        # (6) save results
        save_result(params_trial, model, 
                    select_t1="2023-01-01T00:00:00", select_t2="2026-01-01T00:00:00", show_plot=False)
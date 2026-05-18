#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2026-04-29
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import numpy as np
import pandas as pd
import xarray as xr

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.SedCas.SedCas import SedCas
from func.post_bayesian_inference.thin_posterior import sample_posterior

def prepare_posterior4model():
    
    model_version = "bayesian_inference0dot4"
    posterior_h5_dir = f"{project_root}/functions/{model_version}/sedcas_mcmc_results.h5"

    sampled_theta = sample_posterior(posterior_h5_dir, num_draw=100, burn_in_step=100, fix_seed=True)
    theta = np.mean(sampled_theta, axis=0) # select theta

    theta_names = [
    'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
    'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
    'ls_alpha_v',
    'Qdf', 'max_s2w',
    'channel_storage_cap', 'erosion_k']

    lower = np.array([0.1, 10, 10, 1, 6, 6, 1.1, 0.1, 0.1, 1, 0.01])
    upper = np.array([10, 100, 100, 144, 1008, 1008, 2.0, 1.0, 1.0, 100, 10])
    theta = theta * (upper - lower) + lower # normalize it back to real scale

    posterior_theta = {}
    for key, value in zip(theta_names, theta):
        posterior_theta[key] = value

    return posterior_theta


def load_climate_input4model():

    data_source = "MeteoSwiss"
    station = "Montana (MVE)"
    time_now = UTCDateTime().isoformat()
    
    resolution = 600  # unit is second
    
    data1 = pd.read_csv(f"{project_root}/data/liveshow_cache/climate/climate_2023_2024_2025_t.txt", header=0)
    data2 = pd.read_csv(f"{project_root}/data/liveshow_cache/climate/climate_2026_t.txt", header=0)
    data = pd.concat([data1, data2], ignore_index=True)

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
            "data_source": data_source,
            "station": station,
            "resolution": resolution,
            "resolution_unit": f"seconds",
            "create_time": time_now
        }
    )

    return climate_forcing


def run_sedcas(posterior_theta, climate_forcing):
    
    model_input_params = f"{project_root}/config/SedCas_params/SedCas_input_params_10min_bo_calibrated.yaml"
    model = SedCas(project_root=project_root, model_input_params=model_input_params)
    
    # assign the 2023-2026 data
    model.climate_forcing = climate_forcing

    # region <update the model params>
    model.cfg.w_storage_cap.value[0] = [posterior_theta["w_storage_cap0"]]
    model.cfg.w_storage_cap.value[1] = [posterior_theta["w_storage_cap1"],
                                        posterior_theta["w_storage_cap2"]]

    model.cfg.w_residence_time.value[0] = [posterior_theta["w_residence_time0"]]
    model.cfg.w_residence_time.value[1] = [posterior_theta["w_residence_time1"],
                                           posterior_theta["w_residence_time2"]]

    model.cfg.ls_alpha_v.value = posterior_theta["ls_alpha_v"]

    model.cfg.Qdf.value = posterior_theta["Qdf"]
    model.cfg.max_s2w.value = posterior_theta["max_s2w"]

    model.cfg.channel_storage_cap.value = posterior_theta["channel_storage_cap"]
    model.cfg.erosion_k.value = posterior_theta["erosion_k"]

    # you must update the params then post-processing
    model._params_post_processing()
    # make it as critial value
    model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value
    # endregion

    model.run_hydro()
    model.run_stochastic_simulations(seed=0, num_iteration=100, progress_bars=True)

    # only save the 2026 results
    t1 = "2025-01-01T00:00:00"
    t2 = "2036-01-01T00:00:00"
    p_dir = f"{project_root}/data/liveshow_cache/results"
    os.makedirs(p_dir, exist_ok=True)

    mask = (model.hydro_output.time_str >= t1) & (model.hydro_output.time_str < t2)

    hydro_output = model.hydro_output.isel(time=mask)
    hydro_output.to_netcdf(f"{p_dir}/hydro_output.nc")

    sed_output = model.sed_output.isel(time=mask)
    sed_output.to_netcdf(f"{p_dir}/sed_output.nc")
    
    climate = model.climate_forcing .isel(time=mask)
    climate.to_netcdf(f"{p_dir}/climate_forcing.nc")
    
    
def simulate():
    
    posterior_theta = prepare_posterior4model()
    climate_forcing = load_climate_input4model()
    
    run_sedcas(posterior_theta, climate_forcing)

if __name__ == "__main__":
    
    simulate()
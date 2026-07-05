#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-02T18:08:52
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import json
import numpy as np
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


# import the custom functions
from func.SedCas.mass_balance_checker import mass_balance_checker
from func.SedCas_pred.run_posterior import load_config, run_posterior_sedcas
from func.bayesian_inference.sample_posterior import get_posterior_theta, get_MAP_theta
from func.bayesian_inference.params_boundary import custom_boundary
from func.SedCas.load_climate_input import load_climate_input4model


def simulate(num_iteration=100,
             project_root=project_root, 
             output_dir=f"deploy/liveshow_cache/monitoring",
             select_t1="2004-01-01T00:00:00", select_t2="2050-01-01T00:00:00"):

    model_input_params = "SedCas_input_params_10min_QZ.yaml"
    posterior_h5_dir = Path(project_root) / "func/bayesian_inference/sedcas_mcmc_results.h5"


    # (1) load the climate forcing
    climate_input_path1 = Path(project_root) / f"data/SedCas_input/climate_2004_2025_t.txt"
    df_climate1 = pd.read_csv(climate_input_path1, header=0)
    
    climate_input_path2 = Path(project_root) / f"deploy/liveshow_cache/climate/climate_2026_t.txt"
    df_climate2 = pd.read_csv(climate_input_path2, header=0)
    
    df_climate = pd.concat([df_climate1, df_climate2], axis=0)
    climate_forcing = load_climate_input4model(df_climate, climate_resolution=600, data_source="MeteoSwiss")
    
    
    # (2) preapre posterior
    theta = get_MAP_theta(posterior_h5_dir, burn_in_step=100)

    # normalize it back to real scale
    theta_names, lower, upper = custom_boundary()
    theta = theta * (upper - lower) + lower 
    

    # (3) pack the params
    params_trial = {}
    params_trial["project_root"] = project_root
    params_trial["output_dir"] = output_dir
    params_trial["model_input_params"] = model_input_params
    params_trial["data_type"] = 600
    params_trial["climate_forcing"] = climate_forcing
    
    
    # (4) uppdate the params_trial for current process / thrend
    current_theta = {}
    for theta_name, theta_value in zip(theta_names, theta):
        params_trial[theta_name] = theta_value
        current_theta[theta_name] = theta_value
    msg = f"Current theta is\n: {current_theta} \n"
    params_trial = params_trial | current_theta


    # (5) run the model, this is most expensive time-consuming part
    model = run_posterior_sedcas(params_trial, num_iteration=num_iteration,
                                 
                                 progress_bars=True, save_output=True, 
                                 save_sed_container=False, save_climate_forcing=True,
                                 
                                 fix_ls=False, save_ls=False,
                                 
                                 plot_output=False, show_plot=False,
                                 
                                 select_t1=select_t1, select_t2=select_t2)
    mass_balance_checker(sed_container=model.sed_container, residual=1.0, iteration=None, silence=True)

    # (6) record the meta
    last_update = {f"Latest SedCas Update": f"{UTCDateTime().strftime('%Y-%m-%dT%H:%M:%S')} [UTC+0]"}
    json_path = Path(project_root) / f"deploy/liveshow_cache/monitoring/last_SedCas_update.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(last_update, f)
        
    return msg

if __name__ == "__main__":
    
    simulate()
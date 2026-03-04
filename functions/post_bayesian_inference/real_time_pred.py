#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-03-01
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import argparse

import numpy as np
import pandas as pd

import xarray as xr

import emcee

from multiprocessing import Pool, current_process

from obspy import UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas_bo.opt_main import load_climate_input4model, run1time_sedcas

from functions.SedCas_bo.SedCas import SedCas
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr

from functions.toolkit.physical_unit_converter import unit_converter
from functions.toolkit.loss_func import gaussian_log_likelihood, clean_obs_pre

from functions.toolkit.archive_data import dump_as_row
from functions.post_bayesian_inference.thin_posterior import sample_posterior

def load_config():

    # all input params are stored here and will be updated later
    params_trial = {"project_root": current_dir.parent.parent,
                    "output_dir": f"{current_dir.parent.parent}/pipeline/real_pred/output",
                    "df_volume_file_name": "debris_flow_volume_2004_2022.txt",
                    "data_type": "10-minutes",
                    "model_params":"SedCas_input_params_10min_bo.yaml",
                    "posterior_results_file": "functions/bayesian_inference/sedcas_mcmc_results.h5"}

    os.makedirs(params_trial["output_dir"], exist_ok=True)

    # load the same observed df volume and climate forcing
    y_obs = pd.read_csv(f"{params_trial['project_root']}"
                        f"/data/event_catalog/{params_trial['df_volume_file_name']}",
                        skiprows=6, header=0)
    climate_forcing = load_climate_input4model(data_type=params_trial["data_type"])
    params_trial["y_obs"] = y_obs
    params_trial["climate_forcing"] = climate_forcing

    return params_trial

def log_likelihood(theta, params_trial, eps=1e-10):

    # uppdate the params_trial for current process / thrend
    current_params_trial = params_trial.copy()
    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v', 'Qdf', 'max_s2w', 'channel_storage_cap', 'erosion_k', 'sigma'
    ]
    current_theta = {}
    for theta_name, theta_value in zip(theta_names, theta):
        current_params_trial[theta_name] = theta_value
        current_theta[theta_name] = theta_value

    # run the model, this is most expensive time-consuming part
    model = run1time_sedcas(current_params_trial, num_iteration=120,
                            progress_bars=False, save_output=False, plot_output=False)

    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    y_pred = unit_converter(input=sed_transport_real,
                            catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    #
    y_obs_valid, y_pred_valid = clean_obs_pre(current_params_trial["y_obs"], y_pred, buffer_time=3, failed_prediction=0)
    y_obs = y_obs_valid["Volume[m3]"].values
    y_pred = y_pred_valid["Volume[m3]"].values

    # avoid log(0)
    y_obs = np.clip(y_obs, a_min=eps, a_max=None)
    y_pred = np.clip(y_pred, a_min=eps, a_max=None)

    residual = np.log(y_obs) - np.log(y_pred) # loge based volume residual
    sigma = max(current_params_trial["sigma"], eps) # ensure sigma > 0
    g_log_like = gaussian_log_likelihood(residual, sigma)

    # record
    worker_name = current_process().name
    record = (f"{UTCDateTime.now().isoformat()}\n"
              f"{worker_name}, {g_log_like}, {sigma}, \n"
              f"{', '.join(map(str, y_pred))}, \n"
              f"{current_theta} \n")
    output_dir = f'{current_params_trial["output_dir"]}'
    output_name = f"sedcas_mcmc_record_{worker_name}"

    dump_as_row(output_dir, output_name, record)

    return g_log_like

if __name__ == "__main__":
    # receive the arguments
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--num_worker", default=32, type=int)
    args = parser.parse_args()

    optuna_optimal = {'w_storage_cap0': 0.366,
                      'w_storage_cap1': 69.662,
                      'w_storage_cap2': 31.169,

                      'w_residence_time0': 21.294,
                      'w_residence_time1': 7.391,
                      'w_residence_time2': 23.829,

                      'ls_alpha_v': 1.166, # uniform

                      'Qdf': 0.154,
                      'max_s2w': 0.144,
                      'channel_storage_cap': 11.154,
                      'erosion_k': 1.295,

                      'sigma': 0.629, # represents measurement noise and the model structural error
                      }
    params_trial = load_config()

    posterior_results_file = f"{project_root}/{params_trial['posterior_results_file']}"
    posterior_draws = sample_posterior(posterior_results_file, num_draw=100, burn_in_step=100, fix_seed=True)


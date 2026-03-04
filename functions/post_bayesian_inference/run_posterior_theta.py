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
from functions.post_bayesian_inference.thin_posterior import sample_posterior, maximum_likelihood_theta

from functions.bayesian_inference.main_BI import load_config, log_likelihood


def main():

    params_trial = load_config()
    params_trial["output_dir"] = f"{project_root}/pipeline/real_pred/output"
    os.makedirs(params_trial["output_dir"], exist_ok=True)

    posterior_results_file = f"{project_root}/functions/bayesian_inference/sedcas_mcmc_results.h5"
    theta =  maximum_likelihood_theta(posterior_results_file, burn_in_step=100)

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
                            progress_bars=True, save_output=True, plot_output=False)

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

    residual = np.log(y_obs) - np.log(y_pred)  # loge based volume residual
    sigma = max(current_params_trial["sigma"], eps)  # ensure sigma > 0
    g_log_like = gaussian_log_likelihood(residual, sigma)
    print(g_log_like)

    time_coord = "time_str"
    t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"
    mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
    sed_output_2017 = model.sed_output.isel(time=mask)
    list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                         (time_coord, "channel_storage_Q50"),
                         (time_coord, "sed_transport_real_Q50")]
    fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                      list_of_col_names=list_of_col_names)

if __name__ == "__main__":
    main()

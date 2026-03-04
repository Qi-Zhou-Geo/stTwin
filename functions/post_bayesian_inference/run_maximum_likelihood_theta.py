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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


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
from functions.toolkit.loss_func import calculate_pred_ratio, clean_obs_pre

from functions.toolkit.archive_data import dump_as_row
from functions.post_bayesian_inference.thin_posterior import sample_posterior, maximum_likelihood_theta

from functions.bayesian_inference.main_BI import load_config, log_likelihood

def main(params_trial, theta=None):

    # allow you
    if theta is None:
        posterior_results_file = params_trial["posterior_results_file"]
        theta =  maximum_likelihood_theta(posterior_results_file, burn_in_step=20)
        print(theta)

    # uppdate the params_trial for current process / thrend
    current_params_trial = params_trial.copy()
    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v', 'Qdf', 'max_s2w', 'channel_storage_cap', 'erosion_k'
    ]
    current_theta = {}
    for theta_name, theta_value in zip(theta_names, theta):
        current_params_trial[theta_name] = theta_value
        current_theta[theta_name] = theta_value

    # run the model, this is most expensive time-consuming part
    model = run1time_sedcas(current_params_trial, num_iteration=100,
                            progress_bars=True, save_output=True, plot_output=False)
    # model.sed_output.to_netcdf(f"{project_root}/pipeline/real_pred/output/sed_output.nc")

    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    y_pred = unit_converter(input=sed_transport_real,
                            catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    # select the presults
    y_obs_valid, y_pred_valid = clean_obs_pre(current_params_trial["y_obs"], y_pred,
                                              buffer_time=3, failed_prediction=0, ratio_of_faliure=0.1)
    y_obs = y_obs_valid["Volume[m3]"].values
    y_pred = y_pred_valid["Volume[m3]"].values

    # avoid log(0)
    eps = 1e-10
    y_obs = np.clip(y_obs, a_min=eps, a_max=None)
    y_pred = np.clip(y_pred, a_min=eps, a_max=None)

    residual = np.log10(y_obs) - np.log10(y_pred)
    sigma = 4.34  # fixed sigma, 4.34 is σ=10 in natural log
    g_log_like = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
    pred_ratio = calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1)

    plt.rcParams.update({'font.size': 7,
                         'axes.formatter.limits': (-4, 6),
                         'axes.formatter.use_mathtext': True})

    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    x = np.arange(len(pred_ratio))
    r1 = np.sum(pred_ratio > 1e3)
    r2 = np.sum(pred_ratio < 1e-3)
    y = np.clip(pred_ratio, a_min=1e-3, a_max=1e3)
    plt.title(f"Outside 1e-3 to 1e3: {r1+r2}, g_log_like: {g_log_like}")
    plt.bar(x, y, color="black", alpha=0.5)
    plt.yscale("log")
    plt.ylim(1e-3, 1e3)
    ax.set_ylabel("Prediction Error", fontweight='bold')
    ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{params_trial['output_dir']}/ratio_error.png", dpi=600)  # , transparent=True
    plt.show()
    plt.close(fig=fig)

    sed_container_mean = model.sed_container.mean(dim='iteration', keep_attrs=True)
    with xr.set_options(keep_attrs=True):
        sed_container_mean = sed_container_mean * 4830.0 # add catchment scale

    time_coord = "time_str"
    t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"
    # t1, t2 = "2004-08-24T06:41:00", "2004-08-24T14:41:00"
    t1_epoch = UTCDateTime(t1).timestamp
    t2_epoch = UTCDateTime(t2).timestamp
    sed_output_2017 = sed_container_mean.sel(time=slice(t1_epoch, t2_epoch))
    list_of_col_names = [(time_coord, "hillslope_storage"),
                         (time_coord, "channel_storage"),
                         (time_coord, "sed_transport_real")]
    fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                      list_of_col_names=list_of_col_names)

    return model

if __name__ == "__main__":
    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v',
        'Qdf', 'max_s2w',
        'channel_storage_cap', 'erosion_k'
    ]

    params_trial = load_config()
    params_trial["output_dir"] = f"{project_root}/pipeline/real_pred/bayesian_inference0dot2"
    os.makedirs(params_trial["output_dir"], exist_ok=True)
    params_trial["posterior_results_file"] =  f"{project_root}/functions/bayesian_inference0dot2/sedcas_mcmc_results.h5"

    theta = None #[1.832, 54.202, 40.707, 95.093, 386.850, 386.613, 1.299, 0.455, 0.270, 25.935, 2.958]
    model = main(params_trial=params_trial, theta=theta)

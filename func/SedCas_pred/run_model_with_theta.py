#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-03-01
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import yaml
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
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

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.bayesian_opt.opt_main import load_climate_input4model, run1time_sedcas
from func.SedCas.SedCas import SedCas
from func.toolkit.physical_unit_converter import unit_converter
from func.toolkit.loss_func import calculate_pred_ratio, clean_obs_pre
from func.visulize.plotly_visualize import plotly_multi_time_series_xr


def load_config(project_dir, output_dir, data_type="10-minutes"):

    # all input params are stored here and will be updated later
    params_trial = {"project_root": project_dir,
                    "output_dir": output_dir,
                    "df_volume_file_name": "debris_flow_volume_2004_2022.txt",
                    "data_type": data_type,
                    "model_params":"SedCas_input_params_10min_QZ.yaml",
                    "updated_params":"SedCas_input_params_10min_after_mcmc.yaml"}

    os.makedirs(f'{project_root}/{params_trial["output_dir"]}', exist_ok=True)
    print(f'output_dir: {project_root}/{params_trial["output_dir"]} \n')

    # load the same observed df volume and climate forcing
    y_obs = pd.read_csv(f"{params_trial['project_root']}"
                        f"/data/event_catalog/{params_trial['df_volume_file_name']}",
                        skiprows=6, header=0)
    climate_forcing = load_climate_input4model(data_type=params_trial["data_type"])
    params_trial["y_obs"] = y_obs
    params_trial["climate_forcing"] = climate_forcing

    # for theta bounds
    # in nature (non-log) sapce
    lower_bounds = np.array([0.1, 10, 10, 1, 6, 6, 1.1, 0.1, 0.1, 1, 0.01])
    params_trial["lower_bounds"] = lower_bounds

    # in nature (non-log) sapce
    upper_bounds = np.array([10, 100, 100, 144, 1008, 1008, 2.0, 1.0, 1.0, 100, 10])
    params_trial["upper_bounds"] = upper_bounds

    # params name
    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v',
        'Qdf', 'max_s2w',
        'channel_storage_cap', 'erosion_k'
    ]
    params_trial["theta_names"] = theta_names

    return params_trial

def plot_ratio(current_params_trial, y_pred, sigma):

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
    # sigma = 1 # 1.0 in log10 space = factor of 10 in linear space
    # I expect SedCas model to predict the volume within one order of magnitude of the truth
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
    plt.title(f"Outside 1e-3 to 1e3: {r1 + r2}, g_log_like: {g_log_like}")
    plt.bar(x, y, color="black", alpha=0.5)
    plt.yscale("log")
    plt.ylim(1e-3, 1e3)
    ax.set_ylabel("Prediction Error", fontweight='bold')
    ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{current_params_trial['project_root']}"
                f"/{current_params_trial['output_dir']}"
                f"/ratio_error.png", dpi=600)  # , transparent=True
    # plt.show()
    plt.close(fig=fig)

def save_last_status(model, current_params_trial, current_theta):

    # (1) parepae the I/O
    input_yaml = (f'{project_root}'
                  f'/config/SedCas_params'
                  f'/{current_params_trial["model_params"]}')

    output_yaml = (f'{project_root}'
                  f'/config/SedCas_params'
                  f'/{current_params_trial["updated_params"]}')

    # (2) prepare the new values
    # initial water storage in each reservoir
    initial_w_storage = [
        [float(model.hydro_output["w_storage"][-1, 0, 0].values)],
        [float(model.hydro_output["w_storage"][-1, 1, 1].values),
         float(model.hydro_output["w_storage"][-1, 1, 2].values)]
    ] # if you do not understand, check line 265 in functions/SedCas/SedCas.py

    # convrt to plain Python float type
    w_storage_cap = [[float(v) for v in sublist] for sublist in model.cfg.w_storage_cap.value]
    w_residence_time = [[float(v) for v in sublist] for sublist in model.cfg.w_residence_time.value]

    # Initlal hillslope storage, area-normalized sediment thickness
    initial_hs_storage = model.sed_output["hillslope_storage_Q50"].values[-1]

    # Initlal channel storage, area-normalized sediment thickness
    initial_ch_storage = model.sed_output["channel_storage_Q50"].values[-1]


    new_values = {
        "initial_w_storage": initial_w_storage,
        "w_storage_cap": w_storage_cap,
        "w_residence_time": w_residence_time,
        "initial_hs_storage": initial_hs_storage,
        "initial_ch_storage": initial_ch_storage,
    }
    inherited_parameters = ", ".join(new_values.keys())
    inherited_from_time = model.sed_output["channel_storage_Q50"].coords["time_str"][-1].item()


    for key, value in current_theta.items():
        new_values[key] = value

    # (3) update the original ymal file
    ruamel_yaml = YAML()
    ruamel_yaml.preserve_quotes = True

    with open(input_yaml, "r") as f:
        cfg = ruamel_yaml.load(f)
        # cfg
        #  ├── model_input
        #  │     └── key
        #  ├── model_config
        #  │     └── key
        #  └── model_output
        #        └── key

    for key, val in new_values.items():
        for section in cfg:  # model_input, model_config, model_output
            if key in cfg[section]:
                # cfg[section][key]["value"] = float(val)
                if key in ["initial_w_storage", "w_storage_cap", "w_residence_time"]:
                    value_ref = cfg[section][key]["value"]
                    value_ref[0][0] = val[0][0]
                    value_ref[1][0] = val[1][0]
                    value_ref[1][1] = val[1][1]
                else:
                    cfg[section][key]["value"] = float(val)


    # (5) create ReadMe section
    readme = CommentedMap()
    readme["Last Updated"] = UTCDateTime.now().isoformat()
    readme["Author"] = "QZ"
    readme["inherited_parameters"] = inherited_parameters
    readme["inherited_from_time"] = inherited_from_time
    cfg.insert(0, "ReadMe", readme) # insert at top

    # (5) save the new ymal file
    with open(output_yaml, "w") as f:
        ruamel_yaml.dump(cfg, f)

    print(f"Updated model parameters are saved at:\n"
          f" {output_yaml}\n")


def run_sedcas_once(params_trial, num_iteration=100,
                    progress_bars=False, save_output=True,
                    plot_output=True, show_plot=False,
                    select_t1="2004-02-01T00:00:00", select_t2="2023-01-01T00:00:00"):

    project_root = params_trial["project_root"]
    model = SedCas(project_root=project_root,
                   model_input_params=f"{project_root}/config/SedCas_params/{params_trial['model_params']}")
    # rather do: model.load_climate_input(data_type=data_type)
    model.climate_forcing = params_trial["climate_forcing"]

    # <editor-fold desc="update the model params">
    model.cfg.w_storage_cap.value[0] = [params_trial["w_storage_cap0"]]
    model.cfg.w_storage_cap.value[1] = [params_trial["w_storage_cap1"],
                                        params_trial["w_storage_cap2"]]

    model.cfg.w_residence_time.value[0] = [params_trial["w_residence_time0"]]
    model.cfg.w_residence_time.value[1] = [params_trial["w_residence_time1"],
                                           params_trial["w_residence_time2"]]

    model.cfg.ls_alpha_v.value = params_trial["ls_alpha_v"]

    model.cfg.Qdf.value = params_trial["Qdf"]
    model.cfg.max_s2w.value = params_trial["max_s2w"]

    model.cfg.channel_storage_cap.value = params_trial["channel_storage_cap"]
    model.cfg.erosion_k.value = params_trial["erosion_k"]

    # you must update the params then post-processing
    model._params_post_processing()
    # make it as critial value
    model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value
    # endregion

    model.run_hydro()
    model.run_stochastic_simulations(seed=0, num_iteration=num_iteration, progress_bars=progress_bars)

    # prepare the output
    output_dir = f"{params_trial['project_root']}/{params_trial['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)

    # save the results
    if save_output is True:
        model.hydro_output.to_netcdf(f"{output_dir}/hydro_output.nc")
        model.sed_output.to_netcdf(f"{output_dir}/sed_output.nc")
        model.sed_container.to_netcdf(f"{output_dir}/sed_container.nc")

    # plot it
    if plot_output is True:

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

        # <editor-fold desc="update the model params">
        ## climate forcing
        mask = (model.climate_forcing.time_str >= t1) & (model.climate_forcing.time_str < t2)
        climate_forcing_2017 = model.climate_forcing.isel(time=mask)
        list_of_col_names = [(time_coord, "precipitation"),
                             (time_coord, "temperature"),
                             (time_coord, "sun_radiation")]
        fig = plotly_multi_time_series_xr(xr_dataset=climate_forcing_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_climate_forcing.html")

        ## hydro
        mask = (model.hydro_output.time_str >= t1) & (model.hydro_output.time_str < t2)
        hydro_output_2017 = model.hydro_output.isel(time=mask)

        # SWE
        list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "snow_delta_depth"),
                             (time_coord, "snow_acc"), (time_coord, "snow_melt")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_SWE.html")

        # ET
        list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_ET.html")

        # Q
        list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_discharge.html")

        ## sed
        mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
        sed_output_2017 = model.sed_output.isel(time=mask)

        # landslides
        list_of_col_names = [(time_coord, "ls_Q1"),
                             (time_coord, "ls_Q50"),
                             (time_coord, "ls_Q99")]
        fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_ls.html")

        # sed
        list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                             (time_coord, "channel_storage_Q50"),
                             (time_coord, "sed_transport_real_Q50")]
        fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_sediments.html")
        # endregion

    return model

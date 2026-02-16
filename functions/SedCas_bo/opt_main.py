#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import argparse

from functools import partial
from multiprocessing import Pool

import optuna
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

import zarr
import xarray as xr

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
from functions.SedCas_bo.SedCas import SedCas
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr

from functions.SedCas_bo.physical_unit_converter import unit_converter
from functions.SedCas_bo.loss_func import likehood_loss

from functions.toolkit.archive_data import dump_as_row

def sedcas_plot(params_trial):

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
    model.cfg.h2s_r.value = params_trial["h2s_r"]

    model.cfg.Qdf.value = params_trial["Qdf"]

    model.cfg.max_s2w.value = params_trial["max_s2w"]
    model.cfg.max_s_c.value = params_trial["max_s_c"]

    model.cfg.channel_storage_cap.value = params_trial["channel_storage_cap"]
    model.cfg.erosion_k.value = params_trial["erosion_k"]

    # you must update the params then post-processing
    model._params_post_processing()
    # make it as critial value
    model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value
    # </editor-fold>

    model.run_hydro()
    model.run_stochastic_simulations(seed=0, num_iteration=100, progress_bars=False)


    # plot it
    output_dir = f"{params_trial['project_root']}/{params_trial['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)
    time_coord = "time_str"
    t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"

    # <editor-fold desc="update the model params">
    ## climate forcing
    mask = (model.climate_forcing.time_str >= t1) & (model.climate_forcing.time_str < t2)
    climate_forcing_2017 = model.climate_forcing.isel(time=mask)
    list_of_col_names = [(time_coord, "precipitation"), (time_coord, "temperature"), (time_coord, "sun_radiation")]
    fig = plotly_multi_time_series_xr(xr_dataset=climate_forcing_2017, list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_climate_forcing.html")

    ## hydro
    mask = (model.hydro_output.time_str >= t1) & (model.hydro_output.time_str < t2)
    hydro_output_2017 = model.hydro_output.isel(time=mask)

    # SWE
    list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "snow_delta_depth"),
                         (time_coord, "snow_acc"), (time_coord, "snow_melt")]
    fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_SWE.html")

    # ET
    list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
    fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_ET.html")

    # Q
    list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
    fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_discharge.html")

    ## sed
    mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
    sed_output_2017 = model.sed_output.isel(time=mask)

    # landslides
    list_of_col_names = [(time_coord, "ls_Q1"),
                         (time_coord, "ls_Q50"),
                         (time_coord, "ls_Q99")]
    fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                      list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_ls.html")

    # sed
    list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                         (time_coord, "channel_storage_Q50"),
                         (time_coord, "sed_transport_real_Q50")]
    fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                      list_of_col_names=list_of_col_names)
    fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{t1[:4]}_{t2[:4]}_sediments.html")
    # </editor-fold>

    model.hydro_output.to_netcdf(f"{output_dir}/hydro_output.nc")
    model.sed_output.to_netcdf(f"{output_dir}/sed_output.nc")
    model.sed_container.to_netcdf(f"{output_dir}/sed_container.nc")

def write_results(params_trial, trial_number, total_loss, details_loss):

    output_dir = f"{params_trial['project_root']}/{params_trial['output_dir']}"
    os.makedirs(output_dir, exist_ok=True)
    output_name = f"details_loss"
    details_loss = [f"{UTCDateTime.now().isoformat()}_Trial_{trial_number}"] + details_loss # extend the list
    variable_str = "\n".join(map(str, details_loss))
    dump_as_row(output_dir, output_name, variable_str)

    print(f"{UTCDateTime.now().isoformat()}\n"
          f"Process (ID={os.getpid()}) done trial (number={trial_number}).\n"
          f"Total_loss={total_loss}")

def load_climate_input4model(data_type):

    data_source = "MeteoSwiss"
    station = "Montana (MVE)"
    time_now = UTCDateTime().isoformat()

    if data_type == "default":
        # use the default data from SedCas model
        data = pd.read_csv(f"{self.model_input_dir}/climate.met", sep='\t')

        time_float = [UTCDateTime(i).timestamp for i in data.iloc[:, 0]]
        time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 0]]

        # Extract variables
        precipitation = data.Pr.values
        temperature = data.Ta.values
        sun_radiation = data.Rsw.values

        resolution = 3600  # unit is second
    elif data_type in ["1-hour", "10-minutes"]:

        if data_type == "1-hour":
            climate_frocing_input = "climate_2004_2023_h.txt"
            resolution = 3600  # unit is second
        elif data_type == "10-minutes":
            climate_frocing_input = "climate_2004_2023_t.txt"
            resolution = 600  # unit is second
        else:
            raise ValueError("data_type must be '1-hour' or '10-minutes'")

        data = pd.read_csv(f"{project_root}/data/SedCas_input/{climate_frocing_input}", header=0)

        time_float = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
        time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 1]]

        # Extract variables
        precipitation = data.iloc[:, 2].values
        temperature = data.iloc[:, 3].values
        sun_radiation = data.iloc[:, 4].values

    elif data_type == "sediment":
        pass

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

def objective(trial, params_trial):

    trial_number = trial.number

    # <editor-fold desc="(1) set thg params need to be calibrated">
    # Water storage capacities of HRUs, unit by mm for normalized area
    w_storage_cap0 = trial.suggest_float("w_storage_cap0", 0.1, 10, log=True)  # bedrock
    w_storage_cap1 = trial.suggest_float("w_storage_cap1", 10, 100, log=True)  # forest top
    w_storage_cap2 = trial.suggest_float("w_storage_cap2", 10, 100, log=True)  # forest bottom

    # Mean residence time in saturated condition, unit by time step (10 minutes here)
    w_residence_time0 = trial.suggest_float("w_residence_time0", 1, 288, log=True)  # bedrock, from 10 minutes to 2 days
    w_residence_time1 = trial.suggest_float("w_residence_time1", 6, 1008, log=True)  # forest top, from 1h to 7 days
    w_residence_time2 = trial.suggest_float("w_residence_time2", 1, 2016, log=True)  # forest bottom, from 1h to 14 days

    # Minimum potential landslide volume, unit by m^3
    ls_alpha_v = trial.suggest_float("ls_alpha_v", 1.1, 3) # bigger value -> large landslides become much rarer

    # Sediments deposition rate from hillslope to channel, no physical unit
    h2s_r = trial.suggest_float("h2s_r", 0, 1)
    # Discharge threshold for triggering debris flows, mm (area-normalzied unit) / <time_resolution>
    Qdf = trial.suggest_float("Qdf", 0.01, 10, log=True)  # debris flow

    # Max volumetric sediment to water ratio, no physical unit
    max_s2w = trial.suggest_float("max_s2w", 0.01, 0.99, log=True)
    # Max possible sediment concentration for bedload, no physical unit
    max_s_c = trial.suggest_float("max_s_c", 0.01, 0.99, log=True)

    # channel_storage_cap, mm, aera normalized unit
    channel_storage_cap = trial.suggest_float("channel_storage_cap", 10, 150, log=True)
    # erosion efficiency
    erosion_k = trial.suggest_float("erosion_k", 0.1, 10, log=True)
    # </editor-fold>

    # (2) load model
    project_root = params_trial["project_root"]
    model = SedCas(project_root=project_root,
                   model_input_params=f"{project_root}/config/SedCas_params/{params_trial['model_params']}")
    # rather do: model.load_climate_input(data_type=data_type)
    model.climate_forcing = params_trial["climate_forcing"].copy()

    # <editor-fold desc="(3) update the model params">
    model.cfg.w_storage_cap.value[0] = [w_storage_cap0]
    model.cfg.w_storage_cap.value[1] = [w_storage_cap1, w_storage_cap2]

    model.cfg.w_residence_time.value[0] = [w_residence_time0]
    model.cfg.w_residence_time.value[1] = [w_residence_time1, w_residence_time2]

    model.cfg.ls_alpha_v.value = ls_alpha_v
    model.cfg.h2s_r.value = h2s_r

    model.cfg.Qdf.value = Qdf

    model.cfg.max_s2w.value = max_s2w
    model.cfg.max_s_c.value = max_s_c

    model.cfg.channel_storage_cap.value = channel_storage_cap
    model.cfg.erosion_k.value = erosion_k

    # you must update the params then post-processing
    model._params_post_processing()
    # make it as critial value
    model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value
    # </editor-fold>

    # (4) run the model
    model.run_hydro()
    model.run_stochastic_simulations(seed=0, num_iteration=100, progress_bars=False)
    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    y_pred = unit_converter(input=sed_transport_real,
                            catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    # (5) elevate the loss
    y_obs = params_trial["y_obs"].copy()  # field observed debris flow events and volume
    total_loss, details_loss = likehood_loss(y_obs, y_pred, buffer_time=3, default_loss=1e10)

    # (6) dump the details loss
    write_results(params_trial, trial_number, total_loss, details_loss)


    return total_loss

def run_optimization(num_trials):

    # all input params are stored here and will be updated later
    params_trial = {"project_root": current_dir.parent.parent,
                    "output_dir": "functions/SedCas_bo/output",
                    "df_volume_file_name": "debris_flow_volume_2004_2022.txt",
                    "data_type": "10-minutes",
                    "model_params":"SedCas_input_params_10min_bo.yaml"}

    # load the same observed df volume and climate forcing
    y_obs = pd.read_csv(f"{params_trial['project_root']}"
                        f"/data/event_catalog/{params_trial['df_volume_file_name']}",
                        skiprows=6, header=0)
    climate_forcing = load_climate_input4model(data_type=params_trial["data_type"])
    params_trial["y_obs"] = y_obs
    params_trial["climate_forcing"] = climate_forcing


    # perpare optuna
    storage_file = f"{current_dir}/sedcas_journal.log"
    storage = JournalStorage(JournalFileBackend(file_path=storage_file))

    # prepare the Bayesian Optimization
    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.TPESampler(seed=42 + os.getpid()), # unique for each process
        study_name="sedcas_calibration",
        direction="minimize",
        load_if_exists=True # Useful for multi-process or multi-node optimization.
    )
    objective_with_data = partial(objective, params_trial=params_trial)
    study.optimize(objective_with_data, n_trials=num_trials)

    # best parameters found
    print("Best parameters found:\n")
    for param_name, param_value in study.best_params.items():
        print(f"{param_name} = {param_value:.4f}")
        params_trial[param_name] = param_value
    print(f"min loss = {study.best_value:.4f}")

    # plot the best fiting
    sedcas_plot(params_trial)


if __name__ == "__main__":

    # remove the storage
    storage_file = f"{current_dir}/sedcas_journal.log"
    if os.path.exists(storage_file):
        os.remove(storage_file)
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"Delete old storage file:\n{storage_file}.")

    # receive the arguments
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--num_trials", default=4, type=int)
    parser.add_argument("--num_worker", default=2, type=int)
    args = parser.parse_args()

    # prepare the params
    project_root = current_dir.parent.parent
    num_trials = args.num_trials
    num_worker = args.num_worker
    trials_per_worker = num_trials // num_worker

    print(f"Start: {UTCDateTime.now().isoformat()}, \n"
          f"num_trials={num_trials}, num_worker={num_worker}")

    # multiple process
    with Pool(processes=num_worker) as pool:
        pool.map(run_optimization, [trials_per_worker] * num_worker)

    print(f"End: {UTCDateTime.now().isoformat()}")

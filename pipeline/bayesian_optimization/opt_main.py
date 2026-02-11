#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import optuna
import numpy as np
import pandas as pd


# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas_re.SedCas_re import SedCas
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr

from functions.SedCas_re.physical_unit_converter import unit_converter
from functions.SedCas_re.loss_func import likehood_loss


def sedcas_model(model_params, data_type,

                 min_df_v,
                 w_storage_cap0, w_storage_cap1, w_storage_cap2,
                 w_residence_time0, w_residence_time1, w_residence_time2,
                 Qdf,

                 project_root):

    # (1) initial the SedCas model
    model = SedCas(project_root=project_root,
                   model_input_params=f"{project_root}/config/SedCas_params/{model_params}")

    # update the model params
    model.cfg.min_df_v.value = min_df_v

    model.cfg.w_storage_cap.value[0] = [w_storage_cap0]
    model.cfg.w_storage_cap.value[1] = [w_storage_cap1, w_storage_cap2]

    model.cfg.w_residence_time.value[0] = [w_residence_time0]
    model.cfg.w_residence_time.value[1] = [w_residence_time1, w_residence_time2]

    model.cfg.Qdf.value = Qdf

    # you must update the params then post-processing
    model._params_post_processing()

    # (2) load the climate forcing data
    model.load_climate_input(data_type=data_type)

    # (3) run the hydro model
    model.run_hydro()

    # (4) run the sediment model
    model.run_stochastic_simulations(seed=0, num_iteration=100)

    # (5) select the sediments array
    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    # conver mm to m^3
    y_pred = unit_converter(input=sed_transport_real, catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    return y_pred


def objective(trial, y_obs,
              project_root,
              model_params,
              data_type):

    # (1) set thg params need to be calibrated
    # Minium debris-flow volume, unit by m^3
    min_df_v = trial.suggest_float("min_df_v", 100, 5000, log=True)

    # Water storage capacities of HRUs, unit by mm for normalized area
    w_storage_cap0 = trial.suggest_float("w_storage_cap0", 0.1, 10, log=True)  # bedrock
    w_storage_cap1 = trial.suggest_float("w_storage_cap1", 10, 100, log=True)  # forest top
    w_storage_cap2 = trial.suggest_float("w_storage_cap2", 10, 100, log=True)  # forest bottom

    # Mean residence time in saturated condition, unit by time step
    w_residence_time0 = trial.suggest_float("w_residence_time0", 1, 100, log=True)  # bedrock
    w_residence_time1 = trial.suggest_float("w_residence_time1", 1, 500, log=True)  # forest top
    w_residence_time2 = trial.suggest_float("w_residence_time2", 1, 500, log=True)  # forest bottom

    # Discharge threshold for triggering debris flows
    Qdf = trial.suggest_float("Qdf", 0.1, 5)  # debris flow

    # (2) pass the prams + data to the model and return model predicted
    y_pred = sedcas_model(model_params, data_type,

                          min_df_v,
                          w_storage_cap0, w_storage_cap1, w_storage_cap2,
                          w_residence_time0, w_residence_time1, w_residence_time2,
                          Qdf,

                          project_root)

    # (3) calculate the loss
    total_loss = likehood_loss(y_obs, y_pred)

    return total_loss

def main(project_root, num_trials, num_worker):

    # field observed debris flow events and volume
    file_name = "debris_flow_volume_2004_2022.txt"
    y_obs = pd.read_csv(f"{project_root}/data/event_catalog/{file_name}", skiprows=6, header=0)

    # model default params and input climate forcing data
    model_params = "SedCas_input_params_10min.yaml"
    data_type = "10-minutes"

    storage = "sqlite:///sedcas_calibration.db"
    # prepare the Bayesian Optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage,
        study_name="sedcas_calibration",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, y_obs, project_root, model_params, data_type),
                   n_trials=num_trials, n_jobs=num_worker)

    # best parameters found
    for param_name, param_value in study.best_params.items():
        print(f"{param_name} = {param_value:.4f}")
    print(f"min loss = {study.best_value:.4f}")

if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--num_worker", default=1, type=int)
    args = parser.parse_args()

    # run it
    project_root = current_dir.parent.parent
    n_trials = 500
    num_worker = args.num_worker
    main(project_root, n_trials, num_worker)

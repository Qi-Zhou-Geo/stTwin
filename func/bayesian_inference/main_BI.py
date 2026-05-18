#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
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

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
import sys

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# import the custom functions
from func.SedCas.opt_main import load_climate_input4model, run1time_sedcas

from func.SedCas.SedCas import SedCas

from func.toolkit.physical_unit_converter import unit_converter
from func.toolkit.loss_func import calculate_pred_ratio, clean_obs_pre

from func.toolkit.archive_data import dump_as_row


def check_MCMC_process(sampler, iteration, total_steps, params_trial):

    acceptance_fraction = sampler.acceptance_fraction
    max_log_prob = np.max(sampler.get_log_prob())

    chain = sampler.get_chain(flat=False)
    mean_theta = np.mean(chain[-1, :, :], axis=0)  # last step, mean over walkers
    mean_theta_str = ", ".join([f"{x:.3f}" for x in mean_theta])

    try:
        tau = sampler.get_autocorr_time(tol=0)
        tau_mean = np.mean(tau)
        tau_std = np.std(tau)

        ESS = total_steps / tau
        ESS_mean = np.mean(ESS)

        tau_str = f"{tau_mean:.2f} ± {tau_std:.2f}"
        ESS_str = f"{ESS_mean:.1f}"
    except:
        tau_str = "None"
        ESS_str = "None"

    record = (f"{UTCDateTime.now().isoformat()}\n"
          f"Done Main sampling phase {iteration}.\n"
          f"Mean acceptance fraction: {np.mean(acceptance_fraction):.3f}\n"  # 0.2 – 0.4 is good
          f"Min: {np.min(acceptance_fraction):.3f}, "
          f"Max: {np.max(acceptance_fraction):.3f}\n"
          f"Total iterations: {sampler.iteration}\n"
          f"Max Log-Prob: {max_log_prob:.2f}\n"
          f"{mean_theta_str}\n"
          f"Autocorrelation time (mean): {tau_str}\n"
          f"Mean Effective Sample Size: {ESS_str}\n")

    output_dir = f'{params_trial["output_dir"]}'
    output_name = f"sedcas_mcmc_checkpoint"
    dump_as_row(output_dir, output_name, record)

def load_config():

    # all input params are stored here and will be updated later
    params_trial = {"project_root": current_dir.parent.parent,
                    "output_dir": f"{current_dir.parent.parent}/functions/bayesian_inference0dot4/output",
                    "df_volume_file_name": "debris_flow_volume_2004_2022.txt",
                    "data_type": "10-minutes",
                    "model_params":"SedCas_input_params_10min_bo.yaml"}
    os.makedirs(params_trial["output_dir"], exist_ok=True)

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

    return params_trial

def scale_theta(theta, params_trial, method="raw2normalized"):

    lower = params_trial["lower_bounds"]
    upper = params_trial["upper_bounds"]

    if method == "raw2normalized":
        # normalized to 0-1
        # (Value - Min) / (Max - Min)
        theta = (theta - lower) / (upper - lower)
    elif method == "normalized2raw":
        # magnify 0-1 value to real scale
        # (Normalized_Value * Range) + Min
        theta = theta * (upper - lower) + lower

    return theta

def log_prior(theta):

    if np.any(theta < 0) or np.any(theta > 1):
        value = -np.inf  # log(0) -> This parameter set is impossible. Reject immediately.
    else:
        # why constant number 0? not log[1 / (theta_1_right - theta_1_left)]?
        # acceptance ratio -> https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        # because: Any constant value cancels out in the acceptance ratio.
        value = 0.0  # log(1)

    return value

def log_likelihood(theta, params_trial, eps=1e-10, sigma=1.0):

    # back to real scale for model input
    theta = scale_theta(theta, params_trial, method="normalized2raw")

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
                            progress_bars=False, save_output=False, plot_output=False)

    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    y_pred = unit_converter(input=sed_transport_real,
                            catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    # prepare the event-wise mean volume
    y_obs_valid, y_pred_valid = clean_obs_pre(current_params_trial["y_obs"], y_pred,
                                              buffer_time=3, failed_prediction=0,
                                              ratio_of_faliure=0.01)
    y_obs = y_obs_valid["Volume[m3]"].values
    y_pred = y_pred_valid["Volume[m3]"].values

    # avoid log(0)
    y_obs = np.clip(y_obs, a_min=eps, a_max=None)
    y_pred = np.clip(y_pred, a_min=eps, a_max=None)

    residual = np.log10(y_obs) - np.log10(y_pred)
    # sigma = 1 # 1.0 in log10 space = factor of 10 in linear space
    # I expect SedCas model to predict the volume within one order of magnitude of the truth
    g_log_like = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))

    pred_ratio = calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1)

    # record
    worker_name = current_process().name
    record = (f"{UTCDateTime.now().isoformat()}\n"
              f"{worker_name}, {g_log_like}, {sigma}, \n"
              f"{', '.join(map(str, y_pred))}, \n"
              f"{', '.join(map(str, pred_ratio))}, \n"
              f"{current_theta} \n")
    output_dir = f'{current_params_trial["output_dir"]}'
    output_name = f"sedcas_mcmc_record_{worker_name}"

    dump_as_row(output_dir, output_name, record)

    return g_log_like

def log_posterior(theta, params_trial):

    # the input theta is in range [0, 1]
    lp = log_prior(theta)

    if np.isinf(lp):
        # Reject parameter sets outside the prior bounds
        posterior = -np.inf
    else:
        # Only compute likelihood if parameters are valid under the prior
        try:
            posterior = lp + log_likelihood(theta, params_trial)
        except Exception as e:
            posterior = -np.inf
            # back to real scale for model input
            theta = scale_theta(theta, params_trial, method="normalized2raw")
            print(f"{theta}, {e}")

    return posterior


if __name__ == "__main__":
    # <editor-fold desc="receive the arguments">
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--num_steps1", default=20, type=int)
    parser.add_argument("--num_steps2", default=500, type=int)
    parser.add_argument("--num_worker", default=32, type=int)
    args = parser.parse_args()

    num_steps1 = args.num_steps1
    num_steps2 = args.num_steps2
    num_worker = args.num_worker
    # endregion

    # load the needed params once
    params_trial = load_config()

    # <editor-fold desc="initial the theta">
    # this is based on the results from vdot2
    # Number of steps per walker: 1020
    # num_steps=1020, num_walkers=48, num_params=11
    # Movement rate in last 5 steps: 12.50%
    # Max log-probability: -276.02
    #
    # Parameter means: 1.390, 57.762, 50.997, 131.354, 413.266, 687.626, 1.303, 0.619, 0.296, 31.761, 3.767
    # Parameter stds: 3.616, 24.782, 26.145, 73.261, 297.072, 525.975, 0.129, 0.323, 0.128, 15.924, 2.743
    # Min-Mean-Max acceptance fraction: 0.000, 0.152, 0.184,
    guessed_theta = {'w_storage_cap0': 0.38,
                      'w_storage_cap1': 76,
                      'w_storage_cap2': 19,

                      'w_residence_time0': 70,
                      'w_residence_time1': 447,
                      'w_residence_time2': 191,

                      'ls_alpha_v': 1.20, # uniform

                      'Qdf': 0.21,
                      'max_s2w': 0.14,
                      'channel_storage_cap': 18.7,
                      'erosion_k': 1
                      }

    num_params = len(guessed_theta)
    # 32 CPUs process 32 walkers at a time → second batch of 16 walkers runs immediately after.
    num_walkers = max(2 * num_params, 32, num_worker)

    rng = np.random.default_rng(42 + os.getpid())  # local seed only for this generator
    # uniform sampling the initial theta in [-1, 1]
    initial_noise = rng.standard_normal((num_walkers, num_params))

    theta_value = np.array(list(guessed_theta.values())) # in raw range
    theta0 = scale_theta(theta_value, params_trial, method="raw2normalized")

    initial_theta = theta0 + 0.01 * initial_noise # add 5% noise to the optimal theta
    initial_theta = np.clip(initial_theta, 0, 1)
    # endregion


    filename = f"{current_dir}/sedcas_mcmc_results.h5"
    backend = emcee.backends.HDFBackend(filename)

    with Pool(processes=num_worker) as pool:
        # initial_theta  →  burn-in  →  heated_theta  →  main run  →  posterior samples
        sampler = emcee.EnsembleSampler(
            nwalkers=num_walkers,
            ndim=num_params,
            log_prob_fn=log_posterior,
            args=[params_trial],
            pool=pool,
            backend=backend
        )

        # (1) Burn-in phase
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"Start Burn-in phase.\n")
        heated_theta, _, _ = sampler.run_mcmc(initial_theta, nsteps=num_steps1, progress=False, store=True)
        print("Mean acceptance:", np.mean(sampler.acceptance_fraction))
        # sampler.reset()  # discard burn-in or not
        print(f"{UTCDateTime.now().isoformat()}\n"
              f"Done Burn-in phase.\n")

        # (2) Run the chunk
        chunk_size = 5
        current_theta = None # this tell mcmc -> continue from wherever you are right now
        total_steps = num_steps1
        for i in range(int(num_steps2 / chunk_size)):

            print(f"{UTCDateTime.now().isoformat()}\n"
                  f"Start Main sampling phase {i}.\n")

            sampler.run_mcmc(current_theta, nsteps=chunk_size, progress=False, store=True)

            total_steps = total_steps + chunk_size

            check_MCMC_process(sampler, i, total_steps, params_trial)
            print("Mean acceptance:", np.mean(sampler.acceptance_fraction))

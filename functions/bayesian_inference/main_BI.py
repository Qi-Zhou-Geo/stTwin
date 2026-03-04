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

def load_config():

    # all input params are stored here and will be updated later
    params_trial = {"project_root": current_dir.parent.parent,
                    "output_dir": f"{current_dir.parent.parent}/functions/bayesian_inference/output",
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

    return params_trial

def log_prior(theta):
    # Construct the Prior params distribution $$p(\theta)$$
    # p(\theta \mid y_{\mathrm{obs}}) \propto p( y_{\mathrm{obs}} \mid \theta) \times p(\theta),

    # (1) In theory,
    # here, the uniform distribution (also called "uninformative") priors for the params
    # mathmaticly, for a given model parameter, theta_1, it likes
    # if theta_1_left < theta_1 < theta_1_right:
    #   p(theta_1) = fixed_value = 1 / (theta_1_right - theta_1_left)
    # else:
    #   p(theta_1) = 0

    # (2) In numerical coding,
    # we take the log of the prior probability, not log(theta) itself.
    # For a uniform prior, the log of the constant probability is irrelevant,
    # because it cancels in the MCMC acceptance ratio.
    # So we just return 0.0 for valid parameters.

    (
        w_storage_cap0,
        w_storage_cap1,
        w_storage_cap2,

        w_residence_time0,
        w_residence_time1,
        w_residence_time2,

        ls_alpha_v,

        Qdf,
        max_s2w,

        channel_storage_cap,
        erosion_k,

        sigma,
    ) = theta

    if (
        0.1 <= w_storage_cap0 <= 10 and
        10 <= w_storage_cap1 <= 100 and
        10 <= w_storage_cap2 <= 100 and

        1 <= w_residence_time0 <= 288 and
        6 <= w_residence_time1 <= 1008 and
        1 <= w_residence_time2 <= 2016 and

        1.1 <= ls_alpha_v <= 3 and

        0.1 <= Qdf <= 10 and
        0.1 <= max_s2w <= 1 and

        10 <= channel_storage_cap <= 150 and
        0.1 <= erosion_k <= 10 and

        1e-5 <= sigma <= 5
    ):
        # why constant number 0? not log[1 / (theta_1_right - theta_1_left)]?
        # acceptance ratio -> https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
        # because: Any constant value cancels out in the acceptance ratio.
        value = 0.0  # log(1)
    else:
        value = -np.inf # log(0) -> This parameter set is impossible. Reject immediately.

    return value

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

def log_posterior(theta, params_trial):

    lp = log_prior(theta)
    if np.isinf(lp):
        # Reject parameter sets outside the prior bounds
        posterior = -np.inf
    else:
        # Only compute likelihood if parameters are valid under the prior
        posterior = lp + log_likelihood(theta, params_trial)

    return posterior

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

if __name__ == "__main__":
    # receive the arguments
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--num_steps1", default=20, type=int)
    parser.add_argument("--num_steps2", default=500, type=int)
    parser.add_argument("--num_worker", default=32, type=int)
    args = parser.parse_args()

    num_steps1 = args.num_steps1
    num_steps2 = args.num_steps2
    num_worker = args.num_worker


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
    lower_bounds = [0.1, 10, 10,  1,   6,    6,    1.1, 0.1, 0.1, 10, 0.1, 1e-2] # in non-log sapce
    upper_bounds = [10, 100, 100, 144, 1008, 1008, 2.0, 0.5, 0.5, 30, 10,  1e1] # in non-log sapce

    num_params = len(optuna_optimal)
    # 16 CPUs process 16 walkers at a time → second batch of 16 walkers runs immediately after.
    num_walkers = max(2 * num_params, 32, num_worker)


    rng = np.random.default_rng(42 + os.getpid())  # local seed only for this generator
    # set initial theta around the optuna value
    initial = np.array(list(optuna_optimal.values()))
    initial_theta = initial + rng.normal(loc=0.0, scale=1e-2, size=(num_walkers, num_params))
    initial_theta = np.clip(initial_theta, lower_bounds, upper_bounds)

    params_trial = load_config()

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

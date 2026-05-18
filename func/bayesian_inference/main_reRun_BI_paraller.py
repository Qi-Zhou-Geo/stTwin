#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import shutil
import argparse

import numpy as np
import emcee

from multiprocessing import Pool, current_process
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
from main_BI import load_config, log_posterior, check_MCMC_process # import all functions


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

    # load the existing results from backend
    filename = f"{current_dir}/sedcas_mcmc_results.h5"
    new_h5 = f"{current_dir}/sedcas_mcmc_results_{os.getpid()}.h5"
    shutil.copy(filename, new_h5) # copy from the last hf file
    backend = emcee.backends.HDFBackend(new_h5)

    # read sampler state from backend
    num_walkers, num_params = backend.shape
    completed_iteration = backend.iteration
    print(f"Resuming from backend: {filename}\n"
          f"Steps already completed: {completed_iteration}\n"
          f"num_walkers={num_walkers}, num_params={num_params}\n"
          f"New H5 file: {new_h5}\n")

    # start the new run
    with Pool(processes=num_worker) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=num_walkers,
            ndim=num_params,
            log_prob_fn=log_posterior,
            args=[params_trial],
            pool=pool,
            backend=backend
        )

        # (1) like the burn-in
        # last_pos shape: (nwalkers, ndim)
        last_pos, last_log_prob, last_state = backend.get_last_sample()

        # add "small noise" to re-initialize new independent chain
        rng = np.random.default_rng(42 + os.getpid())  # local seed only for this generator
        initial_noise = 1e-3 * rng.standard_normal((num_walkers, num_params))

        current_theta = np.clip(last_pos + initial_noise, 0, 1)
        # let the sampler re-run 10 steps, do not record the data
        sampler.run_mcmc(current_theta, nsteps=num_steps1, progress=False, store=False)


        # (2) Run the chunk based on existing MCMC chain
        chunk_size = 5
        current_theta = None # this tell mcmc -> continue from wherever you are right now
        total_steps = completed_iteration  # start counter from existing steps

        for i in range(int(num_steps2 / chunk_size)):

            print(f"{UTCDateTime.now().isoformat()}\n"
                  f"Start Re-run sampling phase {i}.\n")

            sampler.run_mcmc(current_theta, nsteps=chunk_size, progress=False, store=True)

            try:
                print("Mean acceptance:", np.mean(sampler.acceptance_fraction))
                tau = sampler.get_autocorr_time(tol=0)
                mean_tau = np.mean(tau)

                if total_steps > 50 * mean_tau:
                    print(f"{UTCDateTime.now().isoformat()}\n"
                          f"Chain is likely long enough for reliable statistics.")
                else:
                    print(f"{UTCDateTime.now().isoformat()}\n"
                          f"Chain too short. Aim for {int(50 * mean_tau)} steps.")

            except Exception as e:
                print(e)

            total_steps = total_steps + chunk_size
            check_MCMC_process(sampler, i, total_steps, params_trial)

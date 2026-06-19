#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-17T16:35:32
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import argparse

import numpy as np
import pandas as pd

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



if __name__ == "__main__":
    
    # region <receive the arguments>
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--model_version", type=str, default="v0dot4")
    parser.add_argument("--theta_draw_idx", type=int, default=0)
    args = parser.parse_args()

    model_version = args.model_version
    theta_draw_idx = args.theta_draw_idx # the idx in the sampled
    # endregion

    project_root, output_dir = project_root, f"{current_dir}/{model_version}"
    
    climate_input, climate_resolution = "climate_2004_2025_t.txt", 600
    model_input_params = "SedCas_input_params_10min_QZ.yaml"
    model_updated_params = "SedCas_input_params_10min_after_mcmc.yaml"
    
    
    posterior_h5_dir = Path(project_root) / "func/bayesian_inference/sedcas_mcmc_results.h5"
    burn_in_step = 100
    num_draw = 21
    
    
    # (1) preapre 2004-2025 climate forcing
    params_trial = load_config(project_root, output_dir,
                               climate_input, climate_resolution, 
                               model_input_params, model_updated_params)
    
    # (2) preapre posterior
    if theta_draw_idx == 0:
        theta = get_MAP_theta(posterior_h5_dir, burn_in_step=burn_in_step)
        fix_ls = False
        save_ls = True
        plot_output = True
    else:
        # retuen shape (num_draw, num_paramsters)
        sampled_theta = get_posterior_theta(posterior_h5_dir, num_draw=num_draw, burn_in_step=burn_in_step, fix_seed=True)
        theta = sampled_theta[theta_draw_idx, :] # select theta
        fix_ls = False
        save_ls = False
        plot_output = False
        
    # normalize it back to real scale
    lower = params_trial["lower_bounds"]
    upper = params_trial["upper_bounds"]
    theta = theta * (upper - lower) + lower 
    
    # (3) pack the params
    params_trial["output_dir"] = f'{params_trial["output_dir"]}/theta_{str(theta_draw_idx + 1).zfill(3)}'
    params_trial["fix_ls"] = fix_ls
    params_trial["save_ls"] = save_ls

    
    # (4) uppdate the params_trial for current process / thrend
    current_params_trial = params_trial.copy()

    current_theta = {}
    for theta_name, theta_value in zip(params_trial["theta_names"], theta):
        current_params_trial[theta_name] = theta_value
        current_theta[theta_name] = theta_value
    print(f"Current theta is\n: {current_theta} \n")


    # (5) run the model, this is most expensive time-consuming part
    # plot_output=True, show_plot=False, save but not automaticlt show it
    model = run_posterior_sedcas(current_params_trial, num_iteration=100,
                                 
                                 progress_bars=True, save_output=True, save_sed_container=True,
                                 
                                 fix_ls=params_trial["fix_ls"], save_ls=params_trial["save_ls"],
                                 
                                 plot_output=plot_output, show_plot=False,
                                 
                                 select_t1="2004-01-01T00:00:00", select_t2="2026-01-01T00:00:00")
    mass_balance_checker(sed_container=model.sed_container, residual=1.0, iteration=None, silence=True)

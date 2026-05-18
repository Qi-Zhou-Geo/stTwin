#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-03-01
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse

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
from func.toolkit.physical_unit_converter import unit_converter
from func.post_bayesian_inference.thin_posterior import sample_posterior
from func.post_bayesian_inference.thin_posterior import maximum_likelihood_theta

# import from the current path
from func.post_bayesian_inference.run_model_with_theta import run_sedcas_once
from func.post_bayesian_inference.run_model_with_theta import load_config, plot_ratio, save_last_status

def main(params_trial, theta, sigma=1, save_last=True):

    # uppdate the params_trial for current process / thrend
    current_params_trial = params_trial.copy()

    current_theta = {}
    for theta_name, theta_value in zip(params_trial["theta_names"], theta):
        current_params_trial[theta_name] = theta_value
        current_theta[theta_name] = theta_value
    print(f"Current theta is\n: {current_theta} \n")


    # run the model, this is most expensive time-consuming part
    # plot_output=True, show_plot=False, save but not automaticlt show it
    model = run_sedcas_once(current_params_trial, num_iteration=100,
                            progress_bars=True, save_output=True,
                            plot_output=True, show_plot=False,
                            select_t1="2004-02-01T00:00:00", select_t2="2023-01-01T00:00:00")

    sed_transport_real = model.sed_container["sed_transport_real"].copy()
    y_pred = unit_converter(input=sed_transport_real,
                            catchment_area=model.cfg.c_area.value,
                            method="area-aggregated")

    plot_ratio(current_params_trial, sigma=sigma, y_pred=y_pred)

    if save_last is True:
        save_last_status(model, current_params_trial, current_theta)

    return model

if __name__ == "__main__":
    # <editor-fold desc="receive the arguments">
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--project_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--posterior_h5_dir", type=str)
    parser.add_argument("--burn_in_step", type=int)
    args = parser.parse_args()

    project_root = args.project_root # not this is may != project root
    output_dir = args.output_dir # functions/bayesian_inference0dot2/sedcas_mcmc_results.h5
    posterior_h5_dir = args.posterior_h5_dir #pipeline/real_pred/bayesian_inference0dot2
    burn_in_step = args.burn_in_step # discard the burn-in step
    # endregion

    params_trial = load_config(project_root, output_dir, data_type="10-minutes")
    # load as normalized theta in [0, 1]
    theta = maximum_likelihood_theta(posterior_h5_dir, burn_in_step=burn_in_step)
    lower = params_trial["lower_bounds"]
    upper = params_trial["upper_bounds"]
    theta = theta * (upper - lower) + lower # normalize it back to real scale

    model = main(params_trial=params_trial, theta=theta)

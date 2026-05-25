#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import emcee
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

import corner

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
from func.bayesian_inference.params_boundary import custom_boundary
from func.SedCas_pred.thin_posterior import sample_posterior, maximum_likelihood_theta
from func.bayesian_inference.main_BI import log_posterior

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


theta_names, lower_bounds, upper_bounds = custom_boundary()
file_name = Path(project_root) / "func" / "bayesian_inference" / "sedcas_mcmc_results.h5"
burn_in = 100

# region <(1) check the Maximum A Posterior theta>
# theta is same shape as the turned theta
max_posterior_theta = maximum_likelihood_theta(posterior_results_file=file_name, burn_in_step=burn_in)
max_posterior_theta = max_posterior_theta * (upper_bounds - lower_bounds) + lower_bounds # scale to real value

max_a_posterior = {}
for theta_name, theta_value in zip(theta_names, max_posterior_theta):
    max_a_posterior[theta_name] = round(theta_value, 2)

# endregion


# region <(2) All theta>
backend = emcee.backends.HDFBackend(file_name)
chain = backend.get_chain(flat=False)  # (num_steps, num_walkers, num_params)
chain = lower_bounds + chain * (upper_bounds - lower_bounds)
chain = chain[burn_in:, :, :] # discard the burn in period
num_steps, num_walkers, num_params = chain.shape
print(f"num_steps={num_steps}, num_walkers={num_walkers}, num_params={num_params}\n")


flat_chain = chain.reshape(-1, num_params)
mean_theta = np.mean(flat_chain, axis=0)
std_theta = np.std(flat_chain, axis=0)
print("Parameter best:", ", ".join([f"{x:.3f}" for x in max_a_posterior.values()]))
print("Parameter mean:", ", ".join([f"{x:.3f}" for x in mean_theta]))
print("Parameter stds:", ", ".join([f"{x:.3f}" for x in std_theta]))
print(f"\n")
# endregion


fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 2)


keys =  ["(a) Bedrock Reservoir", "(b) Forest Upper Reservoir", "(c) Forest Lower Reservoir"]
keys_idx = [0, 1, 3]
for idx, key in enumerate(keys):
    
    ax = plt.subplot(gs[keys_idx[idx]])
    
    current_theta_pdf = flat_chain[:, idx]
    ax.hist(current_theta_pdf, bins=100, alpha=0.5, color="black", density=True)
    ax.axvline(x=max_posterior_theta[idx], color="green", lw=1, linestyle="--", label="Max A Posterior Value")
    ax.axvline(x=mean_theta[idx], color="C0", lw=1, linestyle="--", label="Mean Value")
    ax.set_xlim(lower_bounds[idx], upper_bounds[idx])

    ax.set_xlabel(f"Water Storage Capacity [Area-normalized]", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')

    ax.set_title(label=key, loc="left", fontsize=7, fontweight='bold')
    if idx == 0:
        ax.legend(loc="upper right", fontsize=6)
        ax.set_xlim(0.1, 1)

    
plt.tight_layout()
plt.savefig(f"{current_dir}/water_storage.png", dpi=600)  # , transparent=True
# plt.show()
plt.close(fig=fig)


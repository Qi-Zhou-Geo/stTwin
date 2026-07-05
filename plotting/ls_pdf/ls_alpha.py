#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

import emcee

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion


# region <add Arial font>
import platform, getpass
# Specify the directory containing the Arial font
if platform.system() == "Linux" and getpass.getuser() == "qizhou":

    from matplotlib import font_manager
    font_dirs = ['/storage/vast-gfz-hpc-01/home/qizhou/2python/font']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
# endregion

# import custom func.
from func.toolkit.physical_unit_converter import unit_converter
from func.bayesian_inference.params_boundary import custom_boundary
from func.SedCas_pred.thin_posterior import maximum_likelihood_theta

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})

burn_in = 100
hdf5_file_name = Path(project_root) / "func/bayesian_inference/sedcas_mcmc_results.h5"
backend = emcee.backends.HDFBackend(hdf5_file_name)
chain = backend.get_chain(flat=False)  # (num_steps, num_walkers, num_params)

theta_names, lower_bounds, upper_bounds = custom_boundary()
num_params = len(theta_names)

chain = lower_bounds + chain * (upper_bounds - lower_bounds)
chain = chain[burn_in:, :, :] # discard the burn in period
flat_chain = chain.reshape(-1, num_params)

min_theta = np.min(flat_chain, axis=0)
mean_theta = np.mean(flat_chain, axis=0)
max_theta = np.max(flat_chain, axis=0)
std_theta = np.std(flat_chain, axis=0)

max_posterior_theta = maximum_likelihood_theta(posterior_results_file=hdf5_file_name, burn_in_step=burn_in)
max_posterior_theta = max_posterior_theta * (upper_bounds - lower_bounds) + lower_bounds # scale to real value



fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0])

color = "black"
label = "Synthetic Landslide"
zorder = 2
alpha = 0.1

theta_idx = np.where(theta_names == 'ls_alpha_v')[0][0]
print(f"min: {min_theta[theta_idx]:.3f}, mean: {mean_theta[theta_idx]:.3f}," 
      f"max: {max_theta[theta_idx]:.3f}, std.: {std_theta[theta_idx]:.3f}")

current_theta_pdf = flat_chain[:, theta_idx]
ax.hist(current_theta_pdf, bins=50, alpha=0.5, color="black", density=True)
ax.axvline(x=max_posterior_theta[theta_idx], color="green", lw=1, linestyle="--", 
           label="Max A Posterior\n"+r"$\alpha=$"+f"{max_posterior_theta[theta_idx]:.3f}")

ax.set_xlabel(f"Power-law Exponent of Landslide Volume "+r"$\alpha$", fontweight='bold')
ax.set_ylabel("Probability Density", fontweight='bold')

ax.legend(loc="upper right", fontsize=6)
ax.set_xlim(1.10, 1.40)
ax.set_ylim(0, 9)
ax.grid(axis='both', which='major', color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)


# complementary cumulative distribution function
png_path = Path(current_dir) / f"landslide_alpha.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(png_path, dpi=600)
plt.show()
plt.close(fig)

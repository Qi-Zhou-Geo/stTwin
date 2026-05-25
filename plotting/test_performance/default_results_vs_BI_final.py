#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

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
from func.toolkit.round_timestamp import round_time
from func.toolkit.loss_func import likehood_loss, ratio_loss
from func.toolkit.physical_unit_converter import unit_converter
from func.toolkit.loss_func import calculate_pred_ratio, clean_obs_pre

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


model_verson = "bayesian_inference0dot4"

# region <(1) calaulate the loss for Jacob default>
# this result is use the default params and data
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas_default/sed_output_2004_2018.nc")

sed_transport_real = sed_output["sed_output_catchment_Q50"] # mean values
y_pred = unit_converter(input=sed_transport_real,
                        catchment_area=4.83, # unit is km**2
                        method="area-aggregated")

# select the presults
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)

y_obs_valid, y_pred_valid = clean_obs_pre(event_catalog, y_pred, buffer_time=3, failed_prediction=0, ratio_of_faliure=0)
y_obs = y_obs_valid["Volume[m3]"].values
y_pred = y_pred_valid["Volume[m3]"].values

# avoid log(0)
eps = 1e-10
y_obs = np.clip(y_obs, a_min=eps, a_max=None)
y_pred = np.clip(y_pred, a_min=eps, a_max=None)

residual = np.log10(y_obs) - np.log10(y_pred)
sigma = 1  # fixed sigma, 4.34 is σ=10 in natural log
g_log_like_JH = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
pred_ratio = calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1)
print(f"g_log_like(JH): {g_log_like_JH} ")

pred_ratio_JH = event_catalog.copy() # get the same shape
pred_ratio_JH["ratio"] = pred_ratio_JH["Volume[m3]"] # create a new row
nan_idx = np.where(np.isnan(pred_ratio_JH.iloc[:, 2]) == True)[0]
assert len(nan_idx) + len(pred_ratio) == len(event_catalog), f"print event length is not equal"
idx = np.arange(len(pred_ratio_JH))
new_idx = np.setdiff1d(idx, nan_idx)

pred_ratio_JH.iloc[new_idx, 3] = pred_ratio
# endregion


# region <(2) calaulate the loss for QZ BI optimal>
# this result is use the default params and data
sed_output = xr.open_dataset(f"{project_root}/pipeline/real_pred/{model_verson}/MAP/sed_output.nc")

sed_transport_real = sed_output["sed_transport_real_Q50"] # mean values
y_pred = unit_converter(input=sed_transport_real,
                        catchment_area=4.83, # unit is km**2
                        method="area-aggregated")

# select the presults
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)

y_obs_valid, y_pred_valid = clean_obs_pre(event_catalog, y_pred, buffer_time=3, failed_prediction=0, ratio_of_faliure=0)
y_obs = y_obs_valid["Volume[m3]"].values
y_pred = y_pred_valid["Volume[m3]"].values


# avoid log(0)
eps = 1e-10
y_obs = np.clip(y_obs, a_min=eps, a_max=None)
y_pred = np.clip(y_pred, a_min=eps, a_max=None)

residual = np.log10(y_obs) - np.log10(y_pred)
sigma = 1  # fixed sigma, 4.34 is σ=10 in natural log
g_log_like_QZ = -0.5 * np.sum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
pred_ratio = calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1)
print(f"g_log_like(JH): {g_log_like_QZ} ")

pred_ratio_QZ = event_catalog.copy() # get the same shape
pred_ratio_QZ["ratio"] = pred_ratio_QZ["Volume[m3]"] # create a new row
nan_idx = np.where(np.isnan(pred_ratio_QZ.iloc[:, 2]) == True)[0]
assert len(nan_idx) + len(pred_ratio) == len(event_catalog), f"print event length is not equal"
idx = np.arange(len(pred_ratio_QZ))
new_idx = np.setdiff1d(idx, nan_idx)

pred_ratio_QZ.iloc[new_idx, 3] = pred_ratio
# endregion


# (2) plot it
x = np.arange(len(pred_ratio_QZ))
width = 0.25

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1)
ax = plt.subplot(gs[0])

num_no_prediction_JH = 0
num_no_prediction_QZ = 0
for idx in x:

    # plot QZ after BO
    ratio = pred_ratio_QZ.iloc[idx, 3]
    if np.isnan(ratio):
        # no observed volume
        ratio = 1
        color = "gray"
        label = "No Observation"
    elif (ratio > 1e2) or (ratio < 1e-2):
        # no detection
        ratio = 1
        color = "orange"
        label = "No Volume Prediction (QZ)"
        num_no_prediction_QZ = num_no_prediction_QZ + 1
    else:
        # succssed
        color = "blue"
        label = "With Volume Prediction (QZ)"

    ax.bar(1 + idx + width/2, ratio, width, color=color, alpha=0.7, label=label, zorder=2)


    if idx >= 63:
        # from 2018 event
        pass
    else:
        # plot benchmark (only befor 2017 events)
        ratio = pred_ratio_JH.iloc[idx, 3]
        if np.isnan(ratio):
            # no observed volume
            ratio = 1
            color = "gray"
            label = "No Observation"
        elif (ratio > 1e2) or (ratio < 1e-2):
            # no detection
            ratio = 1
            color = "red"
            label = "No Volume Prediction (JH)"
            num_no_prediction_JH = num_no_prediction_JH + 1
        else:
            # succssed
            color = "C0"
            label = "With Volume Prediction (JH)"

        ax.bar(1 + idx - width/2, ratio, width, color=color, alpha=0.7, label=label, zorder=2)


ax.set_ylim(1e-2, 1e2)
ax.set_xlim(0.5, 102.5)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=3)

ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
# ax.set_xlim(0.5, 62.5) # coment it for 2017

ax.set_ylabel("Ratio of Predicted to Observed Volume", fontweight='bold')
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2022]", fontweight='bold')


# ratio QZ/JH
ax = plt.subplot(gs[1])
ax.set_title(label=f"No Volume Prediction (JH): {num_no_prediction_JH}, Log-Posterior: {g_log_like_JH:.1f}\n"
                   f"No Volume Prediction (QZ): {num_no_prediction_QZ}, Log-Posterior: {g_log_like_QZ:.1f}", fontsize=7)

for idx in x:

    ratio_JH = pred_ratio_JH.iloc[idx, 3]
    ratio_QZ = pred_ratio_QZ.iloc[idx, 3]

    if np.isnan(ratio_JH) or np.isnan(ratio_QZ):
        # no observed volume
        ratio = 1
        color = "black"
        label = "No Observation"
    else:

        if np.isinf(ratio_JH):
            # no detection
            ratio = 1
            color = "red"
            label = "No Volume Prediction (JH)"

        elif np.isinf(ratio_QZ):
            # no detection
            ratio = 1
            color = "orange"
            label = "No Volume Prediction (QZ)"

        else:
            # with detection
            ratio = ratio_QZ / ratio_JH
            color = "C0"
            label = "Volume Prediction (QZ) / Volume Prediction (JH)"

    ax.bar(idx, ratio, width, color=color, alpha=0.7, label=label, zorder=2)


ax.set_ylim(1e-2, 1e2)
ax.set_xlim(0.5, 102.5)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=1)

ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60], [1, 10, 20, 30, 40, 50, 60])

ax.set_ylabel("Prediction Error", fontweight='bold')
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2022]", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/default_results_vs_BI_1000step.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

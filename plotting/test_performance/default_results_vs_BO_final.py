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

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


# <editor-fold desc="(0) load the observed debris flow volume">
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)
event_catalog = event_catalog.iloc[:63, :]

# round the time
y_obs = event_catalog
t_s, t_e = [], []
for event_id in range(len(y_obs)):
    t1 = round_time(y_obs.iloc[event_id, 0])
    t_s.append(t1)

    t2 = round_time(y_obs.iloc[event_id, 1])
    t_e.append(t2)
y_obs.iloc[:, 0] = t_s
y_obs.iloc[:, 1] = t_e
# endregion


# <editor-fold desc="# (1) calaulate the loss for Jacob default">
# this result is use the default params and data
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas_default/sed_output_2004_2018.nc")
dfs = sed_output["sed_output_catchment_Q50"] # mean values
dfs = dfs.to_dataframe().reset_index()
y_pred = dfs[['time_str', 'sed_output_catchment_Q50']]

# convert the volume unit
catchment_area = 4.83  # unit is km^2
volume = unit_converter(input=y_pred.iloc[:, 1], catchment_area=catchment_area, method="area-aggregated")
y_pred.iloc[:, 1] = volume

# calculate the loss
predicted_gap_JH = ratio_loss(y_obs, y_pred, buffer_time=3, ratio_no_obs=np.nan, ratio_no_prediction=np.inf)
# endregion


# <editor-fold desc="# (2) calaulate the loss for QZ BO optimal">
# this result is use the default params and data
sed_output = xr.open_dataset(f"{project_root}/functions/SedCas/output/sed_output.nc")
dfs = sed_output["sed_transport_real_Q50"] # mean values
dfs = dfs.to_dataframe().reset_index()
y_pred = dfs[['time_str', "sed_transport_real_Q50"]]

# convert the volume unit
catchment_area = 4.83  # unit is km^2
volume = unit_converter(input=y_pred.iloc[:, 1], catchment_area=catchment_area, method="area-aggregated")
y_pred.iloc[:, 1] = volume

# calculate the loss
predicted_gap_QZ = ratio_loss(y_obs, y_pred, buffer_time=3, ratio_no_obs=np.nan, ratio_no_prediction=np.inf)
# endregion


# (2) plot it
x = np.arange(len(y_obs))
width = 0.25

fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1)
ax = plt.subplot(gs[0])

num_no_prediction_JH = 0
num_no_prediction_QZ = 0
for idx in x:

    # plot benchmark
    ratio = predicted_gap_JH[idx][-1]
    if np.isnan(ratio):
        # no observed volume
        ratio = 1
        color = "black"
        label = "No Observation (JH)"
    elif np.isinf(ratio):
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


    # plot QZ after BO
    ratio = predicted_gap_QZ[idx][-1]
    if np.isnan(ratio):
        # no observed volume
        ratio = 1
        color = "gray"
        label = "No Observation (QZ)"
    elif np.isinf(ratio):
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


ax.set_title(label=f"No Volume Prediction (JH): {num_no_prediction_JH}\n"
                   f"No Volume Prediction (QZ): {num_no_prediction_QZ}", fontsize=7)
ax.set_ylim(1e-2, 1e2)
ax.set_xlim(0.5, 70)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=3)

ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60], [1, 10, 20, 30, 40, 50, 60])

ax.set_ylabel("Ratio of Predicted to Observed Volume", fontweight='bold')
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')


# ratio QZ/JH
ax = plt.subplot(gs[1])
for idx in x:

    ratio_JH = predicted_gap_JH[idx][-1]
    ratio_QZ = predicted_gap_QZ[idx][-1]

    if np.isnan(ratio_JH) or np.isnan(ratio_QZ):
        # no observed volume
        ratio = 1
        color = "black"
        label = "No Observation (JH)"
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
ax.set_xlim(0.5, 70)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=6, ncol=1)

ax.set_yscale('log')
ax.grid(axis='y', color='red', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60], [1, 10, 20, 30, 40, 50, 60])

ax.set_ylabel("Prediction Error", fontweight='bold')
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/default_results_vs_BI.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

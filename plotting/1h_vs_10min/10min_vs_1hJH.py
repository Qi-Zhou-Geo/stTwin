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
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

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


print(f"Note:\n"
      f"Before you run this, please make sure:\n"
      f"(1) done MCMC, and you have the .h5 file\n"
      f"(2) done re-run the 10-min model with MAP from .h5"
      f"(3) done the 1h model with JH's default params")

#region <load the observed debris flow volume>
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


#region <default 1h JH>
sed_output = pd.read_csv(f"{project_root}/config/SedCas-main/Sediment.out", header=0)
y_pred = sed_output.loc[:, ["D", "Q50"]]

# format the date-time
date_time = []
for i in range(y_pred.shape[0]):
    t = str(y_pred.iloc[i, 0]).replace(" ", "T")
    date_time.append(t)
y_pred.iloc[:, 0] = date_time

# convert the volume unit
catchment_area = 4.83  # unit is km^2
volume = unit_converter(input=y_pred.iloc[:, 1].values, catchment_area=catchment_area, method="area-aggregated")
y_pred.iloc[:, 1] = volume

# calculate the loss
predicted_gap = ratio_loss(y_obs, y_pred, buffer_time=3, ratio_no_obs=np.nan, ratio_no_prediction=np.inf)
# save to local as txt
temp = pd.DataFrame.from_dict(predicted_gap, orient="index",
                              columns=["start_time", "end_time",
                                       "predicted_volume[m^3]", "observed_volume[m^3]", "ratio pre/obs"])

temp.to_csv(f"{current_dir}/JH_1h_original.txt", index=False, na_rep="NaN")

pred_ratio_JH = predicted_gap

# avoid log(0)
eps = 1e-10
y_obs = np.array([v[3] for v in pred_ratio_JH.values()], dtype=float)
y_obs = np.clip(y_obs, a_min=eps, a_max=None)
y_pred = np.array([v[2] for v in pred_ratio_JH.values()], dtype=float)
y_pred = np.clip(y_pred, a_min=eps, a_max=None)

residual = np.log10(y_obs) - np.log10(y_pred)
sigma = 1  # fixed sigma, 4.34 is σ=10 in natural log
g_log_like_JH = -0.5 * np.nansum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
print(f"g_log_like(JH): {g_log_like_JH} ")
# endregion



# region <(2) 10 minutes results after the MCMC>
model_verson = "v0dot4"
sed_output = xr.open_dataset(f"{project_root}/pipeline/run_2004_2025_posterior/{model_verson}/theta_001/sed_output.nc")

sed_transport_real = sed_output["sed_transport_real_Q50"] # mean values
y_pred = unit_converter(input=sed_transport_real,
                        catchment_area=4.83, # unit is km**2
                        method="area-aggregated")

# select the presults
event_catalog = pd.read_csv(f"{project_root}/"
                            f"data/event_catalog/debris_flow_volume_2004_2022.txt",
                            skiprows=6, header=0)
event_catalog = event_catalog.iloc[:63, :]

y_obs_valid, y_pred_valid = clean_obs_pre(event_catalog, y_pred, buffer_time=3, failed_prediction=0, ratio_of_faliure=0)
y_obs = y_obs_valid["Volume[m3]"].values
y_pred = y_pred_valid["Volume[m3]"].values


# avoid log(0)
eps = 1e-10
y_obs = np.clip(y_obs, a_min=eps, a_max=None) # type: ignore
y_pred = np.clip(y_pred, a_min=eps, a_max=None) # type: ignore

residual = np.log10(y_obs) - np.log10(y_pred)
sigma = 1  # fixed sigma, 4.34 is σ=10 in natural log
g_log_like_QZ = -0.5 * np.nansum((residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2))
print(f"g_log_like(QZ): {g_log_like_QZ} ")
pred_ratio = calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1)


pred_ratio_QZ = event_catalog.copy() # get the same shape
pred_ratio_QZ["ratio"] = pred_ratio_QZ["Volume[m3]"] # create a new row
nan_idx = np.where(np.isnan(pred_ratio_QZ.iloc[:, 2]) == True)[0]
assert len(nan_idx) + len(pred_ratio) == len(event_catalog), f"print event length is not equal"
idx = np.arange(len(pred_ratio_QZ))
new_idx = np.setdiff1d(idx, nan_idx)

pred_ratio_QZ.iloc[new_idx, 3] = pred_ratio
pred_ratio_QZ.to_csv(f"{current_dir}/QZ_10min_mcmc.txt", index=False, na_rep="NaN")
# endregion



# (2) plot it
ratio_QZ = np.array(pred_ratio_QZ.iloc[idx, 3].values, dtype=float)
ratio_JH = np.array([v[-1] for v in pred_ratio_JH.values()], dtype=float)

x = np.arange(len(pred_ratio_QZ))
width = 0.25

subplot_idx = ["(a)", "(b)"]
fig = plt.figure(figsize=(6, 6))
gs = gridspec.GridSpec(2, 1)


for subplot_id, tolerance in enumerate([1e2, 1e1]):
    ax = plt.subplot(gs[subplot_id])


    num_no_prediction_JH = 0
    num_no_prediction_QZ = 0
    for idx in x:

        if idx < 63:
            
            # plot QZ‘s results after MCMC
            ratio = ratio_QZ[idx]
            if np.isnan(ratio):
                # no observed volume
                ratio = 1
                color = "gray"
                label = "No Observation"
            elif (ratio > tolerance) or (ratio < 1 / tolerance):
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


            # plot benchmark (only befor 2017 events)
            ratio = ratio_JH[idx]
            if np.isnan(ratio):
                # no observed volume
                ratio = 1
                color = "gray"
                label = "No Observation"
            elif (ratio > tolerance) or (ratio < 1 / tolerance):
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

        else:
            # plot QZ‘s results after MCMC
            ratio = ratio_QZ[idx]
            if np.isnan(ratio):
                # no observed volume
                ratio = 1
                color = "gray"
                label = "No Observation"
            elif (ratio > tolerance) or (ratio < 1 / tolerance):
                # no detection
                ratio = 1
                color = "orange"
                label = "No Volume Prediction (QZ)"
            else:
                # succssed
                color = "blue"
                label = "With Volume Prediction (QZ)"

            ax.bar(1 + idx + width/2, ratio, width, color=color, alpha=0.7, label=label, zorder=2)

    if subplot_id == 0:
        space = "  "
    else:
        space = ""
        
    ax.set_title(label=(
        rf"$\bf{{{subplot_idx[subplot_id]}}}$"
        f" {1 / tolerance} <= threshold <= {tolerance}\n"
        f"No Volume Prediction (JH):  {num_no_prediction_JH}, {space}Log-Posterior: {g_log_like_JH:.1f}\n"
        f"No Volume Prediction (QZ): {num_no_prediction_QZ}, Log-Posterior: {g_log_like_QZ:.1f}"), fontsize=7, loc="left")

    ax.set_ylim(1e-2, 1e2)
    ax.set_xlim(0.5, 63.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ordered_labels = [
        "With Volume Prediction (QZ)",   # row1, col1
        "With Volume Prediction (JH)",   # row2, col1
        "No Volume Prediction (QZ)",     # row1, col2
        "No Volume Prediction (JH)",     # row2, col2
        "No Observation",                # row1, col3
        "",                              # row2, col3 (empty placeholder)
    ]
    ordered_handles = [by_label.get(l, plt.matplotlib.patches.Patch(visible=False)) for l in ordered_labels] # type: ignore
    ax.legend(ordered_handles, ordered_labels, fontsize=6, ncol=3) # type: ignore


    ax.set_yscale('log')
    ax.grid(axis='y', color='black', linestyle='--', lw=0.7, alpha=0.5, zorder=1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 63],
                [1, 10, 20, 30, 40, 50, 60, 63]) # type: ignore
    # ax.set_xlim(0.5, 62.5) # coment it for 2017

    ax.set_ylabel("Ratio of Predicted to Observed Volume", fontweight='bold')
    ax.set_xlabel("", fontweight='bold')

ax = plt.subplot(gs[1])
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')


plt.tight_layout()
plt.savefig(f"{current_dir}/compare_1h_vs_10min.png", dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

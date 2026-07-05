#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-05T12:13:30
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

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


# use this xr dataset template
model_verson = "v0dot4"
sed_output = xr.open_dataset(f"{project_root}/pipeline/run_2004_2025_posterior/{model_verson}/theta_001/sed_output.nc")
sed_transport_real = sed_output["sed_transport_real_Q50"] # mean values


# (1) stastic
# region < Fix MAP, use 1-100 ls, UQ along the ls stochastic dimension > 
cache_name = "fix_MAP__use_ls_1-100"
key = "sed_transport_real"

cache_path = Path(current_dir) / f"cache/{cache_name}_uq_in_{key}.npz"
cache_data = np.load(cache_path, allow_pickle=True)
q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]


assert sed_transport_real.shape == q50.shape, f"Warning! Data shape is not equal."
y_pred = sed_transport_real.copy() # use Q50
y_pred.values = q50


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
pred_ratio_QZ.to_csv(f"{current_dir}/cache/{cache_name}_{key}_compare_obs.txt", index=False, na_rep="NaN")
# endregion


# region < Fix ls = 1 or 50 or 100, use 1-21 posterior, UQ along the MCMC dimension  > 
cache_name = "fix_ls__use_MCMC_1-21"
key = "sed_transport_real"

for ls_idx in [0, 49, 99]:
    cache_path = Path(current_dir) / f"cache/{cache_name}_uq_in_{key}_{ls_idx}.npz"
    cache_data = np.load(cache_path, allow_pickle=True)
    q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]


    assert sed_transport_real.shape == q50.shape, f"Warning! Data shape is not equal."
    y_pred = sed_transport_real.copy() # use Q50
    y_pred.values = q50


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
    pred_ratio_QZ.to_csv(f"{current_dir}/cache/{cache_name}_{key}_compare_obs_{ls_idx}.txt", index=False, na_rep="NaN")
# endregion


# region < use 1-100 ls, use 1-21 posterior, UQ along the ls stochastic and MCMC dimension  > 
cache_name = "use_ls_1-100__use_MCMC_1-21"
key = "sed_transport_real"


cache_path = Path(current_dir) / f"cache/{cache_name}_uq_in_{key}.npz"
cache_data = np.load(cache_path, allow_pickle=True)
q05, q50, q95 = cache_data["q05"], cache_data["q50"], cache_data["q95"]


assert sed_transport_real.shape == q50.shape, f"Warning! Data shape is not equal."
y_pred = sed_transport_real.copy() # use Q50
y_pred.values = q50


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
pred_ratio_QZ.to_csv(f"{current_dir}/cache/{cache_name}_{key}_compare_obs.txt", index=False, na_rep="NaN")
# endregion



# (2) plot it
x = np.arange(len(pred_ratio_QZ))
width = 0.25
subplot_idx = ["(a)", "(b)", "(c)", "(d)"]
tolerance = 1e1

df0 = pd.read_csv(f"{current_dir}/cache/fix_MAP__use_ls_1-100_sed_transport_real_compare_obs.txt", header=0)
ratio_df0 = df0["ratio"].to_numpy().copy()

mask_nan = np.isnan(ratio_df0)
mask_exceed_tolerance = ( (ratio_df0 > tolerance) | (ratio_df0 < 1 / tolerance) ) # use "or" logic
num_exceed_df0 = np.sum(mask_exceed_tolerance)


colors_df0 = np.full(ratio_df0.shape, "C0", dtype=object) # within 1 / tolerance <= ratio <= tolorance
colors_df0[mask_nan] = "gray" # no field obs. volume
colors_df0[mask_exceed_tolerance] = "orange" # ratio < 1 / tolerance, or,  ratio > tolorance


ratio_df0[mask_nan] = 1 # replace the nan as 1, this will be plotted as gary
ratio_df0[mask_exceed_tolerance] = 1 # replace the exceed_tolerance as 1, this will be plotted as red


data_name = ["fix_ls__use_MCMC_1-21_sed_transport_real_compare_obs_0.txt",
             "fix_ls__use_MCMC_1-21_sed_transport_real_compare_obs_49.txt",
             "fix_ls__use_MCMC_1-21_sed_transport_real_compare_obs_99.txt",
             "use_ls_1-100__use_MCMC_1-21_sed_transport_real_compare_obs.txt"]
label_list = [
    "Fixed LS #1 (MAP + 20 posterior samples)",
    "Fixed LS #50 (MAP + 20 posterior samples)",
    "Fixed LS #100 (MAP + 20 posterior samples)",
    "100 stochastic LS (MAP + 20 posterior samples)",
]
fig = plt.figure(figsize=(6, 8))
gs = gridspec.GridSpec(4, 1)


for idx, name in enumerate(data_name):
    ax = plt.subplot(gs[idx])

    df = pd.read_csv(f"{current_dir}/cache/{name}", header=0)
    ratio_df = df["ratio"].to_numpy().copy()

    mask_nan = np.isnan(ratio_df)
    mask_exceed_tolerance = ( (ratio_df > tolerance) | (ratio_df < 1 / tolerance) ) # use "or" logic
    num_exceed_df = np.sum(mask_exceed_tolerance)

    colors_df = np.full(ratio_df.shape, "blue", dtype=object) # within 1 / tolerance <= ratio <= tolorance
    colors_df[mask_nan] = "gray" # no field obs. volume
    colors_df[mask_exceed_tolerance] = "red" # ratio < 1 / tolerance, or,  ratio > tolorance

    ratio_df[mask_nan] = 1 # replace the nan as 1, this will be plotted as gary
    ratio_df[mask_exceed_tolerance] = 1 # replace the exceed_tolerance as 1, this will be plotted as red


    ax.bar(1 + x - width/2, ratio_df0, width, color=colors_df0, alpha=0.7, label="MAP", zorder=2)
    ax.bar(1 + x + width/2, ratio_df,  width, color=colors_df,  alpha=0.7, label="Current Case", zorder=2)

    
    ax.set_ylim(1e-2, 1e2)
    ax.set_xlim(0.5, 63.5)
    
    ax.set_title(label=f"{subplot_idx[idx]} {label_list[idx]}", loc="left", fontsize=7, fontweight='bold')
    ax.text(x=1, y=30, s=(f"No Volume from Benchmark: {num_exceed_df0}\n"
                          f"No Volume from Scenario: {num_exceed_df}"), fontsize=6)

    ax.set_yscale('log')
    ax.grid(axis='y', color='black', linestyle='--', lw=0.7, alpha=0.5, zorder=1)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 63], [1, 10, 20, 30, 40, 50, 60, 63]) # type: ignore

    ax.set_ylabel("", fontweight='bold') #Ratio of Predicted to Observed Volume"
    ax.set_xlabel("", fontweight='bold')

ax = plt.subplot(gs[0])
legend_handles = [
    Patch(facecolor="C0",     alpha=0.7, label="With Volume (Benchmark)"),
    Patch(facecolor="orange", alpha=0.7, label="No Volume (Benchmark)"),
    Patch(facecolor="none", edgecolor="none", label=""),
    
    Patch(facecolor="blue",   alpha=0.7, label="With Volume (Scenario)"),
    Patch(facecolor="red",    alpha=0.7, label="No Volume (Scenario)"),
    Patch(facecolor="gray",   alpha=0.7, label="No Observation"),
    
]
ax.legend(handles=legend_handles, ncol=2, loc="upper right", fontsize=6)

ax = plt.subplot(gs[3])
ax.set_xlabel("Debris Flow Event Index [from 2004 to 2017]", fontweight='bold')

fig.text(x=0, y=0.5, s='Ratio of Predicted to Observed Volume', weight='bold', va='center', rotation='vertical')

plt.tight_layout()
png_path = Path(current_dir) / f"compare_sed_yield.png"
png_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(png_path, dpi=600)  # , transparent=True
plt.show()
plt.close(fig=fig)

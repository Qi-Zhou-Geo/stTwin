#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-14
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import argparse

import pandas as pd
import numpy as np
from scipy.stats import gamma, norm

from obspy import UTCDateTime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 7,
                    'axes.formatter.limits': (-3, 6),
                    'axes.formatter.use_mathtext': True})

# region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion


def load_data():
    # load the data
    data_path = f"{project_root}/data/SedCas_input/prep_1931_2026_m.txt"
    df = pd.read_csv(data_path, header=0, skiprows=5)

    # "reference_timestamp" is UTC+0 time
    time_stamps = [UTCDateTime(ts) for ts in df["reference_timestamp"]]
    # 'rre150m0' is monthly precp with unit [mm per month]
    precip = df["rre150m0"].values

    print(f"Load data from: {data_path}\n"
          f"Data start time: {df.iloc[0, 1]}\n"
          f"Data end time: {df.iloc[-1, 1]}\n\n")
    
    return time_stamps, precip

def extract_months(time_stamps):

    out =  np.array([t.month for t in time_stamps])

    return out

def compute_rolling_precip(precip, spi_scale):

    if spi_scale == 1:
        out = precip.copy()
    else:
        num_t = len(precip)
        out = np.full(num_t, np.nan)

        for t in range(spi_scale - 1, num_t):
            out[t] = np.sum(precip[t - spi_scale + 1:t + 1])

    return out

def fit_gamma(precip_sum, months):

    # only for montly SPI
    shape = np.full(12, np.nan)
    loc = np.full(12, np.nan)
    scale = np.full(12, np.nan)

    for m in range(1, 13):
        idx = (months == m)
        data = precip_sum[idx]

        data = data[~np.isnan(data)]
        data = data[data > 0]

        if len(data) < 10:
            continue

        shape[m - 1], loc[m - 1], scale[m - 1] = gamma.fit(data, floc=0)

    return shape, loc, scale

def compute_spi(precip_sum, months, shape, loc, scale):

    spi = np.full(len(precip_sum), np.nan)

    for t in range(len(precip_sum)):

        x = precip_sum[t]
        if np.isnan(x):
            continue

        m = months[t]
        a = shape[m - 1]
        l = loc[m - 1]
        s = scale[m - 1]

        if np.isnan(a):
            continue

        # Gamma CDF
        if x <= 0:
            cdf = 0.0
        else:
            cdf = gamma.cdf(x, a, loc=l, scale=s)

        # numerical stability
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)

        # pass gamma and get the nornmal distribution
        spi[t] = norm.ppf(cdf)

    return spi

def SPI_pipeline(spi_scale):

    assert spi_scale == 1, f"Warning!\nCurrent framework only support monthly SPI.\nYour spi_scale is {spi_scale}."

    time_stamps, precip = load_data()
    months = extract_months(time_stamps) # retun as list of months index, 1, 2, 3, ..., 12, 1, 2, ...
    precip_sum = compute_rolling_precip(precip, spi_scale) # retuen as the list of monthly sum
    print(f"max precip_sum: {np.nanmax(precip_sum)} \n"
          f"mean precip_sum: {np.nanmean(precip_sum)} \n"
          f"min precip_sum: {np.nanmin(precip_sum)} \n")

    
    params = fit_gamma(precip_sum, months)
    shape, loc, scale = params
    spi = compute_spi(precip_sum, months, shape, loc, scale)

    print(f"Gammar func params:\n" 
        f"spi.shape: {spi.shape}\n"
        f"shape.shape: {shape.shape}\n"
        f"loc.shape: {loc.shape}\n"
        f"scale.shape: {scale.shape}\n")

    out_file = f"{current_dir}/spi_params_spi_month={spi_scale}.npz"
    print(f"out_file: {out_file}\n")
    dump_SPI(out_file, params, mode="save")
    
    return spi, (shape, loc, scale)

def dump_SPI(out_file, params, mode="save"):

    if mode == "save":
        shape, loc, scale = params
        np.savez(out_file, shape=shape, loc=loc, scale=scale)
    elif mode == "load":
        data = np.load(out_file)
        shape, loc, scale = data["shape"], data["loc"], data["scale"]

        return shape, loc, scale

    else:
        raise ValueError(f"Unknown mode {mode}")



def classify_spi(spi):

    # based on:
    # https://drought.emergency.copernicus.eu/data/factsheets/factsheet_spi.pdf

    if spi >= 2.0:
        status = "extremely wet"
    elif spi >= 1.5:
        status = "very wet"
    elif spi >= 1.0:
        status = "moderately wet"

    elif spi >= -1.0:
        status = "normal precipitation"
    elif spi >= -1.5:
        status = "moderately dry"
    elif spi >= -2.0:
        status = "very dry"
    else:
        status = "extremely dry"

    return status

def SPI_usage(precp_montly_total, month, spi_scale):

    spi_params = Path(current_dir) / f"spi_params_spi_month={spi_scale}.npz"
    if spi_params.exists():
        pass
    else:
        # recalculate the SPI params if it is not exist
        SPI_pipeline(spi_scale)
    
    shape, loc, scale = dump_SPI(spi_params, params=None, mode="load")

    # the previous is vector, now we select the right values based on month index
    shape, loc, scale = shape[month - 1], loc[month - 1], scale[month - 1]

    if precp_montly_total <= 0:
        raise ValueError(f"precp_montly_total = {precp_montly_total} can not be negative.")
    else:
        cdf = gamma.cdf(precp_montly_total, shape, loc=loc, scale=scale)

    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    spi_value = norm.ppf(cdf)
    status = classify_spi(spi_value)

    
    print(f"\nThe SPI for\n"
        f"<month={month}>\n"
        f"<precp_montly_total={precp_montly_total}>\n"
        f"SPI value: {spi_value}>\n"
        f"SPI status: {status}")

    return spi_value, status

def plot_SPI_boundary(spi_scale, months = np.arange(1, 13), max_precp=350):

    # histrocial dry and wet year
    data_path = f"{project_root}/data/SedCas_input/prep_1931_2026_y.txt"
    df2 = pd.read_csv(data_path, header=0, skiprows=5)
    time_str = df2.iloc[:, 1].values
    data = df2.iloc[:, 2].values
    dry_year_id = np.where(data == np.min(df2.iloc[:, 2]))[0][0]
    dry_year = time_str[dry_year_id]
    wet_year_id = np.where(data == np.max(df2.iloc[:, 2]))[0][0]
    wet_year = time_str[wet_year_id]
    

    # load the data
    data_path = f"{project_root}/data/SedCas_input/prep_1931_2026_m.txt"
    df = pd.read_csv(data_path, header=0, skiprows=5)
    time_str = df.iloc[:, 1].values
    clip_time_str = np.array([ts[5:] for ts in time_str])
    monthly_total = df.iloc[:, 2].values
    
    mean_monthly_total = [] # mean values of monthly precp total, unit by mm
    for m in months:
        idx = np.where(clip_time_str == f"{str(m).zfill(2)}-01T00:00:00")[0]
        mean_values = np.mean(monthly_total[idx])
        mean_monthly_total.append(mean_values)

    # find the exact values
    id1 = np.where(time_str == dry_year)[0][0]
    id2 = id1 + 12
    dry_year_precp = monthly_total[id1:id2]
    
    id1 = np.where(time_str == wet_year)[0][0]
    id2 = id1 + 12
    wet_year_precp = monthly_total[id1:id2]

    spi_params = Path(current_dir) / f"spi_params_spi_month={spi_scale}.npz"
    shape, loc, scale = dump_SPI(spi_params, params=None, mode="load")


    spi_status = {
        "Extremely Dry (SPI ≤ -2.0)" : (-4.0, -2.0),
        "Very Dry (-2.0 < SPI ≤ -1.5)" : (-2.0, -1.5),
        "Moderately Dry (-1.5 < SPI ≤ -1.0)" : (-1.5, -1.0),
        "Near Normal (-1.0 < SPI ≤ 1.0)" : (-1.0, 1.0),
        "Moderately Wet  (1.0 < SPI ≤ 1.5)" : (1.0, 1.5),
        "Very Wet  (1.5 < SPI ≤ 2.0)" : (1.5, 2.0),
        "Extremely Wet  (2.0 < SPI)" : (2.0, 4.0),
    }

    color_alpha = [("C3", 0.9), ("C3", 0.6), ("C3", 0.3),
                   ("black", 0.1),
                   ("C0", 0.3), ("C0", 0.6), ("C0", 0.9),]
    
    
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])
    
    ax.plot(months, mean_monthly_total, lw=1.5, ls="--", color="black", zorder=3,
            label=f"Observed Mean Monthly Total Precipitation\n"
            f"(Jan. 1931 – Apr 2026; MeteoSwiss Montana Station)")
    
    # ax.plot(months, dry_year_precp, lw=1.5, ls="--", color="C1", zorder=3,
    #     label=f"Driest Year: {dry_year[:4]} (Yearly Total Precp. = {np.sum(dry_year_precp)} mm)")
    
    # ax.plot(months, wet_year_precp, lw=1.5, ls="--", color="C2", zorder=3,
    #     label=f"Wettest Year: {wet_year[:4]} (Yearly Total Precp. = {np.sum(wet_year_precp)} mm)")
    
    df_boundary = [months.reshape(-1, 1)]
    
    for idx, (label, spi_val) in enumerate(spi_status.items()):
        boundary = []
        for b in spi_val:
            pro = norm.cdf(b)
            precp = gamma.ppf(pro, a=shape, loc=loc, scale=scale)
            precp = np.clip(precp, 0, None)
            boundary.append(precp)
        
        low_b, high_b = boundary[0], boundary[1]
        color, alpha = color_alpha[idx]
        
        if idx == 0:
            y = high_b
            ax.plot(months, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=months, y1=np.full(shape=len(months), fill_value=0), y2=y, color=color, alpha=alpha, label=label, zorder=2) 
        elif idx == 6:
            y = low_b
            ax.plot(months, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=months, y1=y, y2=np.full(shape=len(months), fill_value=max_precp), color=color, alpha=alpha, label=label, zorder=2)
        else:
            y = high_b
            ax.plot(months, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=months, y1=low_b, y2=high_b, color=color, alpha=alpha, label=label, zorder=2)

        df_boundary.append(y.reshape(-1, 1))
        
    ax.set_xlim(1, 12)
    ax.set_ylim(0, max_precp)
    ax.set_xticks(months, months)
    
    ax.legend(loc="upper right", fontsize=6)
    
    ax.set_xlabel("Month", fontweight='bold')
    ax.set_ylabel("Monthly Total Precipitation [mm]", fontweight='bold')
    ax.grid(axis='both', which="both", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/spi_boundary_spi_month={spi_scale}.png", dpi=600)
    plt.show()
    plt.close(fig=fig)

    # save to local
    df_boundary = pd.DataFrame(np.hstack(df_boundary), columns=["month", 
                                                               "SPI=-2.0", "SPI=-1.5", "SPI=-1.0", 
                                                               "SPI=+1.0", "SPI=+1.5", "SPI=+2.0",
                                                               "SPI=+4.0"])
    df_boundary.to_csv(f"{current_dir}/SPI_monthly_boundary.txt", index=False)


def main(spi_scale=1):

    # (1) calculate the SPI params and dump it for further usage
    SPI_pipeline(spi_scale)

    # (2) usage example, e.g., 
    precp_montly_total = 50 # unit by mm per month
    month = 4 # april
    spi_value, status = SPI_usage(precp_montly_total, month, spi_scale)

    # (3) test the boundary
    plot_SPI_boundary(spi_scale)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spi_scale", type=int, default=1)
    args = parser.parse_args()

    main(args.spi_scale)

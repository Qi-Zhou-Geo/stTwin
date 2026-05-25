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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

    
plt.rcParams.update({'font.size': 7,
                    'axes.formatter.limits': (-3, 6),
                    'axes.formatter.use_mathtext': True})


def load_data():
    # load the data
    data_path = f"{project_root}/data/SedCas_input/climate_1931_2025_d.txt"
    df = pd.read_csv(data_path, header=0)

    # "reference_timestamp" is UTC+0 time
    time_stamps = [UTCDateTime(ts) for ts in df["timestamp [UTC+0]"]]
    # 'precipitation [mm per time_step]' is daily total precp with unit [mm per day]
    precp = df["precipitation [mm per time_step]"].values

    print(f"Load data from: {data_path}\n"
          f"Data start time: {df.iloc[0, 1]}\n"
          f"Data end time: {df.iloc[-1, 1]}\n\n")
    
    return time_stamps, precp

def compute_rolling_precp(time_stamps, precp, spi_scale=30):

    filtered_time = []
    filtered_precp = []


    # remove Feb 29
    for ts, p in zip(time_stamps, precp):

        if ts.month == 2 and ts.day == 29:
            pass
        else:
            filtered_time.append(ts)
            filtered_precp.append(p)
    filtered_time = np.array(filtered_time)
    filtered_precp = np.array(filtered_precp)


    num_data = len(filtered_precp)

    # Rolling precipitation sums
    precip_roll = np.full(num_data, np.nan) # total sum in the last "spi_scale" window
    for i in range(spi_scale - 1, num_data):
        precip_roll[i] = np.sum(filtered_precp[i - spi_scale + 1 : i + 1])


    # Day-of-year (1–365)
    doy = np.full(num_data, np.nan, dtype=int)
    for i, ts in enumerate(filtered_time):

        doy_i = ts.julday

        # leap year correction after Feb 28
        if ts.year % 4 == 0:
            # century years (…1800, 1900, 2100, 2200…) are NOT leap years 
            # unless divisible by 400
            leap = (
                (ts.year % 100 != 0)
                or
                (ts.year % 400 == 0)
            )

            if leap and ts.month > 2:
                doy_i = doy_i - 1

        doy[i] = doy_i

    return filtered_time, precip_roll, doy

def fit_gamma(precip_roll, doy, spi_scale=30, num_days=365):

    shape = np.full(num_days, np.nan)
    loc = np.full(num_days, np.nan)
    scale = np.full(num_days, np.nan)

    for day in range(1, num_days + 1):
        idx = (doy == day)
        data = precip_roll[idx]

        data = data[~np.isnan(data)]
        data = data[data > 0]

        if len(data) < spi_scale:
            # not has enough data
            pass
        else:
            shape[day - 1], loc[day - 1], scale[day - 1] = gamma.fit(data, floc=0)

    return shape, loc, scale

def compute_spi(precip_roll, doy, shape, loc, scale):

    spi = np.full(len(precip_roll), np.nan)

    for t in range(len(precip_roll)):

        x = precip_roll[t]
        if np.isnan(x):
            continue

        day = doy[t]
        a = shape[day - 1] # shape parameter that controls skewness
        l = loc[day - 1] # location parameter (usually fixed to 0 in SPI)
        s = scale[day - 1] # scale parameter that controls spread

        if np.isnan(a):
            continue

        # Gamma CDF
        if x <= 0:
            cdf = 0.0
        else:
            cdf = gamma.cdf(x, a, loc=l, scale=s)

        # keep numerical stability
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)

        # pass gamma and get the nornmal distribution
        spi[t] = norm.ppf(cdf)

    return spi


def SPI_pipeline(spi_scale):


    time_stamps, precp = load_data()
    filtered_time, precip_roll, doy = compute_rolling_precp(time_stamps, precp, spi_scale)
    print(f"max precip_roll: {np.nanmax(precip_roll)} mm\n"
          f"mean precip_roll: {np.nanmean(precip_roll)} mm\n"
          f"min precip_roll: {np.nanmin(precip_roll)} mm\n")

    params = fit_gamma(precip_roll, doy, spi_scale=spi_scale, num_days=365)
    shape, loc, scale = params
    spi = compute_spi(precip_roll, doy, shape, loc, scale)

    print(f"Gammar func params:\n" 
        f"spi.shape: {spi.shape}\n"
        f"shape.shape: {shape.shape}\n"
        f"loc.shape: {loc.shape}\n"
        f"scale.shape: {scale.shape}\n")

    out_file = f"{current_dir}/spi_params_daily_spi_day={spi_scale}.npz"
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
        status = "normal precpitation"
    elif spi >= -1.5:
        status = "moderately dry"
    elif spi >= -2.0:
        status = "very dry"
    else:
        status = "extremely dry"

    return status

def SPI_usage(precp_spi_scale_total, day_of_year, spi_scale):

    spi_params = Path(current_dir) / f"spi_params_daily_spi_day={spi_scale}.npz"
    if spi_params.exists():
        pass
    else:
        # recalculate the SPI params if it is not exist
        SPI_pipeline(spi_scale)
    
    shape, loc, scale = dump_SPI(spi_params, params=None, mode="load")

    # the previous is vector, now we select the right values based on month index
    shape, loc, scale = shape[day_of_year - 1], loc[day_of_year - 1], scale[day_of_year - 1]

    if precp_spi_scale_total <= 0:
        raise ValueError(f"precp_spi_scale_total = {precp_spi_scale_total} can not be negative.")
    else:
        cdf = gamma.cdf(precp_spi_scale_total, shape, loc=loc, scale=scale)

    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
    spi_value = norm.ppf(cdf)
    status = classify_spi(spi_value)

    
    print(f"\nThe SPI for\n"
        f"<day_of_year={day_of_year}>\n"
        f"<precp_spi_scale_total={precp_spi_scale_total}>\n"
        f"SPI value: {spi_value}>\n"
        f"SPI status: {status}")

    return spi_value, status



def plot_SPI_boundary_full(spi_scale, day_of_year=np.arange(1, 365 + 1), max_precp=350):

    # load the data
    time_stamps, precp = load_data()
    filtered_time, precip_roll, doy = compute_rolling_precp(time_stamps, precp, spi_scale)
    
    mean_daily_total_in_last_spi_scale = []
    for day in day_of_year:
        idx = np.where(doy == day)[0]
        value = precip_roll[idx]
        mean_daily_total_in_last_spi_scale.append(np.nanmean(value))
    print(f"np.sum(mean_daily_total_in_last_spi_scale): {np.sum(mean_daily_total_in_last_spi_scale)}")

    spi_params = Path(current_dir) / f"spi_params_daily_spi_day={spi_scale}.npz"
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
                   ("black", 0.3),
                   ("C0", 0.3), ("C0", 0.6), ("C0", 0.9),]

    df_boundary = [day_of_year.reshape(-1, 1)]

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])


    ax.plot(day_of_year, mean_daily_total_in_last_spi_scale, lw=1.5, ls="--", color="black", zorder=3,
            label=f"Mean Daily Total Precipitation in Last {spi_scale} Days\n"
            f"(Jan. 1931 – Dec. 2025; MeteoSwiss Montana Station)")
    
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
            ax.plot(day_of_year, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=day_of_year, y1=np.full(shape=len(day_of_year), fill_value=0), y2=y, color=color, alpha=alpha, label=label, zorder=2) 
        elif idx == 6:
            y = low_b
            ax.plot(day_of_year, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=day_of_year, y1=y, y2=np.full(shape=len(day_of_year), fill_value=max_precp), color=color, alpha=alpha, label=label, zorder=2)
        else:
            y = high_b
            ax.plot(day_of_year, y, lw=1, color=color, alpha=alpha, zorder=3)
            ax.fill_between(x=day_of_year, y1=low_b, y2=high_b, color=color, alpha=alpha, label=label, zorder=2)

        df_boundary.append(y.reshape(-1, 1))

    ax.set_xlim(1, 365)
    ax.set_ylim(0, max_precp)
    
    x_ticks = []
    for month in range(1, 13):
        # only plot from odd month to even month
        julday1 = UTCDateTime(year=2023, month=month, day=1).julday
        x_ticks.append(julday1)
    x_ticks.append(365)
    ax.set_xticks(x_ticks, x_ticks)

    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlabel("Day of Year", fontweight='bold')
    ax.set_ylabel(f"Rolling Total Precipitation in Last {spi_scale} Days [mm]", fontweight='bold')
    ax.grid(axis='both', which="both", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/spi_boundary_daily_spi_day={spi_scale}.png", dpi=600)
    plt.show()
    plt.close(fig=fig)

    df_boundary = pd.DataFrame(np.hstack(df_boundary), columns=["month", 
                                                                "SPI=-2.0", "SPI=-1.5", "SPI=-1.0", 
                                                                "SPI=+1.0", "SPI=+1.5", "SPI=+2.0", "SPI=+4.0"])
    df_boundary.to_csv(f"{current_dir}/SPI_daily_boundary.txt", index=False)


def plot_spi_boundary(df_boundary, p_syn=None, p_obs=None, spi_scale=30, max_precp=500):
    """
    Plot the SPI boundary
    
    Args:
        df_boundary (data frame): _description_
        p_syn (1D vector): _description_. Defaults to None.
        p_obs (1D vector): _description_. Defaults to None.
        spi_scale (int, optional): _description_. Defaults to 30.
        max_precp (int, optional): _description_. Defaults to 350.

    Returns:
        _type_: _description_
    """
    
    
    day_of_year = df_boundary.iloc[:, 0].values
    spi_cols = df_boundary.columns[1:]

    color_alpha = [
        ("C3", 0.9, "Extremely Dry (SPI ≤ -2.0)"), 
        ("C3", 0.6, "Very Dry (-2.0 < SPI ≤ -1.5)"), 
        ("C3", 0.3, "Moderately Dry (-1.5 < SPI ≤ -1.0)"),
        
        ("black", 0.3, "Near Normal (-1.0 < SPI ≤ 1.0)"),
        
        ("C0", 0.3, "Moderately Wet  (1.0 < SPI ≤ 1.5)"),
        ("C0", 0.6, "Very Wet  (1.5 < SPI ≤ 2.0)"),
        ("C0", 0.9, "Extremely Wet  (2.0 < SPI)"),
    ]

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])
    ax = plt.subplot(gs[0])
    car_ax = plt.subplot(gs[1])
    
    # Optional: plot p_syn
    if p_syn is not None:
        ax.plot(day_of_year, p_syn,
                lw=1, ls="--", color="black",
                label="Synthetic", zorder=4)

    if p_obs is not None:
        ax.plot(day_of_year, p_obs,
                lw=1, ls="--", color="C1",
                label=(f"Mean Daily Total Precipitation\nin Last {spi_scale} Days\n"
                       f"(Jan. 1931 – Dec. 2025;\nMeteoSwiss Montana Station)"), 
                zorder=4)


    # SPI boundaries
    for idx, col in enumerate(spi_cols):
        y = df_boundary[col].values
        color, alpha, label = color_alpha[idx]

        ax.plot(day_of_year, y,
                lw=1, color=color, alpha=alpha, zorder=3)

        # fill logic
        if idx == 0: # value in 0 - SPI+1.0
            ax.fill_between(day_of_year, 0, y, color=color, alpha=alpha, label=label)
        elif idx == len(spi_cols) - 1: # value in SPI-2.0 to inf
            ax.fill_between(day_of_year, y, max_precp, color=color, alpha=alpha, label=label)
        else:
            ax.fill_between(day_of_year, df_boundary[spi_cols[idx - 1]].values, y,
                            color=color, alpha=alpha, label=label)

    # Axis formatting
    ax.set_xlim(1, 365)
    x_ticks = []
    for month in range(1, 13):
        # only plot from odd month to even month
        julday1 = UTCDateTime(year=2023, month=month, day=1).julday
        x_ticks.append(julday1)
    x_ticks.append(365)
    ax.set_xticks(x_ticks, x_ticks)
    ax.set_ylim(0, max_precp)

    ax.legend(loc="upper left", fontsize=6)
    ax.set_xlabel("Day of Year", fontweight="bold")
    ax.set_ylabel(f"Rolling Total Precipitation in Last {spi_scale} days [mm]",
                  fontweight="bold")

    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    car_ax.axis("off")
    car_ax.legend(handles, labels, loc="upper center", fontsize=6)
    
    return fig, ax
  
def archive_data(spi_scale, day_of_year=np.arange(1, 365 + 1), max_precp=350):

    # load the data
    time_stamps, precp = load_data()
    filtered_time, precip_roll, doy = compute_rolling_precp(time_stamps, precp, spi_scale)
    
    mean_daily_total_in_last_spi_scale = []
    for day in day_of_year:
        idx = np.where(doy == day)[0]
        value = precip_roll[idx]
        mean_daily_total_in_last_spi_scale.append(np.nanmean(value))
    print(f"np.sum(mean_daily_total_in_last_spi_scale): {np.sum(mean_daily_total_in_last_spi_scale)}")



    spi_params = Path(current_dir) / f"spi_params_daily_spi_day={spi_scale}.npz"
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


    df_boundary = [day_of_year.reshape(-1, 1)]
    for idx, (label, spi_val) in enumerate(spi_status.items()):
        boundary = []
        for b in spi_val:
            pro = norm.cdf(b)
            precp = gamma.ppf(pro, a=shape, loc=loc, scale=scale)
            precp = np.clip(precp, 0, None)
            boundary.append(precp)

        low_b, high_b = boundary[0], boundary[1]
        if idx == 0:
            y = high_b
        elif idx == 6:
            y = low_b
        else:
            y = high_b

        df_boundary.append(y.reshape(-1, 1))


    output_path = Path(project_root) / "data" / "SPI_boundary"
    
    
    df_boundary = pd.DataFrame(np.hstack(df_boundary), columns=["day_of_year", 
                                                                "SPI=-2.0", "SPI=-1.5", "SPI=-1.0", 
                                                                "SPI=+1.0", "SPI=+1.5", "SPI=+2.0", "SPI=+4.0"])
    df_boundary.to_csv(f"{output_path}/SPI_daily_boundary.txt", index=False)
    
    
    
    df_obs = [day_of_year.reshape(-1, 1)]
    df_obs.append(np.array(mean_daily_total_in_last_spi_scale).reshape(-1, 1))
    df_obs = pd.DataFrame(np.hstack(df_obs), columns=["day_of_year", f"mean_daily_total_precp_in_last{spi_scale}days"])
    df_obs.to_csv(f"{output_path}/SPI_daily_cum_obs.txt", index=False)


    
def main(spi_scale=30):

    # (1) calculate the SPI params and dump it for further usage
    SPI_pipeline(spi_scale)

    # (2) usage example, e.g., 
    precp_spi_scale_total = 20 # unit by mm per day
    day_of_year = 151
    spi_value, status = SPI_usage(precp_spi_scale_total, day_of_year, spi_scale)

    # (3) plot the boundary
    archive_data(spi_scale, day_of_year=np.arange(1, 365 + 1), max_precp=350)
    df_boundary = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_boundary.txt", header=0)
    df_obs = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_cum_obs.txt", header=0)
    p_obs = df_obs.iloc[:, 1].values
    
    fig, ax = plot_spi_boundary(df_boundary, p_syn=None, p_obs=p_obs, spi_scale=spi_scale)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/spi_boundary_daily_spi_day={spi_scale}.png", dpi=600)
    plt.show()
    plt.close(fig=fig)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spi_scale", type=int, default=30, help="unit by day")
    args = parser.parse_args()

    main(args.spi_scale)

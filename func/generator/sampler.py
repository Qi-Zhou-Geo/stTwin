#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent
#
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# endregion

# import the func. from the same folder
from func.generator.storm2drought import storm2drought_generator
from func.generator.load_statistics import sta_loader
from func.SPI.ILL_SPI_daily import plot_spi_boundary

def generate_synthetic(metadata, temp_sta, radiation_sta, 
                       extremely_dry_b, moderately_wet_b,
                       time_t, status_t, sigma_scale, 
                       ref_last29_precp, seed, 
                       plot=False):
    """Generate one year daily synthetic data
    
    Args:
        metadata (_type_): _description_
        precp_sta (_type_): _description_
        temp_sta (_type_): _description_
        radiation_sta (_type_): _description_
        time_t (_type_): _description_
        status_t (_type_): _description_
        sigma_scale (int, optional): _description_. Defaults to 3.

    Returns:
        _type_: _description_
    """

    # (1) create a mapping for the statistics indices
    meta_dict = {}
    for idx, sta_method in enumerate(metadata):
        meta_dict[sta_method] = idx

    # (2) prepare the day indices
    days = np.array(time_t, dtype=int)

    # (3) define sampling boundary
    # region
    is_drought = status_t == 1
    
    t_low = np.where(
        is_drought,
        temp_sta[days, meta_dict["mean"]], # if drought (is_drought == True)
        temp_sta[days, meta_dict["min"]],  # if storm (is_drought == False)
    )

    t_high = np.where(
        is_drought,
        temp_sta[days, meta_dict["mean"]] + sigma_scale * temp_sta[days, meta_dict["std"]],
        temp_sta[days, meta_dict["mean"]],
    )

    r_low = np.where(
        is_drought,
        radiation_sta[days, meta_dict["mean"]], # if drought (is_drought == True)
        radiation_sta[days, meta_dict["min"]],  # if storm (is_drought == False)
    )
    r_high = np.where(
        is_drought,
        radiation_sta[days, meta_dict["mean"]] + sigma_scale * radiation_sta[days, meta_dict["std"]],
        radiation_sta[days, meta_dict["mean"]],
    )
    # endregion

    # (4) batch sampling for temperature and sun radiation
    rng = np.random.default_rng(seed=seed)  # local seed only for this generator
    t_syn = rng.uniform(t_low, t_high)
    r_syn = rng.uniform(r_low, r_high)


    # (5) sample the precp
    p_syn = np.full(shape=len(days), fill_value=0)
    p_syn = np.append(ref_last29_precp, p_syn) # shape by 394 = 365 + 29 or 395 = 366 + 29
    for idx, drought in enumerate(is_drought):

        sum_last_29 = np.sum(p_syn[idx:29+idx]) # default is SPI30
        if drought == True:
            # need dry -> keep 30-day sum below extremely dry boundary (SPI30 = -2.0)
            # extremely_dry_b[idx] - sum_last_29 < 0 -> last 29 days total precp is too wet
            remaining = max(0.0, extremely_dry_b[idx] - sum_last_29)
            p_idx = rng.uniform(low=0, high=remaining)
        else:
            # need wet -> keep 30-day sum above extremely wet boundary (SPI30 = +2.0)
            needed = max(0.0, moderately_wet_b[idx] - sum_last_29)
            p_idx = rng.uniform(low=needed, high=moderately_wet_b[idx])

        p_syn[idx+29] = p_idx
        
    # drop the first 29 elements
    if plot is True:
        plot_SPI(p_syn, spi_window=30)
    p_syn = p_syn[29:]


    # (6) check the p_syn and historical yearly total
    data_path = Path(project_root) / "data" / "SedCas_input" / "prep_1931_2026_y.txt"
    df = pd.read_csv(data_path, skiprows=5, header=0)
    data = df.iloc[:, 2]
    
    if np.sum(p_syn) > np.max(data):
        marker = "too much"
    elif np.sum(p_syn) < np.max(data):
        marker = "too less"
    else:
        marker = "too perfect"
        
    print(f"The model generates <<{marker}>> total precipitation in one year.\n"
            f"The unit of the following values is mm.\n"
            f"np.sum(p_syn)={np.sum(p_syn) :.1f} > np.max(data)={np.max(data) :.1f} \n"
            f"np.min(data)={np.min(data) :.1f}\n" 
            f"np.std(data, ddof=0)={np.std(data, ddof=0) :.1f}\n"
            f"np.mean(data)={np.mean(data) :.1f}\n")

    # (7) prepare return
    synthetic = np.column_stack([p_syn, t_syn, r_syn])

    return synthetic


def daily_sampler(
    cycle_period=30,  # every 60 day
    storm_onset=32,  # start from 1st May, 365 days
    storm2drought_ratio=0.05,
    sigma_scale=3,
    leap_year=False,
    ref_last29_precp=None,
    seed=None,
    plot=False):

    time_t, status_t = storm2drought_generator(
        cycle_period=cycle_period,
        t0=storm_onset,
        Rs2d=storm2drought_ratio,
        leap_year=leap_year
    )
    
    spi_boundary = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_boundary.txt", header=0)
    extremely_dry_b = spi_boundary["SPI=-2.0"].values # represents the "Extremely Dry"
    moderately_wet_b = spi_boundary["SPI=+1.0"].values # represents the "Moderatly Wet"
    
    
    metadata, precp_sta, temp_sta, radiation_sta = sta_loader()
    if leap_year is True:
        precp_sta, temp_sta, radiation_sta = precp_sta[:366, :], temp_sta[:366, :], radiation_sta[:366, :]
        extremely_dry_b = np.append(extremely_dry_b, extremely_dry_b[-1]) # type: ignore
        moderately_wet_b= np.append(moderately_wet_b, moderately_wet_b[-1]) # type: ignore
    else:
        precp_sta, temp_sta, radiation_sta = precp_sta[:365, :], temp_sta[:365, :], radiation_sta[:365, :]


    if ref_last29_precp is None:
        temp = pd.read_csv(f"{project_root}/data/SedCas_input/climate_1931_2025_d.txt", header=0)
        ref_last29_precp = temp.iloc[-29:, 2].values
        
    synthetic = generate_synthetic(metadata, temp_sta, radiation_sta, 
                                   extremely_dry_b, moderately_wet_b,
                                   time_t, status_t, sigma_scale, 
                                   ref_last29_precp=ref_last29_precp, # the very last 29-days daily total precp.
                                   seed=seed, plot=plot)

    return time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic

def plot_month_background(ax, leap=False):
    
    if leap is True:
        year = 2024
    else:
        year = 2025
    
    
    for month in range(1, 13):
        
        if month % 2 != 0:
            # only plot from odd month to even month
            julday1 = UTCDateTime(year=year, month=month, day=1).julday
            julday2 = UTCDateTime(year=year, month=month + 1, day=1).julday
            
            ax.axvspan(julday1, julday2, color='gray', alpha=0.3, zorder=1)


def plot_syn(time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic, sigma_scale, output_name=None):

    fig = plt.figure(figsize=(6, 7))
    gs = gridspec.GridSpec(4, 1)

    ax = plt.subplot(gs[0])
    ax.plot(time_t, status_t, color='black', zorder=3)
    ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')
    storm_onset = np.where(status_t==-1)[0][0]
    ax.axvline(x=storm_onset, color="C3", ls="--", lw=1, label=f"First Storm Onset (" + r"$t_0$" + f"={storm_onset})")
    ax.set_xlim(time_t[0], time_t[-1])
    
    ax.set_xlim(1, 365)
    ax.set_xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 365],
        [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
    )
    plot_month_background(ax, leap=False)
    ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax.set_ylabel("Status", fontweight='bold')
    ax.set_yticks([-1, 1], ["Strom (-1)", "Drought (1)"])
    ax.set_xlabel(f"", fontweight='bold')
    ax.legend(loc="lower left", fontsize='6')




    y_label = [
        f"Daily Total Precipitation\n[mm]",
        f"Daily Mean Temperature\n[" + r"$^{\circ}\mathrm{C}$" + f"]",
        f"Sun Radiation\n[" + r"$\mathrm{W\,m^{-2}}$" + f"]",
    ]

    y_min = [1, -20, 0]
    y_max = [300, 30, 600]
    subplot_label = ["(b)", "(c)", "(d)"]
    
    for idx, data in enumerate([precp_sta, temp_sta, radiation_sta]):

        ax = plt.subplot(gs[idx + 1])
        ax.set_title(f"{subplot_label[idx]}", fontweight='bold', fontsize=7, loc='left')

        max_data = data[:, 0]
        mean_data = data[:, 1]
        min_data = data[:, 2]
        std_data = data[:, 3]
        q5_data = data[:, 4]
        q95_data = data[:, 5]

        x = range(1, len(max_data) + 1)
        y = mean_data
        y1 = mean_data - sigma_scale * std_data
        y2 = mean_data + sigma_scale * std_data
        y3 = min_data
        y4 = max_data

        if idx == 1:
            pass
        else:
            # this for radation
            y1 = np.clip(y1, a_min=0, a_max=np.max(y1))
            y2 = np.clip(y2, a_min=0, a_max=np.max(y2))

        ax.plot(x, synthetic[:, idx], label="Synthetic", color="black", zorder=3)
        ax.plot(x, y, color="C1", label="Mean", zorder=4)

        if idx == 0:
            ax.plot(x, y2, color="C0", label="Mean + Std.", zorder=1)
            ax.plot(x, y4, color="C2", label="Max", zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(1e0, 2e2)
        else:
            ax.fill_between(x, y1, y2, color="C0", label=f"Mean" + r"$\pm$" + f"{sigma_scale}" + r"$\times$" + f"Std.", alpha=0.5, zorder=1)
            ax.fill_between(x, y3, y4, color="C2", label="Min to Max", alpha=0.5, zorder=2)

        if idx in [2]:
            ax.legend(loc="upper left", fontsize="6")
        
        if idx in [0]:
            ax.legend(loc="upper left", fontsize="6", ncol=4)

        ax.set_ylabel(f"{y_label[idx]}", fontweight="bold")
        ax.set_ylim(y_min[idx], y_max[idx])
        
        plot_month_background(ax, leap=False)
        ax.set_xlim(1, 365)
        ax.set_xticks(
            [1, 50, 100, 150, 200, 250, 300, 350, 365],
            [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
        )
        ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax.set_xlabel("Day of Year", fontweight="bold") # (1931–2025, MeteoSwiss)
    plt.tight_layout()
    if output_name is None:
        png_name = Path(current_dir) / "input_scenarios-storm_drought_cycle.png"
    else:
        png_name = output_name
    
    print(f"Figure from <plot_syn> will be saved at: {png_name}")
    plt.savefig(png_name, dpi=600, transparent=True)
    plt.show()
    plt.close(fig=fig)

def plot_SPI(p_syn, spi_window=30):
    
    cumsum = np.cumsum(p_syn)
    cum_p_syn = cumsum.copy()
    cum_p_syn[spi_window:] = cumsum[spi_window:] - cumsum[:-spi_window]
    cum_p_syn = cum_p_syn[29:] # the first 29 data is not from current year
    
    
    df_boundary = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_boundary.txt", header=0)
    df_obs = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_cum_obs.txt", header=0)
    p_obs = df_obs.iloc[:, 1].values
    
    fig, ax = plot_spi_boundary(df_boundary, p_syn=cum_p_syn[:365], p_obs=p_obs[:365], spi_scale=spi_window)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/synthetic_{spi_window}.png", dpi=600)
    plt.show()
    plt.close(fig=fig)
    
    
    fig, ax = plot_spi_boundary(df_boundary, p_obs=p_obs[:365], spi_scale=spi_window)

    plt.tight_layout()
    plt.savefig(f"{current_dir}/synthetic_{spi_window}_no_sum.png", dpi=600)
    plt.show()
    plt.close(fig=fig)

def plot_syn2(time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic, sigma_scale, output_name=None):

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(3, 1)


    y_label = [
        f"Daily Total Precipitation\n[mm]",
        f"Daily Mean Temperature\n[degree]",
        f"Daily Mean Radiation\n[W / m²]",
    ]

    y_min = [1, -20, 0]
    y_max = [100, 30, 600]

    for idx, data in enumerate([precp_sta, temp_sta, radiation_sta]):

        ax = plt.subplot(gs[idx])

        max_data = data[:, 0]
        mean_data = data[:, 1]
        min_data = data[:, 2]
        std_data = data[:, 3]
        q5_data = data[:, 4]
        q95_data = data[:, 5]

        x = range(1, len(max_data) + 1)
        y = mean_data
        y1 = mean_data - sigma_scale * std_data
        y2 = mean_data + sigma_scale * std_data
        y3 = min_data
        y4 = max_data

        if idx == 1:
            pass
        else:
            # this for radation
            y1 = np.clip(y1, a_min=0, a_max=np.max(y1))
            y2 = np.clip(y2, a_min=0, a_max=np.max(y2))

        #ax.plot(x, synthetic[:, idx], label="Synthetic", color="black", zorder=3)
        ax.plot(x, y, color="C1", label="Mean", zorder=4)

        if idx == 0:
            ax.plot(x, y2, color="C0", label="Mean + Std.", zorder=1)
            ax.plot(x, y4, color="C2", label="Max", zorder=2)
            ax.set_yscale("log")
            ax.set_ylim(1e0, 2e2)
        else:
            
            if idx == 2:
                y1 = np.clip(y1, a_min=0, a_max=None)
                y2 = np.clip(y2, a_min=0, a_max=np.max(y2))
                y3 = np.clip(y3, a_min=0, a_max=np.max(y3))
                y4 = np.clip(y4, a_min=0, a_max=np.max(y4))

            ax.fill_between(x, y1, y2, color="C0", label=f"Mean to {sigma_scale}Std.", alpha=0.5, zorder=1)
            ax.fill_between(x, y3, y4, color="C2", label="Min to Max", alpha=0.5, zorder=2)

        if idx in [2]:
            ax.legend(loc="upper left", fontsize="6")

        ax.set_ylabel(f"{y_label[idx]}", fontweight="bold")
        ax.set_ylim(y_min[idx], y_max[idx])
        
        plot_month_background(ax, leap=False)
        ax.set_xlim(1, 365)
        ax.set_xticks(
            [1, 50, 100, 150, 200, 250, 300, 350, 365],
            [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
        )
        ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)

    ax = plt.subplot(gs[2])
    ax.set_xlabel("Day of Year", fontweight="bold") #  (1931–2025, MeteoSwiss MVE Station)
    plt.tight_layout()
    if output_name is None:
        png_name = Path(current_dir) / "statistic_climate_MVE.png"
    else:
        png_name = output_name
        
    plt.savefig(png_name, dpi=600, transparent=True)
    plt.show()
    plt.close(fig=fig)

def main(sigma_scale):
    temp = daily_sampler(sigma_scale = sigma_scale, plot=True)
    time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic = temp
    plot_syn2(time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic, sigma_scale, output_name=None)


if __name__ == "__main__":
    main(sigma_scale=3)

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

# region ### add the sys.path to search for custom modules ###
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
from func.SPI.ILL_SPI_daily import dump_SPI


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

def plot_spi_boundary(ax, df_boundary, p_syn=None, p_obs=None, spi_scale=30, max_precp=400):
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
        
        ("black", 0.3, "Near Normal (-1.0 < SPI ≤ 0)"),
        ("gray", 0.3, "Near Normal (0 < SPI ≤ 1.0)"),
        
        ("C0", 0.3, "Moderately Wet  (1.0 < SPI ≤ 1.5)"),
        ("C0", 0.6, "Very Wet  (1.5 < SPI ≤ 2.0)"),
        ("C0", 0.9, "Extremely Wet  (2.0 < SPI)"),
    ]




    # SPI boundaries
    for idx, col in enumerate(spi_cols):
        y = df_boundary[col].values
        color, alpha, label = color_alpha[idx]

        ax.plot(day_of_year, y, lw=1, color=color, alpha=alpha, zorder=3)

        # fill logic
        if idx == 0: # value in 0 - SPI+1.0
            ax.fill_between(day_of_year, 0, y, color=color, alpha=alpha, label=label)
        elif idx == len(spi_cols) - 1: # value in SPI-2.0 to inf
            ax.fill_between(day_of_year, y, max_precp, color=color, alpha=alpha, label=label)
        else:
            ax.fill_between(day_of_year, df_boundary[spi_cols[idx - 1]].values, y,
                            color=color, alpha=alpha, label=label)

    # Optional: plot p_syn
    if p_syn is not None:
        ax.plot(day_of_year, p_syn,
                lw=1, ls="-", color="black",
                label="Synthetic Rolling 30-day Total", zorder=4)

    if p_obs is not None:
        ax.plot(day_of_year, p_obs,
                lw=1, ls="-", color="C1",
                # label=(f"Mean Daily Total Precipitation in Last {spi_scale} Days\n"
                #        f"(Jan. 1931 – Dec. 2025; MeteoSwiss Montana Station)"), 
                label="Observed Rolling 30-day Total",
                zorder=4)

    for i in range(6):
        ax.plot([], [],
                lw=1, ls="-", color="black",
                label=" ", zorder=4, alpha=0)
        
    
    # set ticks every 50 days
    ax.set_xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 365],
        [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
    )
    ax.set_ylim(0, max_precp)
    ax.set_xlim(1, 365)

    ax.set_ylabel(f"Rolling Total Precipitation\nin Last-{spi_scale}-Day [mm]", fontweight="bold")
    ax.grid(True, ls="--", lw=0.5, alpha=0.5)
    ax.legend(loc="upper center", fontsize=6, ncol=2, bbox_to_anchor=(0.55, 1.0))

    return ax


def plot_syn_with_SPI(
    time_t,
    status_t,
    precp_sta,
    p_syn,
    sigma_scale,
    cycle_period,
    storm2drought_ratio,
    storm_onset_month,
    output_name=None):

    fig = plt.figure(figsize=(6, 6.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 2, 1.5])

    # region < strom-to-drought >
    ax = plt.subplot(gs[0])
    ax.set_title("(a)", fontweight='bold', fontsize=7, loc='left')

    storm_onset = np.where(status_t==-1)[0][0]
    label = (f"Strom-to-Drought (s2d) Cycle\n"
             f"First Storm Onset: " + r"$t_0$" +f"={storm_onset}\n"
             f"s2d Ratio: " + r"$R_{s2d}$" +f"={storm2drought_ratio}\n"
             f"Cycle Period: " + r"$1/f$" +f"={cycle_period}\n"
             f"Num. of Strom Day: {len(np.where(status_t==-1)[0])}")
    ax.plot(time_t, status_t, color='black', zorder=3, label="")
    
    mask = (time_t >= storm_onset) & (time_t < storm_onset + cycle_period)
    label = f"First Cycle\n" + rf"({time_t[storm_onset]} $\leq t < {time_t[storm_onset] + time_t[cycle_period] - 1}$)"
    ax.plot(time_t[mask], status_t[mask], color="green", zorder=4, label=label)
    
    label = r"$t_0$"
    ax.axvline(x=storm_onset, color="C3", ls="--", lw=1, label=r"$t_0$" +f"={time_t[storm_onset]}", zorder=1)
    ax.set_xlim(time_t[0], time_t[-1])

    ax.set_xlim(1, 365)
    ax.set_xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 365],
        [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
    )
    plot_month_background(ax, leap=False)
    ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=5)
    ax.set_ylabel("Status", fontweight='bold')
    ax.set_yticks([-1, 1], ["Strom (-1)", "Drought (1)"])
    ax.set_xlabel(f"", fontweight='bold')
    ax.legend(loc="center left", fontsize='6', ncol=1, bbox_to_anchor=(0.75, 0.5))
    # endregion


    # region < SPI >
    temp = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2004_2023_h.txt", header=0)
    ref_last29_precp = temp.iloc[-29:, 2].values # last 29 precp. from 2022

    spi_window = 30
    p_syn_long = np.append(ref_last29_precp, p_syn) # type: ignore
    cumsum = np.cumsum(p_syn_long)
    cum_p_syn = cumsum.copy()
    cum_p_syn[spi_window:] = cumsum[spi_window:] - cumsum[:-spi_window]
    cum_p_syn = cum_p_syn[29:] # the first 29 data is not from current year
    
    
    df_boundary_path = Path(project_root) / f"data/SPI_boundary/SPI_daily_boundary.txt"
    df_boundary = pd.read_csv(df_boundary_path, header=0)
    df_obs = pd.read_csv(f"{project_root}/data/SPI_boundary/SPI_daily_cum_obs.txt", header=0)
    p_obs = np.array(df_obs.iloc[:, 1])

    ax = plt.subplot(gs[1])
    ax.set_title(f"(b)", fontweight='bold', fontsize=7, loc='left')

    plot_spi_boundary(ax, df_boundary,p_syn=cum_p_syn[:365], p_obs=p_obs[:365], spi_scale=30, max_precp=400)
    # endregion
    
    
    # region < precp >
    data = precp_sta

    ax = plt.subplot(gs[2])
    ax.set_title(f"(c)", fontweight='bold', fontsize=7, loc='left')

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

    ax.plot(x, p_syn, label="Synthetic", color="black", zorder=5)
    ax.plot(x, y, color="C1", label="Mean", zorder=4)

    ax.plot(x, y2, color="C0", label="Mean + " + f"{sigma_scale}" + r"$\times$" + f"Std.", zorder=1)
    ax.plot(x, y4, color="C2", label="Max", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(1e0, 2e2)
    ax.legend(loc="upper right", fontsize="6", ncol=4)

    ax.set_ylabel(f"Daily Total Precipitation\n[mm]", fontweight="bold")
    ax.set_ylim(1, 300)

    plot_month_background(ax, leap=False)
    ax.set_xlim(1, 365)
    ax.set_xticks(
        [1, 50, 100, 150, 200, 250, 300, 350, 365],
        [1, 50, 100, 150, 200, 250, 300, 350, 365], # type: ignore
    )
    ax.grid(axis="both", color="grey", linestyle="--", lw=0.5, alpha=0.5, zorder=1)
    ax.set_xlabel(f"Day of Year", fontweight="bold") # (1931–2025, MeteoSwiss)
    # endregion


    plt.tight_layout()
    if output_name is None:
        png_name = Path(current_dir) / "input_scenarios-storm_drought_cycle.png"
    else:
        png_name = output_name

    print(f"Figure from <plot_syn> will be saved at: {png_name}")
    plt.savefig(png_name, dpi=600, transparent=True)
    plt.show()
    plt.close(fig=fig)



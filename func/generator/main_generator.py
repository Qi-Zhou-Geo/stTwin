#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the func. from the same folder
from func.generator.sampler import daily_sampler, plot_syn
from func.generator.upsampling import upsampler_prcp, upsampler_temp, upsampler_radi
from func.generator.plot_s2d import plot_syn_with_SPI


def s2d_workflow(year_list, 
             cycle_period, 
             storm2drought_ratio, 
             storm_onset_month, 
             storm_onset_day,
             sigma_scale=3,
             plot=False, 
             seed=None, 
             ref_history=True):

    # region <sampling the daily resolution data>
    time_t_l = []
    status_t_l = []
    synthetic_l = []

    
    if isinstance(year_list, tuple):
        scenario_year = list(year_list)
    elif isinstance(year_list, int):
        scenario_year = [year_list]
    else:
        raise ValueError(f"Unsupport data type. type(year_list) is {type(year_list)}.")
    
    
    for year in list(scenario_year):

        if year % 4 == 0:
            leap_year = True
        else:
            leap_year = False


        if seed is None:
            seed = year

        if ref_history is True:
            if year == 2023:
                # ref last 29 precp from 2022
                df_path = Path(project_root) / "data/SedCas_input/climate_2004_2023_t.txt"
                df = pd.read_csv(df_path, header=0)
                ref_last29_precp = np.array(df["precipitation [mm per time_step]"][-29:])
            else:
                # ref sythenstic data, this is from "temp = daily_sampler"
                ref_last29_precp = synthetic[-29:, 0] # type: ignore
        else:
            ref_last29_precp = np.full(shape=29, fill_value=0.0)
        
        storm_onset = UTCDateTime(year=year, month=int(storm_onset_month), day=int(storm_onset_day)).julday

        temp = daily_sampler(
            cycle_period=cycle_period,
            storm_onset=storm_onset,
            storm2drought_ratio=storm2drought_ratio,
            sigma_scale=sigma_scale,
            leap_year=leap_year,
            ref_last29_precp=ref_last29_precp,
            seed=seed,
            plot=plot)


        time_t, status_t, precp_sta, temp_sta, radiation_sta, synthetic = temp
        
        if plot is True:
            output_dir = Path(project_root) / "plotting/what_if_plots"
            output_name1 = f"{output_dir}/{year}_cycle_period={cycle_period}_storm2drought_ratio={storm2drought_ratio}_storm_onset={storm_onset}.png"
            output_name2 = f"{output_dir}/{year}_cycle_period={cycle_period}_storm2drought_ratio={storm2drought_ratio}_storm_onset={storm_onset}_SPI.png"
            os.makedirs(output_dir, exist_ok=True)
            plot_syn(time_t=time_t, status_t=status_t, 
                     precp_sta=precp_sta, temp_sta=temp_sta, radiation_sta=radiation_sta, 
                     synthetic=synthetic, sigma_scale=sigma_scale, 
                     cycle_period=cycle_period, storm2drought_ratio=storm2drought_ratio, storm_onset_month=storm_onset_month, 
                     output_name=output_name1)
            
            plot_syn_with_SPI(time_t=time_t, status_t=status_t,
                              precp_sta=precp_sta, p_syn=synthetic[:, 0],
                              sigma_scale=sigma_scale, 
                              cycle_period=cycle_period, storm2drought_ratio=storm2drought_ratio, storm_onset_month=storm_onset_month,
                              output_name=output_name2)

        time_t_l.append(time_t)
        status_t_l.append(status_t)
        synthetic_l.append(synthetic)

    # merge together
    time_t_arr = np.concatenate(time_t_l, axis=0)
    status_t_arr = np.concatenate(status_t_l, axis=0)
    synthetic_arr = np.concatenate(synthetic_l, axis=0) # column by (p_syn, t_syn, r_syn)
    print(f"time_t_arr.shape = {time_t_arr.shape}")
    print(f"status_t_arr.shape = {status_t_arr.shape}")
    print(f"synthetic_arr.shape = {synthetic_arr.shape}")

    # endregion

    # region <upsampling>
    prec_l = []
    temp_l = []
    radi_l = []
    for day in range(len(time_t_arr)):

        daily_total = synthetic_arr[day, 0] # precipitation
        upsampled_data = upsampler_prcp(daily_total)
        prec_l.append(upsampled_data)

        daily_mean = synthetic_arr[day, 1] # temperature
        upsampled_data = upsampler_temp(daily_mean)
        temp_l.append(upsampled_data)

        daily_mean = synthetic_arr[day, 2] # radiation
        upsampled_data = upsampler_radi(daily_mean)
        radi_l.append(upsampled_data)

    prec = np.concatenate(prec_l, axis=0)
    temp = np.concatenate(temp_l, axis=0)
    radi = np.concatenate(radi_l, axis=0)
    print(f"prec.shap = {prec.shape}")
    print(f"temp.shap = {temp.shape}")
    print(f"radi.shap = {radi.shape}")
    # endregion

    station = np.full(len(prec), "what-if")
    df = pd.read_csv(f"{project_root}/data/SedCas_input/climate_2023_2026_t.txt", header=0)
    id1 = np.where(df.iloc[:, 1] == "2023-01-01T00:00:00")[0][0]
    id2 = np.where(df.iloc[:, 1] == "2025-12-31T23:50:00")[0][0] + 1
    time_str = np.array(df.iloc[id1:id2, 1])

    data = np.concatenate((station.reshape(-1, 1), 
                           time_str.reshape(-1, 1),
                           prec.reshape(-1, 1),
                           temp.reshape(-1, 1),
                           radi.reshape(-1, 1)
                        ), axis = 1)

    file_format = f"CP={int(cycle_period)}_R={storm2drought_ratio:.3f}_M={int(storm_onset_month)}_D={int(storm_onset_day)}"
    scenario_name = f"climate_2023_2026_t_whatif_{file_format}"
    
    scenario_input_path = Path(project_root) / f"data/SedCas_whatif_input/{scenario_name}.txt"
    scenario_input_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(scenario_input_path, data, delimiter=",", fmt="%s", header=",".join(df.columns))


    meta = {
    "archived_tine": UTCDateTime.now().isoformat(),
    "cycle_period": cycle_period,
    "storm_onset_month": storm_onset_month,
    "storm_onset_day": storm_onset_day,
    "storm2drought_ratio": storm2drought_ratio,
    "sigma_scale": sigma_scale}

    scenario_input_meta = Path(project_root) / f"data/SedCas_whatif_input/{scenario_name}_meta.json"
    scenario_input_meta.parent.mkdir(parents=True, exist_ok=True)
    with open(scenario_input_meta, "w") as f:
        json.dump(meta, f, indent=2)
        
    return file_format, data


if __name__ == "__main__":
    file_format, data = s2d_workflow(
        year_list=(2023, 2024, 2025), #(2023), # (2023, 2024, 2025)
        cycle_period=105,  # every N day
        storm2drought_ratio=0.05,  # duration of storm / drought is 0.1
        storm_onset_month=8,  # start from 1st of Febuary
        storm_onset_day=1,
        sigma_scale=3,  # control the std. for temperature and sun radiation
        plot=True,
        seed=None,
        ref_history=True,
    )

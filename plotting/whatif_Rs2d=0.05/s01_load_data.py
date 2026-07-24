#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-17T22:28:21
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse

import numpy as np
import pandas as pd

import xarray as xr

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator


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
from func.SedCas.mass_balance_checker import mass_balance_checker

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-3, 6),
                     'axes.formatter.use_mathtext': True})



def extract_whatif(df_sub, scenario_idx, select_t1, select_t2, model_version, scenario_name="run_whatif_Rs2d=0.05"):

    base_values = df_sub.iloc[scenario_idx, :].values
    i, cp, Rs2d, t0, d = base_values
    
    whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"
    data_dir = Path(project_root) / f"pipeline/{scenario_name}/{model_version}/{whatif_type}"
    
    ds_path = Path(data_dir) / f"theta_001/sed_container.nc" # 001 is MAP
    ds = xr.load_dataset(ds_path)
    mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
    ds = ds.isel(time=mask)

    # return as List[Dict]
    result = mass_balance_checker(sed_container=ds, residual=1.0, iteration=None, silence=True)
    
    df_statistic = pd.DataFrame(result).set_index("iteration")
    print(df_statistic.shape) # row by "iteration", column by sediment state
    
    
    stat_values = []
    stat_col_name = []
    for col in df_statistic.columns:
        data = df_statistic[col] # shape as ("iteration", num_sed_state)
        
        # collapse "iteration"
        data_mean = np.mean(data)
        data_std = np.std(data, ddof=1)

        # axis=0 >> collapse along the 100-iteration dimension
        q05 = np.quantile(a=data, q=0.05, axis=0)
        q50 = np.quantile(a=data, q=0.50, axis=0)
        q95 = np.quantile(a=data, q=0.95, axis=0)

        stat_col_name.append(f"{col}_mean")
        stat_col_name.append(f"{col}_std")
        stat_col_name.append(f"{col}_05")
        stat_col_name.append(f"{col}_50")
        stat_col_name.append(f"{col}_95")

        stat_values.extend([data_mean, data_std, q05, q50, q95])


    stat_values = np.array(stat_values)
    stat_col_name = np.array(stat_col_name)

    npz_path = Path(current_dir) / f"cache/whatif_MAP_{whatif_type}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, stat_values=stat_values, stat_col_name=stat_col_name)


def extract_benchmark(select_t1, select_t2, model_version):

    # monitor_MAP_real
    monitor_MAP_real = Path(project_root) / f'pipeline/run_2004_2025_posterior/{model_version}/theta_001/sed_container.nc'
    ds = xr.load_dataset(monitor_MAP_real)
    mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
    ds = ds.isel(time=mask)

    # return as List[Dict]
    result = mass_balance_checker(sed_container=ds, residual=1.0, iteration=None, silence=True)
    df_statistic = pd.DataFrame(result).set_index("iteration")
    print(df_statistic.shape)

    stat_values = []
    stat_col_name = []
    for col in df_statistic.columns:
        data = df_statistic[col] # shape as ("iteration", num_sed_state)
        
        # collapse "iteration"
        data_mean = np.mean(data)
        data_std = np.std(data, ddof=1)

        # axis=0 >> collapse along the 100-iteration dimension
        q05 = np.quantile(a=data, q=0.05, axis=0)
        q50 = np.quantile(a=data, q=0.50, axis=0)
        q95 = np.quantile(a=data, q=0.95, axis=0)


        stat_col_name.append(f"{col}_mean")
        stat_col_name.append(f"{col}_std")
        stat_col_name.append(f"{col}_05")
        stat_col_name.append(f"{col}_50")
        stat_col_name.append(f"{col}_95")

        stat_values.extend([data_mean, data_std, q05, q50, q95])


    stat_values = np.array(stat_values)
    stat_col_name = np.array(stat_col_name)

    npz_path = Path(current_dir) / f"cache/monitor_MAP_real.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, stat_values=stat_values, stat_col_name=stat_col_name)


def extract_precp(df_sub, select_t1="2023-01-01T00:00:00", select_t2="2025-12-31T23:50:00"):
    
    precp1_path = Path(project_root) / "data/SedCas_input/climate_2023_2026_t.txt"
    precp1 = pd.read_csv(precp1_path, header=0)
    time_str = precp1.iloc[:, 1].values
    id1 = np.where(time_str==select_t1)[0][0]
    id2 = np.where(time_str==select_t2)[0][0]
    precp_real = np.sum(precp1.iloc[id1:id2, 2])

    for scenario_idx in range(len(df_sub)):
        base_values = df_sub.iloc[scenario_idx, :].values
        i, cp, Rs2d, t0, d = base_values
        
        whatif_type = f"CP={int(cp)}_R={Rs2d:.3f}_M={int(t0)}_D={int(d)}"

        # Load generated precp
        # for the period 2023-01-01T00:00:00 to 2025-12-31T23:50:00.
        precp2_path = Path(project_root) / f"data/SedCas_whatif_input/climate_2023_2026_t_whatif_{whatif_type}.txt"
        precp2 = pd.read_csv(precp2_path, header=0)
        time_str = precp2.iloc[:, 1].values
        id1 = np.where(time_str==select_t1)[0][0]
        id2 = np.where(time_str==select_t2)[0][0]
        precp_whatif = np.sum(precp2.iloc[id1:id2, 2])
    
        ratio = precp_whatif / precp_real
        
        npz_path = Path(current_dir) / f"cache/monitor_precp_{whatif_type}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, ratio=ratio, precp_real=precp_real, precp_whatif=precp_whatif)

def main(scenario_idx, scenario_name="run_whatif_Rs2d=0.05", model_version="v0dot4"):
    
    statistic_ratio = Path(project_root) / f"pipeline/{scenario_name}/scenario_bound.txt"
    df = pd.read_csv(statistic_ratio, header=0)

    select_t1, select_t2 = "2023-01-01T00:00:00", "2026-01-01T00:00:00"
    extract_whatif(df, scenario_idx, select_t1, select_t2, model_version)
    
    if scenario_idx == 0:
        extract_benchmark(select_t1, select_t2, model_version)
        extract_precp(df, select_t1="2023-01-01T00:00:00", select_t2="2025-12-31T23:50:00")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--scenario_idx", type=int, default=0)
    args = parser.parse_args()
    
    main(args.scenario_idx, scenario_name="run_whatif_Rs2d=0.05", model_version="v0dot4")
#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse

import os
import numpy as np
import pandas as pd

# region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

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
from func.toolkit.load_sedcas_nc_output import load_data


def benchmark(idx,
              posterior_result_path, 
              posterior_type = "real", 
              key_type = "sed", 
              key = "sed_transport_real_Q50"):
    
    nc_folder = posterior_result_path[posterior_type]
    time_str, sed_arr = load_data(key_type, key, nc_folder, num_draw=50)


    # Note: the whole "sed_arr" does not dump, it is too big
    total_sed_yield = np.sum(sed_arr)

    
    output_path = Path(current_dir) / f"{model_version}"
    os.makedirs(output_path, exist_ok=True)
    outpur_name = Path(output_path) / f"{posterior_type}_{key}.npz"
    np.savez(outpur_name, time_str=time_str, total_sed_yield=total_sed_yield)


def whatif_results(idx, 
                   posterior_result_path,
                   posterior_type = "what-if", 
                   key_type = "sed", 
                   key = "sed_transport_real_Q50"):
    
    scenario_bound = Path(project_root) / "pipeline" / "what_if" / "scenario_bound.txt"
    df = pd.read_csv(scenario_bound, header=0)
        
    temp = df.iloc[idx, :].values
    i, cp, r, m, d = temp

    whatif_type = f"CP={cp}_R={r}_M={m}_D={d}"
    nc_folder = f"{posterior_result_path[posterior_type]}/{whatif_type}"
    time_str, sed_arr = load_data(key_type, key, nc_folder, num_draw=50)

    # Note: the whole "sed_arr" does not dump, it is too big
    total_sed_yield = np.sum(sed_arr)

    output_path = Path(current_dir) / f"{model_version}"
    os.makedirs(output_path, exist_ok=True)
    outpur_name = Path(output_path) / f"{posterior_type}_{key}_{whatif_type}.npz"
    np.savez(outpur_name, time_str=time_str, total_sed_yield=total_sed_yield)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--scenario_idx", type=int)
    parser.add_argument("--model_version", type=str, default="v0dot4")
    args = parser.parse_args()

    scenario_idx = args.scenario_idx
    model_version = args.model_version


    posterior_result_path = {"real": f"/home/qizhou/3paper/stTwin/pipeline/real_monitor/{model_version}",
                            "what-if": f"/home/qizhou/3paper/stTwin/pipeline/what_if/{model_version}"}

    whatif_results(scenario_idx, posterior_result_path, 
                   posterior_type="what-if", key_type="sed", key="sed_transport_real_Q50")

    if scenario_idx == 0:
        benchmark(scenario_idx, posterior_result_path, 
                   posterior_type="real", key_type="sed", key="sed_transport_real_Q50")
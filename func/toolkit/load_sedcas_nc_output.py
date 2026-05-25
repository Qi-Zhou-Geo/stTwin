#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import numpy as np
import pandas as pd
import xarray as xr

#region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent.parent

import sys

sys.path.append(str(project_root))
# endregion


def load_data(key_type, key, posterior_result_path, num_draw=50):
    
    if key_type in ["hydro"]:
        nc_file = "hydro_output.nc"
    elif key_type in ["sed"]:
        nc_file = "sed_output.nc"

    time_str = None
    temp_l = []

    for theta_idx in range(1, num_draw + 1):
        theta_idx = str(theta_idx).zfill(3)

        output = xr.load_dataset(f"{posterior_result_path}/theta_{theta_idx}/{nc_file}")

        if time_str is None:
            time_str = output.coords["time_str"].values

        output = output[key].values
        temp_l.append(output.reshape(-1, 1))
        
        del output

    # shape as (time step 1 -> N, key_value 1 -> num_draw)
    arr = np.hstack(temp_l)

    return time_str, arr


def run_load_data():
    
    key_type = "sed"
    for key in ["sed_transport_real_Q50", "channel_storage_Q50"]:
        time_str, sed_arr = load_data(key_type, key)
        np.savez(f"./{key}.npz", time_str=time_str, key=sed_arr)
    
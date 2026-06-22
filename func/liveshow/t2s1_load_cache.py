#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-22T09:11:38
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import json

import numpy as np
import pandas as pd
import xarray as xr

from obspy import UTCDateTime

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
# Do not need

def load_cache_monitoring(data_type, t1="2025-01-01T00:00:00", t2="2036-01-01T00:00:00"):


    data_path = Path(project_root) / f"deploy/liveshow_cache/monitoring"
    key1 = "output"
    hydro_key = ""
    sed_key = "_Q50"
    
    if data_type in ["hydro", "hydro_output"]:
        data_key = f"hydro_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"modelled_SWE{hydro_key}": "SWE: Modelled Snow-Water-Equivalent\n[mm]",
            f"Qs{hydro_key}": "Qs: Surface Discharge\n[mm]",
            f"Qss{hydro_key}": "Qss: Sub-Surface Discharge\n[mm]"
        }
        
    elif data_type in ["sed", "sed_output"]:
        data_key = f"sed_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"hillslope_storage{sed_key}": "HS: Hillslope Storage\n[mm]",
            f"channel_storage{sed_key}": "CS: Channel Storage\n[mm]",
            f"sed_transport_real{sed_key}": "SY: Sediments Yield\n[mm]"
        }
    else:
        raise ValueError(f"Please check the input <data_type> {data_type}")
    
    ds1 = xr.load_dataset(f"{data_path}/{data_key}")
    ds2 = xr.load_dataset(f"{data_path}/climate_forcing.nc")
        
    mask = (ds1.time_str >= t1) & (ds1.time_str < t2)
    ds1 = ds1.isel(time=mask)
    mask = (ds2.time_str >= t1) & (ds2.time_str < t2)
    ds2 = ds2.isel(time=mask)
    
    ds_merged = xr.merge([ds1, ds2])
    vars_to_keep  = list(vars_dict.keys())
    ds_sub = ds_merged[vars_to_keep]

    msg = (f"<load_cache> with latest data <{ds_sub.coords['time_str'].values[-1]}> \n")

    return msg, ds_sub, vars_dict

def load_cache_whatif(data_type, whatif_type=None, t1="2023-01-01T00:00:00", t2="2026-01-01T00:00:00"):

    data_path = Path(project_root) / "data" / "liveshow_cache" / "whatif" / "v0dot4_stastic"
    key1 = "stastic"
    hydro_key = "_mean"
    sed_key = "_Q50_mean"
    
    if data_type in ["hydro", "hydro_output"]:
        data_key = f"hydro_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"modelled_SWE{hydro_key}": "SWE: Modelled Snow-Water-Equivalent\n[mm]",
            f"Qs{hydro_key}": "Qs: Surface Discharge\n[mm]",
            f"Qss{hydro_key}": "Qss: Sub-Surface Discharge\n[mm]"
        }
        
    elif data_type in ["sed", "sed_output"]:
        data_key = f"sed_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"hillslope_storage{sed_key}": "HS: Hillslope Storage\n[mm]",
            f"channel_storage{sed_key}": "CS: Channel Storage\n[mm]",
            f"sed_transport_real{sed_key}": "SY: Sediments Yield\n[mm]"
        }
    else:
        raise ValueError(f"Please check the input <data_type> {data_type}")
    
    vars_dict_whatif = vars_dict
    ds1 = xr.load_dataset(f"{data_path}/{whatif_type}/{data_key}")
    ds2 = xr.load_dataset(f"{data_path}/{whatif_type}/climate_forcing.nc")
        
    mask = (ds1.time_str >= t1) & (ds1.time_str < t2)
    ds1 = ds1.isel(time=mask)
    mask = (ds2.time_str >= t1) & (ds2.time_str < t2)
    ds2 = ds2.isel(time=mask)
    
    ds_merged = xr.merge([ds1, ds2])
    vars_to_keep  = list(vars_dict.keys())
    ds_sub_whatif = ds_merged[vars_to_keep]

    msg = f"<load_cache> with latest data <{ds_sub_whatif.coords['time_str'].values[-1]}> \n"


    # real-monitoring with fixed ls
    if data_type in ["hydro", "hydro_output"]:
        data_key = f"hydro_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"modelled_SWE_mean": "SWE: Modelled Snow-Water-Equivalent\n[mm]",
            f"Qs_mean": "Qs: Surface Discharge\n[mm]",
            f"Qss_mean": "Qss: Sub-Surface Discharge\n[mm]"
        }
        
    elif data_type in ["sed", "sed_output"]:
        data_key = f"sed_{key1}.nc"
        
        vars_dict = {
            "precipitation": "Prcp: Total Precipitation\nin 10-minute [mm]",
            f"hillslope_storage_Q50_mean": "HS: Hillslope Storage\n[mm]",
            f"channel_storage_Q50_mean": "CS: Channel Storage\n[mm]",
            f"sed_transport_real_Q50_mean": "SY: Sediments Yield\n[mm]"
        }
    else:
        raise ValueError(f"Please check the input <data_type> {data_type}")
    
    vars_dict_monitoring = vars_dict
    ds3 = xr.load_dataset(f"{data_path}/posterior_draw/{data_key}")
    ds4 = xr.load_dataset(f"{data_path}/posterior_draw/climate_forcing.nc")
        
    mask = (ds3.time_str >= t1) & (ds3.time_str < t2)
    ds3 = ds3.isel(time=mask)
    mask = (ds4.time_str >= t1) & (ds4.time_str < t2)
    ds4 = ds4.isel(time=mask)
    
    ds_merged = xr.merge([ds3, ds4])
    vars_to_keep  = list(vars_dict.keys())
    ds_sub_monitoring = ds_merged[vars_to_keep]

    return msg, ds_sub_whatif, vars_dict_whatif, ds_sub_monitoring, vars_dict_monitoring

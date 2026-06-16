#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-11T18:57:17
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import xarray as xr


# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

num_draw = 21 # 1 MAP + 50 draws
model_version = "v0dot4"
data_dir = Path(project_root) / f"pipeline/run_2004_2025_posterior/{model_version}"


select_t1 = "2004-02-01T00:00:00"
select_t2 = "2025-12-31T23:50:00"
output_dir = Path(current_dir) / "output" / f"{select_t1[:4]}-{select_t2[:4]}"
output_dir.mkdir(parents=True, exist_ok=True)

for key in ["ls_Q50", "hillslope_storage_Q50", 
            "channel_storage_Q50", "sed_transport_real_Q50"]:
    
    temp = []
    
    for theta_idx in range(1, num_draw + 1, 1):
        ds_path = Path(data_dir) / f"theta_{theta_idx:03d}"
        ds = xr.load_dataset(f"{ds_path}/sed_output.nc")
        
        
        mask = (ds.time_str >= select_t1) & (ds.time_str <= select_t2)
        ds = ds.isel(time=mask)
        values = ds[key].values.squeeze()
        
        if theta_idx == 1:
            np.savez_compressed(f"{output_dir}/{key}_MAP.npz", values=values.astype(np.float32), time_str=ds.coords["time_str"].values)
        else:
            temp.append(values)

        print(f"Done: {key}, {theta_idx}")
        
    temp = np.stack(temp, axis=1) # stack theta 2-51 as column >> shape as (time, 50)
    np.savez_compressed(f"{output_dir}/{key}_draw.npz", draw=temp.astype(np.float32))
    





select_t1 = "2023-01-01T00:00:00"
select_t2 = "2025-12-31T23:50:00"
output_dir = Path(current_dir) / "output" / f"{select_t1[:4]}-{select_t2[:4]}"
output_dir.mkdir(parents=True, exist_ok=True)

for key in ["ls_Q50", "hillslope_storage_Q50", 
            "channel_storage_Q50", "sed_transport_real_Q50"]:
    
    temp = []
    
    for theta_idx in range(1, num_draw + 1, 1):
        ds_path = Path(data_dir) / f"theta_{theta_idx:03d}"
        ds = xr.load_dataset(f"{ds_path}/sed_output.nc")
        
        
        mask = (ds.time_str >= select_t1) & (ds.time_str <= select_t2)
        ds = ds.isel(time=mask)
        values = ds[key].values.squeeze()
        
        if theta_idx == 1:
            np.savez_compressed(f"{output_dir}/{key}_MAP.npz", values=values.astype(np.float32), time_str=ds.coords["time_str"].values)
        else:
            temp.append(values)

        print(f"Done: {key}, {theta_idx}")
        
    temp = np.stack(temp, axis=1) # stack theta 2-51 as column >> shape as (time, 50)
    np.savez_compressed(f"{output_dir}/{key}_draw.npz", draw=temp.astype(np.float32))
    
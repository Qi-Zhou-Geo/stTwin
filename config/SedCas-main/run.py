#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 16:56:57 2022

@author: hirschbe
"""

from SedCas import SedCas

model = SedCas()
model.load_climate()
model.load_params()
model.run_hydro()
# Add by QZ at 2026-02-09
model.M = 200 # run 200 times
model.run_sediment()
model.save_output()
model.plot_sedyield_monthly()




# Add by QZ at 2026-02-09
import xarray as xr
import numpy as np
import os
from obspy import UTCDateTime

sed = model.sed

time_index = sed.index
time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in time_index]
iteration = np.arange(sed.ls.shape[1])

sed_container = xr.Dataset(

    coords=dict(
        time=time_index,
        iteration=iteration,
        time_str=("time", time_str),
    ),

    data_vars=dict(
        ls=(("time", "iteration"), sed.ls),
        sc=(("time", "iteration"), sed.sc),
        sh=(("time", "iteration"), sed.sh),
        sed_transport_real=(("time", "iteration"), sed.so),
        sopot=(("time", "iteration"), sed.sopot),
        dfs=(("time", "iteration"), sed.dfs),
        dfspot=(("time",), sed.dfspot),
    ),
    attrs=dict(
        description="Sediment ensemble output"
    )
)

time_coord = "time_str"
t1, t2 = "2004-03-01T00:00:00", "2018-01-01T00:00:00"
mask = (sed_container.time_str >= t1) & (sed_container.time_str < t2)
sed_container = sed_container.isel(time=mask)
current_dir = os.getcwd()
os.makedirs(f"{current_dir}/1h", exist_ok=True)
sed_container.to_netcdf(f"{current_dir}/1h/sed_container_{t1[:4]}_{t2[:4]}.nc")


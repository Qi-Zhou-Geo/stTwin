#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

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
from func.toolkit.physical_unit_converter import unit_converter
from func.visulize.plotly_visualize import plotly_multi_time_series_xr
from func.visulize.plotly_visualize import plotly_multi_time_series_shade_xr
from func.SedCas import SedCas

# (1) initial the SedCas model
model_params = "SedCas_input_params_10min_bo.yaml"
model = SedCas(project_root=project_root,
               model_input_params=f"{project_root}/config/SedCas_params/{model_params}")
# you must update the params then post-processing
model._params_post_processing()
# make it as critial value
model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value


# (2) load the climate forcing data
data_type = "10-minutes"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
model.run_hydro()

# # (4) run the sediment model
model.run_stochastic_simulations(seed=0, num_iteration=200)

# update the attrs if the xr is 2024 version
template_sed_container = model._create_sed_dataset(num_iteration=1)
for var in model.sed_container.data_vars:
    model.sed_container[var].attrs = template_sed_container[var].attrs.copy()
    model.sed_output[f"{var}_Q1"].attrs = template_sed_container[var].attrs.copy()
    model.sed_output[f"{var}_Q50"].attrs = template_sed_container[var].attrs.copy()
    model.sed_output[f"{var}_Q99"].attrs = template_sed_container[var].attrs.copy()

# (5-1) visualize
time_coord = "time_str"
t1, t2 = "2004-03-01T00:00:00", "2018-01-01T00:00:00"
output_dir = f"{current_dir}/10min"
os.makedirs(output_dir, exist_ok=True)

# (5-2) hydro
mask = (model.hydro_output.time_str >= t1) & (model.hydro_output.time_str < t2)
hydro_output = model.hydro_output.isel(time=mask)

# (5-2-1) SWE
# modelled_s_depth <-> modelled_SWE
# delta_depth <-> snow_delta_depth
list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "snow_delta_depth"),
                     (time_coord, "snow_acc"), (time_coord, "snow_melt")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
fig.write_html(f"{output_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_SWE.html")


# (5-2-2) ET
# potential_ET <-> PET
# actual_ET <-> AET
list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
fig.write_html(f"{output_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_ET.html")


# (5-2-3) discharge
# discharge <-> Q
# discharge_surface <-> Qs
# discharge_sub_surface <-> Qss
list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
fig.write_html(f"{output_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_discharge.html")




# (5-3) sediemnts
mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output = model.sed_output.isel(time=mask)

# (5-3-1) landslides
list_of_col_names = [(time_coord, "ls_Q1"),
                     (time_coord, "ls_Q50"),
                     (time_coord, "ls_Q99")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output, list_of_col_names=list_of_col_names)
fig.write_html(f"{output_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_ls.html")


# (5-3-2) sediments
# sed_output_catchment_q_Q50 <-> sed_transport_real_Q50
list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_transport_real_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output, list_of_col_names=list_of_col_names)
fig.write_html(f"{output_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_sediments.html")


# (6) dump to local, you need dump all interations
time_coord = "time_str"
t1, t2 = "2004-03-01T00:00:00", "2018-01-01T00:00:00"
mask = (model.sed_container.time_str >= t1) & (model.sed_container.time_str < t2)
sed_container = model.sed_container.isel(time=mask)
sed_container.to_netcdf(f"{output_dir}/sed_container_{t1[:4]}_{t2[:4]}.nc")

mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output = model.sed_output.isel(time=mask)
sed_output.to_netcdf(f"{output_dir}/sed_output_{t1[:4]}_{t2[:4]}.nc")

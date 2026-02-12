#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import pandas as pd
from obspy import UTCDateTime
# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas_re.SedCas_re import SedCas
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr

# initial the SedCas model
model_params = "SedCas_input_params_10min.yaml"
model = SedCas(project_root=project_root,
               model_input_params=f"{project_root}/config/SedCas_params/{model_params}")
model._params_post_processing()
model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value

# (2) load the climate forcing data
data_type = "10-minutes"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
model.run_hydro()

# # (4) run the sediment model
model.run_stochastic_simulations(seed=0, num_iteration=4)
#
# time_coord = "time_str"
# t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"
#
# mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
# sed_output_2017 = model.sed_output.isel(time=mask)
#
#
# list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
#                      (time_coord, "channel_storage_Q50"),
#                      (time_coord, "sed_transport_real_Q50")]
# fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
#                                   list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_sediments.html")
# fig.show()
# sssssss




# loss
from functions.SedCas_re.physical_unit_converter import unit_converter
from functions.SedCas_re.loss_func import likehood_loss
file_name = "debris_flow_volume_2004_2022.txt"
y_obs = pd.read_csv(f"{project_root}/data/event_catalog/{file_name}", skiprows=6, header=0)
sed_transport_real = model.sed_container["sed_transport_real"].copy()
# conver mm to m^3
y_pred = unit_converter(input=sed_transport_real, catchment_area=model.cfg.c_area.value, method="area-aggregated")
total_loss, details_loss = likehood_loss(y_obs, y_pred)
print(f"Loss: {UTCDateTime.now().isoformat()}")

ssss


# # # (5) visualize
time_coord = "time_str"
t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"

# climate forcing
mask = (model.climate_forcing.time_str >= t1) & (model.climate_forcing.time_str < t2)
climate_forcing_2017 = model.climate_forcing.isel(time=mask)
list_of_col_names = [(time_coord, "precipitation"), (time_coord, "temperature"), (time_coord, "sun_radiation")]
fig = plotly_multi_time_series_xr(xr_dataset=climate_forcing_2017, list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_climate_forcing.html")
fig.show()


# hydro
mask = (model.hydro_output.time_str >= t1) & (model.hydro_output.time_str < t2)
hydro_output_2017 = model.hydro_output.isel(time=mask)

# SWE
list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "snow_delta_depth"),
                     (time_coord, "snow_acc"), (time_coord, "snow_melt")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_SWE.html")
fig.show()


# ET
list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_ET.html")
fig.show()


list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_discharge.html")
fig.show()




# sed
mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output_2017 = model.sed_output.isel(time=mask)

# landslides
# list_of_col_names = [(time_coord, "ls_Q1"),
#                      (time_coord, "ls_Q50"),
#                      (time_coord, "ls_Q99")]
# fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
#                                   list_of_col_names=list_of_col_names)
# fig.show()


list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_transport_real_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                  list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_sediments.html")
fig.show()


# sed
# lower_b, mean_x, upper_b = "Q1", "Q50", "Q99"
# list_of_col_names = [(time_coord, "hillslope_storage", lower_b, mean_x, upper_b),
#                      (time_coord, "channel_storage", lower_b, mean_x, upper_b),
#                      (time_coord, "sed_output_catchment", lower_b, mean_x, upper_b)]
# fig = plotly_multi_time_series_shade_xr(xr_dataset=sed_output_2017, list_of_col_names=list_of_col_names)
# fig.show()
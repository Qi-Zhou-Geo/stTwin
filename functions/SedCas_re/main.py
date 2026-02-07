#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

from obspy import UTCDateTime

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas_re.SedCas_re import SedCas
from functions.toolkit.confidence_level_test import statistical_testing
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr
from functions.toolkit.plotly_visualize import plotly_multi_time_series_shade_xr

# initial the SedCas model
model = SedCas(project_root=project_root,
               model_input_params=f"/Users/qizhou/#python/stTwin/config/SedCas_params/SedCas_input_params_re.yaml")

# (2) load the climate forcing data
data_type = "default"  # "2017-2025"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
model.run_hydro()

# # (4) run the sediment model
model.run_stochastic_simulations(seed=0, num_iteration=10)


# # # (5) visualize
time_coord = "time_str"
t1, t2 = "2005-01-01T00:00:00", "2018-01-01T00:00:00"
mask = (model.climate_forcing.time_str >= t1) & (model.climate_forcing.time_str < t2)
climate_forcing_2017 = model.climate_forcing.isel(time=mask)
sss


# climate forcing

list_of_col_names = [(time_coord, "precipitation"), (time_coord, "temperature"), (time_coord, "sun_radiation")]
fig = plotly_multi_time_series_xr(xr_dataset=climate_forcing_2017, list_of_col_names=list_of_col_names)
fig.show()


# hydro

# SWE
list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "delta_depth"),
                     (time_coord, "snow_acc"), (time_coord, "snow_melt")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.show()


# ET
list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.show()


# Q
list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017, list_of_col_names=list_of_col_names)
fig.show()



# ls
mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output_2017 = model.sed_output.isel(time=mask)

list_of_col_names = [(time_coord, "ls_Q1"),
                     (time_coord, "ls_Q50"),
                     (time_coord, "ls_Q99")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                  list_of_col_names=list_of_col_names)
fig.show()






# sedmient
mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output_2017 = model.sed_output.isel(time=mask)


list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_output_catchment_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                  list_of_col_names=list_of_col_names)
fig.show()






# sed
lower_b, mean_x, upper_b = "Q1", "Q50", "Q99"
list_of_col_names = [(time_coord, "hillslope_storage", lower_b, mean_x, upper_b),
                     (time_coord, "channel_storage", lower_b, mean_x, upper_b),
                     (time_coord, "sed_output_catchment", lower_b, mean_x, upper_b)]
fig = plotly_multi_time_series_shade_xr(xr_dataset=sed_output_2017, list_of_col_names=list_of_col_names)
fig.show()
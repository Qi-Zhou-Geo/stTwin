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
from functions.SedCas_bo.SedCas import SedCas
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr

# initial the SedCas model
model_params = "SedCas_input_params_10min_bo.yaml"
model = SedCas(project_root=project_root,
               model_input_params=f"{project_root}/config/SedCas_params/{model_params}")
# you must update the params then post-processing
model._params_post_processing()
# make it as critial value
model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value
# model.cfg.initial_ch_storage.value = model.cfg.channel_storage_cap.value

# (2) load the climate forcing data
data_type = "10-minutes"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
model.run_hydro()

# # (4) run the sediment model
model.run_stochastic_simulations(seed=0, num_iteration=10)

time_coord = "time_str"
t1, t2 = "2004-02-01T00:00:00", "2023-01-01T00:00:00"
mask = (model.sed_output.time_str >= t1) & (model.sed_output.time_str < t2)
sed_output_2017 = model.sed_output.isel(time=mask)
list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_transport_real_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                  list_of_col_names=list_of_col_names)
fig.show()

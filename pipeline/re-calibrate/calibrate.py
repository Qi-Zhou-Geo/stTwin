#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-24
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys
sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.SedCas.SedCas import SedCas
from functions.SedCas.SedCas_new import SedCas_new
from functions.toolkit.plotly_visualize import plotly_multi_time_series_xr
from functions.toolkit.plotly_visualize import plotly_multi_time_series_shade_xr



## test the default data + old methods
time_step = "10min-new-q_df"

# initial the SedCas model
model2 = SedCas_new(project_root=project_root)


# (2) load the climate forcing data
data_type = "2004_2017_t"
model2.load_climate_input(data_type=data_type)

# # (3) run the hydro model
hydro_container = model2.run_hydro()
# model.cfg.print_config_params(check_params="out_hydro_container")

# # (4) run the sediment model
sed_container, sed_container_stats = model2.run_stochastic_simulations(seed=0, num_iteration=100)

t1, t2 = "2005-01-01T00:00:00", "2018-01-01T00:00:00"
hydro_container_2017 = hydro_container.sel(time=slice(t1, t2))

time_coord = "time_str"
list_of_col_names = [(time_coord, "precipitation"), (time_coord, "discharge"), (time_coord, "discharge_surface")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_container_2017, list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/{data_type}_hydro_container_2017_{time_step}.html")


sed_container_stats_2017 = sed_container_stats.sel(time=slice(t1, t2))
list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_output_catchment_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_container_stats_2017,
                                  list_of_col_names=list_of_col_names)
fig.write_html(f"{current_dir}/{data_type}_sed_container_2017_{time_step}.html")

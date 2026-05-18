#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-24
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import pandas as pd

#region ### add the sys.path to search for custom modules ###
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys
sys.path.append(str(project_root))
# endregion


# import the custom functions
from func.toolkit.confidence_level_test import statistical_testing
from func.visulize.plotly_visualize import plotly_multi_time_series_xr
from func.visulize.plotly_visualize import plotly_multi_time_series_shade_xr
# all the following functions are stored under the same path
from SedCas import SedCas


# (0) initial the SedCas model
model = SedCas(project_root=project_root)

# (1) load the pre-calibrated parameters
yaml_file = "SedCas_input_params_default.yaml"
yaml_path = f"{project_root}/config/SedCas_params/{yaml_file}"
model.load_default_params(yaml_path=yaml_path, log_params=False)

# (2) load the climate forcing data
data_type = "default-extend"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
hydro_container = model.run_hydro()

# (4) run the sediment model
sed_container, sed_container_stats = model.run_stochastic_simulations(seed=0, num_iteration=100)

# (5-1) visualize
time_coord = "time_str"
t1, t2 = "2000-01-01T00:00:00", "2023-01-01T00:00:00"


# (5-2) hydro
mask = (hydro_container.time_str >= t1) & (hydro_container.time_str < t2)
hydro_output = hydro_container.isel(time=mask)

# (5-2-1) SWE
# modelled_s_depth <-> modelled_SWE
# delta_depth <-> snow_delta_depth
list_of_col_names = [(time_coord, "modelled_s_depth"), (time_coord, "delta_depth"),
                     (time_coord, "snow_accumulation"), (time_coord, "snow_melt")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_SWE.html")
# fig.show()


# (5-2-2) ET
# potential_ET <-> PET
# actual_ET <-> AET
list_of_col_names = [(time_coord, "albedo"), (time_coord, "potential_ET"), (time_coord, "actual_ET")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_ET.html")
# fig.show()

# (5-2-3) discharge
# discharge <-> Q
# discharge_surface <-> Qs
# discharge_sub_surface <-> Qss
list_of_col_names = [(time_coord, "discharge"), (time_coord, "discharge_surface"), (time_coord, "discharge_sub_surface")]
fig = plotly_multi_time_series_xr(xr_dataset=hydro_output, list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_discharge.html")
# fig.show()



# (5-3) sediemnts
mask = (sed_container_stats.time_str >= t1) & (sed_container_stats.time_str < t2)
sed_output = sed_container_stats.isel(time=mask)

# (5-3-1) landslides
list_of_col_names = [(time_coord, "ls_Q1"),
                     (time_coord, "ls_Q50"),
                     (time_coord, "ls_Q99")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output, list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_ls.html")
# fig.show()

# (5-3-2) sediments
# sed_output_catchment_q_Q50 <-> sed_transport_real_Q50
list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                     (time_coord, "channel_storage_Q50"),
                     (time_coord, "sed_output_catchment_q_Q50")]
fig = plotly_multi_time_series_xr(xr_dataset=sed_output, list_of_col_names=list_of_col_names)
# fig.write_html(f"{current_dir}/resolution_{data_type}_{t1[:4]}_{t2[:4]}_sediments.html")
fig.show()

# (6) dump to local
time_coord = "time_str"
t1, t2 = "2004-03-01T00:00:00", "2023-01-01T00:00:00"
mask = (sed_container_stats.time_str >= t1) & (sed_container_stats.time_str < t2)
sed_output = sed_container_stats.isel(time=mask)
sed_output.to_netcdf(f"sed_output_{t1[:4]}_{t2[:4]}.nc")

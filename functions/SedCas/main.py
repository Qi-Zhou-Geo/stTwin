#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-09-24
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from SedCas import SedCas
from functions.toolkit.confidence_level_test import statistical_testing

# initial the SedCas model
model = SedCas(project_root=project_root)

data_type = "2017-2025"
# load the climate forcing data
model.load_climate(data_type=data_type)

# load the pre-calibrated parameters
model.load_params()

# print model config params
# model.log_config_params()

# run the hydro model
hyd = model.run_hydro()


# run the sediment model
sed = model.run_sediment()

sedout, sed, sed.sopot = model.save_output(h_name=f"Hydro_{data_type}.txt",
                           s_name=f"Sediment_{data_type}.txt")

# model.plot_sedyield_monthly()

# t = np.array(model.sed.index)
# t_str = np.datetime_as_string(t, unit='s')
# id1 = np.where(t_str=="2017-05-19T09:00:00")[0][0]
# id2 = np.where(t_str=="2017-05-19T15:00:00")[0][0] + 1
#
# output_mean, output_ci_range = statistical_testing(input_data=model.sed.so, row_or_column="row", confidence_interval=0.95)
#
# x = np.arange(0, id2-id1)
# plt.scatter(x, output_mean[id1: id2])
#
# plt.fill_between(x,
#                  output_mean[id1: id2]-output_ci_range[id1: id2],
#                  output_mean[id1: id2]+output_ci_range[id1: id2], color='blue', alpha=0.2, label='95% CI')
#
#
# plt.show()


# df = fetch_data4SedCas(station="mve", granularity="T")

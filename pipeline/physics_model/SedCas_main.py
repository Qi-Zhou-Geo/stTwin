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
from functions.toolkit.confidence_level_test import statistical_testing

# initial the SedCas model
model = SedCas(project_root=project_root)


# (1) load the pre-calibrated parameters
model.load_default_params(log_params=False)

# (2) load the climate forcing data
data_type = "default" #"2017-2025"
model.load_climate_input(data_type=data_type)

# (3) run the hydro model
hydro, SWE, PET, HYM = model.run_hydro()

# (4) run the sediment model
sed_container = model.run_sediment(total_iteration=2)

# sedout = model.save_output(h_name=f"Hydro_{data_type}.txt",
#                            s_name=f"Sediment_{data_type}.txt")

# list_of_tuples =
# model.results_visualize(list_of_tuples=None)
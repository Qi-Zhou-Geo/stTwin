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
from SedCas import SedCas
from functions.toolkit.confidence_level_test import statistical_testing

# initial the SedCas model
model = SedCas(project_root=project_root)


# load the pre-calibrated parameters
model.load_default_params(log_params=False)


data_type = "2017-2025"
# load the climate forcing data
model.load_climate(data_type=data_type)


# run the hydro model
hydro, SWE, PET, HYM = model.run_hydro()


# run the sediment model
sed = model.run_sediment()
sdf

sedout, sed, sed.sopot = model.save_output(h_name=f"Hydro_{data_type}.txt",
                                           s_name=f"Sediment_{data_type}.txt")

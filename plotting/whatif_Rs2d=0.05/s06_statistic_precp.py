#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-24T19:16:02
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import numpy as np
import pandas as pd

from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

df_path = Path(project_root) / "data/SedCas_input/climate_1931_2025_d.txt"
df = pd.read_csv(df_path)

df["date"] = pd.to_datetime(df["timestamp [UTC+0]"])

jan = df[(df["date"].dt.month == 1) & (df["date"].dt.day <= 31)]

mean_jan_precip_sum = (
    jan.groupby(jan["date"].dt.year)["precipitation [mm per time_step]"]
    .sum()
    .mean()
)

mean_jan_temperature = jan["temperature [degree]"].mean()

print("Mean January precipitation total:", mean_jan_precip_sum)
print("Mean January daily temperature:", mean_jan_temperature)
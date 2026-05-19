#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-19
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import yaml
import numpy as np
import pandas as pd
from itertools import product

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the func. from the same folder


def cache_bound(output_path, 
                cycle_period_range, storm2drought_ratio_range, 
                storm_onset_month_range, storm_onset_day_range):
    
    # create all parameter combinations
    params = list(product(
        cycle_period_range,
        storm2drought_ratio_range,
        storm_onset_month_range,
        storm_onset_day_range
    ))
    idx = np.arange(0, len(params))
    
    # create dataframe
    arr = np.hstack((idx.reshape(-1, 1), params))
    columns = ["idx", "cycle_period", "storm2drought_ratio", "storm_onset_month", "storm_onset_day"]
    df = pd.DataFrame(arr, columns=columns)
    df.to_csv(output_path, index=False)


def load_what_if_bound():
    
    what_if_bound = Path(project_root) / "config" / "SedCas_params" / "what_if_range.yaml"
    # load YAML file
    with open(what_if_bound, "r") as f:
        data = yaml.safe_load(f)

    cfg = data["cycle_period"]
    cycle_period_range = np.linspace(cfg["value_min"], cfg["value_max"], cfg["value_num"])

    cfg = data["storm2drought_ratio"]
    storm2drought_ratio_range = np.linspace(cfg["value_min"], cfg["value_max"], cfg["value_num"])

    cfg = data["storm_onset_month"]
    storm_onset_month_range = np.linspace(cfg["value_min"], cfg["value_max"], cfg["value_num"])

    cfg = data["storm_onset_day"]
    storm_onset_day_range = np.linspace(cfg["value_min"], cfg["value_max"], cfg["value_num"])
    
    print(f"Note:\n"
          f"Total combinations: {len(cycle_period_range) * len(storm2drought_ratio_range) * len(storm_onset_month_range) * len(storm_onset_day_range)}\n"
          f"len(cycle_period_range)={len(cycle_period_range)}, len(storm2drought_ratio_range)={len(storm2drought_ratio_range)}\n"
          f"len(storm_onset_month_range)={len(storm_onset_month_range)}, len(storm_onset_day_range)={len(storm_onset_day_range)}\n")
    
    return cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range


#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-19
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import yaml
import numpy as np
import pandas as pd
from itertools import product

# region ### add the sys.path to search for custom modules ###
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
    df = pd.DataFrame(arr, columns=columns).astype(
        {
            "idx": int,
            "cycle_period": int,
            "storm2drought_ratio": float,
            "storm_onset_month": int,
            "storm_onset_day": int,
        }
    )
    
    df.to_csv(output_path, index=False)


def load_what_if_bound():
    
    what_if_bound = Path(project_root) / "config" / "SedCas_params" / "what_if_range.yaml"
    # load YAML file
    with open(what_if_bound, "r") as f:
        cfg = yaml.safe_load(f)

    data = cfg["cycle_period"]
    cycle_period_range = np.linspace(data["value_min"], data["value_max"], data["value_num"])
    cycle_period_range = cycle_period_range.astype(int)
    
    data = cfg["storm2drought_ratio"]
    storm2drought_ratio_range = np.linspace(data["value_min"], data["value_max"], data["value_num"])
    storm2drought_ratio_range = storm2drought_ratio_range.astype(float)
    
    data = cfg["storm_onset_month"]
    storm_onset_month_range = np.linspace(data["value_min"], data["value_max"], data["value_num"])
    storm_onset_month_range = storm_onset_month_range.astype(int)
    
    data = cfg["storm_onset_day"]
    storm_onset_day_range = np.linspace(data["value_min"], data["value_max"], data["value_num"])
    storm_onset_day_range = storm_onset_day_range.astype(int)

    return cfg, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range

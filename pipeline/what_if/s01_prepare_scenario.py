#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-04-20
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import json

import pandas as pd
import numpy as np

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
from func.SedCas_whatif.create_bound import load_what_if_bound, cache_bound

def main():
    # (1) create the scenario from yaml config
    cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range = load_what_if_bound()
    
    # (2) save as txt
    output_path = Path(project_root) / "pipeline" / "what_if" / "scenario_bound.txt"
    cache_bound(output_path, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range)

if __name__ == "__main__":
    main()
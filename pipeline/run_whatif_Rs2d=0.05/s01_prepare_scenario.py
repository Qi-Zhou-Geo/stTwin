#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-19T17:27:00
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import argparse
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

# import custom func.
from func.SedCas_whatif.create_bound import load_what_if_bound, cache_bound

def main(method="from_yaml", **kwargs):
    # (1) create the scenario from yaml config
    
    if method == "from_yaml":
        cfg, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range = load_what_if_bound()
    elif method == "manual":
        cycle_period_range = np.array(kwargs["cycle_period_range"], dtype=int)
        storm2drought_ratio_range = np.array(kwargs["storm2drought_ratio_range"], dtype=float)
        storm_onset_month_range = np.array(kwargs["storm_onset_month_range"], dtype=int)
        storm_onset_day_range = np.array(kwargs["storm_onset_day_range"], dtype=int)
    else:
        raise ValueError(f"Please check your method={method}")
    
        
    print(f"Note: Method={method}\n"
          f"Total combinations: {len(cycle_period_range) * len(storm2drought_ratio_range) * len(storm_onset_month_range) * len(storm_onset_day_range)}\n"
          f"len(cycle_period_range)={len(cycle_period_range)}, len(storm2drought_ratio_range)={len(storm2drought_ratio_range)}\n"
          f"len(storm_onset_month_range)={len(storm_onset_month_range)}, len(storm_onset_day_range)={len(storm_onset_day_range)}\n")
    
    
    # (2) save as txt
    output_path = Path(current_dir) / "scenario_bound.txt"
    cache_bound(output_path, cycle_period_range, storm2drought_ratio_range, storm_onset_month_range, storm_onset_day_range)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--method", type=str, default="from_yaml")
    parser.add_argument("--cycle_period_range", type=int, nargs='+', default=None)
    parser.add_argument("--storm2drought_ratio_range", type=float, nargs='+', default=None)
    parser.add_argument("--storm_onset_month_range", type=int, nargs='+', default=None)
    parser.add_argument("--storm_onset_day_range", type=int, nargs='+', default=None)
    args = parser.parse_args()
    
    main(
        method=args.method,
        cycle_period_range=args.cycle_period_range,
        storm2drought_ratio_range=args.storm2drought_ratio_range,
        storm_onset_month_range=args.storm_onset_month_range,
        storm_onset_day_range=args.storm_onset_day_range,
    )

# usage
# python s01_prepare_scenario.py --method "manual" \
#     --cycle_period_range 30 45 60 75 90 105 120 \
#     --storm2drought_ratio_range 0.05 \
#     --storm_onset_month_range 1 2 3 4 5 6 7 8 9 \
#     --storm_onset_day_range 1
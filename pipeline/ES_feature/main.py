#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-01-20
# __author__ = Qi Zhou and Sibashish Dash, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this code without the author's permission

import os
import yaml

import argparse

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# from brokenaxes import brokenaxes

from obspy import Stream, Trace, read
from obspy.core import UTCDateTime  # default is UTC+0 time zone

from scipy.signal import hilbert

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.cal_ES import single_day_ES
from functions.seismic.seismic_data_processing import load_seismic_signal


def main(year, julday, window_size, window_overlap):

    catchment_name, seismic_network = "Illgraben", "9S"
    input_station, input_component = "ILL12", "EHZ"
    f_min, f_max = 1, 50

    t_s = UTCDateTime(year=year, julday=julday)
    t_e = UTCDateTime(year=year, julday=julday + 1)

    st = load_seismic_signal(catchment_name, seismic_network,
                             input_station, input_component,
                             t_s, t_e,
                             f_min=f_min, f_max=f_max,
                             remove_sensor_response=True)

    total_data_volume = (t_e - t_e) * st[0].stats.sampling_rate

    # check whether data gap exits
    if total_data_volume != st[0].stats.sampling_rate:
        print("Total data volume does not match sampling rate")
    else:
        pass

    single_day_ES(st, window_size, window_overlap)


if __name__ == "__main__":
    # sinfo -n node[501-514] -N --Format="Nodelist,CPUsState,AllocMem,Memory,GresUsed,Gres"
    parser = argparse.ArgumentParser(description='input parameters')

    parser.add_argument("--year", default=2020, type=int)
    parser.add_argument("--julday", default=123, type=int)
    parser.add_argument("--window_size", default=60, type=int)
    parser.add_argument("--window_overlap", default=0, type=float)

    args = parser.parse_args()

    main(args.year, args.julday, args.window_size, args.window_overlap)


f = np.load(f"/Users/qizhou/#python/stTwin/data/seismic_temp/seis_energy/9S.ILL12.EHZ.2018.146.npz")
t_str = f["t_str"]
t_float = f["t_float"]
seismic_energy = f["seismic_energy"]

for i in range(5):
    plt.plot(seismic_energy[:, i])
plt.show()
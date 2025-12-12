#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2024-02-23
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
import argparse

import pandas as pd
import numpy as np

from scipy.signal import hilbert, lfilter, butter, spectrogram
from scipy.stats import kurtosis, skew, iqr

from tqdm import tqdm

from datetime import datetime, timezone, timedelta
from obspy import Stream

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object_typeect moves one level up the directory hierarchy
project_root = current_dir.parent.parent
import sys
sys.path.append(str(project_root))
# </editor-fold>

# import the custom functions
from functions.seismic.st2tr import stream_to_trace
from functions.seismic.chunk_st2seq import chunk_data


def get_freq_band_energy(data,
                         sps,
                         freq_lower=(1, 5, 15, 25, 35),
                         freq_upper=(5, 15, 25, 35, 45)):

    '''
    Calculate log10 energy in multiple frequency bands using bandpass filtering + Hilbert transform.

    Args:
        data: array-like, 1D seismic waveform.
        sps: float, data sampling rate (samples per second).
        freq_lower: tuple, frequency band edges
        freq_upper: tuple, frequency band edges

    Returns:

    '''

    # convert to array
    data = np.asarray(data, dtype=float)

    if len(freq_lower) != len(freq_upper):
        raise ValueError(f"freq_lower = {freq_lower} and "
                         f"freq_upper = {freq_upper} must have same length.")

    nf = len(freq_lower)
    seismic_energy = np.empty(nf, dtype=float)

    Nyquist = sps / 2

    for i, (f_low, f_high) in enumerate(zip(freq_lower, freq_upper)):

        # Normalize to Nyquist
        wn = [f_low / Nyquist, f_high / Nyquist]
        # Butterworth bandpass filter
        b, a = butter(N=2, Wn=wn, btype='band')
        # Filter data
        data_filt = lfilter(b, a, data)

        # Hilbert envelope energy (trapz integral)
        analytic_signal = hilbert(data_filt)
        envelope = np.abs(analytic_signal)

        seismic_energy[i] = np.log10(np.trapz(envelope))

    return seismic_energy


def single_day_ES(st, window_size, window_overlap,
                  freq_lower=(1, 5, 15, 25, 35),
                  freq_upper=(5, 15, 25, 35, 45)):

    tr = stream_to_trace(st=st)
    data = tr.data
    data_start_time = tr.stats.starttime.strftime("%Y-%m-%dT%H:%M:%S")
    data_sps = tr.stats.sampling_rate

    # chunk the data to avoid the loop-st.trim
    t_value, t_str, chunk_x = chunk_data(data, data_start_time, data_sps, window_size, window_overlap)
    print(len(t_str), len(data))
    # prepare the empty temp_ES to store the results
    temp_ES = np.empty((len(t_str), len(freq_lower)), dtype=float)
    for i in tqdm(range(len(t_str)), file=sys.stdout):
        data_temp = chunk_x[i]

        es_arr = get_freq_band_energy(data_temp, data_sps,
                                      freq_lower=freq_lower,
                                      freq_upper=freq_upper)
        temp_ES[i] = es_arr

    # save the npy file
    output_path = f"{project_root}/data/seismic_temp/seis_energy"
    os.makedirs(output_path, exist_ok=True)
    output_name = f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel}.{tr.stats.starttime.year}.{tr.stats.starttime.julday:03d}"
    np.savez(f"{output_path}/{output_name}.npz",
             allow_pickle=True,
             t_str=t_str,
             t_float=t_value,
             seismic_energy=temp_ES)

def load_ES_energy(network, station, channel, year, julday,
                   print_output=False,
                   freq_lower=(1, 5, 15, 25, 35),
                   freq_upper=(5, 15, 25, 35, 45)):

    # save the npy file
    output_path = f"{project_root}/data/seismic_temp/seis_energy"
    os.makedirs(output_path, exist_ok=True)
    output_name = f"{network}.{station}.{channel}.{year}.{julday:03d}"

    with np.load(f"{output_path}/{output_name}.npz", "r", allow_pickle=True) as f:
        t_str = f["t_str"]
        t_float = f["t_float"]
        seismic_energy = f["seismic_energy"]


    delta = t_float[1] - t_float[0]
    sps = 1 / delta

    output_temp = np.hstack((t_str.reshape(-1, 1), t_float.reshape(-1, 1), seismic_energy))

    output_header = ["t_str", "t_float"]
    for i, j in zip(freq_lower, freq_upper):
        h = f"seismic_energy{i}-{j}Hz"
        output_header.append(h)

    if print_output is True:
        print(f"{output_name}, delta: {delta}, SPS: {sps}, output_temp.shape: {output_temp.shape}")

    return output_temp, output_header
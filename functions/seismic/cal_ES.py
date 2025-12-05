#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2025-06-19
#__author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import pytz
from datetime import datetime, timedelta

import numpy as np


def chunk_data(data, data_start_time, data_sps, window_size, window_overlap):

    x_seq_length = int(data_sps * window_size)  # x_seq_length unit by data point
    overlap_length = int(x_seq_length * (1 - window_overlap))  # unit by data point

    # prepare the float time stamps
    date_start_time = datetime.strptime(data_start_time, "%Y-%m-%dT%H:%M:%S")
    timestamps = np.array([date_start_time + timedelta(seconds= i / data_sps) for i in range(len(data))])
    timestamps = np.array([(ts - datetime(1970, 1, 1)).total_seconds() for ts in timestamps])
    
    # prepare a 2D data numpy array
    if window_overlap != 0:  # for overlap window
        chunk_t = np.lib.stride_tricks.sliding_window_view(timestamps, x_seq_length)[::overlap_length]
        chunk_x = np.lib.stride_tricks.sliding_window_view(data, x_seq_length)[::overlap_length]
    else: # for none overlap window
        num_windows = len(data) // x_seq_length
        chunk_t = timestamps[:num_windows * x_seq_length].reshape(-1, x_seq_length)
        chunk_x = data[:num_windows * x_seq_length].reshape(-1, x_seq_length)

    t_value = chunk_t[:, 0]
    t_str = [datetime.utcfromtimestamp(ts).strftime('%Y-%m-%dT%H:%M:%S.%f') for ts in t_value]
    chunk_x = chunk_x

    return t_value, t_str, chunk_x


data = np.arange(1, 1440*60)
data_start_time = "2020-10-01T00:00:00"
data_sps = 1
window_size = 60
window_overlap = 0

t_value, t_str, chunk_x = chunk_data(data, data_start_time, data_sps, window_size, window_overlap)

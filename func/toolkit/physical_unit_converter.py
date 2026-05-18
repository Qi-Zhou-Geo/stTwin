#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-09
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np

def unit_converter(input, catchment_area, method):
    '''
    Convert the physical unit for the model input and output

    Args:
        input (float): model input or output param
        catchment_area (float): catchment area, physical unit by km^2
        method (str): distinguishing method, either "area-weighted" or "area-aggregated"

    Returns:
        output (float): param with physical unit
    '''

    try:
        output = input.copy()
    except AttributeError as e:
        output = input

    if method == "area-weighted":
        # area-normalized physical param, unit: m
        output = output / (catchment_area * 1e6)
        # convert m to mm
        output = output * 1e3
    elif method == "area-aggregated":
        # convert mm to m
        output = output / 1e3
        # back to area aggregated, unit is m
        output = output * (catchment_area * 1e6)

    return output

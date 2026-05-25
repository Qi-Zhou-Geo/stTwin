#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-05-13
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo

import numpy as np

def custom_boundary():
    
    theta_names = np.array([
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v', 
        'Qdf', 'max_s2w', 
        'channel_storage_cap', 'erosion_k'], dtype=str)

    lower_bounds = np.array([0.1, 10, 10, # time step, 0.1 step >> 0.1 * 10 minutes = 1 min
                             1, 6, 6, 
                             1.1, 
                             0.1, 0.1, 
                             1, 0.01], dtype=float)

    upper_bounds = np.array([10, 100, 100, 
                             144, 1008, 1008, 
                             2.0, 
                             1.0, 1.0, 
                             100, 10], dtype=float)


    return theta_names, lower_bounds, upper_bounds
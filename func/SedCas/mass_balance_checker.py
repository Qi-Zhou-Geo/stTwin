#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-14T16:26:06
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import xarray as xr


def mass_balance_checker(sed_container, residual=1.0, iteration=None, silence=True):
    
    """
    Check sediment mass balance for each model iteration.

    The balance equation is:
        sed_yield = ls_input + hillslope_storage_loss + channel_storage_loss

    where:
        sed_yield = total sediment exported (sum over period)
        
        ls_input = total landslide input (sum over period)
        hillslope_storage_loss = hillslope storage at start - end
        channel_storage_loss  = channel storage at start - end

    Args:
        sed_container (xr.Dataset): sed_container dataset.
        residual (float): tolerance threshold for imbalance warning, unit by "area-weighted" mm
        iteration (int, optional): iteration to check. If None, check all iterations.
        silence (bool): if True, suppress success messages.
    """
    
    ds = sed_container

    if iteration is None:
        iterations = ds.coords["iteration"].values
    else:
        iterations = [iteration]

    results = []
    for it in iterations:
        sed_yield = ds["sed_transport_real"].sel(iteration=it).values
        sed_out = np.sum(sed_yield)

        ls_input = ds["ls"].sel(iteration=it).values
        ls_input_sum = np.sum(ls_input)

        hs = ds["hillslope_storage"].sel(iteration=it).values
        hs_loss = hs[0] - hs[-1]

        ch = ds["channel_storage"].sel(iteration=it).values
        ch_loss = ch[0] - ch[-1]

        sed_in = ls_input_sum + hs_loss + ch_loss

        if np.abs(sed_in - sed_out) >= residual:
            msg = (f"<mass_balance_checker> Warning! {it}\n"
                   f"Sediment transport may not balance at iteration.\n"
                   f"sed_in={sed_in}, sed_out={sed_out}, residual={residual}\n")
        else:
            msg = f"<mass_balance_checker> All good! {it}"
        
        if silence is True:
            print(msg)
        
        results.append({
            "iteration": it,
            
            "ls_input_sum": ls_input_sum,
            "hs_loss": hs_loss,
            "ch_loss": ch_loss,
            
            "sed_in": sed_in,
            
            "sed_out": sed_out
        })
             
    return results


#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-10
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import warnings
import numpy as np
import pandas as pd
import xarray as xr
from obspy import UTCDateTime

def get_mu_sigma(y_pred, start_time, end_time):
    '''
    Calculate the mean and standard deviation of a given event.

    Args:
        y_pred (xr.Dataset): SedCas predicted sediments transport,
            shape by [time_step, iteration], physical unit m^3.
        start_time (str): event start time, format by %Y-%m-%dT%H:%M:%S
        end_time (str): event end time, format by %Y-%m-%dT%H:%M:%S

    Returns:
        mu (float): mean of the aggregated debris flow volume along the iteration domain
        sigma (float): standard deviation of the aggregated debris flow volume along the iteration domain
    '''

    mask = (y_pred.time_str >= start_time) & (y_pred.time_str < end_time)
    # shape by [time_step, iteration]
    sed_output = y_pred.isel(time=mask).values
    # sum along the time domain -> shape [iteration]
    aggregate_sed = np.sum(sed_output, axis=0)

    if np.any(np.isnan(aggregate_sed)):
        # this make sure np.log(aggregate_sed) work
        raise ValueError("NaN encountered in aggregated sediment volume.")

    if np.any(aggregate_sed <= 0):
        raise RuntimeError(
            f"Model failed for event [{start_time}, {end_time}]: "
            "non-positive aggregated sediment volume."
        )

    log_agg = np.log(aggregate_sed) # transform ensemble to log-space with base e

    # statistics in log-space
    mu = np.mean(log_agg)
    sigma = np.std(log_agg, ddof=1) # ddof=1 -> unbiased sample estimator

    # if sigma < 1e-20:
    #     print(f"Warning! Ensemble sigma is nearly zero (sigma={sigma}).")

    return mu, sigma

def loss_func(volume_obs, mu, sigma, sigma0=1e-3):
    '''
    Calculate the likelihood-based loss for single event level.

    # equal to:
    # \left(
    # \frac{\log(V^\mathrm{obs}_{i}) - \mu_i^{\log} (\theta)} {\sigma^{\log}_i(\theta)}
    # \right)^2
    # + \log \left( (\sigma^{\log}_i(\theta))^2 \right)

    Args:
        volume_obs (float): field bserved debris-flow volume in raw base, unit by m^3
        mu (float): mean of the aggregated debris flow volume along the iteration domain in log base
        sigma (float): standard deviation of the aggregated debris flow volume along the iteration domain in log base
        sigma0 (float): default value is 1e-3
    Returns:
        loss (float): likelihood of the loss for single event level
    '''

    # sigma_eff adds a small variance floor (sigma0) to prevent the likelihood
    # from becoming numerically unstable when the ensemble spread collapses (sigma → 0).
    # This represents irreducible uncertainty
    # and avoids overconfident penalties in the negative log-likelihood.
    sigma_eff = np.sqrt(sigma ** 2 + sigma0 ** 2) #
    loss = ((np.log(volume_obs) - mu) / sigma_eff)**2 + np.log(sigma_eff**2)

    return loss

def likehood_loss(y_obs, y_pred, buffer_time=3, default_loss=1e10):
    """
    Compute the likelihood-based loss aggregated over all debris-flow events.

    Args:
        y_obs (pandas.DataFrame):
            Event-level observation table in which each row corresponds to one debris-flow event:
            - column 0: event start time (datetime-like or ISO string)
            - column 1: event end time (datetime-like or ISO string)
            - column 2: observed debris-flow volume (m^3)

        y_pred (xarray.Dataset):
            Model-predicted sediment transport from SedCas.
            The dataset contains the variable "sed_transport_real"` with dimensions
            [time_step, iteration], representing stochastic realizations of sediment transport (m^3).

        buffer_time (float or int): Time (in hours) added before and after the
            event start and end times to account for potential inaccuracies in the labeled timestamps.

        default_loss (float): if the event faliure, e.g., aggregate_sed <= 0,
            default loss value is 1e-10 or np.nanmean(event_level_loss)

    Returns:
        total_loss (float): Total negative log-likelihood summed over all events with available observed volumes.
    """


    event_level_loss = []
    y_obs = np.array(y_obs)

    if buffer_time is not None:
        # extend the event duration
        for event_id in range(len(y_obs)):
            y_obs[event_id, 0] = (UTCDateTime(y_obs[event_id, 0]) - buffer_time * 3600).strftime("%Y-%m-%dT%H:%M:%S")
            y_obs[event_id, 1] = (UTCDateTime(y_obs[event_id, 1]) + buffer_time * 3600).strftime("%Y-%m-%dT%H:%M:%S")

    details_loss = []
    for event_id in range(len(y_obs)):

        volume_obs = y_obs[event_id, 2]
        if np.isnan(volume_obs): # is True
            # no observed volume, only with event document
            pass
        else:
            start_time, end_time = y_obs[event_id, 0], y_obs[event_id, 1]

            try:
                mu, sigma = get_mu_sigma(y_pred, start_time, end_time)
                loss = loss_func(volume_obs, mu, sigma)
            except RuntimeError:

                mu, sigma = np.nan, np.nan
                if len(event_level_loss) == 0:
                    # in case this situation occurs in the first event
                    loss = default_loss
                else:
                    loss = max(default_loss, np.nanmean(event_level_loss))

            event_level_loss.append(loss)

            # save the details
            record = [default_loss, event_id, volume_obs, start_time, end_time, f"{loss:.1f}", mu, sigma]
            record = ", ".join(map(str, record))
            details_loss.append(record)

    total_loss = np.nansum(event_level_loss)

    return total_loss, details_loss


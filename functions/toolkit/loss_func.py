#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-10
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import numpy as np
import xarray as xr
from obspy import UTCDateTime


def get_mu_sigma(y_pred, start_time, end_time, ratio_of_faliure=0.5):
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

    # (1) select the model predicted in the time period
    mask = (y_pred.time_str >= start_time) & (y_pred.time_str < end_time)
    # shape by [time_step, iteration]
    sed_output = y_pred.isel(time=mask).values
    # sum along the time domain -> shape [iteration]
    aggregate_sed = np.sum(sed_output, axis=0) # this could contain zero

    if np.any(np.isnan(aggregate_sed)):
        # this make sure np.log10(aggregate_sed) work, it is very rare case
        raise RuntimeError("NaN encountered in aggregated sediment volume.")

    # (2) check whether too many zero-prediction
    non_zero_volume_id = aggregate_sed > 0 # model predicted volume > 0
    if np.sum(non_zero_volume_id) > ratio_of_faliure * len(aggregate_sed):
        # at least, it needs {(1-ratio_of_faliure)*100}% non-zero predicted volume for next step
        aggregate_sed = aggregate_sed[non_zero_volume_id]
    else:
        print(f"{start_time} to {end_time}:\n"
              f"num_zero_volume_prediction={len(aggregate_sed) - np.sum(non_zero_volume_id)}\n")
        raise RuntimeError(
            f"Model contains more than {(1-ratio_of_faliure)*100}% zero predicted volume for:"
            f"[{start_time}, {end_time}]: "
        )

    # (3) calulate the mean (mu) and std (sigma) in log10 space
    mean_aggregate_sed = np.mean(aggregate_sed)  # non-log10 based model predicted mean value
    log_agg = np.log10(aggregate_sed) # transform ensemble to log10-space with base e

    # statistics in log-space
    mu = np.mean(log_agg)
    sigma = np.std(log_agg, ddof=1) # ddof=1 -> unbiased sample estimator

    return mu, sigma, mean_aggregate_sed

def likehood_loss_func(volume_obs, mu, sigma, sigma0=0.05):
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
    loss = ((np.log10(volume_obs) - mu) / sigma_eff)**2 + np.log10(sigma_eff**2)

    return loss

def clean_obs_pre(y_obs, y_pred, buffer_time=3, failed_prediction=0.0, ratio_of_faliure=0.5):
    # make sure the time format of y_obs is like:
    # (a): format: "%Y-%m-%dT%H:%M:%S"
    # (b): second is zero, minute is not zero

    y_obs_valid = y_obs.copy()

    # (1) buffer the event
    if buffer_time is None:
        pass
    else:
        delta_t = buffer_time * 3600
        t_format = "%Y-%m-%dT%H:%M:%S"
        # extend the event duration
        for event_id in range(len(y_obs_valid)):
            y_obs_valid.iloc[event_id, 0] = (UTCDateTime(y_obs_valid.iloc[event_id, 0]) - delta_t).strftime(t_format)
            y_obs_valid.iloc[event_id, 1] = (UTCDateTime(y_obs_valid.iloc[event_id, 1]) + delta_t).strftime(t_format)


    # (2) select the event with valid observed volume
    y_obs_valid = y_obs_valid[y_obs_valid["Volume[m3]"].notna()]

    # (3) calculate the model predicted volume for given observation time period
    y_pred_valid = y_obs_valid.copy()
    y_pred_valid["Volume[m3]"] = failed_prediction
    y_pred_valid["Volume[m3]"] = y_pred_valid["Volume[m3]"].astype(float) # force the data type
    for event_id in range(len(y_obs_valid)):
        start_time, end_time = y_obs_valid.iloc[event_id, 0], y_obs_valid.iloc[event_id, 1]

        try:
            # (3-1) succssed prediction
            # return log10 based mu, sigma, and non-log mean_aggregate_sed [m^3]
            mu, sigma, mean_aggregate_sed = get_mu_sigma(y_pred, start_time, end_time, ratio_of_faliure=ratio_of_faliure)
        except RuntimeError:
            # (3-2) all predicted volumes are zero
            # set as zero
            mean_aggregate_sed = failed_prediction

        y_pred_valid.iloc[event_id, 2] = mean_aggregate_sed

    return y_obs_valid, y_pred_valid

def likehood_loss(y_obs, y_pred, buffer_time=3, loss_no_obs=np.nan, low_value=0, high_value=1e4):
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

    y_obs = np.array(y_obs)
    if buffer_time is None:
        pass
    else:
        delta_t = buffer_time * 3600
        # extend the event duration
        for event_id in range(len(y_obs)):
            y_obs[event_id, 0] = (UTCDateTime(y_obs[event_id, 0]) - delta_t).strftime("%Y-%m-%dT%H:%M:%S")
            y_obs[event_id, 1] = (UTCDateTime(y_obs[event_id, 1]) + delta_t).strftime("%Y-%m-%dT%H:%M:%S")

    event_level_loss = []
    details_loss = ["event_index, observed_volume [m^3], start_time, end_time, loss, "
                    "mu (log10 space), sigma (log10 space), mean_aggregate_sed [m^3], predicted_gap"]


    for event_id in range(len(y_obs)):
        volume_obs = y_obs[event_id, 2]
        start_time, end_time = y_obs[event_id, 0], y_obs[event_id, 1]

        if np.isnan(volume_obs): # is True
            # (1) no observed volume, only with event document
            mu, sigma, mean_aggregate_sed = loss_no_obs, loss_no_obs, loss_no_obs
            predicted_gap = loss_no_obs

            loss = loss_no_obs
        else:
            # (2) with observed volume
            try:
                # (2-1) succssed prediction
                # return log10 based mu, sigma
                mu, sigma, mean_aggregate_sed = get_mu_sigma(y_pred, start_time, end_time, ratio_of_faliure=0.5)
                # predicted volume (Q50 or mean) error
                predicted_gap = mean_aggregate_sed / volume_obs
                loss = likehood_loss_func(volume_obs, mu, sigma)
            except RuntimeError:
                # (2-2) all predicted volumes are zero
                # treat it as completely uninformative
                mu = low_value # low value for mu -> alway performing poorly
                sigma = high_value  # high value for sigma -> alway performing poorly
                mean_aggregate_sed = low_value # no prediction
                predicted_gap = high_value # high gap

                if event_id == 0:
                    # in case this situation occurs in the first event
                    loss = high_value
                else:
                    loss = max(low_value, np.mean(event_level_loss))

        event_level_loss.append(loss)

        # save the details
        record = [event_id, f"volume_obs={volume_obs}", start_time, end_time,
                  f"loss={loss:.1f}", mu, sigma,
                  f"volume_pred={mean_aggregate_sed:.1f}", f"{predicted_gap:.4f}"]
        record = ", ".join(map(str, record))
        details_loss.append(record)

    # do not consider
    total_loss = np.nansum(event_level_loss)

    return total_loss, details_loss

def ratio_loss(y_obs, y_pred, buffer_time=3, ratio_no_obs=np.nan, ratio_no_prediction=np.inf):
    # the input y_pred is event-level arragated volume


    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    if buffer_time is None:
        pass
    else:
        delta_t = buffer_time * 3600
        # extend the event duration
        for event_id in range(len(y_obs)):
            y_obs[event_id, 0] = (UTCDateTime(y_obs[event_id, 0]) - delta_t).strftime("%Y-%m-%dT%H:%M:%S")
            y_obs[event_id, 1] = (UTCDateTime(y_obs[event_id, 1]) + delta_t).strftime("%Y-%m-%dT%H:%M:%S")


    predicted_gap = {}

    for event_id in range(len(y_obs)):
        volume_obs = y_obs[event_id, 2]
        start_time, end_time = y_obs[event_id, 0], y_obs[event_id, 1]

        if np.isnan(volume_obs): # is True
            # (1) no observed volume, only with event document
            ratio = ratio_no_obs
            mean_aggregate_sed = ratio_no_obs
        else:
            # (2) with observed volume
            id1 = np.where(y_pred[:, 0] == start_time)[0][0]
            id2 = np.where(y_pred[:, 0] == end_time)[0][0]
            mean_aggregate_sed = np.sum(y_pred[id1:id2, 1])

            if mean_aggregate_sed != 0:
                # (2-1) succssed prediction
                ratio = mean_aggregate_sed / volume_obs
            else:
                # (2-2) all predicted volumes are zero
                ratio = ratio_no_prediction

        predicted_gap[event_id] = [start_time, end_time, mean_aggregate_sed, volume_obs, ratio]

    return predicted_gap

def gaussian_log_likelihood(residual, sigma):

    # \begin{equation}
    # \log[p( y_{\mathrm{obs}} \mid \theta)] = - \frac{1}{2} \times \sum_i [
    # \frac{(y_{\mathrm{obs}}(t_i) - y_{\mathrm{pre}}(t_i))^2}{\sigma^2} + \log{(2\pi\sigma^2)} ] ,
    # \label{likehood_log}
    # % the -1/2 and pi come from Normal distribution
    # \end{equation}

    g_log_like = -0.5 * np.sum( (residual / sigma) ** 2 + np.log(2 * np.pi * sigma ** 2) )

    return g_log_like


def calculate_pred_ratio(y_obs, y_pred, for_none_obs_ratio=1):

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    # initialize with the default ratio
    pred_ratio = np.full_like(y_obs, for_none_obs_ratio, dtype=float)

    # Mask for valid (non-NaN) observations
    mask = ~np.isnan(y_obs) # ~ flips True ↔ False.

    # Compute ratio only for valid entries
    pred_ratio[mask] = y_pred[mask] / y_obs[mask]

    return pred_ratio
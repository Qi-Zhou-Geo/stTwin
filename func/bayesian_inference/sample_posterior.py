#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-07-10T10:59:54
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import emcee
import numpy as np

import shutil
import tempfile

def get_posterior_theta(posterior_results_file, num_draw=100, burn_in_step=100, fix_seed=True):
    """
    Extract thinned posterior samples from an emcee HDF backend.

    This function loads an existing MCMC chain, removes burn-in samples,
    applies thinning to reduce autocorrelation, and randomly selects a
    subset of posterior samples for prediction or uncertainty analysis.

    Args:
        posterior_results_file (str): Path to the emcee HDF backend file containing stored MCMC results.

        num_draw (int, optional): Number of posterior parameter sets to randomly sample for downstream prediction.
            Defaults to 100.

        burn_in_step (int, optional): Number of initial MCMC steps to discard as burn-in.
            Defaults to 100.

        fix_seed (bool, optional): If True, use a fixed random seed (42) for reproducible posterior sampling.
                                   If False, sampling is stochastic.
                                   Defaults to True.

    Returns:
        np.ndarray: Array of shape (num_draw, num_params) containing randomly selected posterior parameter sets.

    Notes:
        - Thinning factor is fixed to 5 to reduce autocorrelation.
        - If num_draw exceeds available posterior samples,
            it will be automatically reduced to the maximum available.
        - When fix_seed=True, repeated calls will return identical draws.
    """

    # 1 create temp file, just in case of mutiple I/O
    temp_dir = os.path.dirname(posterior_results_file)
    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp:
        temp_path = temp.name

    try:
        # 2 copy snapshot
        shutil.copy2(posterior_results_file, temp_path)

        # 3 read from snapshot
        backend = emcee.backends.HDFBackend(temp_path)
        flat_samples = backend.get_chain(discard=burn_in_step, thin=5, flat=True)

        num_available = len(flat_samples) # type: ignore
        num_draw = min(num_draw, num_available)

        # 4 set the seed
        if fix_seed is True:
            # fixed seed for reproducibility
            rng = np.random.default_rng(42)
        else:
            # stochastic sampling
            rng = np.random.default_rng()
        indices = rng.choice(num_available, size=num_draw, replace=False)

        # 5 get the final sampled posterior params
        sampled_theta = flat_samples[indices] # type: ignore

        return sampled_theta

    finally:
        # this step will be executed regardless of whether an error occurs in the try block
        if os.path.exists(temp_path) is True:
            os.remove(temp_path)


def get_MAP_theta(posterior_results_file, burn_in_step=100):
    
    # get the maximum_likelihood_theta
    
    backend = emcee.backends.HDFBackend(posterior_results_file)

    flat_samples = backend.get_chain(discard=burn_in_step, flat=True)
    flat_log_prob = backend.get_log_prob(discard=burn_in_step, flat=True)

    max_index = np.argmax(flat_log_prob) # type: ignore
    theta_map = flat_samples[max_index] # type: ignore

    return theta_map


#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-03-01
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import emcee
import numpy as np

def sample_posterior(posterior_results_file, num_draw=100, burn_in_step=100, fix_seed=True):
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

    backend = emcee.backends.HDFBackend(posterior_results_file)
    flat_samples = backend.get_chain(discard=burn_in_step, thin=5, flat=True)

    num_available = len(flat_samples)
    num_draw = min(num_draw, num_available)

    if fix_seed:
        # fixed seed for reproducibility
        rng = np.random.default_rng(42)
        indices = rng.choice(num_available, size=num_draw, replace=False)
    else:
        # stochastic sampling
        indices = np.random.choice(num_available, size=num_draw, replace=False)

    posterior_draws = flat_samples[indices]

    return posterior_draws

def maximum_likelihood_theta(posterior_results_file, burn_in_step=100):

    backend = emcee.backends.HDFBackend(posterior_results_file)

    flat_samples = backend.get_chain(discard=burn_in_step, flat=True)
    flat_log_prob = backend.get_log_prob(discard=burn_in_step, flat=True)

    max_index = np.argmax(flat_log_prob)
    theta_map = flat_samples[max_index]

    return theta_map


#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import emcee
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

import corner

#region ### add the sys.path to search for custom modules ###
from pathlib import Path
current_file = Path(__file__).resolve()
current_dir = current_file.parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys
sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.post_bayesian_inference.thin_posterior import sample_posterior, maximum_likelihood_theta
from main_BI import log_posterior

plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})

def traces_vs_step(num_walkers, num_params, chain, max_posterior_theta, mean_theta,
                   theta_names, lower_bounds, upper_bounds):

    # first 6 params
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 1)
    axes = []
    for i in range(6):
        ax = plt.subplot(gs[i])
        axes.append(ax)
        for w in range(num_walkers):
            ax.plot(chain[:, w, i], alpha=0.5, color="black")

        # ax.set_ylim(lower_bounds[i], upper_bounds[i])
        ax.axhline(y=max_posterior_theta[i], color="green", lw=1, linestyle="--")
        ax.axhline(y=mean_theta[i], color="red", lw=1, linestyle="--")
        ax.set_ylabel(f"Param {i}", fontweight='bold')
        ax.text(x=0, y=ax.get_ylim()[1]*0.99, s=f"{theta_names[i]}")
        if i < 5:#num_params - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Step after Burn-in", fontweight='bold')

    plt.tight_layout()
    output_dir = f"{current_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trace_plot_1.png", dpi=600)  # , transparent=True
    # plt.show()
    plt.close(fig=fig)

    # last 6 params
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(5, 1)
    axes = []
    for i in range(6, 11):
        ax = plt.subplot(gs[i-6])
        axes.append(ax)
        for w in range(num_walkers):
            ax.plot(chain[:, w, i], alpha=0.5, color="black")

        # ax.set_ylim(lower_bounds[i], upper_bounds[i])
        ax.axhline(y=max_posterior_theta[i], color="green", lw=1, linestyle="--")
        ax.axhline(y=mean_theta[i], color="red", lw=1, linestyle="--")
        ax.set_ylabel(f"Param {i}", fontweight='bold')
        ax.text(x=0, y=ax.get_ylim()[1]*0.99, s=f"{theta_names[i]}")
        if i < num_params - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Step after Burn-in", fontweight='bold')

    plt.tight_layout()
    output_dir = f"{current_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trace_plot_2.png", dpi=600)  # , transparent=True
    # plt.show()
    plt.close(fig=fig)

def posterior_pdf(current_theta_pdf, theta_names, max_posterior_theta, mean_theta,
                  theta_idx, lower_bounds, upper_bounds):

    fig = plt.figure(figsize=(3.5, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    ax.hist(current_theta_pdf, bins=50, alpha=0.5, color="black", density=True)
    ax.axvline(x=max_posterior_theta[theta_idx], color="green", lw=1, linestyle="--", label="Max A Posterior Value")
    ax.axvline(x=mean_theta[theta_idx], color="red", lw=1, linestyle="--", label="Mean Value")
    ax.set_xlim(lower_bounds[theta_idx], upper_bounds[theta_idx])

    ax.set_xlabel(f"Parameter {theta_names[theta_idx]}", fontweight='bold')
    ax.set_ylabel("Probability Density", fontweight='bold')

    ax.legend(loc="upper right", fontsize=6)

    plt.tight_layout()
    output_dir = f"{current_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{str(theta_idx).zfill(2)}_{theta_names[theta_idx]}.png", dpi=600)  # , transparent=True
    # plt.show()
    plt.close(fig=fig)

def posterior_kde(whole_chain, theta_names, max_posterior_theta, mean_theta,
                  lower_bounds, upper_bounds):

    num_steps, num_walkers, num_params = whole_chain.shape

    for theta_idx in range(num_params):

        fig = plt.figure(figsize=(3.5, 3))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

        for walker_idx in range(num_walkers):
            x = whole_chain[:, walker_idx, theta_idx]
            sns.kdeplot(x, color="black", alpha=0.5, clip=(lower_bounds[theta_idx], upper_bounds[theta_idx]))

        ax.axvline(x=max_posterior_theta[theta_idx], color="green", lw=1, linestyle="--", label="Max A Posterior Value")
        ax.axvline(x=mean_theta[theta_idx], color="red", lw=1, linestyle="--", label="Mean Value")
        ax.set_xlim(lower_bounds[theta_idx], upper_bounds[theta_idx])

        ax.set_xlabel(f"Parameter {theta_names[theta_idx]}", fontweight='bold')
        ax.set_ylabel("KDE", fontweight='bold')

        ax.legend(loc="upper right", fontsize=6)

        plt.tight_layout()
        output_dir = f"{current_dir}/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{str(theta_idx).zfill(2)}_{theta_names[theta_idx]}_KDE.png", dpi=600)  # , transparent=True
        # plt.show()
        plt.close(fig=fig)



def check_mcmc_results(file_name, burn_in=50):

    print(f"{UTCDateTime.now().isoformat()} Loading backend: {file_name}\n")

    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v', 'Qdf', 'max_s2w', 'channel_storage_cap', 'erosion_k', 'sigma'
    ]
    # in nature (non-log) sapce
    lower_bounds = np.array([0.1, 10, 10, 1, 6, 6, 1.1, 0.1, 0.1, 1, 0.01])
    # in nature (non-log) sapce
    upper_bounds = np.array([10, 100, 100, 144, 1008, 1008, 2.0, 1.0, 1.0, 100, 10])

    # <editor-fold desc="(1) check the Maximum A Posterior theta">
    # theta is same shape as the turned theta
    max_posterior_theta = maximum_likelihood_theta(posterior_results_file=file_name, burn_in_step=burn_in)
    max_posterior_theta = max_posterior_theta * (upper_bounds - lower_bounds) + lower_bounds # scale to real value

    max_a_posterior = {}
    for theta_name, theta_value in zip(theta_names, max_posterior_theta):
        max_a_posterior[theta_name] = round(theta_value, 2)
    # endregion


    backend = emcee.backends.HDFBackend(file_name)
    chain = backend.get_chain(flat=False)  # (num_steps, num_walkers, num_params)
    chain = lower_bounds + chain * (upper_bounds - lower_bounds)
    chain = chain[burn_in:, :, :] # discard the burn in period
    num_steps, num_walkers, num_params = chain.shape

    # <editor-fold desc="(2) Basic statistic">
    # Basic stastic
    print(f"Number of steps per walker: {backend.iteration}")
    print(f"Number of Burn-in: {burn_in}")
    print(f"num_steps={num_steps}, num_walkers={num_walkers}, num_params={num_params}")

    moves = np.any(np.diff(chain[-5:], axis=0) != 0, axis=-1)
    current_acc_rate = np.mean(moves)
    print(f"Movement rate in last 5 steps: {current_acc_rate:.2%}")

    # Max log-prob
    log_prob = backend.get_log_prob(flat=False)  # (nsteps, nwalkers)
    max_log_prob = np.max(log_prob)
    print(f"Max log-probability: {max_log_prob:.2f}\n")

    # Flatten the chain for rough posterior
    flat_chain = chain.reshape(-1, num_params)
    mean_theta = np.mean(flat_chain, axis=0)
    std_theta = np.std(flat_chain, axis=0)
    print("Parameter best:", ", ".join([f"{x:.3f}" for x in max_a_posterior.values()]))
    print("Parameter mean:", ", ".join([f"{x:.3f}" for x in mean_theta]))
    print("Parameter stds:", ", ".join([f"{x:.3f}" for x in std_theta]))
    print(f"\n")
    # endregion

    # <editor-fold desc="(2) Acceptance fraction">
    sampler = emcee.EnsembleSampler(num_walkers, num_params, log_posterior, backend=backend)

    acceptance_fraction = sampler.acceptance_fraction # shape as (num_walkers)
    print(f"Min-Mean-Max acceptance fraction: "
          f"{np.min(acceptance_fraction):.3f}, "
          f"{np.mean(acceptance_fraction):.3f}, "
          f"{np.max(acceptance_fraction):.3f}, \n"
          f"A typical <good> acceptance_fraction range is: 0.15 to 0.5 for MCMC.")
    # endregion

    traces_vs_step(num_walkers, num_params, chain, max_posterior_theta, mean_theta,
                   theta_names, lower_bounds, upper_bounds)

    for theta_idx in range(num_params):
        current_theta_pdf = flat_chain[:, theta_idx]
        posterior_pdf(current_theta_pdf, theta_names, max_posterior_theta, mean_theta,
                      theta_idx, lower_bounds, upper_bounds)

    whole_chain = chain
    posterior_kde(whole_chain, theta_names, max_posterior_theta, mean_theta,
                  lower_bounds, upper_bounds)
    # --------------------------
    # Histogram plots
    # --------------------------
    flat_chain = chain[:, :, :].reshape(-1, num_params)

    fig = plt.figure(figsize=(6, 6))

    corner.corner(
        flat_chain,
        labels=theta_names,
        bins=40,
        smooth=1.0,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
    )

    output_dir = f"{current_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(f"{output_dir}/corner_plot.png", dpi=300)
    plt.show()
    plt.close(fig=fig)


    # 3. Calculate the maximum log-prob at each step across all walkers
    log_probs = backend.get_log_prob()
    max_log_probs = np.max(log_probs, axis=1)
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    ax.plot(np.abs(max_log_probs), color='black', linewidth=1, zorder=3)
    ax.set_ylabel("Negative Log-Likehood", fontweight='bold')
    ax.set_xlabel("Number of Step", fontweight='bold')
    # ax.set_yscale('log')
    # ax.set_ylim(1e3, 2e4)
    ax.grid(axis='y', which="both", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    # Highlight the "Burn-in" vs "Plateau"
    plt.axvspan(0, burn_in, color='gray', alpha=0.2, label='Initial Burn-in', zorder=2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Log-Likehood.png", dpi=600)
    plt.show()
    plt.close(fig=fig)

file_name = f"{current_dir}/sedcas_mcmc_results_984877.h5"
check_mcmc_results(file_name, burn_in=100)
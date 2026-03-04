#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2026-02-23
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os

import emcee
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

import corner

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path
current_dir = Path(__file__).resolve().parent

# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

import sys
sys.path.append(str(project_root))
# </editor-fold>


plt.rcParams.update({'font.size': 7,
                     'axes.formatter.limits': (-4, 6),
                     'axes.formatter.use_mathtext': True})


def check_mcmc_results(file_name, burn_in = 20):
    print(f"{UTCDateTime.now().isoformat()} Loading backend: {file_name}\n")

    backend = emcee.backends.HDFBackend(file_name)
    chain = backend.get_chain(flat=False)  # (num_steps, num_walkers, num_params)

    # Basic stastic
    num_steps, num_walkers, num_params = chain.shape
    print(f"Number of steps per walker: {backend.iteration}")
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
    print("Parameter means:", ", ".join([f"{x:.3f}" for x in mean_theta]))
    print("Parameter stds:", ", ".join([f"{x:.3f}" for x in std_theta]))



    # Acceptance fraction
    try:
        af = backend.get_sampler_state()['acceptance_fraction']
        mean_af = np.mean(af)
        print(f"Mean acceptance fraction: {mean_af:.3f}")
        print(f"Min acceptance fraction: {np.min(af):.3f}")
        print(f"Max acceptance fraction: {np.max(af):.3f}\n")
    except Exception:
        print("Could not retrieve acceptance fraction directly from backend. Use sampler object if available.\n")

    # --------------------------
    # Trace plots
    # --------------------------
    theta_names = [
        'w_storage_cap0', 'w_storage_cap1', 'w_storage_cap2',
        'w_residence_time0', 'w_residence_time1', 'w_residence_time2',
        'ls_alpha_v', 'Qdf', 'max_s2w', 'channel_storage_cap', 'erosion_k', 'sigma'
    ]

    # first 6 params
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(6, 1)
    axes = []
    for i in range(6):
        ax = plt.subplot(gs[i])
        axes.append(ax)
        for w in range(num_walkers):
            ax.plot(chain[:, w, i], alpha=0.5)
        ax.set_ylabel(f"Param {i}", fontweight='bold')
        ax.text(x=0, y=ax.get_ylim()[1]*0.99, s=f"{theta_names[i]}")
        if i < 5:#num_params - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Step", fontweight='bold')

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
            ax.plot(chain[:, w, i], alpha=0.5)
        ax.set_ylabel(f"Param {i}", fontweight='bold')
        ax.text(x=0, y=ax.get_ylim()[1]*0.99, s=f"{theta_names[i]}")
        if i < num_params - 1:
            ax.set_xticklabels([])

    axes[-1].set_xlabel("Step", fontweight='bold')

    plt.tight_layout()
    output_dir = f"{current_dir}/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/trace_plot_2.png", dpi=600)  # , transparent=True
    # plt.show()
    plt.close(fig=fig)


    # --------------------------
    # Histogram plots
    # --------------------------
    for i in range(num_params):

        fig = plt.figure(figsize=(3.5, 3))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0])

        ax.hist(flat_chain[:, i], bins=50, alpha=0.5, color="black", density=True)

        ax.set_xlabel(f"Parameter {theta_names[i]}", fontweight='bold')
        ax.set_ylabel("Probability Density", fontweight='bold')

        plt.tight_layout()
        output_dir = f"{current_dir}/plots"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{str(i).zfill(2)}_{theta_names[i]}.png", dpi=600)  # , transparent=True
        # plt.show()
        plt.close(fig=fig)


    # --------------------------
    # Histogram plots
    # --------------------------
    flat_chain = chain[burn_in:, :, :].reshape(-1, num_params)

    panel_size = 2.5
    figsize = (num_params * panel_size, num_params * panel_size)
    fig = plt.figure(figsize=figsize)

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

    plt.savefig(f"{output_dir}/corner_plot.png", dpi=800)
    plt.show()
    plt.close(fig=fig)


    # 3. Calculate the maximum log-prob at each step across all walkers
    log_probs = backend.get_log_prob()
    max_log_probs = np.max(log_probs, axis=1)
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0])

    ax.plot(np.abs(max_log_probs), color='black', linewidth=1, zorder=3)
    ax.set_ylabel("Negative Log-Probability", fontweight='bold')
    ax.set_xlabel("Number of Step", fontweight='bold')
    # ax.set_yscale('log')
    # ax.set_ylim(1e3, 2e4)
    ax.grid(axis='y', which="both", color='grey', linestyle='--', lw=0.5, alpha=0.5, zorder=1)
    # Highlight the "Burn-in" vs "Plateau"
    plt.axvspan(0, burn_in, color='gray', alpha=0.2, label='Initial Burn-in', zorder=2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Log-Probability.png", dpi=600)
    plt.show()
    plt.close(fig=fig)

file_name = f"{current_dir}/sedcas_mcmc_results.h5"
check_mcmc_results(file_name)
#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).

import ast
import pickle

import os
import yaml

import pandas as pd
import numpy as np
import xarray as xr

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from obspy import UTCDateTime

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions
from functions.toolkit.log_infor import log_print

from functions.SedCas.model_config import ModelConfig
from functions.SedCas import hydro_model as SedCas_h_model
from functions.SedCas import sediment_model as SedCas_s_model
from functions.SedCas import transport_model as SedCas_t_model

from functions.download_MeteoSwiss.fetch_data import fetch_data4SedCas



print()


def a():
    data = pd.read_csv(f"/Users/qizhou/#python/stTwin/data/SedCas_input/climate_2004_2017_h.txt", header=0)

    time = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
    time_str = [UTCDateTime(i).isoformat() for i in data.iloc[:, 1]]

    # Extract variables
    precipitation = data.iloc[:, 2].values
    temperature = data.iloc[:, 3].values
    sun_radiation = data.iloc[:, 4].values

    data_source = "MeteoSwiss"
    station = data.iloc[0, 0]  # station name
    resolution = time[1] - time[0]  # unit is second
    time_now = UTCDateTime().isoformat()

    climate_forcing = xr.Dataset(
        coords={
            "time": ("time", np.array(time)),  # numeric UTC+0 time
            "time_str": ("time", np.array(time_str)),  # string UTC+0 time
        },
        data_vars={
            "precipitation": ("time", precipitation,
                              {"units": f"mm per {resolution} s", "description": "Total precipitation"}),

            "temperature": ("time", temperature,
                            {"units": f"°C per {resolution} s", "description": "Air temperature"}),

            "sun_radiation": ("time", sun_radiation,
                              {"units": "W/m^2", "description": "Incoming solar radiation"})
        },
        attrs={
            "source": data_source,
            "station": station,
            "resolution": f"{resolution} s",
            "create_time": time_now
        }
    )

    return climate_forcing



class SedCas_new():

    def __init__(self, project_root, model_input_params="default"):

        # find the project path
        self.project_root = project_root
        print(f"In class <SedCas> initial:\n"
              f"Project root: {self.project_root}")

        # set the path for input varilables and output varilables
        self.model_input_dir = f"{self.project_root}/data/SedCas_input"
        self.model_output_dir = f"{self.project_root}/data/SedCas_output"
        os.makedirs(self.model_output_dir, exist_ok=True)

        # the model config params
        self.cfg = ModelConfig(model_input_params)
        # you can check the params by:
        # self.cfg.print_config_params(check_params="all") # for all params
        # self.cfg.print_config_params(check_params="in_prec") # for "in_prec" param

    # for model output

    def _create_time_label(self):

        # length of time series
        num_data = len(self.cfg.in_prec.value)

        # set time stamps, if self.prec.index is UTC+0, then the following is UTC+0
        time = self.cfg.in_prec.value.index.to_numpy(dtype="datetime64[ns]")
        time_str = self.cfg.in_prec.value.index.strftime("%Y-%m-%dT%H:%M:%S")

        return num_data, time_str, time

    def _create_hydro_dataset(self):

        num_data, time_str, time = self._create_time_label()

        # sed_container by shape(number of time series, number of simulations)
        hydro_container = xr.Dataset(
            coords={"time": time,
                    "time_str": ("time", time_str)
                    },

            data_vars={
                # discharge [mm]
                "discharge": ("time", np.zeros(num_data),
                              {"units": "mm", "full name": "Total discharge"}),

                # discharge from overland flow [mm]
                "discharge_surface": ("time", np.zeros(num_data),
                                      {"units": "mm", "full name": "Discharge from overland flow"}),

                # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
                "discharge_sub_surface": ("time",np.zeros(num_data),
                                          {"units": "mm", "full name": "Discharge from subsurface flow"}),

                # state of soil water storage [mm]
                "soil_water_storage": ("time", np.zeros(num_data),
                                       {"units": "mm", "full name": "Soil water storage"}),

                # snow depth SWE [mm]
                "snow_depth": ("time", np.zeros(num_data),
                               {"units": "mm", "full name": "Snow depth"}),

                # snow accumulation [mm]
                "snow_accumulation": ("time", np.zeros(num_data),
                                      {"units": "mm", "full name": "Snow accumulation"}),

                # Potential evapotranspiration [mm]
                "potential_ET": ("time", np.zeros(num_data),
                                 {"units": "mm", "full name": "Potential evapotranspiration"}),

                # Actual evapotranspiration [mm]
                "actual_ET": ("time", np.zeros(num_data),
                              {"units": "mm", "full name": "Actual evapotranspiration"}),

                # precipitation [mm]
                "precipitation": ("time", np.zeros(num_data),
                                  {"units": "mm", "full name": "Precipitation"}),

                # temperature [degree C]
                "temperature": ("time", np.zeros(num_data),
                                {"units": "degree C", "full name": "Temperature"})
            }

        )

        return hydro_container

    def _create_sed_dataset(self, num_iteration=None):

        num_data, time_str, time = self._create_time_label()

        # sed_container by shape(number of time series, number of simulations)
        sed_container = xr.Dataset(
            coords={"time": time,
                    "time_str": ("time", time_str),
                    "iteration": np.arange(num_iteration)},

            data_vars={
                # sediment input from landslides
                "ls": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                       {"units": "mm", "full name": "Generated large landslide"}),

                # sediment hillslope storage [mm]
                "hillslope_storage": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                      {"units": "mm", "full name": "Sediment stored in hillslope"}),

                # sediment channel storage [mm]
                "channel_storage": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                    {"units": "mm", "full name": "Sediment stored in channel"}),

                # sediment catchment output [mm]
                "sed_output_catchment": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                         {"units": "mm", "full name": "Sediment transfered out catchment"}),

                # potential sediment catchment output [mm], i.e. transport-limited case
                "sed_output_catchment_q": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                           {"units": "mm", "full name": "???"}),

                # debris flows, from 'so' values above threshold and summed consecutive values
                "dfs": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                        {"units": "mm", "full name": "???"}),

                # debris flows potential (only 1 because only 1 climate)
                "df_potential": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                 {"units": "mm", "full name": "???"}),
            }
        )

        return sed_container


    # for model input
    def load_climate_input(self, data_type="default"):

        if data_type == "default":
            # use the default data from SedCas model
            df = pd.read_csv(f"{self.model_input_dir}/climate.met", sep='\t')
            df.D = pd.to_datetime(df.D)

            # assign the parameters by column
            df.index = df.D
            self.cfg.in_prec.value = df.Pr
            self.cfg.in_temp.value = df.Ta
            self.cfg.in_sun.value = df.Rsw
        elif data_type == "near-real-time":
            df = fetch_data4SedCas(station="mve", time_resolution="10 minutes", time_period="Today")
            resolution = "10 minutes"

            # assign the parameters by column
            df.index = df["timestamp [UTC+0]"]
            self.cfg.in_prec.value = df[f"precipitation [mm per {resolution}]"]
            self.cfg.in_temp.value = df["temperature [degree]"]
            self.cfg.in_sun.value = df["sun radiation [W per squared m]"]
        elif data_type == "2017-2025":
            df = pd.read_csv(f"{self.model_input_dir}/climate_2017_2025.txt", sep=',', header=0)
            resolution = "Hourly"

            # assign the parameters by column
            df.index = pd.to_datetime(df["timestamp [UTC+0]"])
            self.cfg.in_prec.value = df[f"precipitation [mm per {resolution}]"]
            self.cfg.in_temp.value = df["temperature [degree]"]
            self.cfg.in_sun.value = df["sun radiation [W per squared m]"]
        else:
            try:
                df = pd.read_csv(f"{self.model_input_dir}/climate_{data_type}.txt", sep=',', header=0)
                if data_type == "2004_2017_h":
                    resolution = "Hourly"
                else:
                    resolution = "10 minutes"

                # assign the parameters by column
                df.index = pd.to_datetime(df["timestamp [UTC+0]"])
                self.cfg.in_prec.value = df[f"precipitation [mm per {resolution}]"]
                self.cfg.in_temp.value = df["temperature [degree]"]
                self.cfg.in_sun.value = df["sun radiation [W per squared m]"]
            except FileNotFoundError:
                print("Error! Please check the <data_type>.")

        print(f"input data (climate.met) shape: {df.shape}")

        return df


    # for single model componment
    def run_hydro(self):

        # running the individual HRUs
        SWE = []  # snow water equivalent
        PET = []  # potential evapotranspiration
        HYM = []  # hydrological model output
        for HRU_id in range(self.cfg.num_HRU.value):
            s_w_e = SedCas_h_model.snow_water_equivalent(temperature=self.cfg.in_temp.value.copy(),
                                                         precipitation=self.cfg.in_prec.value.copy(),
                                                         melt_rate_f=self.cfg.snow_melt_r.value,
                                                         T_theta_a=self.cfg.snow_acc.value,
                                                         T_theta_m=self.cfg.snow_melt.value,
                                                         snow_albedo=self.cfg.snow_albedo_y.value[HRU_id],
                                                         soil_albedo=self.cfg.snow_albedo_n.value[HRU_id])
            SWE.append(s_w_e)
            # print("s_w_e", s_w_e.columns, s_w_e.shape)

            p_e_t = SedCas_h_model.cal_actual_evap(temperature=self.cfg.in_temp.value.copy(),
                                                   sps_temperature=self.cfg.delta_t.value,
                                                   radiation=self.cfg.in_sun.value.copy(),
                                                   cloud_cover_r=self.cfg.cloud_cover_r.value,
                                                   albedo=s_w_e.albedo,
                                                   elevation=self.cfg.c_elevation.value,
                                                   U=self.cfg.r_humidity.value)
            PET.append(p_e_t)
            # print("p_e_t", p_e_t.shape)

            h = SedCas_h_model.h_model(snow=s_w_e,
                                       PET=p_e_t,
                                       precipitation=self.cfg.in_prec.value.copy(),
                                       temperature=self.cfg.in_temp.value.copy(),
                                       alpha=self.cfg.et.value,
                                       num_reservoir=len(self.cfg.epsilon_w.value[HRU_id]),
                                       params={'k': self.cfg.delta_t_water.value[HRU_id],
                                               'Scap': self.cfg.epsilon_w.value[HRU_id],
                                               'S0': [0, 0]}
                                       )
            HYM.append(h)
            # print("h", h.columns, h.shape)

        # lumped hydrology: area-weighted aggregation
        hydro = SedCas_h_model.lump_h_model(HYM,
                                            num_HRU=self.cfg.num_HRU.value,
                                            shares=self.cfg.ratio_HRU.value,
                                            log_print=None)

        # update the attribute
        self.hydro = hydro

        hydro_container = self._create_hydro_dataset()
        # <editor-fold desc="add the params to hydro_container">
        # discharge [mm]
        hydro_container["discharge"].values = hydro["Q"].values
        # discharge from overland flow [mm]
        hydro_container["discharge_surface"].values = hydro["Qs"].values
        # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
        hydro_container["discharge_sub_surface"].values = hydro["Qss"].values
        # state of soil water storage [mm]
        hydro_container["soil_water_storage"].values = hydro["Vw"].values
        # snow depth SWE [mm], already in input
        hydro_container["snow_depth"].values = hydro["snow_depth"].values
        # snow accumulation [mm]
        hydro_container["snow_accumulation"].values = hydro["snowacc"].values
        # Potential evapotranspiration [mm]
        hydro_container["potential_ET"].values = hydro["PET"].values
        # Actual evapotranspiration [mm]
        hydro_container["actual_ET"].values = hydro["AET"].values
        # precipitation [mm]
        hydro_container["precipitation"].values = hydro["Pr"].values
        # temperature [degree C]
        hydro_container["temperature"].values = hydro["temperature"].values
        # </editor-folder>

        self.cfg.out_hydro_container.value = hydro_container

        return hydro_container

    def run_sediment(self, seed_i, iteration, sed_container):

        # large landslides, generate the time series area-normalized landslide thickness
        # shape by [time, landslides magnitude[mm]]
        large_ls = SedCas_s_model.generate_large_ls(ls_trigger=self.cfg.ls_trigger_m.value,
                                                    temperature=self.cfg.in_temp.value.copy(),
                                                    prec=self.cfg.in_prec.value.copy(),
                                                    snow=self.hydro.snow_depth, # model
                                                    theta_sd=self.cfg.ls_trigger_SWE.value,
                                                    theta_prec=self.cfg.ls_trigger_r.value,
                                                    theta_sa=self.cfg.snow_acc.value,
                                                    theta_ls_freeze=self.cfg.ls_trigger_f.value,
                                                    min_ls_volume=self.cfg.ls_min_v.value,
                                                    alpha=self.cfg.ls_alpha_v.value,
                                                    cutoff=self.cfg.ls_max_v.value,
                                                    area=self.cfg.c_area.value,
                                                    seed=seed_i)

        num_large_ls = len(large_ls[large_ls.mag > 0])

        # small landslides, generate the time series area-normalized landslide thickness
        # shape by [time, landslides magnitude[mm]]
        num_days = len(self.cfg.in_prec.value.resample('24h').sum())
        small_ls = SedCas_s_model.generate_small_ls(num_days=num_days,
                                                    num_large_ls=num_large_ls,
                                                    min_ls_volume=self.cfg.ls_min_v.value,
                                                    area=self.cfg.c_area.value,
                                                    seed=seed_i,
                                                    mu=3.36, sigma=1.18, ratio=3.36)

        # date index for small landslides
        small_ls.index = large_ls.index

        # mix water and sediments
        sed_run = SedCas_t_model.trans_model(large_ls_t=large_ls,
                                             small_ls_t=small_ls,
                                             hyd=self.hydro,
                                             Q_theta=self.cfg.q_df.value,
                                             s_max=self.cfg.max_s2w.value,
                                             d_h=self.cfg.h2s_r.value,
                                             hs_theta=self.cfg.epsilon_h.value,
                                             area=self.cfg.c_area.value,
                                             method='exp',
                                             ls_trigger=self.cfg.ls_trigger_m.value,
                                             rainfall_triggered_ls_theta=self.cfg.ls_trigger_r.value,

                                             initial_hs_storage=self.cfg.epsilon_h.value,  # careful this
                                             initial_ch_storage=0,

                                             mindf=self.cfg.min_df_v.value,
                                             smax_nodf=self.cfg.max_s_c.value,
                                             b=self.cfg.scaling_b.value)

        # <editor-fold desc="add the params to sed_container">
        # sediment input from landslides
        sed_container["ls"][:, iteration] = sed_run.ls.values
        # hillslope storage time series [mm]
        sed_container["hillslope_storage"][:, iteration] = sed_run.hillslope_storage.values
        # channel storage time series [mm]
        sed_container["channel_storage"][:, iteration] = sed_run.channel_storage.values
        # catchment sediment output time series [mm]
        sed_container["sed_output_catchment"][:, iteration] = sed_run.sed_output_catchment.values
        # potential sediment output based on discharge [mm]
        sed_container["sed_output_catchment_q"][:, iteration] = sed_run.sed_output_catchment_q.values
        # debris flows, sediment output above minimum threshold and concentration of debris flows[mm]
        sed_container["dfs"][:, iteration] = sed_run.dfs.values
        # debris flows potential (only 1 because only 1 climate)
        sed_container["df_potential"][:, iteration] = sed_run.dfspot.values

        # </editor-folder>

        return sed_run


    # for combine model together
    def run_stochastic_simulations(self, seed=0, num_iteration=None):

        # sediment module with stochastic landslide magnitudes
        if num_iteration is None:
            # default is 100 times
            num_iteration = self.cfg.num_iteration.value
        else:
            self.cfg.num_iteration.value = int(num_iteration)

        sed_container = self._create_sed_dataset(num_iteration=num_iteration)
        for iteration in tqdm(range(num_iteration),
                              desc="running sediment model by stochastic simulations",
                              file=sys.stdout):
            sed_run = self.run_sediment(seed_i=seed, iteration=iteration, sed_container=sed_container)
            seed = seed + 1

        # calculate the stastic values
        sed_container_stats = self.post_process_quantiles(xr_dataset=sed_container)
        self.cfg.out_sed_container.value = sed_container_stats

        return sed_container, sed_container_stats


    # for results post-process
    def post_process_quantiles(self, xr_dataset, quants=(1, 50, 99)):

        q_values = [q / 100 for q in quants]
        q_names = [f"Q{q}" for q in quants]

        # new dataset to store quantile results
        new_xr = xr.Dataset(coords=xr_dataset.coords)

        for var in xr_dataset.data_vars:

            # only compute for variables that have 'iteration' dimension
            if "iteration" in xr_dataset[var].dims:

                q_da = xr_dataset[var].quantile(q_values, dim="iteration")

                # add each quantile as new variable
                for qi, qname in zip(q_values, q_names):
                    new_xr[f"{var}_{qname}"] = q_da.sel(quantile=qi)

        return new_xr

    def post_process_visualize(self, list_of_tuples=None):
        """
        list_of_tuples: list of tuples
            [(df, x_col, y_col), ...]
        """
        from functions.toolkit.plotly_visualize import plotly_multi_time_series
        from functions.toolkit.plotly_visualize import plotly_multi_time_series_with_shade

        if list_of_tuples is None:
            temp = self.hydro.copy()
            temp = temp.reset_index()
            temp = temp.rename(columns={"D": "UTC+0DateTime"})

            list_of_tuples = [(temp, "UTC+0DateTime", "Pr"), (temp, "UTC+0DateTime", "Q")]

            temp = self.sedout.copy()
            temp = temp.reset_index()
            temp = temp.rename(columns={"D": "UTC+0DateTime"})

            list_of_tuples.append([temp, "UTC+0DateTime", "Qstl"])
        else:
            list_of_tuples = list_of_tuples

        plotly_multi_time_series(list_of_tuples,
                                 width=2000,
                                 height_per_panel=300,
                                 shared_title=None)

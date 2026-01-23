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

from functions.SedCas import hydro_model as SedCas_h_model
from functions.SedCas import sediment_model as SedCas_s_model
from functions.SedCas import transport_model as SedCas_t_model

from functions.download_MeteoSwiss.fetch_data import fetch_data4SedCas


class SedCas():

    def __init__(self, project_root):

        # find the project path
        self.project_root = project_root
        print(f"In class <SedCas> initial:\n"
              f"Project root: {self.project_root}")

        # set the path for model params, input varilables and output varilables
        self.model_params_dir = f"{self.project_root}/config/SedCas_params"
        self.model_input_dir = f"{self.project_root}/data/SedCas_input"
        self.model_output_dir = f"{self.project_root}/data/SedCas_output"
        os.makedirs(self.model_output_dir, exist_ok=True)

        # the model params
        self.prec = None  # Precipitation time series [mm/h]
        self.temperature = None  # Temperature time series [degree C]
        self.sun_radiation = None  # Sun radiation [W/m^2]

        # the model output
        self.hydro = None
        self.sedout = None

    # for model output
    def _create_time_label(self):

        # length of time series
        num_data = len(self.prec)

        # set time stamps, if self.prec.index is UTC+0, then the following is UTC+0
        time = self.prec.index.to_numpy(dtype="datetime64[ns]")
        time_str = self.prec.index.strftime("%Y-%m-%dT%H:%M:%S")

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

    # for model params
    def print_config_params(self):
        # you can run this function whenever you want to check the model attributy by:
        # model.log_config_params()

        # logout the instance attributes (attributes of the SedCas model instance)
        for k, v in vars(self).items():

            if k in ["model_params_dir", "model_input_dir", "model_output_dir"]:
                print(f"SedCas model I/O dir \n"
                      f"<{k}>: {v} \n")
            elif k in ["Pr", "Ta", "Rsw"]:
                print(f"SedCas model forcing (input varilables) \n"
                      f"<{k}>: {v.shape}, values[0]={v.iloc[0]}, values[-1]={v.iloc[-1]} \n")
            else:
                print(f"SedCas model params \n"
                      f"<{k}>: {v} \n")

    def load_default_params(self, log_params=True):

        # 1) load the pre-defined model params
        yaml_path = f"{self.model_params_dir}/SedCas_input_params.yaml"
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        params = data["input_params"]

        # assign each parameter as class attribute
        for key, val in params.items():
            # equal to -> self.key_name = val,
            # but the "key_name" is automaticly set as defiend "key_name" in params
            setattr(self, key, val)

        # 2) post-processing to get another two more model params
        # normalizing hillslope sediment storage by catchment area considering packing density
        self.shcap = self.shcap * (self.rho_dry / self.rho_b) / self.area * 1e-3

        # smallest possible sediment amount in debirs flow
        # NOTE: this is only a constraint for the model, the smallest modelled debris flow volume is given by qdf and smax_nodf
        self.mindf = self.minDF * self.smax_nodf / self.area * 1e-3

        # 3) print out the loaded model params
        if log_params is True:
            self.log_config_params()

        return self.mindf

    # for model input
    def load_climate_input(self, data_type="default"):

        if data_type == "default":
            # use the default data from SedCas model
            df = pd.read_csv(f"{self.model_input_dir}/climate.met", sep='\t')
            df.D = pd.to_datetime(df.D)

            # assign the parameters by column
            df.index = df.D
            self.prec = df.Pr
            self.temperature = df.Ta
            self.sun_radiation = df.Rsw
        elif data_type == "near-real-time":
            df = fetch_data4SedCas(station="mve", time_resolution="10 minutes", time_period="Today")
            resolution = "10 minutes"

            # assign the parameters by column
            df.index = df["timestamp [UTC+0]"]
            self.prec = df[f"precipitation [mm per {resolution}]"]
            self.temperature = df["temperature [degree]"]
            self.sun_radiation = df["sun radiation [W per squared m]"]
        elif data_type == "2017-2025":
            df = pd.read_csv(f"{self.model_input_dir}/climate_2017_2025.txt", sep=',', header=0)
            resolution = "Hourly"

            # assign the parameters by column
            df.index = pd.to_datetime(df["timestamp [UTC+0]"])
            self.prec = df[f"precipitation [mm per {resolution}]"]
            self.temperature = df["temperature [degree]"]
            self.sun_radiation = df["sun radiation [W per squared m]"]
        else:
            print("Error! Please check the <data_type>.")

        print(f"input data (climate.met) shape: {df.shape}")

        return df

    # for model componment
    def run_hydro(self, sps_temperature=1, cloud_cover_r=1, U=0.8):

        # running the individual HRUs
        SWE = []  # snow water equivalent
        PET = []  # potential evapotranspiration
        HYM = []  # hydrological model output
        for HRU_id in range(self.num_HRU):
            s_w_e = SedCas_h_model.snow_water_equivalent(temperature=self.temperature.copy(),
                                                         precipitation=self.prec.copy(),
                                                         melt_rate_f=self.mrate,
                                                         T_theta_a=self.Tsa,
                                                         T_theta_m=self.Tsm,
                                                         snow_albedo=self.Asnow[HRU_id],
                                                         soil_albedo=self.Anosnow[HRU_id])
            SWE.append(s_w_e)
            # print("s_w_e", s_w_e.columns, s_w_e.shape)

            p_e_t = SedCas_h_model.cal_actual_evap(temperature=self.temperature,
                                                   sps_temperature=sps_temperature,
                                                   radiation=self.sun_radiation,
                                                   cloud_cover_r=cloud_cover_r,
                                                   albedo=s_w_e.albedo,
                                                   elevation=self.Ele,
                                                   U=U)
            PET.append(p_e_t)
            # print("p_e_t", p_e_t.shape)

            h = SedCas_h_model.h_model(snow=s_w_e,
                                       PET=p_e_t,
                                       precipitation=self.prec,
                                       temperature=self.temperature,
                                       alpha=self.alphaET,
                                       num_reservoir=len(self.Vwcaps[HRU_id]),
                                       params={'k': self.ks[HRU_id], 'Scap': self.Vwcaps[HRU_id], 'S0': [0, 0]})
            HYM.append(h)
            # print("h", h.columns, h.shape)

        # lumped hydrology: area-weighted aggregation
        hydro = SedCas_h_model.lump_h_model(HYM,
                                            num_HRU=self.num_HRU,
                                            shares=self.shares,
                                            log_print=log_print)

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

        return hydro_container

    def run_sediment(self, seed_i, iteration, sed_container):

        # large landslides, generate the time series area-normalized landslide thickness
        # shape by [time, landslides magnitude[mm]]
        large_ls = SedCas_s_model.generate_large_ls(ls_trigger=self.LStrig,
                                                    temperature=self.temperature,
                                                    prec=self.prec,
                                                    snow=self.hydro.snow_depth,
                                                    theta_sd=self.Tsd, theta_prec=self.Tpr, theta_sa=self.Tsa,
                                                    theta_ls_freeze=self.Tfreeze,
                                                    min_ls_volume=self.ls_xmin, alpha=self.ls_alpha,
                                                    cutoff=self.ls_cutoff,
                                                    area=self.area,
                                                    seed=seed_i)

        num_large_ls = len(large_ls[large_ls.mag > 0])

        # small landslides, generate the time series area-normalized landslide thickness
        # shape by [time, landslides magnitude[mm]]
        num_days = len(self.prec.resample('24h').sum())
        small_ls = SedCas_s_model.generate_small_ls(num_days=num_days,
                                                    num_large_ls=num_large_ls,
                                                    min_ls_volume=self.ls_xmin,
                                                    area=self.area,
                                                    seed=seed_i,
                                                    mu=3.36, sigma=1.18, ratio=3.36)

        # date index for small landslides
        small_ls.index = large_ls.index

        # mix water and sediments
        sed_run = SedCas_t_model.trans_model(large_ls_t=large_ls,
                                             small_ls_t=small_ls,
                                             hyd=self.hydro,
                                             Q_theta=self.qdf,
                                             s_max=self.smax,
                                             d_h=self.rhc,
                                             hs_theta=self.shcap,
                                             area=self.area,
                                             method='exp',
                                             ls_trigger=self.LStrig,
                                             rainfall_triggered_ls_theta=self.Tpr,

                                             initial_hs_storage=self.shcap,  # careful this
                                             initial_ch_storage=0,

                                             mindf=self.mindf,
                                             smax_nodf=self.smax_nodf,
                                             b=self.b)

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

    def run_stochastic_simulations(self, seed=0, num_iteration=None):

        # sediment module with stochastic landslide magnitudes
        if num_iteration is None:
            # default is 100 times
            num_iteration = self.num_iteration
        else:
            self.num_iteration = int(num_iteration)

        sed_container = self._create_sed_dataset(num_iteration=num_iteration)
        for iteration in tqdm(range(num_iteration),
                              desc="running sediment model by stochastic simulations",
                              file=sys.stdout):
            sed_run = self.run_sediment(seed_i=seed, iteration=iteration, sed_container=sed_container)
            seed = seed + 1

        # calculate the stastic values
        sed_container_stats = self.post_process_quantiles_1(xr_dataset=sed_container)

        return sed_container, sed_container_stats

    # for results post-process
    def post_process_quantiles(self, xr_dataset, var_name_list,
                               quants=(5, 50, 95),
                               start_time=None, end_time=None):

        # quants=(1, 5, 10, 25, 50, 75, 90, 95, 99)
        for var_name in var_name_list:
            data = xr_dataset[var_name]

            if start_time is not None or end_time is not None:
                data = data.sel(time=slice(start_time, end_time))

            q = [x / 100 for x in quants]
            q_name = [f"Q{int(x * 100)}" for x in q]

            result = data.quantile(q, dim="iteration")
            result = result.assign_coords(quantile=q_name)

        return result

    def post_process_quantiles_1(self, xr_dataset, quants=(1, 50, 99)):

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

    def run_sediment_one_func(self, seed=0, total_iteration=None):

        # sediment module with stochastic landslide magnitudes
        if total_iteration is None:
            # default is 100 times
            total_iteration = self.num_iteration
        else:
            self.num_iteration = int(total_iteration)

        # initialization of variables for stochastic sediment supply

        # length of time series
        num_data = len(self.prec)
        # number of days
        num_days = len(self.prec.resample('24h').sum())

        # <editor-fold desc="Add params">
        class sed_container:
            pass

        # index for date-time
        sed_container.index = self.prec.index

        # sediment input from landslides
        sed_container.ls = np.zeros([num_data,
                                     self.num_iteration])  # self.num_iteration is number of stochastic simulations for hillslope module

        # sediment hillslope storage [mm]
        sed_container.hillslope_storage = np.zeros([num_data, self.num_iteration])

        # sediment channel storage [mm], shape(time series length, number of simulations)
        sed_container.channel_storage = np.zeros([num_data, self.num_iteration])

        # sediment catchment output [mm]
        sed_container.sed_output_catchment = np.zeros([num_data, self.num_iteration])

        # potential sediment catchment output [mm], i.e. transport-limited case
        sed_container.sed_output_catchment_q = np.zeros([num_data, self.num_iteration])

        # debris flows, from 'so' values above threshold and summed consecutive values
        sed_container.dfs = np.zeros([num_data, self.num_iteration])

        # debris flows potential (only 1 because only 1 climate)
        sed_container.dfspot = np.zeros(num_data)
        # </editor-fold>

        for iteration in tqdm(range(total_iteration),
                              desc="running sediment model by stochastic simulations",
                              file=sys.stdout):
            # large landslides, generate the time series area-normalized landslide thickness
            # shape by [time, landslides magnitude[mm]]
            large_ls = SedCas_s_model.generate_large_ls(ls_trigger=self.LStrig,
                                                        temperature=self.temperature,
                                                        prec=self.prec,
                                                        snow=self.hydro.snow_depth,
                                                        theta_sd=self.Tsd, theta_prec=self.Tpr, theta_sa=self.Tsa,
                                                        theta_ls_freeze=self.Tfreeze,
                                                        min_ls_volume=self.ls_xmin, alpha=self.ls_alpha,
                                                        cutoff=self.ls_cutoff,
                                                        area=self.area,
                                                        seed=seed)

            num_large_ls = len(large_ls[large_ls.mag > 0])

            # small landslides, generate the time series area-normalized landslide thickness
            # shape by [time, landslides magnitude[mm]]
            small_ls = SedCas_s_model.generate_small_ls(num_days=num_days,
                                                        num_large_ls=num_large_ls,
                                                        min_ls_volume=self.ls_xmin,
                                                        area=self.area,
                                                        seed=seed,
                                                        mu=3.36, sigma=1.18, ratio=3.36)

            # date index for small landslides
            small_ls.index = large_ls.index

            # mix water and sediments
            sed_run = SedCas_t_model.trans_model(large_ls_t=large_ls,
                                                 small_ls_t=small_ls,
                                                 hyd=self.hydro,
                                                 Q_theta=self.qdf,
                                                 s_max=self.smax,
                                                 d_h=self.rhc,
                                                 hs_theta=self.shcap,
                                                 area=self.area,
                                                 method='exp',
                                                 ls_trigger=self.LStrig,
                                                 rainfall_triggered_ls_theta=self.Tpr,

                                                 initial_hs_storage=self.shcap,  # careful this
                                                 initial_ch_storage=0,

                                                 mindf=self.mindf,
                                                 smax_nodf=self.smax_nodf,
                                                 b=self.b)

            sed_container.ls[:, iteration] = sed_run.ls.values
            sed_container.hillslope_storage[:,
            iteration] = sed_run.hillslope_storage  # hillslope storage time series [mm]
            sed_container.channel_storage[:, iteration] = sed_run.channel_storage  # channel storage time series [mm]
            sed_container.sed_output_catchment[:,
            iteration] = sed_run.sed_output_catchment  # catchment sediment output time series [mm]
            sed_container.sed_output_catchment_q[:,
            iteration] = sed_run.sed_output_catchment_q  # potential sediment output based on discharge [mm]
            sed_container.dfs[:,
            iteration] = sed_run.dfs  # debris flows, sediment output above minimum threshold and concentration of debris flows[mm]

            seed += 1

        sed_container.dfspot[:] = sed_run.dfspot

        self.sed_container = sed_container

        return sed_container

    def save_output(self, h_name="Hydro_new.txt", s_name="Sediment_new.txt"):

        if h_name is None:
            pass
        else:
            file_name = f"{self.model_output_dir}/{h_name}"
            # arr = self.hydro.iloc[:, 0] # conver to ISO format
            # arr = pd.to_datetime(arr).dt.strftime('%Y-%m-%dT%H:%M:%S')
            # self.hydro.iloc[:, 0] = arr
            self.hydro.to_csv(file_name, header=True)
            log_print(f"Save file: {file_name} \n \n")

        if s_name is None:
            pass
        else:
            sedout = pd.DataFrame(index=self.sed.index)
            quants = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            for q in quants:
                col_name = f"Q{q}"
                sedout[col_name] = np.percentile(self.sed.catchment_output, q,
                                                 axis=1)  # catchment sediment output time series [mm]

            sedout['Qstl'] = self.sed.sopot[:, 0]  # debris flows potential

            file_name = f"{self.model_output_dir}/{s_name}"
            # arr = sedout.iloc[:, 0] # conver to ISO format
            # arr = pd.to_datetime(arr).dt.strftime('%Y-%m-%dT%H:%M:%S')
            # sedout.iloc[:, 0] = arr
            sedout.to_csv(file_name, header=True)
            log_print(f"Save file: {file_name} \n \n")

            # update the attribute
            self.sedout = sedout

        return sedout

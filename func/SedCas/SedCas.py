#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-17T17:07:18
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).

import os

import pandas as pd
import numpy as np
import xarray as xr

from tqdm import tqdm

from obspy import UTCDateTime

#region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion



# import the custom functions
from func.SedCas.model_config import ModelConfig, ConfigItem
from func.SedCas import hydro_model as SedCas_h_model
from func.SedCas import sediment_model as SedCas_s_model
from func.SedCas import transport_model as SedCas_t_model

from func.download_MeteoSwiss.fetch_data import fetch_data4SedCas

class SedCas():

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

        self.climate_forcing = None

    def _params_post_processing(self):

        # post-processing to get another two more model params
        # normalizing hillslope sediment storage by catchment area considering packing density
        self.cfg.hillslope_storage_cap.value = (
                self.cfg.hillslope_storage_cap.value *
                (self.cfg.rho_sediment.value / self.cfg.rho_bedrock.value) / self.cfg.c_area.value * 1e-3
        )

        # smallest possible sediment amount in debirs flow
        # NOTE: this is only a constraint for the model,
        # the smallest modelled debris flow volume is given by q_df and max_s_c
        self.cfg.min_df_v.value = self.cfg.min_df_v.value * self.cfg.max_s_c.value / self.cfg.c_area.value * 1e-3

        # add new params
        bl_params = SedCas_t_model.define_bedload_params(Qdf=self.cfg.Qdf.value,
                                                         min_df_v=self.cfg.min_df_v.value,
                                                         max_s_c=self.cfg.max_s_c.value,
                                                         bedload_param_b=self.cfg.bedload_param_b.value)
        Qbl, bedload_param_a = bl_params
        self.cfg.Qbl = ConfigItem(
            name="Qbl",
            value=Qbl,
            attrs={'unit': 'mm / <time_resolution>',
                   'description': 'Discharge threshold for triggering bedload,'
                                  'if Qs > Qbf, bedload could be generated',
                   'calibrate': 'Yes, follow https://doi.org/10.1029/2020JF005739'}
        )

        self.cfg.bedload_param_a = ConfigItem(
            name="bedload_param_a",
            value=bedload_param_a,
            attrs={'unit': 'None',
                   'description': 'Shape parameter for bedload transport,'
                                  'bedload_scaling_param_a will be calculated based on bedload_scaling_param_b',
                   'calibrate': 'Yes, follow https://doi.org/10.1029/2020JF005739'}

        )

    # for model input
    def load_climate_input(self, data_type):

        data_source = "MeteoSwiss"
        station = "Montana (MVE)"
        time_now = UTCDateTime().isoformat()

        if data_type == "default":
            # use the default data from SedCas model
            data = pd.read_csv(f"{self.model_input_dir}/climate_1999_2017_h.txt", sep='\t')

            time_float = [UTCDateTime(i).timestamp for i in data.iloc[:, 0]]
            time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 0]]

            # Extract variables
            precipitation = data.Pr.values
            temperature = data.Ta.values
            sun_radiation = data.Rsw.values

            resolution = 3600  # unit is second
        elif data_type in ["1-hour", "10-minutes"]:

            if data_type == "1-hour":
                climate_frocing_input = "climate_2004_2023_h.txt"
                resolution = 3600  # unit is second
            elif data_type == "10-minutes":
                climate_frocing_input = "climate_2004_2023_t.txt"
                resolution = 600  # unit is second
            else:
                raise ValueError("data_type must be '1-hour' or '10-minutes'")

            data = pd.read_csv(f"{project_root}/data/SedCas_input/{climate_frocing_input}", header=0)

            time_float = [UTCDateTime(i).timestamp for i in data.iloc[:, 1]]
            time_str = [UTCDateTime(i).strftime("%Y-%m-%dT%H:%M:%S") for i in data.iloc[:, 1]]

            # Extract variables
            precipitation = data.iloc[:, 2].values
            temperature = data.iloc[:, 3].values
            sun_radiation = data.iloc[:, 4].values

        elif data_type == "sediment":
            pass


        climate_forcing = xr.Dataset(
            coords={
                "time": ("time", np.array(time_float)),  # numeric UTC+0 time
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
                "data_source": data_source,
                "station": station,
                "resolution": resolution,
                "resolution_unit": f"seconds",
                "create_time": time_now
            }
        )

        self.climate_forcing = climate_forcing
        return climate_forcing

    # for model output
    def _load_climate_meta(self):

        time_float = self.climate_forcing.time.values  # note: values with s
        time_str = self.climate_forcing.time_str.values
        num_data = len(time_str)
        num_HRU = self.cfg.num_HRU.value  # note: value without s

        resolution = time_float[1] - time_float[0]

        data_source = self.climate_forcing.attrs["data_source"]
        station = self.climate_forcing.attrs["station"]

        time_now = UTCDateTime().isoformat()

        return time_float, time_str, num_data, num_HRU, resolution, data_source, station, time_now

    def _create_hydro_dataset(self):

        temp = self._load_climate_meta()
        time_float, time_str, num_data, num_HRU, resolution, data_source, station, time_now = temp

        num_reservoir = self.cfg.num_reservoir.value
        # hydro_output by dimenssion time, HRU_id
        hydro_container = xr.Dataset(
            coords={
                "time": ("time", np.array(time_float)),  # numeric UTC+0 time
                "time_str": ("time", np.array(time_str)),  # string UTC+0 time
                "HRU_id": np.arange(num_HRU),
                # 1 in bedrock HRU (id=0), 2 in forest HRU (top id=1, bottom id=2)
                "reservoir_id": np.arange(num_reservoir),
            },
            data_vars={
                # (1) snow_water_equivalent
                # snow depth SWE
                "modelled_SWE": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                                 {"units": f"mm per {resolution} s",
                                  "description": "Modelled snow-water-equivalent depth"}),

                # Snow pack changes
                "snow_delta_depth": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                                     {"units": f"mm per {resolution} s", "description": "Snow pack changes"}),

                # snow accumulation
                "snow_acc": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                             {"units": f"mm per {resolution} s", "description": "Snow accumulation"}),
                # snow melt
                "snow_melt": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                              {"units": f"mm per {resolution} s", "description": "Snow melting"}),

                # albedo
                "albedo": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                           {"units": "None",
                            "description": "Snow (when covered with snow) or soil (when no more snow existing) albedo, 0->low, 1->high"}),

                # (2)
                # Potential evapotranspiration [mm]
                "PET": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                        {"units": f"mm per {resolution} s", "description": "Potential evapotranspiration"}),

                # Actual evapotranspiration [mm]
                "AET": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                        {"units": f"mm per {resolution} s", "description": "Actual evapotranspiration"}),

                # (3)
                # discharge from subsurface flow (outflow from last bucket in the cascasde) [mm]
                "Qss": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                        {"units": f"mm per {resolution} s",
                         "description": "Discharge from subsurface flow (normalized by catchment area)"}),

                # discharge from overland flow
                "Qs": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                       {"units": f"mm per {resolution} s",
                        "description": "Discharge from overland (surface) flow (normalized by catchment area)"}),

                # discharge
                "Q": (("time", "HRU_id"), np.zeros((num_data, num_HRU)),
                      {"units": f"mm per {resolution} s",
                       "description": "Total discharge (normalized by catchment area)"}),

                # water stored in reservoir 0 (top), reservoir 1 (after-top), w_storage_n (bottom)
                # when you want to select a specific reservoir:
                # for bedrock with one reservoir:
                #   model.hydro_output["w_storage"][:, 0, 0] # first HRU and first reservoir, the rest of is zero
                # for forest with two reservoir:
                #   model.hydro_output["w_storage"][:, 1, 1] # second HRU and first (top) reservoir
                #   model.hydro_output["w_storage"][:, 1, 2] # second HRU and second (bottom) reservoir
                "w_storage": (("time", "HRU_id", "reservoir_id"), np.zeros((num_data, num_HRU, num_reservoir)),
                              {"units": f"mm per {resolution} s",
                               "description": "Total water stored in each reservoir.\n",
                               "reservoir_id=0": "top reservoir in bedrock HRU\n",
                               "reservoir_id=1": "top reservoir in forest HRU\n",
                               "reservoir_id=2": "bottom reservoi in forest HRU\n"}),

            },
            attrs={
                "resolution": resolution,
                "resolution_unit": f"seconds",
                "create_time": time_now
            }
        )

        return hydro_container

    def _create_sed_dataset(self, num_iteration=None):

        temp = self._load_climate_meta()
        time_float, time_str, num_data, num_HRU, resolution, data_source, station, time_now = temp

        # sed_container by shape(number of time series, number of simulations)
        sed_container = xr.Dataset(
            coords={"time": time_float,
                    "time_str": ("time", time_str),
                    "iteration": np.arange(num_iteration)},
            data_vars={
                # sediment input from landslides
                "ls": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                       {"units": f"mm per {resolution} s",
                        "description": "Real landslide input (normalized by catchment area)"}),
                
                # remobilized landslids
                "ls_remobilize": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                       {"units": f"mm per {resolution} s",
                        "description": "Remobilized landslide (normalized by catchment area)"}),

                # sediment hillslope storage
                "hillslope_storage": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                      {"units": f"mm per {resolution} s",
                                       "description": "Sediment stored in hillslope (normalized by catchment area)"}),

                # sediment channel storage
                "channel_storage": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                    {"units": f"mm per {resolution} s",
                                     "description": "Sediment stored in channel (normalized by catchment area)"}),

                # sediment catchment output
                "sed_transport_real": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                       {"units": f"mm per {resolution} s",
                                        "description": "Actual sediment transfered out catchment (normalized by catchment area)"}),

                # sediment catchment output
                "sed_transport_theory": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                         {"units": f"mm per {resolution} s", "description":
                                             "Theoretical sediment transfered out catchment (normalized by catchment area)"}),

                # sediment catchment output
                "sed_limited": (("time", "iteration"), np.zeros([num_data, num_iteration]),
                                {"units": "0(False)_1(True)",
                                 "description": "Status of sediment limited in channel storage (normalized by catchment area)"}),
            },
            attrs={
                "resolution": resolution,
                "resolution_unit": f"seconds",
                "create_time": time_now
            }
        )

        return sed_container

    # for single model componment
    def run_hydro(self):

        self.hydro_container = self._create_hydro_dataset()
        # loop the individual HRUs
        for HRU_id in range(self.cfg.num_HRU.value):

            # region <snow_water_equivalent>
            s_w_e = SedCas_h_model.snow_water_equivalent(temperature=self.climate_forcing.temperature.values.copy(),
                                                         # values with s
                                                         precipitation=self.climate_forcing.precipitation.values.copy(),
                                                         snow_melt_r=self.cfg.snow_melt_r.value,  # value without s
                                                         T_theta_a=self.cfg.snow_acc_theta.value,
                                                         T_theta_m=self.cfg.snow_melt_theta.value,
                                                         snow_albedo=self.cfg.snow_albedo_y.value[HRU_id],
                                                         soil_albedo=self.cfg.snow_albedo_n.value[HRU_id]
                                                         )

            modelled_SWE, snow_delta_depth, snow_acc, snow_melt, albedo = s_w_e
            # update the hydro_output
            self.hydro_container["modelled_SWE"].loc[:, HRU_id] = modelled_SWE
            self.hydro_container["snow_delta_depth"].loc[:, HRU_id] = snow_delta_depth
            self.hydro_container["snow_acc"].loc[:, HRU_id] = snow_acc
            self.hydro_container["snow_melt"].loc[:, HRU_id] = snow_melt
            self.hydro_container["albedo"].loc[:, HRU_id] = albedo
            # endregion

            # region <potential evapotranspiration>
            p_e_t = SedCas_h_model.potential_et(temperature=self.climate_forcing.temperature.values.copy(),
                                                # values with s
                                                sun_radiation=self.climate_forcing.sun_radiation.values.copy(),
                                                albedo=self.hydro_container["albedo"].loc[:, HRU_id].values.copy(),
                                                sps_temperature=self.climate_forcing.attrs["resolution"],
                                                cloud_cover_r=self.cfg.cloud_cover_r.value,  # value without s
                                                elevation=self.cfg.c_elevation.value,
                                                relative_humidity=self.cfg.r_humidity.value
                                                )

            PET = p_e_t
            # update the hydro_output
            self.hydro_container["PET"].loc[:, HRU_id] = PET
            # endregion

            # region <hydrological process>
            h = SedCas_h_model.h_model(snow_acc=self.hydro_container["snow_acc"].loc[:, HRU_id].values.copy(),
                                       # values with s
                                       snow_melt=self.hydro_container["snow_melt"].loc[:, HRU_id].values.copy(),
                                       temperature=self.climate_forcing.temperature.values.copy(),
                                       precipitation=self.climate_forcing.precipitation.values.copy(),
                                       PET=self.hydro_container["PET"].loc[:, HRU_id].values,
                                       alpha=self.cfg.et.value,  # value without s
                                       initial_w_storage=self.cfg.initial_w_storage.value[HRU_id],
                                       w_storage_cap=self.cfg.w_storage_cap.value[HRU_id],
                                       w_residence_time=self.cfg.w_residence_time.value[HRU_id]
                                       )

            AET, Qss, Qs, Q, w_storage = h
            # update the hydro_output
            self.hydro_container["AET"].loc[:, HRU_id] = AET
            self.hydro_container["Qss"].loc[:, HRU_id] = Qss
            self.hydro_container["Qs"].loc[:, HRU_id] = Qs
            self.hydro_container["Q"].loc[:, HRU_id] = Q
            # this will contain all reservoirs,
            # row prepresent time, from column_0 to column_n represent reservoir from top to bottom
            if HRU_id == 0:
                idx = [0]
            elif HRU_id == 1:
                idx = [1, 2]  # [top, bottom reservoir]
            else:
                print("error")
                exit()

            self.hydro_container["w_storage"].loc[:, HRU_id, idx] = w_storage
            # endregion

        # lumped hydrology: area-weighted aggregation
        self.hydro_output = SedCas_h_model.area_weight_aggregate(hydro_container=self.hydro_container,
                                                                 weights=self.cfg.area_ratio_HRU.value)

    def run_sediment(self, seed_i, iteration, sed_container, fix_ls=False, save_ls=False):

        if fix_ls is True:
            output_ls = Path(project_root) / "data" / "SedCas_ls"
            cached_ls = f"{iteration:03d}.nc" # "000.nc" # 
            print(f"{UTCDateTime.now().isoformat()}\nLoad cached landslides from: {output_ls}/{cached_ls}")
            
            ds_ls = xr.open_dataset(f"{output_ls}/{cached_ls}") 
            
            # as 'pandas.core.frame.DataFrame'
            # shape by [time, landslides magnitude[mm]]
            large_ls = ds_ls["large_ls"].to_pandas()
            small_ls = ds_ls["small_ls"].to_pandas()
        else:
            print(f"{UTCDateTime.now().isoformat()}\nUse generated landslides.")
            # region <generate the large landslides>
            # generate the time series area-normalized landslide thickness

            temperature = pd.Series(
                self.climate_forcing.temperature.values,
                index=pd.to_datetime(self.climate_forcing.time_str.values)
            )
            prec = pd.Series(
                self.climate_forcing.precipitation.values,
                index=pd.to_datetime(self.climate_forcing.time_str.values)
            )
            snow = pd.Series(
                self.hydro_output.modelled_SWE.values,
                index=pd.to_datetime(self.climate_forcing.time_str.values)
            )

            # shape by [time, landslides magnitude[mm]]
            large_ls = SedCas_s_model.generate_large_ls(ls_trigger=self.cfg.ls_trigger_m.value,
                                                        temperature=temperature,  # need pd series for resample
                                                        prec=prec,
                                                        snow=snow,  # need pd series for resample
                                                        theta_sd=self.cfg.ls_trigger_SWE.value,
                                                        theta_prec=self.cfg.ls_trigger_r.value,
                                                        theta_sa=self.cfg.snow_acc_theta.value,
                                                        theta_ls_freeze=self.cfg.ls_trigger_f.value,
                                                        ls_min_v=self.cfg.ls_min_v.value,
                                                        ls_alpha_v=self.cfg.ls_alpha_v.value,
                                                        cutoff=self.cfg.ls_max_v.value,
                                                        area=self.cfg.c_area.value,
                                                        seed=seed_i)

            num_large_ls = len(large_ls[large_ls.mag > 0])
            # endregion

            # region <generate the small landslides>
            # generate the time series area-normalized landslide thickness
            # shape by [time, landslides magnitude[mm]]
            num_days = len(prec.resample('24h').sum())
            small_ls = SedCas_s_model.generate_small_ls(num_days=num_days,
                                                        num_large_ls=num_large_ls,
                                                        ls_min_v=self.cfg.ls_min_v.value,
                                                        area=self.cfg.c_area.value,
                                                        seed=seed_i,
                                                        mu=3.36, sigma=1.18, ratio=3.36)

            # date index for small landslides
            small_ls.index = large_ls.index
            # endregion

            if save_ls is True:
                ds_ls = xr.Dataset({
                    "large_ls": (("time", "ls_id"), large_ls.values),
                    "small_ls": (("time", "ls_id"), small_ls.values),
                    },
                    coords={"time": large_ls.index, "ls_id": large_ls.columns}
                )

                output_ls = Path(project_root) / "data" / "SedCas_ls"
                os.makedirs(output_ls, exist_ok=True)
                ds_ls.to_netcdf(f"{output_ls}/{iteration:03d}.nc")
            

        # region <mix water and sediments>
        # desired_freq unit by minutes
        desired_freq = self.climate_forcing.attrs['resolution'] / 60  # divide 60 -> convert second to minute
        sed_run = SedCas_t_model.trans_model(large_ls=large_ls.copy(),
                                             small_ls=small_ls.copy(),
                                             Qs=self.hydro_output["Qs"].values.copy(),
                                             modelled_SWE=self.hydro_output.modelled_SWE.values.copy(),

                                             desired_freq=desired_freq,

                                             h2s_r=self.cfg.h2s_r.value,
                                             initial_hs_storage=self.cfg.initial_hs_storage.value,
                                             initial_ch_storage=self.cfg.initial_ch_storage.value,

                                             hillslope_storage_cap=self.cfg.hillslope_storage_cap.value,
                                             channel_storage_cap=self.cfg.channel_storage_cap.value,
                                             erosion_k=self.cfg.erosion_k.value,

                                             ls_min_v=self.cfg.ls_min_v.value,
                                             ls_alpha_v=self.cfg.ls_alpha_v.value,
                                             ls_max_v=self.cfg.ls_max_v.value,
                                             c_area=self.cfg.c_area.value,

                                             bedload_param_a=self.cfg.bedload_param_a.value,
                                             bedload_param_b=self.cfg.bedload_param_b.value,
                                             max_s2w=self.cfg.max_s2w.value,

                                             Qbl=self.cfg.Qbl.value,
                                             Qdf=self.cfg.Qdf.value,
                                             entrainment=self.cfg.entrainment.value
                                             )
        
        ls_real_input, ls_remobilize, hillslope_storage, channel_storage, sed_transport_real, sed_transport_theory, sed_limited = sed_run
        
        # endregion

        # region <add the params to sed_container>
        # landslids input
        sed_container["ls"][:, iteration] = ls_real_input
        # remobilized landslids
        sed_container["ls_remobilize"][:, iteration] = ls_remobilize
        # hillslope storage time series
        sed_container["hillslope_storage"][:, iteration] = hillslope_storage
        # channel storage time series
        sed_container["channel_storage"][:, iteration] = channel_storage
        # actual catchment sediment output time series
        sed_container["sed_transport_real"][:, iteration] = sed_transport_real
        # theoretical catchment sediment output time series
        sed_container["sed_transport_theory"][:, iteration] = sed_transport_theory
        # sediments limited status
        sed_container["sed_limited"][:, iteration] = sed_limited
        # endregion

        return sed_run

    # for combine model together
    def run_stochastic_simulations(self, seed=0, num_iteration=None, progress_bars=True, fix_ls=False, save_ls=None):

        # sediment module with stochastic landslide magnitudes
        if num_iteration is None:
            # default is 100 times
            num_iteration = self.cfg.num_iteration.value
        else:
            self.cfg.num_iteration.value = int(num_iteration)

        self.sed_container = self._create_sed_dataset(num_iteration=num_iteration)

        # set iterator
        iterator = range(num_iteration)
        if progress_bars is True:
            iterator = tqdm(iterator,
                            desc="running sediment model by stochastic simulations",
                            file=sys.stdout)
        # loop
        for iteration in iterator:
            sed_run = self.run_sediment(seed_i=seed, iteration=iteration, sed_container=self.sed_container, fix_ls=fix_ls, save_ls=save_ls)
            seed = seed + 1

        # calculate the stastic values
        self.sed_output = self.post_process_quantiles(xr_dataset=self.sed_container)


    # for results post-process
    def post_process_quantiles(self, xr_dataset, quants=(1, 50, 99)):

        q_names = [f"Q{q}" for q in quants]

        # new dataset to store quantile results
        new_xr = xr.Dataset(coords=xr_dataset.coords)

        for var in xr_dataset.data_vars:

            # only compute for variables that have 'iteration' dimension
            if "iteration" in xr_dataset[var].dims:

                da = xr_dataset[var]
                data = da.values
                iter_axis = da.dims.index("iteration")
                attrs = da.attrs.copy()

                # use percentile instead of xarray.quantile, faster
                q_np = np.percentile(data, quants, axis=iter_axis)
                std_np = np.std(data, axis=iter_axis, ddof=1)

                # dimensions without iteration
                new_dims = tuple(d for d in da.dims if d != "iteration")

                # add quantiles
                for i, qname in enumerate(q_names):
                    new_var = xr.DataArray(q_np[i],
                                           dims=new_dims,
                                           coords={d: da.coords[d] for d in new_dims},
                                           attrs=attrs.copy()
                                           )
                    new_xr[f"{var}_{qname}"] = new_var

                # add standard deviation
                new_std = xr.DataArray(std_np,
                                       dims=new_dims,
                                       coords={d: da.coords[d] for d in new_dims},
                                       attrs=attrs.copy()
                                       )
                new_xr[f"{var}_std"] = new_std

        return new_xr

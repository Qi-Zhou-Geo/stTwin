#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-21T16:38:16
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd


from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

import numpy as np
import pandas as pd

from obspy import UTCDateTime

# region ### add the sys.path to search for custom modules ###
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
current_dir = current_file.parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy
project_root = current_dir.parent.parent

sys.path.append(str(project_root))
# endregion

# import the custom functions
from func.SedCas.SedCas import SedCas
from func.SedCas.load_climate_input import load_climate_input4model
from func.bayesian_inference.params_boundary import custom_boundary
from func.visulize.plotly_visualize import plotly_multi_time_series_xr


def load_config(project_dir, output_dir, 
                climate_input, climate_resolution,
                model_input_params, model_updated_params):

    # all input params are stored here and will be updated later
    params_trial = {"project_root": project_dir,
                    "output_dir": output_dir,
                    
                    "data_type": f"{climate_resolution}s",
                    "climate_input": climate_input,
                    "climate_resolution": climate_resolution,
                    
                    "model_input_params": model_input_params,
                    "model_updated_params": model_updated_params}

    # prepare the output dir
    output_dir = Path(params_trial["project_root"]) / params_trial["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)


    # load the climate forcing
    climate_input_path = Path(project_root) / f"data/SedCas_input/{params_trial['climate_input']}"
    df_climate = pd.read_csv(climate_input_path, header=0)
    climate_forcing = load_climate_input4model(df_climate, climate_resolution, 
                                data_source="MeteoSwiss", 
                                station="Montana (MVE)"
                                )
    params_trial["climate_forcing"] = climate_forcing


    # for theta bounds
    theta_names, lower_bounds, upper_bounds = custom_boundary()
    
    params_trial["theta_names"] = theta_names
    # in nature (non-log) sapce
    params_trial["lower_bounds"] = lower_bounds
    params_trial["upper_bounds"] = upper_bounds

    return params_trial


def save_last_status(model, current_params_trial, current_theta):

    # (1) parepae the I/O
    input_yaml = (f'{project_root}'
                  f'/config/SedCas_params'
                  f'/{current_params_trial["model_params"]}')

    output_yaml = (f'{project_root}'
                  f'/config/SedCas_params'
                  f'/{current_params_trial["updated_params"]}')

    # (2) prepare the new values
    # initial water storage in each reservoir
    initial_w_storage = [
        [float(model.hydro_output["w_storage"][-1, 0, 0].values)],
        [float(model.hydro_output["w_storage"][-1, 1, 1].values),
         float(model.hydro_output["w_storage"][-1, 1, 2].values)]
    ] # if you do not understand, check line 265 in functions/SedCas/SedCas.py

    # convrt to plain Python float type
    w_storage_cap = [[float(v) for v in sublist] for sublist in model.cfg.w_storage_cap.value]
    w_residence_time = [[float(v) for v in sublist] for sublist in model.cfg.w_residence_time.value]

    # Initlal hillslope storage, area-normalized sediment thickness
    initial_hs_storage = model.sed_output["hillslope_storage_Q50"].values[-1]

    # Initlal channel storage, area-normalized sediment thickness
    initial_ch_storage = model.sed_output["channel_storage_Q50"].values[-1]

    new_values = {
        "initial_w_storage": initial_w_storage,
        "w_storage_cap": w_storage_cap,
        "w_residence_time": w_residence_time,
        "initial_hs_storage": initial_hs_storage,
        "initial_ch_storage": initial_ch_storage,
    }
    inherited_parameters = ", ".join(new_values.keys())
    inherited_from_time = model.sed_output["channel_storage_Q50"].coords["time_str"][-1].item()

    for key, value in current_theta.items():
        new_values[key] = value

    # (3) update the original ymal file
    ruamel_yaml = YAML()
    ruamel_yaml.preserve_quotes = True

    with open(input_yaml, "r") as f:
        cfg = ruamel_yaml.load(f)
        # cfg
        #  ├── model_input
        #  │     └── key
        #  ├── model_config
        #  │     └── key
        #  └── model_output
        #        └── key

    for key, val in new_values.items():
        for section in cfg:  # model_input, model_config, model_output
            if key in cfg[section]:
                # cfg[section][key]["value"] = float(val)
                if key in ["initial_w_storage", "w_storage_cap", "w_residence_time"]:
                    value_ref = cfg[section][key]["value"]
                    value_ref[0][0] = val[0][0]
                    value_ref[1][0] = val[1][0]
                    value_ref[1][1] = val[1][1]
                else:
                    cfg[section][key]["value"] = float(val)

    # (5) create ReadMe section
    readme = CommentedMap()
    readme["Last Updated"] = UTCDateTime.now().isoformat()
    readme["Author"] = "QZ"
    readme["inherited_parameters"] = inherited_parameters
    readme["inherited_from_time"] = inherited_from_time
    cfg.insert(0, "ReadMe", readme) # insert at top

    # (5) save the new ymal file
    with open(output_yaml, "w") as f:
        ruamel_yaml.dump(cfg, f)

    print(f"Updated model parameters are saved at:\n"
          f" {output_yaml}\n")


def run_posterior_sedcas(params_trial, num_iteration=100,
                         
                         progress_bars=False, save_output=True, save_sed_container=False,
                         
                         fix_ls=False, save_ls=None,
                         
                         plot_output=True, show_plot=False,
                         
                         select_t1="2004-02-01T00:00:00", select_t2="2023-01-01T00:00:00",
                         time_coord="time_str"
                         ):

    project_root = params_trial["project_root"]
    model = SedCas(project_root=project_root,
                   model_input_params=f"{project_root}/config/SedCas_params/{params_trial['model_input_params']}")
    # Do NOT do:
    # model.load_climate_input(data_type=data_type)
    model.climate_forcing = params_trial["climate_forcing"]

    # region <update the model params>
    model.cfg.w_storage_cap.value[0] = [params_trial["w_storage_cap0"]] # type: ignore
    model.cfg.w_storage_cap.value[1] = [params_trial["w_storage_cap1"], # type: ignore
                                        params_trial["w_storage_cap2"]]

    model.cfg.w_residence_time.value[0] = [params_trial["w_residence_time0"]] # type: ignore
    model.cfg.w_residence_time.value[1] = [params_trial["w_residence_time1"], # type: ignore
                                           params_trial["w_residence_time2"]]

    model.cfg.ls_alpha_v.value = params_trial["ls_alpha_v"] # type: ignore

    model.cfg.Qdf.value = params_trial["Qdf"] # type: ignore
    model.cfg.max_s2w.value = params_trial["max_s2w"] # type: ignore

    model.cfg.channel_storage_cap.value = params_trial["channel_storage_cap"] # type: ignore
    model.cfg.erosion_k.value = params_trial["erosion_k"] # type: ignore

    # you must update the params then post-processing
    model._params_post_processing()
    # make it as critial value
    model.cfg.initial_hs_storage.value = model.cfg.hillslope_storage_cap.value # type: ignore
    # endregion

    # prepare the output dir
    output_dir = Path(params_trial["project_root"]) / params_trial["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # run the model
    model.run_hydro()
    model.run_stochastic_simulations(seed=0, num_iteration=num_iteration, progress_bars=progress_bars, fix_ls=fix_ls, save_ls=save_ls)

    # save the results
    if save_output is True:
        ds = model.hydro_output
        mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
        ds = ds.isel(time=mask)
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
        ds.to_netcdf(f"{output_dir}/hydro_output.nc", engine="h5netcdf", encoding=encoding)
        
        ds = model.sed_output
        mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
        ds = ds.isel(time=mask)
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
        ds.to_netcdf(f"{output_dir}/sed_output.nc", engine="h5netcdf", encoding=encoding)

    if save_sed_container is True:
        # this is super big ds file
        ds = model.sed_container
        mask = (ds.time_str >= select_t1) & (ds.time_str < select_t2)
        ds = ds.isel(time=mask)
        encoding = {var: {"zlib": True, "complevel": 4, "dtype": "float32"} for var in ds.data_vars}
        ds.to_netcdf(f"{output_dir}/sed_container.nc", engine="h5netcdf", encoding=encoding)


    # plot it
    if plot_output is True:

        # update the attrs if the xr is 2024 version
        template_sed_container = model._create_sed_dataset(num_iteration=1)
        for var in model.sed_container.data_vars:
            model.sed_container[var].attrs = template_sed_container[var].attrs.copy()
            model.sed_output[f"{var}_Q1"].attrs = template_sed_container[var].attrs.copy()
            model.sed_output[f"{var}_Q50"].attrs = template_sed_container[var].attrs.copy()
            model.sed_output[f"{var}_Q99"].attrs = template_sed_container[var].attrs.copy()

        
        # region <save plot as html>

        ## climate forcing
        mask = (model.climate_forcing.time_str >= select_t1) & (model.climate_forcing.time_str < select_t2)
        climate_forcing_2017 = model.climate_forcing.isel(time=mask)
        list_of_col_names = [(time_coord, "precipitation"),
                             (time_coord, "temperature"),
                             (time_coord, "sun_radiation")]
        fig = plotly_multi_time_series_xr(xr_dataset=climate_forcing_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_climate_forcing.html")

        ## hydro
        mask = (model.hydro_output.time_str >= select_t1) & (model.hydro_output.time_str < select_t2)
        hydro_output_2017 = model.hydro_output.isel(time=mask)

        # SWE
        list_of_col_names = [(time_coord, "modelled_SWE"), (time_coord, "snow_delta_depth"),
                             (time_coord, "snow_acc"), (time_coord, "snow_melt")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_SWE.html")

        # ET
        list_of_col_names = [(time_coord, "albedo"), (time_coord, "PET"), (time_coord, "AET")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_ET.html")

        # Q
        list_of_col_names = [(time_coord, "Q"), (time_coord, "Qs"), (time_coord, "Qss")]
        fig = plotly_multi_time_series_xr(xr_dataset=hydro_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_discharge.html")

        ## sed
        mask = (model.sed_output.time_str >= select_t1) & (model.sed_output.time_str < select_t2)
        sed_output_2017 = model.sed_output.isel(time=mask)

        # landslides
        list_of_col_names = [(time_coord, "ls_Q1"),
                             (time_coord, "ls_Q50"),
                             (time_coord, "ls_Q99")]
        fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_ls.html")

        # sed
        list_of_col_names = [(time_coord, "hillslope_storage_Q50"),
                             (time_coord, "channel_storage_Q50"),
                             (time_coord, "sed_transport_real_Q50")]
        fig = plotly_multi_time_series_xr(xr_dataset=sed_output_2017,
                                          list_of_col_names=list_of_col_names,
                                          show_plot=show_plot)
        fig.write_html(f"{output_dir}/resolution_{params_trial['data_type']}_{select_t1[:4]}_{select_t2[:4]}_sediments.html")
        # endregion

    return model

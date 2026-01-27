#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = 2025-09-24
# __author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# __find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# __note__ = This code is adapted from SedCas (Author: Jacob Hirschberg, Created: 2022-02-03, Source: https://github.com/jacobhirschberg/SedCas)
#           and is distributed under the terms of the GNU General Public License v3.0 (GPL-3.0).

import yaml
from dataclasses import dataclass

# <editor-fold desc="add the sys.path to search for custom modules">
from pathlib import Path

current_dir = Path(__file__).resolve().parent
# using ".parent" on a "pathlib.Path" object moves one level up the directory hierarchy

project_root = current_dir.parent.parent
import sys

sys.path.append(str(project_root))
# </editor-fold>


# import the custom functions


@dataclass
# Python automatically creates: __init__(), __repr__(), __eq__()
class ConfigItem:
    name: str # keep exactly as the ymal file structure
    value: any
    attrs: dict

class ModelConfig:

    def __init__(self, model_input_params: str):

        if model_input_params == "default":
            yaml_path = f"{project_root}/config/SedCas_params/SedCas_input_params_c.yaml"
        else:
            yaml_path = model_input_params

        # load YAML file
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # create ConfigItem objects for each section
        for p in ["model_input", "model_config", "model_output"]:
            for key, section in data[p].items():
                setattr(self, key, ConfigItem(**section))

    def print_config_params(self, check_params=None):

        if check_params is None:
            for key, item in self.__dict__.items():
                record = (f"{key}:"
                          f"  name: {type(item.name).__name__}>>{item.name}\n"
                          f"  value: {type(item.value).__name__}>>{item.value}\n"
                          f"  attrs: {type(item.attrs).__name__}>>{item.attrs}\n \n")
                print(record)
        else:
            key = check_params
            if key in self.__dict__:
                item = self.__dict__[check_params]
                record = (f"{key}:\n"
                          f"  name: {type(item.name).__name__}>>{item.name}\n"
                          f"  value: {type(item.value).__name__}>>{item.value}\n"
                          f"  attrs: {type(item.attrs).__name__}>>{item.attrs}\n \n")
                print(record)
            else:
                print(f"Parameter '{check_params}' not found.")

def main(model_input_params):
    '''
    Load the model parameters from yaml file

    Args:
        model_input_params: str, path to the model parameters

    Returns:

    '''

    cfg = ModelConfig(model_input_params)

    return cfg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='input parameters')
    parser.add_argument("--model_input_params", type=str)
    args = parser.parse_args()

    main(args.model_input_params)

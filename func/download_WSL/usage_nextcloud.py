#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = Last modified: 2026-08-15T11:29:35
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import yaml
import urllib.parse
from obspy import UTCDateTime

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
from func.download_WSL.fetch_nextcloud import data_exchange, list_folder




key_path = Path(project_root) / "config/gfz_nextcloud_key.yml"
with open(key_path, "r") as f:
      config = yaml.safe_load(f)
      
      base_url = config[f"base_url"]
      share_token = config[f"share_token"]
      pass_word = config[f"pass_word"]



julday = 167 # change this as julday

# Local file
local_path = Path(project_root) / f"data/seismic/{julday}" / "9S.ILL12..EHZ_15_15_56.MSEED"
local_filename = local_path.name


# Remote Nexecloud file
remote_dir = f"{julday:03d}"
remote_dir = urllib.parse.quote(remote_dir) # URL encode
remote_dir_url = f"{base_url}/{remote_dir}"


remote_path = f"{remote_dir}/{local_filename}"
remote_path = urllib.parse.quote(remote_path) # URL encode
remote_file_url = f"{base_url}/{remote_path}"


# usage 1
purpose = "upload"
response = data_exchange(purpose, local_path, remote_dir_url, remote_file_url, share_token, pass_word)
print(f"{UTCDateTime.now().isoformat()}\n"
      f"{purpose.capitalize()} done: status_code={response.status_code}\n")

# usage 2
purpose = "download"
response = data_exchange(purpose, local_path, remote_dir_url, remote_file_url, share_token, pass_word)
print(f"{UTCDateTime.now().isoformat()}\n"
      f"{purpose.capitalize()} done: status_code={response.status_code}\n")


file_name = list_folder(remote_dir_url, share_token, pass_word)
print(len(file_name))
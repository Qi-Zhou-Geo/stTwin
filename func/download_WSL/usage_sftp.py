#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = Last modified: 2026-08-15T11:29:52
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd

import os
import yaml
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
from func.download_WSL.fetch_sftp import connect_sftp, data_exchange, list_folder



key_path = Path(project_root) / "config/gfz_sftp_key.yml"
with open(key_path, "r") as f:
      config = yaml.safe_load(f)
      
      host = config[f"host"]
      port = config[f"port"]
      username = config[f"username"]
      password = config[f"password"]
      private_key_path = config[f"private_key_path"]
      remote_dir = config[f"remote_dir"]
      
      if private_key_path == "None":
            private_key_path = None


julday = 167
seismic_data_list = os.listdir(Path(project_root) / f"data/seismic/167")
seismic_data_list = sorted(seismic_data_list )

for seismic_data in seismic_data_list:

      # Local file dir. and file path
      local_file_dir = Path(project_root) / f"data/seismic/{str(julday).zfill(3)}"
      local_file_dir = f"{local_file_dir}"

      local_file_path = Path(project_root) / f"data/seismic/{str(julday).zfill(3)}" / seismic_data
      local_file_path = f"{local_file_path}"


      # Remote file dir. and file path
      remote_file_dir = Path(remote_dir) / f"wsl_gfz/{str(julday).zfill(3)}"
      remote_file_dir = f"{remote_file_dir}"

      remote_file_path = Path(remote_dir) / f"wsl_gfz/{str(julday).zfill(3)}" / seismic_data
      remote_file_path = f"{remote_file_path}"


      # build the SFTP connection
      sftp, transport = connect_sftp(host, port, username, password, private_key_path)


      # usage 1
      purpose = "upload"
      msg = data_exchange(sftp, purpose, local_file_path, remote_file_dir, remote_file_path)
      print(f"{UTCDateTime.now().isoformat()}\n"
            f"{purpose.capitalize()}: {seismic_data}\n"
            f"{msg}\n")

      # usage 2
      purpose = "download"
      msg = data_exchange(sftp, purpose, local_file_path, remote_file_dir, remote_file_path)
      print(f"{UTCDateTime.now().isoformat()}\n"
            f"{purpose.capitalize()}: {seismic_data}\n"
            f"{msg}\n")

      # close the sftp
      sftp.close() # type: ignore
      transport.close()


# build the SFTP connection
sftp, transport = connect_sftp(host, port, username, password, private_key_path)

# usage 3 
remote_file_dir = Path(remote_dir) / f"wsl_gfz/{str(julday).zfill(3)}"
remote_file_dir = f"{remote_file_dir}"
msg, file_name = list_folder(sftp, remote_file_dir)
print(file_name)

# close the sftp
sftp.close() # type: ignore
transport.close()

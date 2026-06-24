#!/usr/bin/python
# -*- coding: UTF-8 -*-

# __modification time__ = Last modified: 2026-06-24T13:22:07
# __author__ = Qi Zhou, GFZ Helmholtz Centre for Geosciences
# __find me__ = qi.zhou@gfz.de, qi.zhou.geo@gmail.com, https://github.com/Qi-Zhou-Geo
# Please do not distribute this functions without the author's permission

import os
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
from func.download_WSL.fetch_sftp import connect_sftp, list_folder, data_exchange


def sftp_metadata():
    
    # meta data
    host = "ftp.gfz.de"
    port = 22
    username = "anonymous"
    password = "anonymous@gfz.de"
    private_key_path = None
    remote_dir = "/pub/incoming/aiPmp4XnEuWxdfLRYf9yf7XRuncWdHHP_2026061710000319"
    
    return host, port, username, password, private_key_path, remote_dir


def check_new_data_name():
    
    # time now
    t_now = UTCDateTime.now()
    julday = t_now.julday
    hour = t_now.hour
    minute = t_now.minute

    # is current time in the mid-night: t_now <= Year-Month-Day2T00:05:00
    in_range = (hour == 0 and minute <= 5)
    if in_range is True:
        # check yesterday and today
        j_list = [julday - 1, julday]
    else:
        # only check today
        j_list = [julday]
    
    
    # build the SFTP connection
    host, port, username, password, private_key_path, remote_dir = sftp_metadata()
    sftp, transport = connect_sftp(host, port, username, password, private_key_path)

    # check remote file
    all_remote_data = []
    for j in j_list:
        remote_file_dir = Path(remote_dir) / f"wsl_gfz/{str(j).zfill(3)}"
        remote_file_dir = f"{remote_file_dir}"
        msg, file_name = list_folder(sftp, remote_file_dir)
        
        # replace the previous message
        msg = f"Find: {len(file_name)} files in {remote_file_dir}"
        
        file_name = sorted(file_name)
        
        # add julday marker
        temp = [str(j) + "=" + x for x in file_name]
        all_remote_data = all_remote_data + temp

    
    # close the sftp
    sftp.close() # type: ignore
    transport.close()
    
    
    # check local file
    all_local_data = []
    for j in j_list:
        local_file_dir = Path(project_root) / f"deploy/liveshow_cache/seismic/{str(julday).zfill(3)}"
        local_file_dir.mkdir(parents=True, exist_ok=True)
        local_file_dir = f"{local_file_dir}"
        file_name = os.listdir(local_file_dir)
        file_name = sorted(file_name)
        
        # add julday marker
        temp = [str(j) + "=" + x for x in file_name]
        all_local_data = all_local_data + temp
        
    
    # new file in remote, but not in local
    new_data_name = list(set(all_remote_data) - set(all_local_data))
    
    if len(new_data_name) == 0:
        is_new_data = False
    else:
        is_new_data = True
    
    return is_new_data, new_data_name, msg # type: ignore


def download_new_data(new_data_name, logger):
    
    # build the SFTP connection
    host, port, username, password, private_key_path, remote_dir = sftp_metadata()
    sftp, transport = connect_sftp(host, port, username, password, private_key_path)

    for file_name in new_data_name:

        julday, seismic_data = file_name.split("=")
        
        # Local file dir. and file path
        local_file_dir = Path(project_root) / f"deploy/liveshow_cache/seismic/{str(julday).zfill(3)}"
        local_file_dir.mkdir(parents=True, exist_ok=True)
        local_file_dir = f"{local_file_dir}"

        local_file_path = Path(project_root) / f"deploy/liveshow_cache/seismic/{str(julday).zfill(3)}" / seismic_data
        local_file_path = f"{local_file_path}"


        # Remote file dir. and file path
        remote_file_dir = Path(remote_dir) / f"wsl_gfz/{str(julday).zfill(3)}"
        remote_file_dir = f"{remote_file_dir}"

        remote_file_path = Path(remote_dir) / f"wsl_gfz/{str(julday).zfill(3)}" / seismic_data
        remote_file_path = f"{remote_file_path}"


        # usage 2
        purpose = "download"
        msg = data_exchange(sftp, purpose, local_file_path, remote_file_dir, remote_file_path)
        print(f"{purpose.capitalize()}: {seismic_data}\n"
            f"{msg}\n")
        logger.info(msg)

    # close the sftp
    sftp.close() # type: ignore
    transport.close()


# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 10:54:37 2025

@author: rmgu
"""

import datetime as dt
from pathlib import Path
import re

import numpy as np
import rasterio
import xarray as xr

from meteo_utils import ecmwf_utils as eu
from meteo_utils import solar_irradiance as sun


METEO_DATA_FIELDS = ["TA", "EA", "WS", "PA", "SW-IN", "LW-IN"]
DAILY_VARS = ["ETr+"]
DAILY_SUB_VARS = ["ETr", "SW-IN-DD"]


def process_single_date(elev_input_file,
                        slope_input_file,
                        aspect_input_file,
                        acq_datetime,
                        dst_folder=None,
                        svf_input_file=None,
                        blending_height=100,
                        edh_credentials_file=Path.home()/".netrc"):
    """

    Parameters
    ----------
    elev_input_file : str or Path
        Path to a GDAL compatible Digital Elevation Model
    slope_input_file : str or Path
        Path to a GDAL compatible slope image (degrees)
    aspect_input_file : str or Path
        Path to a GDAL compatible aspect image (0 for flat surfaces))
    acq_datetime : datetime
        Acquistion date and time
    dst_folder : str, optional
        Path to the destination folder in which meteo products will be stored.
    svf_input_file : str, optional
        Path to a GDAL compatible Sky View Fraction image (0-1)
    blending_height : float, optional
        Elevation above ground level at which meteo products will be generated, default=100 magl
    edh_credentials_file: str or Path
        Path to Earth Data Hub credentials file - https://earthdatahub.destine.eu/getting-started

    Returns
    -------
    output : dict
        Dictionary of arrays with the output meteo products
    """

    with open(edh_credentials_file, "r") as f:
        lines = f.read().splitlines()
        pat = re.search("edh_pat_.*", lines[-1]).group(0)

    with rasterio.open(elev_input_file, "r") as fp:
        bounds = fp.bounds
        latitude_slice = slice(bounds[3]+0.25, bounds[1]-0.25)
        longitude_slice = [bounds[i] if bounds[i] >= 0 else bounds[i]+360 for i in [0, 2]]
        longitude_slice = slice(longitude_slice[0]-0.25, longitude_slice[1]+0.25)
        profile = fp.profile
        profile.update({"dtype": "float32", "nodata": np.nan})

    dst_folder = Path(dst_folder)
    print('Reading meteorological data from the Earth Data Hub')

    ds = xr.open_dataset(
        f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    # Sentinel-3 acquisition should be around 10-11 local time and we want to capture a full day
    min_time = acq_datetime - dt.timedelta(hours=13)
    max_time = acq_datetime + dt.timedelta(hours=16)
    time_slice = slice(np.datetime64(f"{min_time:%Y-%m-%dT%H:00:00.0}"),
                       np.datetime64(f"{max_time:%Y-%m-%dT%H:00:00.0}"))
    meteo_ds = ds.sel(latitude=latitude_slice,
                      longitude=longitude_slice,
                      valid_time=time_slice)

    time_zone = sun.angle_average(bounds[0], bounds[2]) / 15.0
    output = eu.get_ECMWF_data(
        meteo_ds,
        acq_datetime,
        METEO_DATA_FIELDS + DAILY_VARS,
        elev_input_file,
        blending_height,
        aod550_data_file=None,
        time_zone=time_zone,
        is_forecast=False,
        slope_file=slope_input_file,
        aspect_file=aspect_input_file,
        svf_file=svf_input_file,
    )

    if dst_folder:
        for param, data in output.items():
            if param == "SW-IN":
                for i, var1 in enumerate(["DIR", "DIF"]):
                    for j, var2 in enumerate(["PAR", "NIR"]):
                        param = f"{var2}-{var1}"
                        dst_file = dst_folder / f"{acq_datetime:%Y%m%dT%H%M%S}_{param}.tif"
                        print(f"Saving {param} to {dst_file}")
                        with rasterio.open(dst_file, "w", **profile) as meteo:
                            meteo.write(data[i][j], 1)
            elif param in METEO_DATA_FIELDS:
                dst_file = dst_folder / f"{acq_datetime:%Y%m%dT%H%M%S}_{param}.tif"
                print(f"Saving {param} to {dst_file}")
                with rasterio.open(dst_file, "w", **profile) as meteo:
                    meteo.write(data, 1)
            elif param == "ETr+":
                for i, sub_param in enumerate(["ETr", "TA-MIN-DD", "TA-MAX-DD", "SW-IN-DD",
                                               "EA-DD", "WS-DD", "PA-DD"]):
                    if sub_param in DAILY_SUB_VARS:
                        dst_file = dst_folder / f"{acq_datetime:%Y%m%d}_{sub_param}.tif"
                        print(f"Saving {sub_param} to {dst_file}")
                        with rasterio.open(dst_file, "w", **profile) as meteo:
                            meteo.write(data[i], 1)
            else:
                dst_file = dst_folder / f"{acq_datetime:%Y%m%d}_{param}.tif"
                print(f"Saving {param} to {dst_file}")
                with rasterio.open(dst_file, "w", **profile) as meteo:
                    meteo.write(data, 1)

    return output


if __name__ == "__main__":

    process_single_date(
        elev_input_file=Path() / "test" / "dem.tif",
        slope_input_file=Path() / "test" / "slope.tif",
        aspect_input_file=Path() / "test" / "aspect.tif",
        acq_datetime=dt.datetime(2020, 6, 20, 12, 0, 0),
        dst_folder=Path() / "test" / "test_era5_edh",
        svf_input_file=Path() / "test" / "svf.tif",
    )

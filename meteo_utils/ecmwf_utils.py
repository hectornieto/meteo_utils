# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:47:50 2018

@author: rmgu
"""

import datetime as dt
import os
import re
import cdsapi
import xarray as xr
import rioxarray
from cfgrib import xarray_store
import numpy as np
import pandas as pd
from osgeo import gdal
import yaml
from pyTSEB import energy_combination_ET as pet
from pyTSEB import meteo_utils as met

from . import dem_utils as du
from . import gdal_utils as gu
from . import solar_irradiance as solar

DRY_LAPSE_RATE = 9.8e-3  # K m-1
ENVIRONMENTAL_LAPSE_RATE = 6.5e-3  # K m-1
# Dew temperature lapse rate
TDEW_LAPSE_RATE = 2e-3  # K m-1

SZA_THRESHOLD_CORRECTION = 75
TCWV_MIDLATITUDE_SUMMER = 2.85
RURAL_AOT_25KM = 0.2
# Table 8.3 Roughness lengths for momentum and heat associated with high
# and low vegetation types.
#               Index:[ Vegetation type, H/L, z0m, z0h]
ECMWF_ZO_LUT = {1: ['Crops, mixed farming', 'L', 0.25, 0.25e-2],
                2: ['Short grass', 'L', 0.2, 0.2e-2],
                3: ['Evergreen needleleaf trees', 'H', 2.0, 2.0],
                4: ['Deciduous needleleaf trees', 'H', 2.0, 2.0],
                5: ['Deciduous broadleaf trees', 'H', 2.0, 2.0],
                6: ['Evergreen broadleaf trees', 'H', 2.0, 2.0],
                7: ['Tall grass', 'L', 0.47, 0.47e-2],
                8: ['Desert', '-', 0.013, 0.013e-2],
                9: ['Tundra', 'L', 0.034, 0.034e-2],
                10: ['Irrigated crops', 'L', 0.5, 0.5e-2],
                11: ['Semidesert', 'L', 0.17, 0.17e-2],
                12: ['Ice caps and glaciers', '-', 1.3e-3, 1.3e-4],
                13: ['Bogs and marshes', 'L', 0.83, 0.83e-2],
                14: ['Inland water', '-', '-', '-'],
                15: ['Ocean', '-', '-', '-'],
                16: ['Evergreen shrubs', 'L', 0.10, 0.10e-2],
                17: ['Deciduous shrubs', 'L', 0.25, 0.25e-2],
                18: ['Mixed forests/woodland', 'H', 2.0, 2.0],
                19: ['Interrupted forest', 'H', 1.1, 1.1],
                20: ['Water and land mixtures', 'L', '-', '-']}

LOW_LC_TYPES = (1, 2, 7, 10, 11, 13, 16, 17)
HIGH_LC_TYPES = (3, 4, 5, 6, 18, 19)

# Acceleration of gravity (m s-2)
GRAVITY = 9.80665
# Ratio of the molecular weight of water vapor to dry air
EPSILON = 0.622
# Gas constant of dry air (J kg-1 K-1)
R_D = 287.04
# Gas constant of wet air (J kg-1 K-1)
R_W = 461.5

# Baseline hours for beginning of ECWMF integrated values
HOURS_FORECAST_CAMS = (1, 13)
HOURS_FORECAST_ERA5 = range(0, 25)

ADS_CREDENTIALS_FILE = os.path.join(os.path.expanduser("~"),
                                    '.adsapirc')

DEM_NANS = [-32768, -9999, -999]

GRIB_KWARGS ={'engine': 'cfgrib',
              'time_dims': ['valid_time'],
              'ignore_keys': ['edition'],
              'extra_coords': {'expver': 'valid_time'}}

def download_CDS_data(dataset,
                      product_type,
                      date_start,
                      date_end,
                      variables,
                      target,
                      overwrite=False,
                      area=None):
    s = {"variable": variables, 
         "data_format": "grib",
         "download_format": "unarchived"}
    s["date"] = date_start.strftime("%Y-%m-%d") + "/" + date_end.strftime(
        "%Y-%m-%d")
    if area:
        s["area"] = area
    if dataset == "reanalysis-era5-single-levels" or dataset == "reanalysis-era5-land":
        if product_type:
            s["product_type"] = product_type
        if "ensemble" in product_type:
            s["time"] = [str(t).zfill(2) + ":00" for t in range(0, 24, 3)]
        else:
            s["time"] = [str(t).zfill(2) + ":00" for t in range(0, 24, 1)]
    elif dataset == "cams-global-reanalysis-eac4":
        s["time"] = [str(t).zfill(2) + ":00" for t in range(0, 24, 3)]
    else:
        raise ValueError("Unknown CDS dataset : " + dataset)

    # Connect to the server and download the data
    if not os.path.exists(target) or overwrite:
        c = cdsapi.Client(quiet=True, progress=False)
        c.retrieve(dataset, s, target)
    print("Downloaded")


def download_ADS_data(dataset,
                      date_start,
                      date_end,
                      variables,
                      target,
                      overwrite=False,
                      area=None):
    with open(ADS_CREDENTIALS_FILE, 'r') as f:
        credentials = yaml.safe_load(f)

    s = {"variable": variables, 
         "data_format": "grib"}
    s["date"] = date_start.strftime("%Y-%m-%d") + "/" + date_end.strftime(
        "%Y-%m-%d")

    if dataset == "cams-global-atmospheric-composition-forecasts":
        s["type"] = "forecast"
        s['time'] = ['00:00', '12:00']
        s['leadtime_hour'] = [str(i + 1) for i in range(12)]
    elif dataset == "cams-global-reanalysis-eac4":
        s["time"] = [str(t).zfill(2) + ":00" for t in range(0, 24, 3)]

    if area:
        s["area"] = area

    # Connect to the server and download the data
    if not os.path.exists(target) or overwrite:
        c = cdsapi.Client(url=credentials['url'], key=credentials['key'], quiet=True, progress=False)
        c.retrieve(dataset, s, target)
    print("Downloaded")


def get_ECMWF_data(ecmwf_data_file,
                   timedate_UTC,
                   meteo_data_fields,
                   elev_file,
                   z_bh,
                   slope_file,
                   aspect_file,
                   svf_file=None,
                   aod550_data_file=None,
                   time_zone=0,
                   is_forecast=False):

    timedate_UTC = timedate_UTC.replace(tzinfo=dt.timezone.utc)
    # Find midnight in local time and convert to UTC time
    date_local = (timedate_UTC + dt.timedelta(
        hours=time_zone)).date()
    midnight_local = dt.datetime.combine(
        date_local, dt.time()).replace(tzinfo=dt.timezone.utc)
    midnight_UTC = midnight_local - dt.timedelta(hours=time_zone)

    ftime = timedate_UTC.hour + timedate_UTC.minute / 60
    xds = xr.open_dataset(ecmwf_data_file, **GRIB_KWARGS)

    xds.rio.write_crs(4326, inplace=True).rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude",
        inplace=True).rio.write_coordinate_system(inplace=True)

    if "valid_time" in xds.variables:
        dates = xds["valid_time"].values
    else:
        dates = xds["time"].values
    beforeI, afterI, frac = _bracketing_dates(dates, timedate_UTC)
    if beforeI is None:
        return None

    lat = xds["latitude"].values
    lon = xds["longitude"].values
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.astype(float), lats.astype(float)

    if aod550_data_file is not None:
        ads = xr.open_dataset(aod550_data_file, **GRIB_KWARGS)
        # Stack the forecast time dimensions
        if is_forecast:
            ads = ads.stack(dim=["forecast_reference_time", "forecast_period"])
        ads.rio.write_crs(4326, inplace=True).rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude",
            inplace=True).rio.write_coordinate_system(inplace=True)
        if "valid_time" in ads.variables:
            datesaot = ads["valid_time"].values
        else:
            datesaot = ads["time"].values
    else:
        ads = None

    elev_data = gu.raster_data(elev_file).astype(float)
    elev_data[np.isin(elev_data, DEM_NANS)] = np.nan
    output = dict()
    for field in meteo_data_fields:
        if field == "TA":
            t2m, gt, proj = _getECMWFTempInterpData(xds, "t2m",
                                                    beforeI, afterI, frac)
            if "z" in xds.variables:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(xds, "z", beforeI,
                                                      afterI, frac)
                z /= GRAVITY
            else:
                _, _, _, _, extent, _ = gu.raster_info('NETCDF:"' + ecmwf_data_file + '":t2m')
                outDs = gdal.Warp("",
                                  elev_file,
                                  format="MEM",
                                  dstSRS=proj,
                                  xRes=gt[1],
                                  yRes=gt[5],
                                  outputBounds=extent,
                                  resampleAlg="average")
                z = outDs.GetRasterBand(1).ReadAsArray()
                del outDs

            sp, gt, proj = _getECMWFTempInterpData(xds, "sp",
                                                   beforeI, afterI, frac)
            # Resample dataset and calculate actual blending height temperature based on input
            # elevation data
            z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                    resample_alg="bilinear")
            t2m = _ECMWFRespampleData(t2m, gt, proj, template_file=elev_file)
            data = calc_air_temperature_blending_height(t2m,
                                                        elev_data + z_bh,
                                                        z_0=z + 2)

        elif field == "EA":
            d2m, gt, proj = _getECMWFTempInterpData(xds, "d2m",
                                                    beforeI, afterI, frac)
            if "z" in xds.variables:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(xds, "z", beforeI,
                                                      afterI, frac)
                z /= GRAVITY
            else:
                _, _, _, _, extent, _ = gu.raster_info(
                    'NETCDF:"' + ecmwf_data_file + '":t2m')
                outDs = gdal.Warp("",
                                  elev_file,
                                  format="MEM",
                                  dstSRS=proj,
                                  xRes=gt[1],
                                  yRes=gt[5],
                                  outputBounds=extent,
                                  resampleAlg="average")
                z = outDs.GetRasterBand(1).ReadAsArray()
                del outDs

            z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                    resample_alg="bilinear")
            d2m = _ECMWFRespampleData(d2m, gt, proj, template_file=elev_file)
            d2m = calc_dew_temperature_blending_height(d2m,
                                                       elev_data,
                                                       z_0=z + 2.0)
            data = calc_vapour_pressure(d2m)

        elif field == "WS":
            if "u100" in xds.variables and "v100" in xds.variables:
                u100, gt, proj = _getECMWFTempInterpData(xds, "u100",
                                                         beforeI, afterI, frac)
                v100, gt, proj = _getECMWFTempInterpData(xds, "v100",
                                                         beforeI, afterI, frac)
                # Combine the two components of wind speed and calculate speed at blending height
                ws = calc_wind_speed(u100, v100)
                z_u = 100
            else:
                u10, gt, proj = _getECMWFTempInterpData(xds, "u10",
                                                         beforeI, afterI, frac)
                v10, gt, proj = _getECMWFTempInterpData(xds, "v10",
                                                         beforeI, afterI, frac)
                # Combine the two components of wind speed and calculate speed at blending height
                ws = calc_wind_speed(u10, v10)
                z_u = 10
            if z_bh == z_u:
                data = _ECMWFRespampleData(ws, gt, proj, template_file=elev_file)
            else:
                ws = _ECMWFRespampleData(ws, gt, proj, template_file=elev_file)
                if "sr" in xds.variables:
                    z_0M, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "sr",
                                                             beforeI, afterI,
                                                             frac)
                    z_0M = _ECMWFRespampleData(z_0M, gt, proj, template_file=elev_file)
                elif "fsr" in xds.variables:
                    z_0M, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "fsr",
                                                             beforeI, afterI,
                                                             frac)
                    z_0M = _ECMWFRespampleData(z_0M, gt, proj, template_file=elev_file)

                else:
                    z_0M = np.full_like(ws, 0.25)
                # Calculate wind speed at blending height.
                data = calc_windspeed_blending_height(ws, z_0M, z_bh)

        elif field == "PA":
            sp, gt, proj = _getECMWFTempInterpData(xds, "sp",
                                                   beforeI, afterI, frac)
            if "z" in xds.variables:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(xds,
                                                      "z",beforeI, afterI, frac)
                z /= GRAVITY
                z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                        resample_alg="bilinear")

            else:
                proj, _, _, _, extent, _ = gu.raster_info(
                    'NETCDF:"' + ecmwf_data_file + '":t2m')
                outDs = gdal.Warp("",
                                  elev_file,
                                  format="MEM",
                                  dstSRS=proj,
                                  xRes=gt[1],
                                  yRes=gt[5],
                                  outputBounds=extent,
                                  resampleAlg="average")
                z = outDs.GetRasterBand(1).ReadAsArray()
                z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                        resample_alg="bilinear")
                del outDs

            t0, gt, proj = _getECMWFTempInterpData(xds, "t2m",
                                                   beforeI, afterI, frac)

            t0 = _ECMWFRespampleData(t0, gt, proj, template_file=elev_file)
            # Convert pressure from pascals to mb
            sp = _ECMWFRespampleData(sp, gt, proj, template_file=elev_file)
            # Calcultate pressure at 0m datum height
            sp = calc_pressure_height(sp, t0, elev_data, z_0=z)
            data = calc_pressure_mb(sp)

        elif field == "TCWV":
            tcwv, gt, proj = _getECMWFTempInterpData(xds, "tcwv",
                                                     beforeI, afterI, frac)
            data = calc_tcwv_cm(tcwv)
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "TP":
            before, after, frac = _bracketing_dates(dates, timedate_UTC)
            if is_forecast:
                if timedate_UTC.hour in HOURS_FORECAST_CAMS:
                    ref_hour = -1
                else:
                    ref_hour = after - 1
            else:
                ref_hour = -1
            data, gt, proj = _get_ECMWF_cummulative_data(
                xds, "strd", ref_hour, after)
            data = data * 1000
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "LW-IN":
            before, after, frac = _bracketing_dates(dates, timedate_UTC)
            if is_forecast:
                if timedate_UTC.hour in HOURS_FORECAST_CAMS:
                    ref_hour = -1
                else:
                    ref_hour = after - 1
            else:
                ref_hour = -1
            data, gt, proj = _get_ECMWF_cummulative_data(
                xds, "strd", ref_hour, after)

            data = data / 3600.
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "SW-IN":
            t, gt, proj = _getECMWFTempInterpData(xds, "t2m",
                                                  beforeI, afterI, frac)
            t = _ECMWFRespampleData(t, gt, proj, elev_file)
            if "z" in xds.variables:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(xds, "z",
                                                      beforeI,
                                                      afterI, frac)
                z /= GRAVITY
                z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                        resample_alg="bilinear")

            else:
                proj, _, _, _, extent, _ = gu.raster_info(
                    'NETCDF:"' + ecmwf_data_file + '":t2m')
                outDs = gdal.Warp("",
                                  elev_file,
                                  format="MEM",
                                  dstSRS=proj,
                                  xRes=gt[1],
                                  yRes=gt[5],
                                  outputBounds=extent,
                                  resampleAlg="average")
                z = outDs.GetRasterBand(1).ReadAsArray()
                z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                        resampleAlg="bilinear")
                del outDs

            sp, _, _ = _getECMWFTempInterpData(xds, "sp",
                                               beforeI, afterI, frac)
            sp = _ECMWFRespampleData(sp, gt, proj, template_file=elev_file)
            # Calcultate pressure at 0m datum height
            sp = calc_pressure_height(sp, t, elev_data, z_0=z)
            sp = calc_pressure_mb(sp)
            if "tcwv" in xds.variables:
                tcwv, _, _ = _getECMWFTempInterpData(xds, "tcwv",
                                                     beforeI, afterI, frac)
                tcwv = calc_tcwv_cm(tcwv)
                tcwv = _ECMWFRespampleData(tcwv, gt, proj, elev_file)

            elif aod550_data_file is not None:
                b, a, f = _bracketing_dates(datesaot, timedate_UTC)
                tcwv, gtaot, projaot = _getECMWFTempInterpData(ads,
                                                               "tcwv", b, a, f)
                tcwv = calc_tcwv_cm(tcwv)
                tcwv = _ECMWFRespampleData(tcwv, gtaot, projaot, elev_file)
            else:
                tcwv = np.full_like(sza, TCWV_MIDLATITUDE_SUMMER)

            if "aod550" in xds.variables:
                aot550, gtaot, projaot = _getECMWFTempInterpData(
                    xds,"aod550", beforeI, afterI, frac)
                aot550 = _ECMWFRespampleData(aot550, gtaot, projaot, elev_file)
            elif aod550_data_file is not None:
                b, a, f = _bracketing_dates(datesaot, timedate_UTC)
                aot550, gtaot, projaot = _getECMWFTempInterpData(
                    ads, "aod550", b, a, f)
                aot550 = _ECMWFRespampleData(aot550, gtaot, projaot, elev_file)
            else:
                aot550 = np.full_like(sza, RURAL_AOT_25KM)

            doy = float(timedate_UTC.strftime("%j"))
            sza, saa = met.calc_sun_angles(lats, lons, 0, doy, ftime)
            sza = _ECMWFRespampleData(sza, gt, proj, elev_file)
            eb, ed = solar.calc_global_horizontal_radiance_clear_sky(doy,
                                                                     sza,
                                                                     aot550,
                                                                     tcwv,
                                                                     sp,
                                                                     t,
                                                                     altitude=elev_data,
                                                                     calc_diffuse=True)

            saa = _ECMWFRespampleData(saa, gt, proj, elev_file,
                                      resample_alg="bilinear")
            non_shaded = du.non_occluded_terrain(elev_file, saa, sza,
                                                 smooth_radius=1)
            slope = gu.raster_data(slope_file)
            slope[slope < 0] = np.nan
            aspect = gu.raster_data(aspect_file)
            aspect[aspect < 0] = np.nan
            lat_hr = _ECMWFRespampleData(lats,
                                         gt, proj, elev_file,
                                         resample_alg="bilinear")
            lon_hr = _ECMWFRespampleData(lons,
                                         gt, proj, elev_file,
                                         resample_alg="bilinear")
            illum_f = du.inclination_factors(lat_hr, slope, aspect)
            if svf_file is not None:
                svf = gu.raster_data(svf_file)
                svf[svf < 0] = np.nan
            else:
                svf = np.ones(slope.shape)

            data = calc_radiance_tilted_surface_op(eb,
                                                   ed,
                                                   doy,
                                                   ftime,
                                                   lon_hr,
                                                   sza,
                                                   illum_f[0],
                                                   illum_f[1],
                                                   illum_f[2],
                                                   svf,
                                                   non_shaded=non_shaded)

        elif field == "AOT":
            if aod550_data_file is not None:
                b, a, f = _bracketing_dates(datesaot, timedate_UTC)
                aod550, gt, proj = _getECMWFTempInterpData(ads,
                                                           "aod550", b, a, f)
            else:
                aod550, gt, proj = _getECMWFTempInterpData(xds,
                                                           "aod550", beforeI,
                                                           afterI, frac)

            data = _ECMWFRespampleData(aod550, gt, proj, template_file=elev_file)

        elif field == "SW-IN-DD":
            # Interpolate solar irradiance over 24 hour period starting at midnight local time
            data, gt, proj = _getECMWFSolarData(xds,
                                                midnight_UTC,
                                                elev_file,
                                                slope_file,
                                                aspect_file,
                                                svf_file=svf_file,
                                                time_window=24,
                                                is_forecast=is_forecast,
                                                aot_ds=ads)


        elif field == "ETr":
            data = daily_reference_et(xds,
                                      timedate_UTC,
                                      elev_file,
                                      slope_file,
                                      aspect_file,
                                      svf_file=svf_file,
                                      time_zone=time_zone,
                                      z_u=10,
                                      z_t=2,
                                      is_forecast=is_forecast,
                                      aot_ds=ads)[0]

        elif field == "TP-DD":
            data, gt, proj = _get_cummulative_data(ecmwf_data_file,
                                                   "tp",
                                                   midnight_UTC,
                                                   elev_file,
                                                   time_window=24,
                                                   is_forecast=is_forecast)
            # Convert to mm
            data = data * 1000

        elif field == "TA-DD":
            data = mean_temperature(xds,
                                    elev_file,
                                    timedate_UTC,
                                    time_zone=time_zone,
                                    z_t=2)

        output[field] = data

    return output


def calc_air_temperature_blending_height(ta, z_1, z_0=2.0):
    """
    # First move dry air adiabiatically
    ta_bh = ta - DRY_LAPSE_RATE * (z_1 - z_0)

    # Get cases when condensation occurs
    sat = np.logical_and.reduce((np.isfinite(ta_bh),
                                 np.isfinite(td),
                                 ta_bh < td))
    # Get condensation elevation of those cases
    z_sat = z_0[sat] - (td[sat] - ta[sat]) / DRY_LAPSE_RATE
    # Compute adiabatic lapse rate of moist air
    ea = calc_vapour_pressure(td[sat])
    p = calc_pressure_height(p[sat], ta[sat], ea, z_0[sat], z_sat)
    lapse_rate_moist = met.calc_lapse_rate_moist(td[sat].astype(np.float32),
                                                 ea.astype(np.float32),
                                                 p.astype(np.float32))
    # Move moist air adiabatically from elevation when condensation occurs
    # to destination height
    ta_bh[sat] = td[sat] - lapse_rate_moist * (z_1[sat] - z_sat)"""
    """
    ea = calc_vapour_pressure(td)
    lapse_rate_moist = met.calc_lapse_rate_moist(ta.astype(np.float32),
                                                 ea.astype(np.float32),
                                                 p.astype(np.float32))
    ta_bh = ta - lapse_rate_moist * (z_1 - z_0)
    """
    ta_bh = ta - ENVIRONMENTAL_LAPSE_RATE * (z_1 - z_0)
    return ta_bh


def calc_dew_temperature_blending_height(td, z_1, z_0=2.0):
    lapse_rate = TDEW_LAPSE_RATE
    td_bh = td - lapse_rate * (z_1 - z_0)
    return td_bh


def calc_pressure_height(sp_0, t_0, z_1, z_0=0.0):
    sp = sp_0 * calc_sp1_sp0(t_0, z_1, z_0=z_0)
    return sp


def calc_sp1_sp0(t_0, z_1, z_0=0):
    sp1_sp0 = np.exp(-GRAVITY * (z_1 - z_0) / (R_D * t_0))
    return sp1_sp0


def calc_vapour_pressure(td):
    # output in mb
    td = td - 273.15
    e = 6.11 * np.power(10, (7.5 * td) / (237.3 + td))
    return e


def calc_wind_speed(u, v):
    ws = (u ** 2 + v ** 2) ** 0.5
    ws = np.maximum(ws, 1.0)
    return ws


def calc_windspeed_blending_height(ws_z, z_0M=0.25, z=10, z_bh=100.0):
    '''Calculates the windspeed at blending height

    Parameters
    ----------
    ws_z : windspeed measured at zm (m s-1)
    z_0M : aerodynamic roughness lenght for momentum (m)
    z : Measrurement height of wind speed
    z_bh : blending height (m) default=100

    Returns
    -------
    u_blend : wind speed (m s-1)

    References
    ----------
    based on Allen 2007 eq 31'''

    u_blend = ws_z * np.log(z_bh / z_0M) / np.log(z / z_0M)
    return u_blend


def calc_pressure_mb(sp):
    # Convert from pascals to mb
    sp_mb = sp / 100.0
    return sp_mb


def calc_tcwv_cm(tcwv):
    # Conert from from kg/m**2 to g/cm**2
    return tcwv / 10.0


def calc_solar_irradiance(aot_filename,
                          tcwv_filename,
                          p_filename,
                          ta_filename,
                          ea_filename,
                          sza_filename,
                          elev_filename):
    # Get DOY from SZA filename
    match = re.search("S3_(\d{4})(\d{2})(\d{2})T\d{6}_.*\.tif", sza_filename)
    year = match.group(1)
    month = match.group(2)
    day = match.group(3)
    doy = dt.date(int(year), int(month), int(day)).timetuple().tm_yday

    # Read input data
    sza = gu.raster_data(sza_filename)
    elev = gu.raster_data(elev_filename)
    aot = gu.raster_data(aot_filename)
    tcwv = gu.raster_data(tcwv_filename)  # tcwv - in g/cm**2
    p = gu.raster_data(p_filename)  # p - in mb
    ta = gu.raster_data(ta_filename)
    ea = gu.raster_data(ea_filename)

    # Calculate S_dn using Ineichen equaiton
    sdn = solar.calc_global_horizontal_radiance_clear_sky(doy, sza, aot, tcwv, p,
                                                          ta, altitude=elev)
    return sdn


def _getECMWFTempInterpData(ds, var_name, before, after, f_time):
    # Get some metadata
    gt, proj = _get_gdal_params(ds, var_name)
    data_before = ds[var_name][before]
    data_after = ds[var_name][after]
    # Perform temporal interpolation
    data = data_before * f_time + data_after * (1.0 - f_time)

    return data.values, gt, proj

def _get_ECMWF_cummulative_data(ds,var_name, before, after):
    # Get some metadata
    gt, proj = _get_gdal_params(ds, var_name)
    if before == -1:
        data_before = 0
    else:
        data_before = ds[var_name][before]
    data_after = ds[var_name][after]
    # Perform temporal interpolation
    data = data_after - data_before

    return data.values, gt, proj


def _ECMWFRespampleData(data, gt, proj, template_file,
                        resample_alg="cubicspline"):
    # Subset and reproject to the template file extent and projection
    ds_out = gu.save_image(data, gt, proj, "MEM")
    ds_out_proj = gu.resample_with_gdalwarp(ds_out, template_file,
                                            resample_alg=resample_alg)
    data = ds_out_proj.GetRasterBand(1).ReadAsArray()
    del ds_out_proj, ds_out

    return data


def _get_cummulative_data(xds,
                          var_name,
                          date_time,
                          elev_file,
                          time_window=24,
                          is_forecast=True):

    date_time = np.datetime64(date_time)
    if "valid_time" in xds.variables:
        dates = xds["valid_time"].values
    else:
        dates = xds["time"].values
    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, date_time)
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates,
            date_time + dt.timedelta(hours=time_window))

    gt, proj = _get_gdal_params(xds, var_name)

    if is_forecast:
        hours_forecast_radiation = HOURS_FORECAST_CAMS
    else:
        hours_forecast_radiation = HOURS_FORECAST_ERA5

    if pd.to_datetime(dates[date_0]).hour in hours_forecast_radiation :
        # The reference will be zero for the next forecast or always zero for ERA5
        data_ref = 0
    else:
        data_ref = xds[var_name][date_0].values - xds[var_name][date_0 - 1].values
        data_ref[np.isnan(data_ref)] = 0

    # Initialize output variable
    cummulated_value = 0.
    for date_i in range(date_0 + 1, date_1 + 1):
        # Read the right time layers
        data = xds[var_name][date_i].values
        data[np.isnan(data)] = 0
        interval = (data - data_ref)
        cummulated_value = cummulated_value + interval
        if pd.to_datetime(dates[date_i]).hour in hours_forecast_radiation:
            # The reference will be zero for the next forecast or always zero for ERA5
            data_ref = 0
        else:
            # the reference is the cummulated value since the begining of the forecast in CAMS
            data_ref = data.copy()

    cummulated_value = _ECMWFRespampleData(cummulated_value, gt, proj, elev_file)
    return cummulated_value, gt, proj


def _getECMWFSolarData(xds,
                       date_time,
                       elev_file,
                       slope_file,
                       aspect_file,
                       svf_file=None,
                       time_window=24,
                       is_forecast=False,
                       aot_ds=None):

    # Open the netcdf time dataset
    if "valid_time" in xds.variables:
        dates = xds["valid_time"].values
    else:
        dates = xds["time"].values
    if not isinstance(aot_ds, type(None)):
        datesaot = aot_ds["valid_time"].values

    lat = xds["latitude"].values
    lon = xds["longitude"].values
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.astype(float), lats.astype(float)

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, date_time)
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates,
            date_time + dt.timedelta(hours=time_window))

    gt, proj = _get_gdal_params(xds, "ssrd")

    # Read slope and aspect for optional correcion for illumination over tilted surfaces
    slope = gu.raster_data(slope_file)
    slope[slope < 0] = np.nan
    aspect = gu.raster_data(aspect_file)
    aspect[aspect < 0] = np.nan
    lat_hr = _ECMWFRespampleData(lats,
                                 gt, proj, elev_file,
                                 resample_alg="bilinear")
    illum_f = du.inclination_factors(lat_hr, slope, aspect)
    if svf_file is not None:
        svf = gu.raster_data(svf_file)
        svf[svf < 0] = np.nan
    else:
        svf = np.ones(slope.shape)

    if is_forecast:
        hours_forecast_radiation = HOURS_FORECAST_CAMS
    else:
        hours_forecast_radiation = HOURS_FORECAST_ERA5

    if pd.to_datetime(dates[date_0]).hour in hours_forecast_radiation :
        # The reference will be zero for the next forecast or always zero for ERA5
        data_ref = 0
    else:
        data_ref = xds["ssrd"][date_0].values - xds["ssrd"][date_0 - 1].values
        data_ref[np.isnan(data_ref)] = 0

    # Initialize output variable
    cummulated_value = 0.
    for date_i in range(date_0 + 1, date_1 + 1):
        date = pd.to_datetime(dates[date_i])
        # Read the right time layers
        data = xds["ssrd"][date_i].values
        data[np.isnan(data)] = 0
        sdn = np.absolute(data - data_ref)
        ftime = date.hour + date.minute / 60
        doy = date.day_of_year
        sza, saa = met.calc_sun_angles(lats, lons, 0, doy, ftime)
        sza = np.clip(sza, 0., 90.)
        if np.any(sza <= SZA_THRESHOLD_CORRECTION):
            sp = xds["sp"][date_i].values
            t = xds["t2m"][date_i].values
            ea = xds["d2m"][date_i].values
            if "aod550" in xds.variables:
                aot550 = xds["aod550"][date_i].values
            elif aot_ds is not None:
                b, a, f = _bracketing_dates(datesaot, dates[date_i])
                aot550, gtaot, projaot = _getECMWFTempInterpData(
                    aot_ds,"aod550", b, a, f)
                tempds = gu.save_image(aot550, gtaot, projaot, "MEM")
                y_size, x_size = data.shape
                extent = [gt[0], gt[3] + gt[5] * y_size,
                          gt[0] + gt[1] * x_size, gt[3]]
                aot550 = gdal.Warp("", tempds,
                                   format="MEM", xRes=gt[1], yRes=gt[5],
                                   outputBounds=extent, resampleAlg="bilinear")
                del tempds
                aot550 = aot550.GetRasterBand(1).ReadAsArray()
            else:
                aot550 = np.full_like(sza, RURAL_AOT_25KM)
            tcwv = xds["tcwv"][date_i].values
            z_0 = xds["z"][date_i].values
            sp = calc_pressure_mb(sp)
            ea = calc_vapour_pressure(ea)
            z_0 /= GRAVITY
            tcwv = calc_tcwv_cm(tcwv)
            # Convert to W m-2
            sdn = sdn / 3600.
            sdn = correct_solar_irradiance(doy,
                                           ftime,
                                           sza,
                                           sdn,
                                           z_0,
                                           sp,
                                           t,
                                           ea,
                                           proj,
                                           gt,
                                           elev_file,
                                           illum_f=illum_f,
                                           saa=saa,
                                           lon=lons,
                                           tcwv=tcwv,
                                           aot550=aot550,
                                           svf=svf)
            # Convert to J m-2
            sdn = sdn * 3600.
        else:
            sdn = _ECMWFRespampleData(sdn, gt, proj, elev_file)

        cummulated_value = cummulated_value + sdn
        if pd.to_datetime(dates[date_i]).hour in hours_forecast_radiation :
            # The reference will be zero for the next forecast or always zero for ERA5
            data_ref = 0
        else:
            # the reference is the cummulated value since the begining of the forecast in CAMS
            data_ref = data.copy()

    # Convert to average W m^-2
    cummulated_value /= (time_window * 3600.)
    return cummulated_value, gt, proj


def _bracketing_dates(date_list, target_date):
    date_list = list(date_list)
    target_date = np.datetime64(target_date)
    try:
        before = [x for x in date_list if (target_date - x) >= 0]
        if len(before) > 0:
            before = max(before)
        else:
            before = date_list[0]

        after = [x for x in date_list if (target_date - x) <= 0]
        if len(after) > 0:
            after = min(after)
        else:
            after = date_list[-1]

    except ValueError as e:
        print(e)
        return None, None, np.nan
    if before == after:
        frac = 1
    else:
        frac = float((after - target_date)) / float((after - before))
    return date_list.index(before), date_list.index(after), frac


def daily_reference_et(xds,
                       timedate_UTC,
                       elev_file,
                       slope_file,
                       aspect_file,
                       svf_file=None,
                       time_zone=0,
                       z_u=10,
                       z_t=2,
                       is_forecast=False,
                       aot_ds=None):

    # Find midnight in local time and convert to UTC time
    date_local = (timedate_UTC + dt.timedelta(hours=time_zone)).date()
    midnight_local = dt.datetime.combine(
        date_local, dt.time()).replace(tzinfo=dt.timezone.utc)
    midnight_UTC = midnight_local - dt.timedelta(hours=time_zone)
    if "valid_time" in xds.variables:
        dates = xds["valid_time"].values
    else:
        dates = xds["time"].values
    lat = xds["latitude"].values
    lon = xds["longitude"].values
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.astype(float), lats.astype(float)

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, midnight_UTC)
    if not date_0:
        date_0 = 0
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates,
            midnight_UTC + dt.timedelta(hours=24))

    elev_data = gu.raster_data(elev_file).astype(float)
    elev_data[np.isin(elev_data, DEM_NANS)] = np.nan
    # Read spatial reference
    gt, proj = _get_gdal_params(xds, "t2m")

    # Initialize stack variables
    t_max = np.full(elev_data.shape, -99999.)
    t_min = np.full(elev_data.shape, 99999.)
    p_array = []
    u_array = []
    ea_array = []
    es_array = []
    for date_i in range(date_0 + 1, date_1 + 1):
        # Read the right time layers
        t_air = xds["t2m"][date_i].values
        td = xds["d2m"][date_i].values
        u = xds["u%i" % z_u][date_i].values
        v = xds["v%i" % z_u][date_i].values
        p = xds["sp"][date_i].values
        t_air, td, u, v, p = map(np.ma.filled,
                                 [t_air, td, u, v, p],
                                 5 * [np.nan])
        ea = td.copy()
        ea[np.isfinite(ea)] = met.calc_vapor_pressure(ea[np.isfinite(ea)])
        valid = np.logical_and(np.isfinite(u), np.isfinite(v))
        u[valid] = np.sqrt(u[valid] ** 2 + v[valid] ** 2)
        del v, valid
        p[np.isfinite(p)] = calc_pressure_mb(p[np.isfinite(p)])

        z = xds["z"][date_i].values / GRAVITY

        # Calculate dew temperature and vapour pressure at elevation height
        z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                resample_alg="bilinear")
        td = _ECMWFRespampleData(td, gt, proj, template_file=elev_file)
        td = calc_dew_temperature_blending_height(td, elev_data, z_0=z)
        p = _ECMWFRespampleData(p, gt, proj, template_file=elev_file)
        u = _ECMWFRespampleData(u, gt, proj, template_file=elev_file)
        # Resample dataset and calculate actual blending height temperature based on input
        # elevation data
        t_air = _ECMWFRespampleData(t_air, gt, proj, template_file=elev_file)

        t_air = calc_air_temperature_blending_height(t_air, elev_data + z_t,
                                                     z_0=z + 2.0)

        # Calcultate pressure at elevation height
        ea = met.calc_vapor_pressure(td)
        p = calc_pressure_height(p, t_air, elev_data, z_0=z)
        # Avoid sprading possible NaNs
        valid = np.isfinite(t_air)
        t_max[valid] = np.maximum(t_max[valid], t_air[valid])
        t_min[valid] = np.minimum(t_min[valid], t_air[valid])
        es_array.append(met.calc_vapor_pressure(t_air))
        ea_array.append(ea)
        p_array.append(p)
        u_array.append(u)

    sdn_mean, _, _ = _getECMWFSolarData(xds,
                                        midnight_UTC,
                                        elev_file,
                                        slope_file,
                                        aspect_file,
                                        svf_file=svf_file,
                                        time_window=24,
                                        is_forecast=is_forecast,
                                        aot_ds=aot_ds)

    t_max[t_max == -99999] = np.nan
    t_min[t_min == 99999] = np.nan

    # Compute daily means
    t_mean = 0.5 * (t_max + t_min)
    ea_mean = np.nanmean(np.array(ea_array), axis=0)
    # es_mean = 0.5 * (met.calc_vapor_pressure(t_max) + met.calc_vapor_pressure(t_min))
    es_mean = np.nanmean(np.array(es_array), axis=0)
    u_mean = np.nanmean(np.array(u_array), axis=0)
    p_mean = np.nanmean(np.array(p_array), axis=0)

    lats = _ECMWFRespampleData(lats, gt, proj, template_file=elev_file,
                              resample_alg="bilinear")
    doy = float(timedate_UTC.strftime("%j"))
    f_cd = pet.calc_cloudiness(sdn_mean, lats, elev_data, doy)
    le = pet.pet_fao56(t_mean, u_mean, ea_mean, es_mean, p_mean, sdn_mean, z_u, z_t,
                       f_cd=f_cd, is_daily=True)

    et_ref = met.flux_2_evaporation(le, t_mean, 24)

    return et_ref, t_min, t_max, sdn_mean, ea_mean, u_mean, p_mean


def mean_temperature(xds,
                     elev_file,
                     timedate_UTC,
                     time_zone=0,
                     z_t=2):
    if "valid_time" in xds.variables:
        dates = xds["valid_time"].values
    else:
        dates = xds["time"].values
    # Find midnight in local time and convert to UTC time
    date_local = (timedate_UTC + dt.timedelta(hours=time_zone)).date()
    midnight_local = dt.datetime.combine(
        date_local, dt.time()).replace(tzinfo=dt.timezone.utc)
    midnight_UTC = midnight_local - dt.timedelta(hours=time_zone)

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, midnight_UTC)
    if not date_0:
        date_0 = 0
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates,
            midnight_UTC + dt.timedelta(hours=24))

    elev_data = gu.raster_data(elev_file).astype(float)
    elev_data[np.isin(elev_data, DEM_NANS)] = np.nan
    # Get spatial reference
    gt, proj = _get_gdal_params(xds, "t2m")

    # Initialize stack variables
    t_max = np.full(elev_data.shape, -99999.)
    t_min = np.full(elev_data.shape, 99999.)
    for date_i in range(date_0 + 1, date_1 + 1):
        # Read the right time layers
        t_air = xds["t2m"][date_i].values
        t_air = np.ma.filled(t_air, np.nan)
        z = xds["z"][date_i].values / GRAVITY
        # Calculate dew temperature and vapour pressure at elevation height
        z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                resample_alg="bilinear")
        t_air = _ECMWFRespampleData(t_air, gt, proj, template_file=elev_file)
        t_air = calc_air_temperature_blending_height(t_air, elev_data + z_t,
                                                     z_0=z + 2.0)

        # Avoid spreading possible NaNs
        valid = np.isfinite(t_air)
        t_max[valid] = np.maximum(t_max[valid], t_air[valid])
        t_min[valid] = np.minimum(t_min[valid], t_air[valid])

    t_max[t_max == -99999] = np.nan
    t_min[t_min == 99999] = np.nan

    # Compute daily mean
    t_mean = 0.5 * (t_max + t_min)
    return t_mean


def correct_solar_irradiance(doy,
                             ftime,
                             sza,
                             sdn,
                             z_0,
                             press,
                             t,
                             ea,
                             proj,
                             gt,
                             elev_file,
                             tcwv=None,
                             aot550=None,
                             illum_f=None,
                             saa=None,
                             lon=None,
                             svf=1):
    out_proj, out_gt = gu.raster_info(elev_file)[:2]
    if aot550 is None:
        aot550 = np.full(sza.shape, 0.2)
    if tcwv is None:
        tcwv = np.full(sza.shape, 1.0)

    sdn_cs = solar.calc_global_horizontal_radiance_clear_sky(doy, sza, aot550,
                                                             tcwv, press, t,
                                                             altitude=z_0,
                                                             calc_diffuse=False)
    sdn_cs = np.maximum(sdn_cs, 0.)
    # Cloudiness factor
    fci_cl = np.clip(sdn / sdn_cs, 0., 1)
    z_1 = gu.raster_data(elev_file).astype(float)
    z_1[np.isin(z_1, DEM_NANS)] = np.nan
    sdn = _ECMWFRespampleData(sdn, gt, proj, elev_file)
    sdn[~np.isfinite(z_1)] = np.nan
    sza = _ECMWFRespampleData(sza, gt, proj, elev_file, resample_alg="bilinear")
    valid = np.logical_and(sza <= SZA_THRESHOLD_CORRECTION, np.isfinite(z_1))
    if np.any(valid):
        # Ressample dataset
        aot550 = _ECMWFRespampleData(aot550, gt, proj, elev_file)
        tcwv = _ECMWFRespampleData(tcwv, gt, proj, elev_file)
        press = _ECMWFRespampleData(press, gt, proj, elev_file)
        t = _ECMWFRespampleData(t, gt, proj, elev_file)
        ea = _ECMWFRespampleData(ea, gt, proj, elev_file)

        fci_cl = _ECMWFRespampleData(fci_cl, gt, proj, elev_file)
        fci_cl = np.clip(fci_cl, 0., 1)
        if illum_f is not None and saa is not None and lon is not None:
            # Correct for illumination overt tilted surfaces
            saa = _ECMWFRespampleData(saa, gt, proj, elev_file, resample_alg="bilinear")
            lon = _ECMWFRespampleData(lon, gt, proj, elev_file, resample_alg="bilinear")
            calc_diffuse = True
            bdn_cs, ddn_cs = solar.calc_global_horizontal_radiance_clear_sky(
                doy,
                sza[valid],
                aot550[valid],
                tcwv[valid],
                press[valid],
                t[valid],
                altitude=z_1[valid],
                calc_diffuse=calc_diffuse,
                method=solar.REST2)
            bdn_cs = np.maximum(np.sum(bdn_cs, axis=0), 0)
            ddn_cs = np.maximum(np.sum(ddn_cs, axis=0), 0)
            sdn_cs = bdn_cs + ddn_cs
            beam_ratio = bdn_cs / sdn_cs
            # Compute the diffuse fraction considering
            # the cloudiness and clear-sky beam ratio
            skyl = np.clip(1. - fci_cl[valid] * beam_ratio, 0, 1)
            bdn = sdn_cs * fci_cl[valid] * (1 - skyl)
            ddn = sdn_cs * fci_cl[valid] * skyl
            # First get areas with are not occluded by protuding terrain elements
            non_shaded = du.non_occluded_terrain(elev_file, saa, sza,
                                                 smooth_radius=1)
            # non_shaded = np.ones(lon.shape, dtype=bool)
            # Then correct for incidence solar angle on the tilted surface
            bdn, ddn = calc_radiance_tilted_surface_op(bdn,
                                                       ddn,
                                                       doy,
                                                       ftime,
                                                       lon[valid],
                                                       sza[valid],
                                                       illum_f[0][valid],
                                                       illum_f[1][valid],
                                                       illum_f[2][valid],
                                                       svf[valid],
                                                       non_shaded=non_shaded[valid])
            sdn[valid] = bdn + ddn
        else:
            calc_diffuse = False
            sdn_cs = solar.calc_global_horizontal_radiance_clear_sky(doy,
                                                                     sza[valid],
                                                                     aot550[valid],
                                                                     tcwv[valid],
                                                                     press[valid],
                                                                     t[valid],
                                                                     altitude=z_1[valid],
                                                                     calc_diffuse=calc_diffuse)
            sdn_cs = np.maximum(sdn_cs, 0)
            sdn[valid] = sdn_cs * fci_cl[valid]
    return sdn


def calc_radiance_tilted_surface_op(bdn,
                                    ddn,
                                    doy,
                                    ftime,
                                    lon,
                                    sza,
                                    f1,
                                    f2,
                                    f3,
                                    svf=1,
                                    non_shaded=None):
    """ Corrects both beam and diffuse radiation on illumination angle"""

    # Compute the incidence angle and incidence ratio
    cos_theta_i = incidence_angle_tilted_optimized(doy, ftime, lon, f1, f2, f3)
    inc_ratio = cos_theta_i / np.cos(np.radians(sza))

    # The global solar radiation contains at least the diffuse part
    # We first consider occlusion effects in diffuse radiation
    # by using the sky-view factor variable
    ddn = ddn * svf
    if non_shaded is None:
        non_shaded = np.ones(bdn.shape, dtype=bool)
    non_shaded = np.logical_and(non_shaded, inc_ratio > 0)
    if bdn.ndim != inc_ratio.ndim:  # Case when we use PAR and NIR arrays
        bdn[:, non_shaded] = bdn[:, non_shaded] * inc_ratio[non_shaded]
        bdn[:, ~non_shaded] = 0
    else:
        bdn[non_shaded] = bdn[non_shaded] * inc_ratio[non_shaded]
        bdn[~non_shaded] = 0
    return bdn, ddn


def incidence_angle_tilted_optimized(doy, ftime, lon, f1, f2, f3, stdlon=0):
    # Get the declination and hour angle
    delta = du.declination_angle(doy)
    # Hour angle is considered negative before noon, positive after noon
    omega = du.hour_angle(ftime, delta, lon, stdlon=stdlon)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_theta_i = sin_delta * f1 \
                  + cos_delta * cos_omega * f2 \
                  + cos_delta * sin_omega * f3

    return cos_theta_i


def iterception_bastiaansen(lai, rain, f_c=1, specific_storage=0.2):
    """

    Parameters
    ----------
    lai : float or ndarray
        Local Leaf Area Index
    rain : float or ndarray
        Total rainfall (mm)
    f_c : float or ndarray
        Nadir-viewing fractional cover. Default=1 indicates homogeneous canopy
    specific_storage : float or ndarray
        Specific canopy storage per unit leaf area (mm). Default=0.2

    Returns
    -------
    i_mm : float or ndarray
        Intercepted rainfall (mm)
    """
    # Vegetation storage capacity
    max_storage = specific_storage * lai
    i_mm = max_storage * (1. - 1. / (1. + f_c * rain / max_storage))

    return i_mm


def iterception_calder(lai, rain, f_c=1, specific_storage=0.2, q=0.5):
    """

    Parameters
    ----------
    lai : float or ndarray
        Local Leaf Area Index
    rain : float or ndarray
        Total rainfall (mm)
    f_c : float or ndarray
        Nadir-viewing fractional cover. Default=1 indicates homogeneous canopy
    specific_storage : float or ndarray
        Specific canopy storage per unit leaf area (mm). Default=0.2
    q : float or ndarray:
        Throughfall coefficient. Indicates the mean number of rainfall drops
        that are effectively retained per canopy element. Default=0.5

    Returns
    -------
    i_mm : float or ndarray
        Intercepted rainfall (mm)
    """
    # Vegetation storage capacity
    max_storage = specific_storage * lai

    i_mm = max_storage * (1. - np.exp(-q * f_c * rain / max_storage))

    return i_mm

def _get_gdal_params(ds, var_name):
    proj = ds[var_name].spatial_ref.crs_wkt
    ds[var_name].rio.transform()
    gt = ds.rio.transform()
    gt = [gt[2], gt[0], gt[1], gt[5], gt[3], gt[4]]
    return gt, proj

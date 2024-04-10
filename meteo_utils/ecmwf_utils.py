# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 14:47:50 2018

@author: rmgu
"""

import datetime
import os
import re

import cdsapi
import netCDF4
import numpy as np
from osgeo import osr, gdal
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
HOURS_FORECAST_CAMS = (0, 12, 24)
HOURS_FORECAST_INTERIM = (0, 26)

ADS_CREDENTIALS_FILE = os.path.join(os.path.expanduser("~"),
                                    '.adsapirc')


def download_CDS_data(dataset,
                      product_type,
                      date_start,
                      date_end,
                      variables,
                      target,
                      overwrite=False, area=None):
    s = {"variable": variables, "format": "netcdf"}
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

    s = {"variable": variables, "format": "netcdf"}
    s["date"] = date_start.strftime("%Y-%m-%d") + "/" + date_end.strftime(
        "%Y-%m-%d")

    if dataset == "cams-global-atmospheric-composition-forecasts":
        s["type"] = "forecast"
        s['time'] = ['00:00', '12:00']
        s['leadtime_hour'] = [str(i) for i in range(12)]
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
                   slope_file=None,
                   aspect_file=None,
                   svf_file=None,
                   aod550_data_file=None,
                   time_zone=0,
                   ecmwf_dataset='CAMS_FC'):

    ftime = timedate_UTC.hour + timedate_UTC.minute / 60
    ncfile = netCDF4.Dataset(ecmwf_data_file, 'r')
    # Find the location of bracketing dates
    time = ncfile.variables['time']
    dates = netCDF4.num2date(time[:], time.units, time.calendar)
    beforeI, afterI, frac = _bracketing_dates(dates, timedate_UTC)
    if beforeI is None:
        return False
    del time
    lat = ncfile.variables["latitude"]
    lon = ncfile.variables["longitude"]
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.data.astype(float), lats.data.astype(float)
    ncfile.close()

    if aod550_data_file is not None:
        ncfile = netCDF4.Dataset(aod550_data_file, 'r')
        time = ncfile.variables['time']
        datesaot = netCDF4.num2date(time[:], time.units, time.calendar)
        del time
        ncfile.close()

    output = {}
    for field in meteo_data_fields:
        if field == "TA":
            t2m, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "t2m",
                                                    beforeI, afterI, frac)
            elev_data = gu.raster_data(elev_file)
            if "land" not in ecmwf_dataset:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "z", beforeI,
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

            sp, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "sp",
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
            d2m, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "d2m",
                                                    beforeI, afterI, frac)
            elev_data = gu.raster_data(elev_file)
            if "land" not in ecmwf_dataset:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "z", beforeI,
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

        elif field == "U":
            u100, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "u100",
                                                     beforeI, afterI, frac)
            v100, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "v100",
                                                     beforeI, afterI, frac)
            # Combine the two components of wind speed and calculate speed at blending height
            ws100 = calc_wind_speed(u100, v100)
            if z_bh == 100:
                data = _ECMWFRespampleData(ws100, gt, proj, template_file=elev_file)
            else:
                ws100 = _ECMWFRespampleData(ws100, gt, proj, template_file=elev_file)
                z_0M, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "fsr",
                                                         beforeI, afterI,
                                                         frac)
                z_0M = _ECMWFRespampleData(z_0M, gt, proj, template_file=elev_file)
                # Calculate wind speed at blending height.
                data = calc_windspeed_blending_height(ws100, z_0M, z_bh)

        elif field == "P":
            sp, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "sp",
                                                   beforeI, afterI, frac)
            if "land" not in ecmwf_dataset:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "z",
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
                                        resample_alg="bilinear")
                del outDs

            t0, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "t2m",
                                                   beforeI, afterI, frac)

            t0 = _ECMWFRespampleData(t0, gt, proj, template_file=elev_file)
            # Convert pressure from pascals to mb
            elev_data = gu.raster_data(elev_file)
            sp = _ECMWFRespampleData(sp, gt, proj, template_file=elev_file)
            # Calcultate pressure at 0m datum height
            sp = calc_pressure_height(sp, t0, elev_data, z_0=z)
            data = calc_pressure_mb(sp)

        elif field == "TCWV":
            tcwv, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "tcwv",
                                                     beforeI, afterI, frac)
            data = calc_tcwv_cm(tcwv)
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "TP":
            data, gt, proj = _getECMWFTempInterpData(ecmwf_data_file,
                                                     "tp",
                                                     afterI, afterI, 1)
            # Convert to mm
            data = data * 1000
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "LDN":
            data, gt, proj = _getECMWFTempInterpData(ecmwf_data_file,
                                                     "strd",
                                                     afterI, afterI, 1)

            # Convert cummulated value to instantaneous Wm-2
            if "land" in ecmwf_dataset or "reanalysis" in ecmwf_dataset:
                time_step = 1
            else:
                time_step = 3

            data = data / (time_step * 3600.)
            data = _ECMWFRespampleData(data, gt, proj, template_file=elev_file)

        elif field == "SDN":
            t, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "t2m",
                                                  beforeI, afterI, frac)
            t = _ECMWFRespampleData(t, gt, proj, elev_file)
            if "land" not in ecmwf_dataset:
                # Get geopotential height at which Ta is calculated
                z, gt, proj = _getECMWFTempInterpData(ecmwf_data_file, "z",
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

            elev_data = gu.raster_data(elev_file)
            sp, _, _ = _getECMWFTempInterpData(ecmwf_data_file, "sp",
                                               beforeI, afterI, frac)
            sp = _ECMWFRespampleData(sp, gt, proj, template_file=elev_file)
            # Calcultate pressure at 0m datum height

            sp = calc_pressure_height(sp, t, elev_data, z_0=z)
            sp = calc_pressure_mb(sp)
            if "land" not in ecmwf_dataset:
                tcwv, _, _ = _getECMWFTempInterpData(ecmwf_data_file, "tcwv",
                                                     beforeI, afterI, frac)
                tcwv = calc_tcwv_cm(tcwv)
                tcwv = _ECMWFRespampleData(tcwv, gt, proj, elev_file)

            elif aod550_data_file is not None:
                b, a, f = _bracketing_dates(datesaot, timedate_UTC)
                tcwv, gtaot, projaot = _getECMWFTempInterpData(
                    aod550_data_file,"aod550", b, a, f)
                tcwv = calc_tcwv_cm(tcwv)
                tcwv = _ECMWFRespampleData(tcwv, gtaot, projaot, elev_file)
            else:
                tcwv = TCWV_MIDLATITUDE_SUMMER

            if aod550_data_file is not None:
                b, a, f = _bracketing_dates(datesaot, timedate_UTC)
                aot550, gtaot, projaot = _getECMWFTempInterpData(
                    aod550_data_file,"aod550", b, a, f)
                aot550 = _ECMWFRespampleData(aot550, gtaot, projaot, elev_file)
            else:
                aot550 = RURAL_AOT_25KM

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

            # Then correct for incidence solar angle on the tilted surface
            if slope_file is not None and aspect_file is not None:
                ftime = timedate_UTC.hour + timedate_UTC.minute / 60

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
                aod550File = netCDF4.Dataset(aod550_data_file)
                time = aod550File.variables['time']
                dates = netCDF4.num2date(time[:], time.units, time.calendar)
                b, a, f = _bracketing_dates(dates, timedate_UTC)
                del time
                aod550File.close()
                aod550, gt, proj = _getECMWFTempInterpData(aod550_data_file,
                                                           "aod550", b, a, f)
            else:
                aod550, gt, proj = _getECMWFTempInterpData(ecmwf_data_file,
                                                           "aod550", beforeI,
                                                           afterI, frac)

            data = _ECMWFRespampleData(aod550, gt, proj, template_file=elev_file)

        elif field == "SDNday":
            # Find midnight in local time and convert to UTC time
            date_local = (timedate_UTC + datetime.timedelta(
                hours=time_zone)).date()
            midnight_local = datetime.datetime.combine(date_local,
                                                       datetime.time())
            midnight_UTC = midnight_local - datetime.timedelta(hours=time_zone)
            # Interpolate solar irradiance over 24 hour period starting at midnight local time
            data, gt, proj = _getECMWFSolarData(ecmwf_data_file,
                                                "ssrd",
                                                midnight_UTC,
                                                elev_file,
                                                slope_file=slope_file,
                                                aspect_file=aspect_file,
                                                svf_file=svf_file,
                                                time_window=24,
                                                dataset=ecmwf_dataset,
                                                aotfile=aod550_data_file)


        elif field == "ETref":
            data = daily_reference_et(ecmwf_data_file,
                                      elev_file,
                                      timedate_UTC,
                                      slope_file=slope_file,
                                      aspect_file=aspect_file,
                                      svf_file=svf_file,
                                      time_zone=time_zone,
                                      z_u=10,
                                      z_t=2,
                                      ecmwf_dataset=ecmwf_dataset,
                                      aot_data_file=aod550_data_file)[0]

        elif field == "TPday":
            data, gt, proj = _get_cummulative_data(ecmwf_data_file,
                                                   "tp",
                                                   timedate_UTC,
                                                   elev_file,
                                                   time_window=24,
                                                   dataset=ecmwf_dataset)
            # Convert to mm
            data = data * 1000

        elif field == "TAday":
            data = mean_temperature(ecmwf_data_file,
                                    elev_file,
                                    timedate_UTC,
                                    time_zone=time_zone,
                                    z_t=2,
                                    ecmwf_dataset=ecmwf_dataset)

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


def calc_windspeed_blending_height(u10, z_0M, z_bh=100.0):
    '''Calculates the windspeed at blending height

    Parameters
    ----------
    u10 : 10m windspeed (m s-1)
    z_0M : aerodynamic roughness lenght for momentum (m)
    z_bh : blending height (m) default=100

    Returns
    -------
    u_blend : wind speed (m s-1)

    References
    ----------
    based on Allen 2007 eq 31'''

    u_blend = u10 * np.log(z_bh / z_0M) / np.log(10.0 / z_0M)
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
    doy = datetime.date(int(year), int(month), int(day)).timetuple().tm_yday

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


def _getECMWFTempInterpData(ncfile, var_name, before, after, f_time):
    # Get some metadata
    ds = gdal.Open('NETCDF:"' + ncfile + '":' + var_name)
    gt = ds.GetGeoTransform()
    del ds
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()
    ds = netCDF4.Dataset(ncfile)

    data_before_test = ds.variables[var_name][:]
    data_before_test = data_before_test[before, :]
    # Read the right time layers
    after_pos = [after]
    if "expver" in ds.variables.keys():
        for expver_pos, data_before in enumerate(data_before_test):
            # Check and expver array is not empty
            if np.any(np.isfinite(data_before)):
                after_pos.append(expver_pos)
                break
    else:
        data_before = data_before_test.copy()

    data_after = ds.variables[var_name][:]
    for i in after_pos:
        data_after = data_after[i, :]
    ds.close()
    # Perform temporal interpolation
    data = data_before * f_time + data_after * (1.0 - f_time)

    return data, gt, proj


def _ECMWFRespampleData(data, gt, proj, template_file,
                        resample_alg="cubicspline"):
    # Subset and reproject to the template file extent and projection
    ds_out = gu.save_image(data, gt, proj, "MEM")
    ds_out_proj = gu.resample_with_gdalwarp(ds_out, template_file,
                                            resample_alg=resample_alg)
    data = ds_out_proj.GetRasterBand(1).ReadAsArray()
    del ds_out_proj, ds_out

    return data


def _getECMWFIntegratedData(ncfile,
                            var_name,
                            date_time,
                            elev_file,
                            time_window=24,
                            dataset='ERA5_reanalysis'):
    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ncfile, 'r')
    time = fid.variables['time']
    dates = netCDF4.num2date(time[:], time.units, time.calendar)

    if "ERA" in dataset:
        hours_forecast_radiation = HOURS_FORECAST_INTERIM
    else:
        hours_forecast_radiation = HOURS_FORECAST_CAMS

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, date_time)
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates, date_time + datetime.timedelta(
        hours=time_window))

    proj, gt = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))[:2]
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()

    # Get the value of the begnning of the forecast as baseline reference for CAMS
    # The accumulations in the short forecasts (from 06 and 18 UTC) of ERA5 are
    # treated differently compared with those in ERA-Interim
    # (where they are from the beginning of the forecast to the forecast step).
    # In the short forecasts of ERA5 the accumulations are since the previous post processing
    # (archiving)
    # Read the right time layers
    data_ref = fid.variables[var_name][:]
    pos = [date_0]
    if "expver" in fid.variables.keys():
        for expver_pos, data_before in enumerate(data_ref[date_0]):
            # Check and expver array is not empty
            if np.any(np.isfinite(data_before)):
                pos.append(expver_pos)
                break

    if dates[date_0].hour in hours_forecast_radiation \
            or "reanalysis" in dataset \
            or "ensemble" in dataset:
        # The reference will be zero for the next forecast or always zero for ERA5
        data_ref = 0
    else:
        # the reference is the cummulated value since the begining of the forecast in CAMS or ERA5-Land
        for i in pos:
            data_ref = data_ref[i, :]
        data_ref[np.isnan(data_ref)] = 0
        data_ref[data_ref < 0] = 0

    # Initialize output variable
    cummulated_value = 0.
    for date_i in range(date_0 + 1, date_1 + 1):
        if "expver" in fid.variables.keys():
            pos = [date_i, expver_pos]
        else:
            pos = [date_i]
        # Read the right time layers
        data = fid.variables[var_name][:]
        for i in pos:
            data = data[i, :]
        data[np.isnan(data)] = 0
        data[data < 0] = 0
        hourly = (data - data_ref)

        hourly = _ECMWFRespampleData(hourly, gt, proj, elev_file, resample_alg="bilinear")

        cummulated_value += hourly

        if dates[date_i].hour in hours_forecast_radiation \
                or "reanalysis" in dataset \
                or "ensemble" in dataset:
            # The reference will be zero for the next forecast or always zero for ERA5
            data_ref = 0
        else:
            # the reference is the cummulated value since the begining of the forecast in CAMS
            data_ref = data.copy()

    fid.close()

    return cummulated_value, gt, proj


def _get_cummulative_data(ncfile,
                          var_name,
                          date_time,
                          elev_file,
                          time_window=24,
                          dataset='ERA5_reanalysis'):
    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ncfile, 'r')
    time = fid.variables['time']
    dates = netCDF4.num2date(time[:], time.units, time.calendar)

    if "ERA" in dataset:
        hours_forecast_radiation = HOURS_FORECAST_INTERIM
    else:
        hours_forecast_radiation = HOURS_FORECAST_CAMS

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, date_time)
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates, date_time + datetime.timedelta(
        hours=time_window))

    proj, gt = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))[:2]
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()

    # Get the value of the begnning of the forecast as baseline reference for CAMS
    # The accumulations in the short forecasts (from 06 and 18 UTC) of ERA5 are
    # treated differently compared with those in ERA-Interim
    # (where they are from the beginning of the forecast to the forecast step).
    # In the short forecasts of ERA5 the accumulations are since the previous post processing
    # (archiving)
    # Read the right time layers
    data_ref = fid.variables[var_name][:]
    pos = [date_0]
    if "expver" in fid.variables.keys():
        for expver_pos, data_before in enumerate(data_ref[date_0]):
            # Check and expver array is not empty
            if np.any(np.isfinite(data_before)):
                pos.append(expver_pos)
                break

    if dates[date_0].hour in hours_forecast_radiation \
            or "reanalysis" in dataset \
            or "ensemble" in dataset:
        # The reference will be zero for the next forecast or always zero for ERA5
        data_ref = 0
    else:
        # the reference is the cummulated value since the begining of the forecast in CAMS or ERA5-Land
        for i in pos:
            data_ref = data_ref[i, :]
        data_ref[np.isnan(data_ref)] = 0

    # Initialize output variable
    cummulated_value = 0.
    for date_i in range(date_0 + 1, date_1 + 1):
        if "expver" in fid.variables.keys():
            pos = [date_i, expver_pos]
        else:
            pos = [date_i]
        # Read the right time layers
        data = fid.variables[var_name][:]
        for i in pos:
            data = data[i, :]
        data[np.isnan(data)] = 0
        interval = (data - data_ref)
        cummulated_value = cummulated_value + interval

        if dates[date_i].hour in hours_forecast_radiation \
                or "reanalysis" in dataset \
                or "ensemble" in dataset:
            # The reference will be zero for the next forecast or always zero for ERA5
            data_ref = 0
        else:
            # the reference is the cummulated value since the begining of the forecast in CAMS
            data_ref = data.copy()

    cummulated_value = _ECMWFRespampleData(cummulated_value, gt, proj, elev_file)
    fid.close()
    return cummulated_value, gt, proj


def _getECMWFSolarData(ncfile,
                       var_name,
                       date_time,
                       elev_file,
                       slope_file=None,
                       aspect_file=None,
                       svf_file=None,
                       time_window=24,
                       dataset='ERA5_reanalysis',
                       aotfile=None):
    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ncfile, 'r')
    time = fid.variables['time']
    lat = fid.variables["latitude"]
    lon = fid.variables["longitude"]
    lons, lats = np.meshgrid(lon, lat)
    lons, lats = lons.data.astype(float), lats.data.astype(float)
    dates = netCDF4.num2date(time[:], time.units, time.calendar)

    if "land" in dataset or "reanalysis" in dataset:
        time_step = 1
    else:
        time_step = 3

    if "ERA" in dataset:
        hours_forecast_radiation = HOURS_FORECAST_INTERIM
    else:
        hours_forecast_radiation = HOURS_FORECAST_CAMS

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, date_time)
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates, date_time + datetime.timedelta(
        hours=time_window))

    proj, gt = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))[:2]
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()

    # Read slope and aspect for optional correcion for illumination over tilted surfaces
    if slope_file is not None and aspect_file is not None:
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
    else:
        illum_f = None
        svf = None

    # Get the value of the begnning of the forecast as baseline reference for CAMS
    # The accumulations in the short forecasts (from 06 and 18 UTC) of ERA5 are
    # treated differently compared with those in ERA-Interim
    # (where they are from the beginning of the forecast to the forecast step).
    # In the short forecasts of ERA5 the accumulations are since the previous post processing
    # (archiving)
    # Read the right time layers
    data_ref = fid.variables[var_name][:]
    pos = [date_0]
    if "expver" in fid.variables.keys():
        for expver_pos, data_before in enumerate(data_ref[date_0]):
            # Check and expver array is not empty
            if np.any(np.isfinite(data_before)):
                pos.append(expver_pos)
                break

    if dates[date_0].hour in hours_forecast_radiation \
            or "reanalysis" in dataset \
            or "ensemble" in dataset:
        # The reference will be zero for the next forecast or always zero for ERA5
        data_ref = 0
    else:
        # the reference is the cummulated value since the begining of the forecast in CAMS or ERA5-Land
        for i in pos:
            data_ref = data_ref[i, :]
        data_ref[np.isnan(data_ref)] = 0

    if aotfile is not None:
        fidaot = netCDF4.Dataset(aotfile, 'r')
        timeaot = fidaot.variables['time']
        datesaot = netCDF4.num2date(timeaot[:], timeaot.units, timeaot.calendar)
        fidaot.close()

    # Initialize output variable
    cummulated_value = 0.
    for date_i in range(date_0 + 1, date_1 + 1):
        if "expver" in fid.variables.keys():
            pos = [date_i, expver_pos]
        else:
            pos = [date_i]
        # Read the right time layers
        data = fid.variables[var_name][:]
        for i in pos:
            data = data[i, :]
        data[np.isnan(data)] = 0
        ftime = dates[date_i].hour + dates[date_i].minute / 60
        doy = float(dates[date_i].strftime("%j"))
        sza, saa = met.calc_sun_angles(lats, lons, 0, doy, ftime)
        sza = np.clip(sza, 0., 90.)
        sdn = (data - data_ref)
        if np.any(sza <= SZA_THRESHOLD_CORRECTION):
            sp = fid.variables["sp"][:]
            t = fid.variables["t2m"][:]
            ea = fid.variables["d2m"][:]
            for i in pos:
                sp = sp[i, :]
                t = t[i, :]
                ea = ea[i, :]
            sp = calc_pressure_mb(sp)
            ea = calc_vapour_pressure(ea)

            # The time step value is the difference between  the actual timestep value and the previous
            # value
            if "land" not in dataset:
                tcwv = fid.variables["tcwv"][:]
                z_0 = fid.variables["z"][:]
                for i in pos:
                    tcwv = tcwv[i, :]
                    z_0 = z_0[i, :]
                z_0 /= GRAVITY
                tcwv = calc_tcwv_cm(tcwv)
            elif aotfile is not None:
                b, a, f = _bracketing_dates(datesaot, dates[date_i])
                tcwv, gtaot, projaot = _getECMWFTempInterpData(aotfile, "tcwv", b, a, f)
                _, _, sizeX, sizeY, extent, _ = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))
                tempds = gu.save_image(tcwv, gtaot, projaot, "MEM")
                tcwv = gdal.Warp("", tempds, format="MEM", xRes=gt[1], yRes=gt[5],
                                 outputBounds=extent, resampleAlg="bilinear")
                del tempds
                tcwv = tcwv.GetRasterBand(1).ReadAsArray()
                z_0, gtaot, projaot = _getECMWFTempInterpData(aotfile, "z", b, a, f)
                z_0 /= GRAVITY
                _, _, sizeX, sizeY, extent, _ = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))
                tempds = gu.save_image(z_0, gtaot, projaot, "MEM")
                z_0 = gdal.Warp("", tempds, format="MEM", xRes=gt[1], yRes=gt[5],
                                outputBounds=extent, resampleAlg="bilinear")
                del tempds
                z_0 = z_0.GetRasterBand(1).ReadAsArray()
            else:
                tcwv = TCWV_MIDLATITUDE_SUMMER
                z_0 = gu.raster_data(elev_file)
            if aotfile is not None:
                b, a, f = _bracketing_dates(datesaot, dates[date_i])
                aot550, gtaot, projaot = _getECMWFTempInterpData(aotfile, "aod550", b, a, f)
                _, _, sizeX, sizeY, extent, _ = gu.raster_info("NETCDF:%s:%s" % (ncfile, var_name))
                tempds = gu.save_image(aot550, gtaot, projaot, "MEM")
                aot550 = gdal.Warp("", tempds, format="MEM", xRes=gt[1], yRes=gt[5],
                                   outputBounds=extent, resampleAlg="bilinear")
                del tempds
                aot550 = aot550.GetRasterBand(1).ReadAsArray()
            else:
                aot550 = RURAL_AOT_25KM

            # Convert to W m-2
            sdn = sdn / (time_step * 3600.)
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
            sdn = sdn * (time_step * 3600.)
        else:
            sdn = _ECMWFRespampleData(sdn, gt, proj, elev_file)

        cummulated_value = cummulated_value + sdn

        if dates[date_i].hour in hours_forecast_radiation \
                or "reanalysis" in dataset \
                or "ensemble" in dataset:
            # The reference will be zero for the next forecast or always zero for ERA5
            data_ref = 0
        else:
            # the reference is the cummulated value since the begining of the forecast in CAMS
            data_ref = data.copy()

    fid.close()
    # Convert to average W m^-2
    cummulated_value /= (time_window * 3600.)

    return cummulated_value, gt, proj


def _bracketing_dates(date_list, target_date):
    date_list = list(date_list)
    try:
        before = max(
            [x for x in date_list if (target_date - x).total_seconds() >= 0])
        after = min(
            [x for x in date_list if (target_date - x).total_seconds() <= 0])
    except ValueError:
        return None, None, np.nan
    if before == after:
        frac = 1
    else:
        frac = float((after - target_date).total_seconds()) / float(
            (after - before).total_seconds())
    return date_list.index(before), date_list.index(after), frac


def daily_reference_et(ecmwf_data_file,
                       elev_file,
                       timedate_UTC,
                       time_zone=0,
                       z_u=10,
                       z_t=2,
                       slope_file=None,
                       aspect_file=None,
                       svf_file=None,
                       ecmwf_dataset='ERA5_reanalysis',
                       aot_data_file=None):
    # Find midnight in local time and convert to UTC time
    date_local = (timedate_UTC + datetime.timedelta(hours=time_zone)).date()
    midnight_local = datetime.datetime.combine(date_local, datetime.time())
    midnight_UTC = midnight_local - datetime.timedelta(hours=time_zone)

    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ecmwf_data_file, 'r')
    time = fid.variables['time']
    lat = fid.variables["latitude"][...]
    lon = fid.variables["longitude"][...]
    lon, lat = np.meshgrid(lon, lat, indexing='xy')
    lon, lat = lon.data.astype(float), lat.data.astype(float)
    dates = netCDF4.num2date(time[:], time.units, time.calendar)

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, midnight_UTC)
    if not date_0:
        date_0 = 0
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates, midnight_UTC + datetime.timedelta(
        hours=24))

    elev_data = gu.raster_data(elev_file)
    # Open air temperature dataset
    t2m_fid = gdal.Open('NETCDF:"' + ecmwf_data_file + '":t2m')
    gt = t2m_fid.GetGeoTransform()
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()
    del t2m_fid

    # Initialize stack variables
    t_max = np.full(elev_data.shape, -99999.)
    t_min = np.full(elev_data.shape, 99999.)
    p_array = []
    u_array = []
    ea_array = []
    es_array = []
    for date_i in range(date_0 + 1, date_1 + 1):
        # Read the right time layers
        t_air = fid.variables["t2m"][:]
        td = fid.variables["d2m"][:]
        u = fid.variables["u%i" % z_u][:]
        v = fid.variables["v%i" % z_u][:]

        p = fid.variables["sp"][:]

        if "expver" in fid.variables.keys():
            pos = [date_i, 0]
        else:
            pos = [date_i]
        # Calcultate temperature at 0m datum height
        for i in pos:
            t_air = t_air[i]
            td = td[i]
            u = u[i]
            v = v[i]
            p = p[i]
        t_air, td, u, v, p = map(np.ma.filled,
                                 [t_air, td, u, v, p],
                                 5 * [np.nan])
        ea = td.copy()
        ea[np.isfinite(ea)] = met.calc_vapor_pressure(ea[np.isfinite(ea)])
        valid = np.logical_and(np.isfinite(u), np.isfinite(v))
        u[valid] = np.sqrt(u[valid] ** 2 + v[valid] ** 2)
        del v, valid
        p[np.isfinite(p)] = calc_pressure_mb(p[np.isfinite(p)])

        if "land" not in ecmwf_dataset:
            z = fid.variables["z"][:]
            for i in pos:
                z = z[i]
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

    sdn_mean, _, _ = _getECMWFSolarData(ecmwf_data_file,
                                        "ssrd",
                                        midnight_UTC,
                                        elev_file,
                                        slope_file=slope_file,
                                        aspect_file=aspect_file,
                                        svf_file=svf_file,
                                        time_window=24,
                                        dataset=ecmwf_dataset,
                                        aotfile=aot_data_file)

    fid.close()

    t_max[t_max == -99999] = np.nan
    t_min[t_min == 99999] = np.nan

    # Compute daily means
    t_mean = 0.5 * (t_max + t_min)
    ea_mean = np.nanmean(np.array(ea_array), axis=0)
    # es_mean = 0.5 * (met.calc_vapor_pressure(t_max) + met.calc_vapor_pressure(t_min))
    es_mean = np.nanmean(np.array(es_array), axis=0)
    u_mean = np.nanmean(np.array(u_array), axis=0)
    p_mean = np.nanmean(np.array(p_array), axis=0)

    lat = _ECMWFRespampleData(lat, gt, proj, template_file=elev_file,
                              resample_alg="bilinear")
    doy = float(timedate_UTC.strftime("%j"))
    f_cd = pet.calc_cloudiness(sdn_mean, lat, elev_data, doy)
    le = pet.pet_fao56(t_mean, u_mean, ea_mean, es_mean, p_mean, sdn_mean, z_u, z_t,
                       f_cd=f_cd, is_daily=True)

    et_ref = met.flux_2_evaporation(le, t_mean, 24)

    return et_ref, t_min, t_max, sdn_mean, ea_mean, u_mean, p_mean


def mean_temperature(ecmwf_data_file,
                     elev_file,
                     timedate_UTC,
                     time_zone=0,
                     z_t=2,
                     ecmwf_dataset='ERA5_reanalysis'):
    # Find midnight in local time and convert to UTC time
    date_local = (timedate_UTC + datetime.timedelta(hours=time_zone)).date()
    midnight_local = datetime.datetime.combine(date_local, datetime.time())
    midnight_UTC = midnight_local - datetime.timedelta(hours=time_zone)

    # Open the netcdf time dataset
    fid = netCDF4.Dataset(ecmwf_data_file, 'r')
    time = fid.variables['time']
    dates = netCDF4.num2date(time[:], time.units, time.calendar)

    # Get the time right before date_time, to ose it as integrated baseline
    date_0, _, _ = _bracketing_dates(dates, midnight_UTC)
    if not date_0:
        date_0 = 0
    # Get the time right before the temporal witndow set
    date_1, _, _ = _bracketing_dates(dates, midnight_UTC + datetime.timedelta(
        hours=24))

    elev_data = gu.raster_data(elev_file)
    # Open air temperature dataset
    t2m_fid = gdal.Open('NETCDF:"' + ecmwf_data_file + '":t2m')
    gt = t2m_fid.GetGeoTransform()
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    proj = sr.ExportToWkt()
    del t2m_fid

    # Initialize stack variables
    t_max = np.full(elev_data.shape, -99999.)
    t_min = np.full(elev_data.shape, 99999.)
    for date_i in range(date_0 + 1, date_1 + 1):
        # Read the right time layers
        t_air = fid.variables["t2m"][:]
        if "expver" in fid.variables.keys():
            pos = [date_i, 0]
        else:
            pos = [date_i]
        # Calcultate temperature at 0m datum height
        for i in pos:
            t_air = t_air[i]
        t_air = np.ma.filled(t_air, np.nan)
        if "land" not in ecmwf_dataset:
            z = fid.variables["z"][:]
            for i in pos:
                z = z[i]
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

        # Calculate dew temperature and vapour pressure at elevation height
        z = _ECMWFRespampleData(z, gt, proj, template_file=elev_file,
                                resample_alg="bilinear")
        t_air = _ECMWFRespampleData(t_air, gt, proj, template_file=elev_file)

        t_air = calc_air_temperature_blending_height(t_air, elev_data + z_t,
                                                     z_0=z + 2.0)

        # Avoid sprading possible NaNs
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
    z_1 = gu.raster_data(elev_file)

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


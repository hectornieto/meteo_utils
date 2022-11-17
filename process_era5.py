from pathlib import Path
import datetime as dt
from osgeo import gdal
from pyproj import Proj
import cdsapi
from meteo_utils import ecmwf_utils as eu
from meteo_utils import solar_irradiance as sun
from meteo_utils import dem_utils as du

METEO_DATA_FIELDS = ["TA", "EA", "U", "P", "AOT", "TCWV", "SDN"]
CDS_VARIABLES = ['100m_u_component_of_wind', '100m_v_component_of_wind',
                 '10m_u_component_of_wind', '10m_v_component_of_wind',
                 '2m_dewpoint_temperature', '2m_temperature',
                 'surface_pressure',
                 'surface_solar_radiation_downward_clear_sky',
                 'surface_solar_radiation_downwards',
                 'surface_thermal_radiation_downwards',
                 'total_column_water_vapour',
                 'geopotential']

ADS_VARIABLES = ["total_aerosol_optical_depth_550nm"]

DAILY_VARS = ["ETref", "SDNday"]

def process_single_date(elev_input_file,
                        slope_input_file,
                        aspect_input_file,
                        date_int,
                        acq_time,
                        dst_folder=None,
                        svf_input_file=None,
                        blending_height=100):
    """

    Parameters
    ----------
    elev_input_file : str
        Path to a GDAL compatible Digital Elevation Model
    slope_input_file : str
        Path to a GDAL compatible slope image (degrees)
    aspect_input_file : str
        Path to a GDAL compatible aspect image (0 for flat surfaces)
    date_int : int or str
        Acquisition date (YYYYMMDD)
    acq_time : float
        Acquistion time in decimal hour
    dst_folder : str, optional
        Path to the destination folder in which meteo products will be stored.
    svf_input_file : str, optional
        Path to a GDAL compatible Sky View Fraction image (0-1)
    blending_height : float, optional
        Elevation above ground level at which meteo products will be generated, default=100 magl

    Returns
    -------
    output : dict
        Dictionary of arrays with the output meteo products
    """

    dst_folder = Path(dst_folder)
    print(f"Downloading \"{', '.join(CDS_VARIABLES)}\" from the Copernicus Climate Store")
    fid = gdal.Open(elev_input_file, gdal.GA_ReadOnly)
    gt = fid.GetGeoTransform()
    proj = fid.GetProjection()
    p = Proj(proj)
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * fid.RasterXSize
    miny = maxy + gt[5] * fid.RasterYSize
    del fid
    date_obj = dt.datetime.strptime(str(date_int), "%Y%m%d")
    date_ini = (date_obj - dt.timedelta(1))
    date_end = date_obj + dt.timedelta(1)
    date_str = f"{date_ini.strftime('%Y-%m-%d')} / {date_end.strftime('%Y-%m-%d')}"
    # Area is North, West, South, East
    extent_geo = p(minx, maxy, inverse=True), p(maxx, miny, inverse=True)
    area = f"{extent_geo[0][1] + 1} / {extent_geo[0][0] - 1} / " \
           f"{extent_geo[1][1] - 1} / {extent_geo[1][0] + 1}"
    print(f"Querying products for extent {area}\n"
          f"..and dates {date_obj - dt.timedelta(1)} to {date_obj + dt.timedelta(1)}")

    # Connect to the server and download the data
    s = {'format': 'netcdf',
         'variable': CDS_VARIABLES,
         'date': date_str,
         'area': area,
         'product_type': 'reanalysis',
         'time': [str(t).zfill(2) + ":00" for t in range(0, 24, 1)]
         }

    c = cdsapi.Client(quiet=True, progress=False)
    cds_target = str(dst_folder / f"{date_int}_era5.nc")
    print(f"Saving into {cds_target}")
    c.retrieve('reanalysis-era5-single-levels', s, cds_target)
    print(f"Saved to file {cds_target}")

    # If we are working on the currrent year we need to query the forecast cams data, otherwise download reanalysis
    if date_obj.year == dt.date.today().year:
        dataset = "cams-global-atmospheric-composition-forecasts"
    else:
        dataset = "cams-global-reanalysis-eac4"

    print(f"Downloading \"{', '.join(ADS_VARIABLES)}\" from the Copernicus Atmospheric Store")
    ads_target = str(dst_folder / f"{date_int}_cams.nc")
    eu.download_ADS_data(dataset,
                         date_obj - dt.timedelta(1),
                         date_obj + dt.timedelta(1),
                         ADS_VARIABLES,
                         ads_target,
                         overwrite=False,
                         area=area)
    print(f"Saved to file {ads_target}")

    print("Computing Solar Zenith Angle")
    lats, lons = du.latlon_from_dem(elev_input_file, output=None)
    doy = float(date_obj.strftime("%j"))
    sza = sun.calc_sun_angles(lats, lons, 0, doy, acq_time)[0]
    sza_file = str(dst_folder / f"{date_int}T{acq_time}_SZA.tif")
    driver = gdal.GetDriverByName("MEM")
    dims = sza.shape
    ds = driver.Create("MEM", dims[1], dims[0], 1, gdal.GDT_Float32)
    ds.SetProjection(proj)
    ds.SetGeoTransform(gt)
    ds.GetRasterBand(1).WriteArray(sza)
    del sza
    driver_opt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
    gdal.Translate(sza_file, ds, format="GTiff",
                   creationOptions=driver_opt, stats=True)
    del ds

    date_obj = date_obj + dt.timedelta(hours=acq_time)
    time_zone = sun.angle_average(extent_geo[0][0], extent_geo[1][0]) / 15.
    print(f"Processing ECMWF data for UTC time {date_obj}\n"
          "This may take some time...")

    meteo_data_fields = METEO_DATA_FIELDS + DAILY_VARS
    output = eu.get_ECMWF_data(cds_target,
                               date_obj,
                               meteo_data_fields,
                               elev_input_file,
                               sza_file,
                               blending_height,
                               aod550_data_file=ads_target,
                               time_zone=time_zone,
                               ecmwf_dataset="ERA5_reanalysis",
                               slope_file=slope_input_file,
                               aspect_file=aspect_input_file,
                               svf_file=svf_input_file)

    out_dict = {}
    if dst_folder:
        for param, array in output.items():
            if param not in DAILY_VARS:
                filename = f"{date_int}T{acq_time}_{param.upper()}.tif"
            else:
                filename = f"{date_int}_{param.upper()}.tif"
            dst_file = str(dst_folder / filename)
            print(f"Saving {param} to {dst_file}")
            driver = gdal.GetDriverByName("MEM")
            dims = array.shape
            ds = driver.Create("MEM", dims[1], dims[0], 1, gdal.GDT_Float32)
            ds.SetProjection(proj)
            ds.SetGeoTransform(gt)
            ds.GetRasterBand(1).WriteArray(array)
            driver_opt = ['COMPRESS=DEFLATE', 'PREDICTOR=1', 'BIGTIFF=IF_SAFER']
            gdal.Translate(dst_file, ds, format="GTiff",
                           creationOptions=driver_opt, stats=True)

        del ds

    return output


if __name__ == "__main__":

    elev_input_file = "/path/to/the/dem/file"
    slope_input_file = "/path/to/the/slope/file" # In degrees
    aspect_input_file = "/path/to/the/aspect/file"  # 0 for flat
    svf_input_file = "/path/to/the/sky/view/fraction/file"  # 0-1
    elev_input_file = "./test/dem.tif"
    slope_input_file = "./test/slope.tif"
    aspect_input_file = "./test/aspect.tif"
    svf_input_file = "./test/svf.tif"

    date_int = 20210112  # YYYYMMDD
    acq_time = 11.5  # Decimal hour
    dst_folder = "./test"
    blending_height = 100  # m above ground level at which meteo will be produced
    process_single_date(elev_input_file,
                        slope_input_file,
                        aspect_input_file,
                        date_int,
                        acq_time,
                        dst_folder,
                        svf_input_file=svf_input_file,
                        blending_height=100)
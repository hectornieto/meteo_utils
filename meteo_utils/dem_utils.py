# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 16:48:56 2018

@author: hector
"""

import glob
import os
import os.path as pth

import numpy as np
from osgeo import osr, gdal
import skimage.morphology as morph
from . import gdal_utils as gu

# Average conversion scale from degrees to meters
DEG_2_METERS = 111120

# Terrain occlusion threshold
OCCLUSION_THRES = 0

# Path to SAGA saga_cmd binary file
SAGA_CMD_PATH = "saga_cmd"

# Process SRTM tiles so that they are of the same resolution, extent and projection as HR dataset
# and derive slope and aspect.


def srtm_dem(data_dir_dem, template_file, output_dem, srtm_nodata=-32762):
    # Find all SRTM tiles in the DEM directory, create virtual raster and the reproject,
    # resample and subset it based of the HR template.
    srtm_tiles = glob.glob(os.path.join(data_dir_dem, "srtm_??_??.tif"))
    if srtm_tiles:
        if len(srtm_tiles) > 1:
            raw_dem = os.path.join(data_dir_dem, 'temp.tif')
            raw_dem = gu.merge_raster_layers(srtm_tiles, raw_dem)
        else:
            raw_dem = srtm_tiles[0]
        out_fid = gu.resample_with_gdalwarp(raw_dem, template_file,
                                            resample_alg='cubic')
        dem_data = out_fid.GetRasterBand(1).ReadAsArray().astype(np.float)

        dem_data[dem_data == srtm_nodata] = np.nan
        gu.save_image(dem_data,
                      out_fid.GetGeoTransform(),
                      out_fid.GetProjection(),
                      output_dem)

        if os.path.exists(os.path.join(data_dir_dem, 'temp.tif')):
            os.remove(os.path.join(data_dir_dem, 'temp.tif'))

def process_dem(data_dir_DEM, template_HR, output_dataset_name, overwrite_existing):

    dem_file = os.path.join(data_dir_DEM, '%s_dem_hr.tif' % output_dataset_name)

    if os.path.isfile(dem_file) and not overwrite_existing:
        print('%s already processed' % dem_file)
    else:
        # Find all SRTM tiles in the DEM directory, create virtual raster and the reproject,
        # resample and subset it based of the HR template. This assumes that the tiles are already
        # present since they are downloaded by Sen2Cor.
        srtm_dem(data_dir_DEM, template_HR, dem_file)

    # Calculate slope and aspect using GDAL
    slope_file = os.path.join(data_dir_DEM, '%s_slope_hr.tif' % output_dataset_name)
    if not os.path.exists(slope_file) or overwrite_existing:
        slope_from_dem(dem_file, slope_file)
    aspect_file = os.path.join(data_dir_DEM, '%s_aspect_hr.tif' % output_dataset_name)
    if not os.path.exists(aspect_file) or overwrite_existing:
        aspect_from_dem(dem_file, aspect_file)

    return dem_file, slope_file, aspect_file


def slope_from_dem(dem_file_path, output=None, scale=None):

    if not output:
        output = dem_file_path.replace('.tif', '_slope.tif')
    if scale is None:
        proj = gu.raster_info(dem_file_path)[0]
        srs = osr.SpatialReference()
        srs.ImportFromWkt(proj)
        if srs.GetAttrValue('UNIT') == "degree":
            # Covert degrees pixel size into meters for trigonometric computations
            scale = DEG_2_METERS
        else:
            scale = 1

    gdal.DEMProcessing(output, dem_file_path, "slope",
                       slopeFormat="degree",
                       computeEdges=True,
                       scale=scale)


def aspect_from_dem(dem_file_path, output=None):

    if not output:
        output = pth.splitext(dem_file_path)[0]+'_aspect.tif'

    gdal.DEMProcessing(output, dem_file_path, "aspect",
                       computeEdges=True,
                       zeroForFlat=True)



def declination_angle(doy):
    ''' Calculates the Earth declination angle

    Parameters
    ----------
    doy : float or int
        day of the year

    Returns
    -------
    declination : float
        Declination angle (radians)
    '''
    declination = np.radians(23.45) * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)

    return declination


def hour_angle(ftime, declination, lon, stdlon=0):
    '''Calculates the hour angle

    Parameters
    ----------
    ftime : float
        Time of the day (decimal hours)
    declination : float
        Declination angle (radians)
    lon : float
        longitude of the site (degrees).
    stdlon : float
        Longitude of the standard meridian that represent the ftime time zone

    Returns
    w : float
        hour angle (radians), negative before noon, positive after noon
    '''

    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
          3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.radians((solar_time - 12.0) * 15.)

    return w


def incidence_angle_tilted(lat, lon, doy, ftime, stdlon=0, aspect=0, slope=0):
    ''' Calculates the incidence solar angle over a tilted flat surface

    Parameters
    ----------
    lat :  float or array
        latitude (degrees)
    lon :  float or array
        longitude (degrees)
    doy : int
        day of the year
    ftime : float
        Time of the day (decimal hours)
    stdlon : float
        Longitude of the standard meridian that represent the ftime time zone
    aspect : float or array
        surface azimuth angle, measured clockwise from north (degrees)
    slope : float or array
        slope angle (degrees)

    Returns
    -------
    cos_theta_i : float or array
        cosine of the incidence angle
    '''

    # Get the dclination and hour angle
    delta = declination_angle(doy)
    # Hour angle is considered negative before noon, positive after noon
    omega = hour_angle(ftime, delta, lon, stdlon=stdlon)

    # Convert remaining angles into radians,
    # aspect is considered the azimuth from the South being westward positive
    lat, aspect, slope = map(np.radians, [lat, aspect - 180, slope])

    f1, f2, f3 = inclination_factors(lat, slope, aspect)

    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)
    cos_theta_i = sin_delta * f1 \
                  + cos_delta * cos_omega * f2 \
                  + cos_delta * sin_omega * f3

    return cos_theta_i


def shading(dem_file, saa, sza, max_chunk_size=1e9):
    """
    Adapted from https://github.com/zoran-cuckovic/QGIS-terrain-shading
    Parameters
    ----------
    dem_file
    saa
    sza
    max_chunk_size

    Returns
    -------

    """
    def mean_angle(angle_array, degrees=True):
        if degrees is True:
            angle_array = np.radians(angle_array)
        cos_i = np.nanmean(np.cos(angle_array))
        sin_i = np.nanmean(np.sin(angle_array))
        mean_angle = np.arctan2(sin_i, cos_i)
        if degrees is True:
            mean_angle = np.degrees(mean_angle)
        return mean_angle

    def window_loop(shape, chunk, axis=0, reverse=False, overlap=0):
        """
        Construct a frame to extract chunks of data from gdal
        (and to insert them properly to a numpy matrix)
        """
        xsize, ysize = shape if axis == 0 else shape[::-1]

        if reverse:
            steps = np.arange(xsize // chunk, -1, -1)
            begin = xsize
        else:
            steps = np.arange(1, xsize // chunk + 2)
            begin = 0

        x, y, x_off, y_off = 0, 0, xsize, ysize

        for step in steps:

            end = min(int(chunk * step), xsize)
            if reverse:
                x, x_off = end, begin - end
            else:
                x, x_off = begin, end - begin

            if overlap and x >= overlap: x -= overlap * int(step)

            begin = end

            if x_off < chunk:
                in_view = np.s_[:, : x_off] if not axis else np.s_[: x_off, :]
            else:
                in_view = np.s_[:]

            if not axis:
                gdal_take = (x, y, x_off, y_off)
            else:
                gdal_take = (y, x, y_off, x_off)

            yield in_view, gdal_take
    # We need to work with a single beam direction
    direction = mean_angle(saa)
    steep = (45 <= direction <= 135 or 225 <= direction <= 315)

    s = direction % 90
    if s > 45:
        s = 90 - s

    slope = np.tan(np.radians(s))  # matrix shear slope

    # ! attention: x in gdal is y dimension un numpy (the first dimension)
    dem_fid = gdal.Open(dem_file, gdal.GA_ReadOnly)
    xsize, ysize = dem_fid.RasterXSize, dem_fid.RasterYSize
    gt = dem_fid.GetGeoTransform()
    proj = dem_fid.GetProjection()
    # Compute the average pixel sixe
    pixel_size = 0.5 * np.abs(gt[1]) + 0.5 * np.abs(gt[5])
    # Check whether elevation projection units are degrees and convert to meters
    srs = osr.SpatialReference()
    srs.ImportFromWkt(proj)
    if srs.GetAttrValue('UNIT') == "degree":
        # Covert degrees pixel size into meters for trigonometric computations
        pixel_size *= DEG_2_METERS
    # adjust for pixel rotation (pixel diagonal = 45°)
    # (diagonal is clipped by light path, under right angle,
    # which means it becomes the adjacent, and the pixel size the hypothenuse.)
    pixel_size *= np.cos(np.radians(s))

    tilt = np.tan(np.radians(90. - sza))

    chunk = min(int(max_chunk_size // (ysize if steep else xsize)),
                (xsize if steep else ysize))

    if s % 45 > 0:
        # Determine the optimal chunk size (estimate!).
        # The problem is to carry rasterized lines
        # from one chunk to another.
        # So, set chunk size according to minimum rasterisation error
        c = (np.arange(1, chunk) * slope) % 1  # %1 to get decimals only
        c[c > 0.5] -= 1
        # this is not ideal : we cannot predict where it would stop
        chunk -= np.argmin(np.round(abs(c), decimals=2)[::-1]) + 1

    # 3) -------   SHEAR MATRIX (INDICES) -------------
    chunk_slice = (ysize, chunk) if steep else (chunk, xsize)
    indices_y, indices_x = np.indices(chunk_slice)
    mx_z = np.zeros(chunk_slice)
    mx_z[:] = -99999

    # this is all upside down ...
    rev_y = 90 <= direction <= 270
    rev_x = not 180 <= direction <= 360

    if rev_y:
        indices_y = indices_y[::-1, :]
    if not rev_x:
        indices_x = indices_x[:, ::-1]

    off_a = indices_x + indices_y * slope
    off_b = indices_y + indices_x * slope

    if steep:
        axis = 0
        # construct a slope to simulate sun angle
        # elevations will be draped over this 2D slope
        off = off_a[:, ::-1]  # indices_y * slope + indices_x[:, ::-1]

        src_y = indices_x[:, ::-1]
        src_x = np.round(off_b).astype(int)

    else:
        axis = 1
        off = off_b[:, ::-1]  # indices_x[:,::-1] * slope + indices_y

        src_x = indices_y
        src_y = np.round(off_a).astype(int)

    src = np.s_[src_y, src_x]

    # Distance to slope's perpendicular is (x + y)/2 * root(2) for 45°
    # because where x = y we have a square with a*root(2) diagonal
    # However, in a rotated pixel grid, the slope is cutting through pixel diagonals
    # so we take cosinus (pixel_size) to get this shortened distance
    off *= pixel_size * np.nanmean(tilt)

    # create a matrix to hold the sheared matrix
    mx_temp = np.zeros(((np.max(src_y) + 1), np.max(src_x) + 1))

    t_y, t_x = mx_temp.shape

    # carrying lines from one chunk to the next (fussy...)
    if steep:
        l = np.s_[-1, : ysize]
        f = np.s_[0, t_x - ysize:]
    else:
        l = np.s_[t_y - xsize:, -1]
        f = np.s_[:xsize, 0]

    last_line = np.zeros((ysize if steep else xsize))

    # 4 -----   LOOP THOUGH DATA CHUNKS AND CALCULATE -----------------
    counter = 0
    out_array = np.zeros([ysize, xsize])
    for dem_view, gdal_coords in window_loop(
            shape=(xsize, ysize),
            chunk=chunk,
            axis=not steep,
            reverse=rev_x if steep else rev_y,
            overlap=1):

        mx_z[dem_view] = dem_fid.ReadAsArray(*gdal_coords).astype(float)

        # should handle better NoData !!
        # nans will destroy the accumulation sequence
        mask = np.isnan(mx_z)
        mx_z[mask] = -9999

        mx_temp[src] = mx_z + off

        mx_temp[f] += -last_line  # shadows have negative values, so *-1
        # accumulate maximum shadow depths
        mx_temp -= np.maximum.accumulate(mx_temp, axis=axis)
        # first line has shadow od zero depth (nothing to accum), so copy from previous chunk
        mx_temp[f] = last_line

        last_line[:] = mx_temp[l]  # save for later

        out = mx_temp[src]
        out[mask] = np.nan
        counter += 1
        out_array[gdal_coords[1]:gdal_coords[1] + gdal_coords[3],
                  gdal_coords[0]:gdal_coords[0] + gdal_coords[2]] = out[dem_view]


    return out_array


def latlon_from_dem(dem_file_path, output=None):
    if not output:
        output = dem_file_path.replace('.tif', '_latlon.tif')

    proj, gt, size_x, size_y = gu.raster_info(dem_file_path)[0:4]
    dims = size_y, size_x
    rows, cols = np.indices(dims)
    lon = gt[0] + cols * gt[1] + rows * gt[2]
    lat = gt[3] + cols * gt[4] + rows * gt[5]
    srcSRS = osr.SpatialReference()
    srcSRS.ImportFromWkt(proj)
    dstSRS = osr.SpatialReference()
    dstSRS.ImportFromEPSG(4326)
    # GDAL decided to reverse the coordinate transformation axis order
    # https://github.com/OSGeo/gdal/issues/1546
    dstSRS.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(srcSRS, dstSRS)
    array = np.stack([lon.reshape(-1), lat.reshape(-1)]).T
    coords = np.array(transform.TransformPoints(array)).T

    lon = coords[0].reshape(dims)
    lat = coords[1].reshape(dims)
    array = np.dstack([lat, lon])
    gu.save_image(array, gt, proj, output, no_data_value=np.nan)
    return lat, lon


def non_occluded_terrain(dem_file, saa, sza, smooth_radius=1):
    # Get areas that are shaded by obstacles around
    occlusion = shading(dem_file, saa, sza)
    non_shaded = np.logical_and(np.isfinite(occlusion), occlusion >= OCCLUSION_THRES)
    del occlusion
    # Smooth the occlusion map
    if smooth_radius > 0:
        se = morph.disk(smooth_radius)
        non_shaded = morph.binary_closing(non_shaded, footprint=se)
    return non_shaded


def inclination_factors(lat, slope, aspect):
    # Convert remaining angles into radians,
    # aspect is considered the azimuth from the South being westward positive
    lat, aspect, slope = map(np.radians, [lat, aspect - 180, slope])

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_slope = np.cos(slope)
    sin_slope = np.sin(slope)
    cos_az = np.cos(aspect)
    sin_az = np.sin(aspect)

    f1 = sin_lat * cos_slope - cos_lat * sin_slope * cos_az
    f2 = cos_lat * cos_slope + sin_lat * sin_slope * cos_az
    f3 = sin_slope * sin_az
    return f1, f2, f3


# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:25:09 2018

@author: rmgu
"""

import math
import numpy as np
import tempfile

from osgeo import ogr, osr, gdal
import xarray

from pyDMS.pyDMSUtils import saveImg, getRasterInfo, openRaster, resampleWithGdalWarp


def raster_data(raster, bands=1):
    fid, closeOnExit = openRaster(raster)
    if type(bands) == int:
        bands = [bands]
    if bands is None:
        n_bands = fid.RasterCount
        bands = range(1, n_bands+1)
    data = None
    for band in bands:
        if data is None:
            data = fid.GetRasterBand(band).ReadAsArray()
        else:
            data = np.dstack((data, fid.GetRasterBand(band).ReadAsArray()))

    if closeOnExit:
        fid = None

    return data


def raster_info(raster):
    return getRasterInfo(raster)


def save_image(data, geotransform, projection, filename, no_data_value=np.nan):
    return saveImg(data, geotransform, projection, filename, noDataValue=no_data_value)


def resample_with_gdalwarp(src, template, resample_alg):
    return resampleWithGdalWarp(src, template, resampleAlg=resample_alg)


def merge_raster_layers(input_list, output_filename, separate=False):
    merge_list = []
    for input_file in input_list:
        bands = raster_info(input_file)[5]
        # GDAL Build VRT cannot stack multiple multi-band images, so they have to be split into
        # multiple singe-band images first.
        if bands > 1:
            for band in range(1, bands+1):
                temp_filename = tempfile.mkstemp(suffix="_"+str(band)+".vrt")[1]
                gdal.BuildVRT(temp_filename, [input_file], bandList=[band])
                merge_list.append(temp_filename)
        else:
            merge_list.append(input_file)
    fp = gdal.BuildVRT(output_filename, merge_list, separate=separate)
    return fp


def get_subset(roi_shape, raster_proj_wkt, raster_geo_transform):

    # Find extent of ROI in roiShape projection
    roi = ogr.Open(roi_shape)
    roi_layer = roi.GetLayer()
    roi_extent = roi_layer.GetExtent()

    # Convert the extent to raster projection
    roi_proj = roi_layer.GetSpatialRef()
    raster_proj = osr.SpatialReference()
    raster_proj.ImportFromWkt(raster_proj_wkt)
    transform = osr.CoordinateTransformation(roi_proj, raster_proj)
    point_UL = ogr.CreateGeometryFromWkt("POINT (" +
                                         str(min(roi_extent[0], roi_extent[1])) + " " +
                                         str(max(roi_extent[2], roi_extent[3])) + ")")
    point_UL.Transform(transform)
    point_UL = point_UL.GetPoint()
    point_LR = ogr.CreateGeometryFromWkt("POINT (" +
                                         str(max(roi_extent[0], roi_extent[1])) + " " +
                                         str(min(roi_extent[2], roi_extent[3])) + ")")
    point_LR.Transform(transform)
    point_LR = point_LR.GetPoint()

    # Get pixel location of this extent
    ulX = raster_geo_transform[0]
    ulY = raster_geo_transform[3]
    pixel_size = raster_geo_transform[1]
    pixel_UL = [max(int(math.floor((ulY - point_UL[1]) / pixel_size)), 0),
                max(int(math.floor((point_UL[0] - ulX) / pixel_size)), 0)]
    pixel_LR = [int(round((ulY - point_LR[1]) / pixel_size)),
                int(round((point_LR[0] - ulX) / pixel_size))]

    # Get projected extent
    point_proj_UL = (ulX + pixel_UL[1]*pixel_size, ulY - pixel_UL[0]*pixel_size)
    point_proj_LR = (ulX + pixel_LR[1]*pixel_size, ulY - pixel_LR[0]*pixel_size)

    # Create a subset from the extent
    subset_proj = [point_proj_UL, point_proj_LR]
    subset_pix = [pixel_UL, pixel_LR]

    return subset_pix, subset_proj


def read_subset(source, subset_pix):
    if type(source) is np.ndarray:
        data = source[subset_pix[0][0]:subset_pix[1][0], subset_pix[0][1]:subset_pix[1][1]]
    elif type(source) == int or type(source) == float:
        data = np.zeros((subset_pix[1][0] - subset_pix[0][0],
                         subset_pix[1][1] - subset_pix[0][1])) + source
    # Otherwise it should be a file path
    else:
        fp = gdal.Open(source)
        data = fp.GetRasterBand(1).ReadAsArray(subset_pix[0][1],
                                               subset_pix[0][0],
                                               subset_pix[1][1] - subset_pix[0][1],
                                               subset_pix[1][0] - subset_pix[0][0])
        fp = None
    return data


# Save pyTSEB input dataset to an NetCDF file
def save_dataset(dataset, gt, proj, output_filename, roi_vector=None, attrs={},
                 compression={'zlib': True, 'complevel': 6}):

    # Get the raster subset extent
    if roi_vector is not None:
        subset_pix, subset_proj = get_subset(roi_vector, proj, gt)
        # NetCDF and GDAL geocoding are off by half a pixel so need to take this into account.
        pixel_size = gt[1]
        subset_proj = [[subset_proj[0][0] + 0.5*pixel_size, subset_proj[0][1] - 0.5*pixel_size],
                       [subset_proj[1][0] + 0.5*pixel_size, subset_proj[1][1] - 0.5*pixel_size]]
    else:
        shape = dataset["LAI"].shape
        subset_pix = [[0, 0], [shape[0], shape[1]]]
        subset_proj = [[gt[0] + gt[1]*0.5, gt[3] + gt[5]*0.5],
                       [gt[0] + gt[1]*(shape[1]+0.5), gt[3]+gt[5]*(shape[0]+0.5)]]

    # Create xarray DataSet
    x = np.linspace(subset_proj[0][1], subset_proj[1][1], subset_pix[1][0] - subset_pix[0][0],
                    endpoint=False)
    y = np.linspace(subset_proj[0][0], subset_proj[1][0], subset_pix[1][1] - subset_pix[0][1],
                    endpoint=False)
    ds = xarray.Dataset({}, coords={'x': (['x'], x),
                                    'y': (['y'], y),
                                    'crs': (['crs'], [])})
    ds.crs.attrs['spatial_ref'] = proj

    # Save the data in the DataSet
    encoding = {}
    for name in dataset:
        data = read_subset(dataset[name], subset_pix)
        ds = ds.assign(temporary=xarray.DataArray(data, coords=[ds.coords['x'], ds.coords['y']],
                                                  dims=('x', 'y')))
        ds["temporary"].attrs['grid_mapping'] = 'crs'
        ds = ds.rename({'temporary': name})
        encoding[name] = compression

    ds.attrs = attrs

    # Save dataset to file
    ds.to_netcdf(output_filename, encoding=encoding)


def prj_to_src(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    return src


def prj_to_epsg(prj):
    src = osr.SpatialReference()
    src.ImportFromWkt(prj)
    epsg = int(src.GetAttrValue("AUTHORITY", 1))
    return epsg


def get_map_coordinates(row, col, geoTransform):
    X = geoTransform[0]+geoTransform[1]*col+geoTransform[2]*row
    Y = geoTransform[3]+geoTransform[4]*col+geoTransform[5]*row
    return X, Y


def get_pixel_coordinates(X, Y, geoTransform):
    row = (Y - geoTransform[3]) / geoTransform[5]
    col = (X - geoTransform[0]) / geoTransform[1]
    return int(row), int(col)


def convert_coordinate(input_coordinate, input_src, output_src=None, Z_in=0):
    ''' Coordinate conversion between two coordinate systems

    Parameters
    ----------
    input_coordinate : tuple
        input coordinate (x,y)
    inputEPSG : int
        EPSG coordinate code of input coordinates
    outputEPSG : int
       EPSG coordinate code of output coordinates
    Z_in : float
        input altitude, default=0

    Returns
    -------
    X_out : float
        output X coordinate
    Y_out : float
        output X coordinate
    Z_out : float
        output X coordinate
    '''

    if not output_src:
        output_src = osr.SpatialReference()
        output_src.ImportFromEPSG(4326)

    # create coordinate transformation
    coordTransform = osr.CoordinateTransformation(input_src, output_src)

    # transform point
    X_out, Y_out, Z_out = coordTransform.TransformPoint(input_coordinate[0],
                                                        input_coordinate[1],
                                                        Z_in)

    # print point in EPSG 4326
    return X_out, Y_out, Z_out


def update_nan(file, src_nodata_value=-32768, dst_nodata_value=np.nan):
    fid = gdal.Open(file, gdal.GA_Update)
    bands = fid.RasterCount
    for band in range(bands):
        band_ds = fid.GetRasterBand(band + 1)
        data = band_ds.ReadAsArray()
        data[data == src_nodata_value] = dst_nodata_value
        band_ds.WriteArray(data)
        band_ds.SetNoDataValue(dst_nodata_value)
        band_ds.FlushCache()

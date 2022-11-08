# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:42:40 2017

@author: hector
"""

import numpy as np
from . import ecmwf_utils as eu
from . import dem_utils as du


# Calculate S_dn using Ineichen equaiton
def calc_global_horizontal_radiance_clear_sky(doy, sza, aot_550, pw, press, t,
                                              altitude=0,
                                              solar_constant=1367.7,
                                              calc_diffuse=False):
    # Calculate extraterrestrial solar irradiance
    sdn_0 = calc_extraterrestrial_radiation(doy, solar_constant)
    # Calculate the air mass
    air_mass = calc_air_mass_kasten(sza)

    # Calculate Linke turbidity index
    p0_p = eu.calc_sp1_sp0(t, 0, z_0=altitude)
    tl = calc_Linke_turbidity_Ineichen(aot_550, pw, p0_p)
    tl = np.maximum(1, tl)

    # Calculate Kasten coefficients
    f_h1 = np.exp(-altitude / 8000.0)
    f_h2 = np.exp(-altitude / 1250.0)

    # Calculate Ineichen and Perez coefficients
    c_g1 = 5.09e-5 * altitude + 0.868
    c_g2 = 3.92e-5 * altitude + 0.0387

    sdn = c_g1 * sdn_0 * np.cos(np.radians(sza)) * \
          np.exp(-c_g2 * air_mass * (f_h1 + f_h2 * (tl - 1.0)))  # * np.exp(0.01 * air_mass ** 1.8)
    if calc_diffuse is True:
        # correct Linke turbidity factor for very low tl:
        # Appendix A
        turbid = tl < 2
        tl[turbid] = tl[turbid] - 0.25 * np.sqrt(2. - tl[turbid])

        # Eq. 8, adapted for an horizontal surface
        b = 0.664 + 0.163 / f_h1
        b_dn = b * sdn_0 * np.cos(np.radians(sza)) * np.exp(-0.09 * air_mass * (tl - 1.0))

        # Appendix A, adapted for an horizontal surface
        b_dn_2 = sdn * (1. - (0.1 - 0.2 * np.exp(-tl))
                        / (0.1 + 0.88 / f_h1))
        b_dn_2[~np.isfinite(b_dn_2)] = sdn[~np.isfinite(b_dn_2)]
        bdn = np.minimum(b_dn, b_dn_2)
        bdn = np.clip(bdn, 0, sdn)
        # Get diffuse ratio
        beam_ratio = np.clip(bdn / sdn, 0, 1)
        sdn = [sdn, beam_ratio]
    return sdn


def calc_air_mass_planeparallel(sza):
    sza = np.clip(sza, 0, 89)
    air_mass = 1.0 / np.cos(np.radians(sza))
    return air_mass


def calc_air_mass_kasten(sza):
    """
    Parameters
    ----------
    sza : float or array
        Solar Zenith Angle (degrees)

    Returns
    -------
    air_mass : float or array
        Relative air_mass

    References
    ----------
    ..[Kasten_1989] Fritz Kasten and Andrew T. Young, Revised optical air mass tables
        and approximation formula,
         Appl. Opt. 28, 4735-4738 (1989)
    """
    # Convert zennith angle to solar elevation
    sza = 90.0 - sza
    sza = np.clip(sza, 0, 90)
    # define constant parameters
    a = 0.50572
    b = 6.07995  # in degrees
    c = 1.6364
    air_mass = 1.0 / (np.sin(np.radians(sza)) + a * (sza + b) ** -c)
    return air_mass


def calc_Linke_turbidity_Ineichen(aot_550, pw, p0_p):

    tl = 3.91 * np.exp(0.689 * p0_p) * aot_550 + 0.376 * np.log(pw) + 2 + \
         0.54 * p0_p - 0.5 * p0_p ** 2 + 0.16 * p0_p ** 3
    return tl


def calc_extraterrestrial_radiation(DOY, solar_constant=1367.7):
    sdn_0 = solar_constant * (1.0 + 0.033 * np.cos(2.0 * np.pi * DOY / 365.0))
    return sdn_0


def calc_radiance_tilted_surface(bdn, ddn,
                                 doy, ftime,
                                 lat, lon, sza,
                                 slope, aspect,
                                 non_shaded=None):
    """ Corrects both beam and diffuse radiation on illumination angle"""

    # Compute the incidence angle and incidence ratio
    cos_theta_i = du.incidence_angle_tilted(lat, lon, doy, ftime,
                                            stdlon=0,
                                            aspect=aspect,
                                            slope=slope)
    inc_ratio = cos_theta_i / np.cos(np.radians(sza))

    # The global solar radiation contains at least the diffuse part
    # We do not consider occlusion effects in diffuse radiation
    sdn = ddn.copy()
    if non_shaded is None:
        non_shaded = np.ones(bdn.shape, dtype=bool)
    non_shaded = np.logical_and(non_shaded, inc_ratio > 0)
    sdn[non_shaded] = sdn[non_shaded] + bdn[non_shaded] * inc_ratio[non_shaded]
    return sdn


def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    '''Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float
        latitude of the site (degrees).
    long : float
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    sza : float
        Sun Zenith Angle (degrees).
    saa : float
        Sun Azimuth Angle (degrees).

    '''

    # Calculate declination
    declination = 0.409 * np.sin((2.0 * np.pi * doy / 365.0) - 1.39)
    EOT = 0.258 * np.cos(declination) - 7.416 * np.sin(declination) - \
        3.648 * np.cos(2.0 * declination) - 9.228 * np.sin(2.0 * declination)
    LC = (stdlon - lon) / 15.
    time_corr = (-EOT / 60.) + LC
    solar_time = ftime - time_corr
    # Get the hour angle
    w = np.asarray((solar_time - 12.0) * 15.)
    # Get solar elevation angle
    sin_thetha = np.cos(np.radians(w)) * np.cos(declination) * np.cos(np.radians(lat)) + \
                 np.sin(declination) * np.sin(np.radians(lat))
    sun_elev = np.arcsin(sin_thetha)
    # Get solar zenith angle
    sza = np.pi / 2.0 - sun_elev
    sza = np.asarray(np.degrees(sza))
    # Get solar azimuth angle
    cos_phi = np.asarray(
        (np.sin(declination) * np.cos(np.radians(lat)) -
         np.cos(np.radians(w)) * np.cos(declination) * np.sin(np.radians(lat))) /
        np.cos(sun_elev))
    saa = np.zeros(sza.shape)
    saa[w <= 0.0] = np.degrees(np.arccos(cos_phi[w <= 0.0]))
    saa[w > 0.0] = 360. - np.degrees(np.arccos(cos_phi[w > 0.0]))
    return np.asarray(sza), np.asarray(saa)

def angle_average(*angles):
    angles = [np.radians(i) for i in angles]
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    mean_angle = np.degrees(mean_angle)
    return mean_angle


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:42:40 2017

@author: hector
"""

import numpy as np
from . import ecmwf_utils as eu
from . import dem_utils as du

ALBEDO = [0.92,  0.84]
GROUND_ALBEDO = 0.15
STANDARD_PRESSURE_MILLIBARS = 1013.15
UN = 0.0003  # atm*cm, from [Gueymard 2008], p. 280
E0_N = [635.4, 709.7]
REST2 = 0
INEICHEN = 1


def calc_global_horizontal_radiance_clear_sky(doy,
                                              sza,
                                              aot_550,
                                              pw,
                                              press,
                                              t=298.15,
                                              altitude=0,
                                              calc_diffuse=False,
                                              method=REST2):
    if method == REST2:
        sdn = calc_global_horizontal_radiance_clear_sky_rest2(sza,
                                                              aod_550=aot_550,
                                                              precipitable_water_cm=pw,
                                                              pressure_millibars=press,
                                                              alpha=[1.3, 1.3])
        if calc_diffuse == False:
            sdn = sdn[0][0] + sdn[0][1] + sdn[1][0] + sdn[1][1]

    else:
        sdn = calc_global_horizontal_radiance_clear_sky_ineichen(doy,
                                                                 sza,
                                                                 aot_550,
                                                                 pw,
                                                                 press,
                                                                 t,
                                                                 altitude=altitude,
                                                                 solar_constant=E0_N[0] + E0_N[1],
                                                                 calc_diffuse=calc_diffuse)
        if calc_diffuse == True:
            sdn = sdn[0] * sdn[1], sdn[0] * (1 - sdn[1])

    return sdn


def calc_global_horizontal_radiance_clear_sky_ineichen(doy,
                                                       sza,
                                                       aot_550,
                                                       pw,
                                                       press,
                                                       t,
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


def calc_Linke_turbidity_Ineichen(aot_550, pw, p0_p):
    tl = 3.91 * np.exp(0.689 * p0_p) * aot_550 + 0.376 * np.log(pw) + 2 + \
         0.54 * p0_p - 0.5 * p0_p ** 2 + 0.16 * p0_p ** 3
    return tl


def calc_extraterrestrial_radiation(DOY, solar_constant=1367.7):
    sdn_0 = solar_constant * (1.0 + 0.033 * np.cos(2.0 * np.pi * DOY / 365.0))
    return sdn_0


def calc_air_mass_planeparallel(sza):
    sza = np.clip(sza, 0, 89)
    air_mass = 1.0 / np.cos(np.radians(sza))
    return air_mass


def calc_air_mass_kasten(sza):
    """Computes the air mass based on zenith angle
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



def calc_sun_angles(lat, lon, stdlon, doy, ftime):
    '''Calculates the Sun Zenith and Azimuth Angles (SZA & SAA).

    Parameters
    ----------
    lat : float or array_like
        latitude of the site (degrees).
    lon : float or array_like
        longitude of the site (degrees).
    stdlng : float
        central longitude of the time zone of the site (degrees).
    doy : float
        day of year of measurement (1-366).
    ftime : float
        time of measurement (decimal hours).

    Returns
    -------
    sza : float or array_like
        Sun Zenith Angle (degrees).
    saa : float or array_like
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


def calc_global_horizontal_radiance_clear_sky_rest2(sza,
                                                    aod_550=0.1,
                                                    precipitable_water_cm=5.0,
                                                    pressure_millibars=STANDARD_PRESSURE_MILLIBARS,
                                                    ozone_atm_cm=0.35,
                                                    nitrogen_atm_cm=0.0002,
                                                    alpha=[1.3, 1.3]):
    """ Spectral (PAR, IR) global irradiance

    Parameters
    ----------
    sza : float or array_like(M)
        Solar zenith angle (Degrees)
    pressure_millibars : float or array_like(M), optional
        Surface pressure (mp)
    precipitable_water_cm : float or array_like(M), optional
        Total column water vapour (cm)
    aod_550 : float or array_like(M), optional
        Aerosol Optical Depth at 550nm
    ozone_atm_cm : float or array_like, optional
        Ozone concentration(atm/cm)
    nitrogen_atm_cm : float or array_like(M), optional
        Nitrogen concentration (atm/cm)
    alpha : tuple of 2 floats or array_like(2. M), optional
        Spectral Angström’s wavelength exponent

    Returns
    -------
    eb : tuple of 2 float or array_like(2, M)
        Beam irradiance (W m²) for PAR and IR
    ed : tuple of 2 float or array_like(2, M)
        Diffuse irradiance  (W m²)

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    m_a = calc_air_mass_kasten(sza)
    effective_wavelength = effective_aerosol_wavelength(m_a, alpha, aod_550)

    tau_a = aerosol_optical_depth(aod_550,
                                  effective_wavelength,
                                  alpha)

    eb = beam_irradiance(sza,
                         tau_a,
                         pressure_millibars=pressure_millibars,
                         precipitable_water_cm= precipitable_water_cm,
                         ozone_atm_cm=ozone_atm_cm,
                         nitrogen_atm_cm=nitrogen_atm_cm)

    edb = diffuse_irradiance(sza, tau_a)
    edd = backscattered_diffuse_irradiance(eb,
                                           edb,
                                           aod_550,
                                           turbidity_alpha=alpha)

    return eb, edb + edd


def beam_irradiance(sza,
                    tau_a,
                    pressure_millibars=STANDARD_PRESSURE_MILLIBARS,
                    precipitable_water_cm=5.0,
                    ozone_atm_cm=0.35,
                    nitrogen_atm_cm=0.0002):
    """Spectral (PAR, IR) beam irradiance

    Parameters
    ----------
    sza : float or array_like(M)
        Sun zenith angle (Degrees)
    tau_a : tuple of floats(2) or tuple of array_like(2, M), optional
        Spectral aerosol Optical Depth
    pressure_millibars : float or array_like(M), optional
        Surface pressure (mp)
    precipitable_water_cm : float or array_like(M), optional
        Total column water vapour (cm)
    ozone_atm_cm : float or array_like(M), optional
        Ozone concentration(atm/cm)
    nitrogen_atm_cm : float or array_like(M), optional
        Nitrogen concentration (atm/cm)

    Returns
    -------
    ebd : array(2, M)
        Spectral (PAR, IR) beam irradiance (W m-2)

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """

    # Eq. 3 of [Gueymard_2008]_
    e_bn = direct_normal_irradiance(sza,
                                    tau_a,
                                    pressure_millibars=pressure_millibars,
                                    precipitable_water_cm=precipitable_water_cm,
                                    ozone_atm_cm=ozone_atm_cm,
                                    nitrogen_atm_cm=nitrogen_atm_cm)
    e_bn = np.maximum(e_bn, 0)
    return e_bn * np.maximum(np.cos(np.radians(sza)), 0)


def direct_normal_irradiance(sza,
                             tau_a,
                             pressure_millibars=STANDARD_PRESSURE_MILLIBARS,
                             precipitable_water_cm=5.0,
                             ozone_atm_cm=0.35,
                             nitrogen_atm_cm=0.0002):
    """
    Computes the direct normal irradiance (Eq. 3 of [Gueymard_2008]_)

    Parameters
    ----------
    sza : float or array_like(M)
        Solar zenith angle (Degrees)
    tau_a : tuple of 2 floats or tuple of array_like(2, M),
        Spectral aerosol optical depth
    pressure_millibars : float or array_like(M), optional
        Surface pressure (mp)
    ozone_atm_cm : float or array_like(M), optional
        Ozone concentration(atm/cm)
    nitrogen_atm_cm : float or array_like(M), optional
        Nitrogen concentration (atm/cm)
    precipitable_water_cm : float or array_like(M), optional
        Total column water vapour (cm)


    Returns
    -------
    tuple of 2 float or array(2, M)
        PAR and IR direct normal irradiance

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    air_mass = calc_air_mass_kasten(sza)
    air_mass_prime = air_mass * pressure_millibars / STANDARD_PRESSURE_MILLIBARS
    t_r = rayleigh_transmittance(air_mass_prime)
    t_g = gas_transmittance(air_mass_prime)
    t_o = ozone_transmittance(air_mass, ozone_atm_cm)
    # is water_optical_mass really used for nitrogen calc?
    t_n = nitrogen_transmittance(air_mass, u_n=nitrogen_atm_cm)
    t_w = water_vapor_transmittance(air_mass, precipitable_water_cm)
    t_a = aerosol_transmittance(air_mass, tau_a)
    return np.moveaxis(np.array(E0_N) *
                       np.moveaxis((t_r * t_g * t_o * t_n * t_w * t_a),0, -1),
                       -1, 0)


def diffuse_irradiance(sza,
                       tau_a,
                       pressure_millibars=STANDARD_PRESSURE_MILLIBARS,
                       precipitable_water_cm=5.0,
                       ozone_atm_cm=0.35,
                       nitrogen_atm_cm=0.0002):
    """Incident diffuse irradiance on a perfectly absorbing
    ground (i.e., with zero albedo) is defined as

    Parameters
    ----------
    sza : float or array_like(M)
        Solar zenith angle (Degrees)
    tau_a : tuple of 2 floats or tuple of array_like(2, M)
        Spectral aerosol optical depth
    air_mass : float or array_like(M), optional
        Relative air mass

    Returns
    -------
    e_dpi : tuple of 2 float or array(2, M)
        Spectral (VIS, IR) indicent diffuse irradiance

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    air_mass = calc_air_mass_kasten(sza)
    # m_a = optical_mass_aerosol(sza)
    # m_o = optical_mass_ozone(sza)
    air_mass_prime = air_mass * pressure_millibars / STANDARD_PRESSURE_MILLIBARS
    t_o = ozone_transmittance(air_mass, ozone_atm_cm)
    t_g = gas_transmittance(air_mass_prime)
    t_n = nitrogen_transmittance(np.full_like(sza, 1.66), nitrogen_atm_cm)
    t_w = water_vapor_transmittance(np.full_like(sza, 1.66),
                                    precipitable_water_cm)
    t_ray = rayleigh_transmittance(air_mass_prime)
    t_a = aerosol_transmittance(air_mass, tau_a)
    t_as = aerosol_scattering_transmittance(air_mass, tau_a)
    b_r = rayleigh_extinction_forward_scattering_fraction(np.full_like(sza,
                                                                       1.66))
    b_a = aerosol_forward_scatterance_factor(np.full_like(sza, air_mass))
    f = aerosol_scattering_correction_factor(air_mass, tau_a)

    # Eq. 9 of [Gueymard_2008]_
    e_on = np.moveaxis(np.array(E0_N) * np.expand_dims(np.cos(np.radians(sza)), -1),
                       -1, 0)

    # Eq. 8 of [Gueymard_2008]_
    e_dpi = t_o * t_g * t_n * t_w * (b_r * (1 - t_ray) * t_a ** 0.25 +
                                   b_a * f * t_ray * (1 - t_as ** 0.25)) * e_on
    e_dpi = np.maximum(e_dpi, 0)
    return e_dpi


def backscattered_diffuse_irradiance(e_b,
                                     e_dpi,
                                     aod,
                                     turbidity_alpha,
                                     rhog=GROUND_ALBEDO):
    """
    backscattered contribution to diffuse irradiance

    Parameters
    ----------
    e_b : tuple of 2 floats or array_like(2, M)
        Beam spectral irradiance (W m²)
    e_dpi : tuple of 2 floats or array_like(2, M)
        Incident diffuse spectral irradiance (W m²)
    tubidity_beta : float or array_like
         Angström’s turbidity coefficient (i.e., AOD at 1 $\mu$m).
    rhog : tuple of 2 floats or array_like(2, M)
        Spectral ground albedo

    Returns
    -------
    e_ddi : tuple of 2 float or array_like(2, M)
        Backscattered diffuse spectral irradiance (W m²)

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """

    rhos = sky_albedo(turbidity_alpha, aod)
    e_ddi = rhog * rhos * (e_b + e_dpi) / (1 - rhog * rhos)
    e_ddi = np.maximum(e_ddi, 0)
    return e_ddi


def aerosol_forward_scatterance_factor(sza):
    """

    Parameters
    ----------
    sza : float or array_like(M)
        Solar zenith angle (Degrees)

    Returns
    -------
    f : float or array_like(M)
        aerosol_forward_scatterance_factor

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    return 1 - np.exp(-0.6931 - 1.8326 * np.cos(np.radians(sza)))


def aerosol_optical_depth(turbidity_beta,
                          effective_wavelength,
                          turbidity_alpha):
    """

    Parameters
    ----------
    turbidity_beta : float or array_like(M)
        Angström’s turbidity coefficient (i.e., AOD at 1 $\mu$m).
    effective_wavelength : tuple of 2 float or array_like(2, M)
        Efective wavelength for PAR and IR
    turbidity_alpha : tuple of 2 float or array_like(2, M)
        Spectral Angström’s wavelength exponent for PAR and IR

    Returns
    -------
    aot : tuple of 2 float or array_like(2, M)
        Spectral (PAR; IR) optical depth

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    effective_wavelength = np.moveaxis(np.asarray(effective_wavelength),
                                       0, -1)
    turbidity_alpha = np.asarray(turbidity_alpha)
    aot = turbidity_beta * np.moveaxis(effective_wavelength ** -turbidity_alpha, -1, 0)
    return aot

def aerosol_scattering_correction_factor(ma, tau_a):
    """
    correction factor introduced to compensate for multiple scattering effects
    Parameters
    ----------
    ma : float or array_like(M)
        Aerosol air mass
    tau_a : tuple of 2 floats or array_like(2, M)
        Spectral (PAR, IR) aerosol optical depth

    Returns
    -------
    tuple of 2 float or array_like (2, M)
        correction factor

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    g0 = (3.715 + 0.368 * ma + 0.036294 * ma ** 2) / \
        (1 + 0.0009391 * ma ** 2)
    g1 = (-0.164 - 0.72567 * ma + 0.20701 * ma ** 2) / \
        (1 + 0.001901 * ma ** 2)
    g2 = (-0.052288 + 0.31902 * ma + 0.17871 * ma ** 2) / \
        (1 + 0.0069592 * ma ** 2)
    h0 = (3.4352 + 0.65267 * ma + 0.00034328 * ma ** 2) / \
        (1 + 0.034388 * ma ** 1.5)
    h1 = (1.231 - 1.63853 * ma + 0.20667 * ma ** 2) / \
        (1 + 0.1451 * ma ** 1.5)
    h2 = (0.8889 - 0.55063 * ma + 0.50152 * ma ** 2) / \
        (1 + 0.14865 * ma ** 1.5)
    return np.array([(g0 + g1 * tau_a[0]) / (1 + g2 * tau_a[0]),
                     (h0 + h1 * tau_a[1]) / (1 + h2 * tau_a[1])])


def aerosol_transmittance(ma, tau_a):
    """Spectral aerosol transmittance

    Parameters
    ----------
    ma : float or array_like
        Relative air mass
    tau_a : float or array_like
        Spectral aerosol optical depth (AOD) along  vertical atmospheric column

    Returns
    -------
    float or array_like
        spectral aerosol transmittance

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    # Eq. 7a of [Gueymard_2008]_
    return np.exp(-ma * tau_a)


def aerosol_scattering_transmittance(ma, tau_a, albedo=ALBEDO):
    """Aerosol scattering transmittance

    Parameters
    ----------
    ma : float or array_like
        Relative air mass
    tau_a : tuple of 2 float or array_like(2, M)
        Spectral aerosol optical depth (AOD) along  vertical atmospheric column
    albedo : tuple of 2 float or array_like(2, M), optional
        Spectral Single-scattering albedo

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Aerosol scattering transmittance for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    # Eq. 7b of [Gueymard_2008]_
    return np.exp(-ma * np.moveaxis(np.array(albedo) *
                                    np.moveaxis(tau_a, 0, -1),
                                    -1, 0))


def effective_aerosol_wavelength(ma, alpha, beta):
    """Computes the aerosol efective wavelength for PAR and IR

    Parameters
    ----------
    ma : float or array
        Aerosol air mass
    alpha : tuple of 2 float or array_like(2, M)
        Spectral Angström’s wavelength exponent for PAR and IR
    beta : float or array_like(M)
        Angström’s turbidity coefficient (i.e., AOD at 1 $\mu$m).

    Returns
    -------
    effective_wavelength : tuple of 2 float or array_like(2, M)
        Efective wavelength for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    # This function has an error somewhere.
    # It returns negative values sometimes,
    # but wavelength should always be positive.
    ua = np.log(1 + ma * beta)
    d0 = 0.57664 - 0.024743 * alpha[0]
    d1 = (0.093942 - 0.2269 * alpha[0] + 0.12848 * alpha[0] ** 2) / \
         (1 + 0.6418 * alpha[0])
    d2 = (-0.093819 + 0.36668 * alpha[0] - 0.12775 * alpha[0] ** 2) / \
        (1 - 0.11651 * alpha[0])
    d3 = alpha[0] * (0.15232 - 0.087214 * alpha[0] + 0.012664 * alpha[0] ** 2) / \
        (1 - 0.90454 * alpha[0] + 0.26167 * alpha[0] ** 2)

    e0 = (1.183 - 0.022989 * alpha[1] + 0.020829 * alpha[1] ** 2) / \
         (1 + 0.11133 * alpha[1])
    e1 = (-0.50003 - 0.18329 * alpha[1] + 0.23835 * alpha[1] ** 2) / \
         (1 + 1.6756 * alpha[1])
    e2 = (-0.50001 + 1.1414 * alpha[1] + 0.0083589 * alpha[1] ** 2) / \
         (1 + 11.168 * alpha[1])
    e3 = (-0.70003 - 0.73587 * alpha[1] + 0.51509 * alpha[1] ** 2) / \
         (1 + 4.7665 * alpha[1])
    return [(d0 + d1 * ua + d2 * ua ** 2) / (1 + d3 * ua ** 2),
            (e0 + e1 * ua + e2 * ua ** 2) / (1 + e3 * ua)]


def gas_transmittance(m_rprime):
    """Gas transmittance

    Parameters
    ----------
    m_rprime : float or array_like(M)
        Relative air mass

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Gas transmittance for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    t_g = np.array([(1 + 0.95885 * m_rprime + 0.012871 * m_rprime ** 2) /
                    (1 + 0.96321 * m_rprime + 0.015455 * m_rprime ** 2),
                    (1 + 0.27284 * m_rprime - 0.00063699 * m_rprime ** 2) /
                    (1 + 0.30306 * m_rprime)])

    return t_g


def nitrogen_transmittance(m_w, u_n=0.0002):
    """Nitrogen transmittance

    Parameters
    ----------
    m_w : float or array_like(M)
        Water vapour air mass
    u_n : float or array_like(M), optional
        Nitrogen concentration (atm - cm)

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Nitrogen transmittance for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    g1 = (0.17499 + 41.654 * u_n - 2146.4 * u_n ** 2) / \
         (1 + 22295.0 * u_n ** 2)
    g2 = u_n * (-1.2134 + 59.324 * u_n) / (1 + 8847.8 * u_n ** 2)
    g3 = (0.17499 + 61.658 * u_n + 9196.4 * u_n ** 2) / \
         (1 + 74109.0 * u_n ** 2)
    t_n = np.array([np.minimum(1.0,
                               (1 + g1 * m_w + g2 * m_w ** 2) / (1 + g3 * m_w)),
                    np.ones_like(m_w)])
    return t_n


def ozone_transmittance(mo, uo=0.35):
    """Ozone transmittance

    Parameters
    ----------
    mo ; float or array_lik(M)
        Ozone air mass
    uo : float or array_like(M), optional
        Ozone concentration (atm-cm)

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Ozone transmittance for PAR and IR
    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """

    f1 = uo * (10.979 - 8.5421 * uo) / (1 + 2.0115 * uo + 40.189 * uo ** 2)
    f2 = uo * (-0.027589 - 0.005138 * uo) / \
        (1 - 2.4857 * uo + 13.942 * uo ** 2)
    f3 = uo * (10.995 - 5.5001 * uo) / (1 + 1.6784 * uo + 42.406 * uo ** 2)
    return np.array([(1 + f1 * mo + f2 * mo ** 2) / (1 + f3 * mo),
                     np.ones_like(mo)])


def rayleigh_extinction_forward_scattering_fraction(m_r):
    """

    Parameters
    ----------
    m_r : float or array_like(M)
        Rayleight air mass

    Returns
    -------
    typle of 2 float or array_like(2, M)
        Rayleigh extinction Forward Scattering Fraction

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    return np.array([0.5 * (0.89013 - 0.049558 * m_r + 0.000045721 * m_r ** 2),
                     np.full_like(m_r, 0.5)])


def rayleigh_transmittance(m_rprime):
    """Rayleigh transmittance

    Parameters
    ----------
    m_rprime : float or array_like(M)
        Relative air mass

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Rayleigh Trasmittance

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    t_r = np.array([(1 + 1.8169 * m_rprime + 0.033454 * m_rprime ** 2) /
                    (1 + 2.063 * m_rprime + 0.31978 * m_rprime ** 2),
                    (1 - 0.010394 * m_rprime) /
                    (1 - 0.00011042 * m_rprime ** 2)])

    return t_r


def sky_albedo(alpha, beta):
    """Single scattering sky albedo

    Parameters
    ----------
    alpha : tuple of 2 float or array_like(2, M)
        Spectral Angström’s wavelength exponent for PAR and IR
    beta : float or array_like(M)
        Angström’s turbidity coefficient (i.e., AOD at 1 $\mu$m).
    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Single scattering sky albedo for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    rhos_par = ((0.13363 + 0.00077358 * alpha[0] + beta *
                (0.37567+ 0.22946 * alpha[0]) / (1 - 0.10832 * alpha[0])) /
                (1 + beta * (0.84057 + 0.68683 * alpha[0]) /
                                       (1 - 0.08158 * alpha[0])))

    rhos_nir = ((0.010191 + 0.00085547 * alpha[1] + beta *
                (0.14618 + 0.062758 * alpha[1]) / (1 - 0.19402 * alpha[1])) /
                (1 + beta * (0.58101 + 0.17426 * alpha[1]) /
                 (1 - 0.17586 * alpha[1])))
    return np.array([rhos_par, rhos_nir])


def water_vapor_transmittance(mw, w):
    """Water vapour transmittance

    Parameters
    ----------
    mw : float or array_like(M)
        Water vapour air mass
    w : float or array_like(M)
        Total Column Water Vapour (cm)

    Returns
    -------
    tuple of 2 float or array_like(2, M)
        Water vapour transmittance for PAR and IR

    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    coeffs = water_vapor_transmittance_coefficients(w)
    t_par = (1 + coeffs[0] * mw) / (1 + coeffs[1] * mw)
    t_nir = (1 + coeffs[2] * mw + coeffs[3] * mw ** 2) / \
            (1 + coeffs[4] * mw + coeffs[5] * mw ** 2)
    return np.array([t_par, t_nir])


def water_vapor_transmittance_coefficients(w):
    """Gets the empirical coefficients needed for computing
    the water vapour transmittance

    Parameters
    ----------
    w : float or array_like
        Total Column Water Vapour (cm)
    Returns
    -------
    tuple of 6 float or array_like(5, M)
        Empirical coefficients for computing the water vapour transmittance
    References
    ----------
    .. [Gueymard_2008] Christian A. Gueymard, (2008)
        High-performance solar radiation model for cloudless-sky irradiance,
        illuminance, and photosynthetically active radiation –
        Validation with a benchmark dataset.
        Solar Energy, Volume 82, Issue 3, Pages 272-285,
        https://doi.org/10.1016/j.solener.2007.04.008.
    """
    h1 = w * (0.065445 + 0.00029901 * w) / (1 + 1.2728 * w)
    h2 = w * (0.065687 + 0.0013218 * w) / (1 + 1.2008 * w)
    c1 = w * (19.566 - 1.6506 * w + 1.0672 * w ** 2) / \
         (1 + 5.4248 * w + 1.6005 * w ** 2)
    c2 = w * (0.50158 - 0.14732 * w + 0.047584 * w ** 2) / \
         (1 + 1.1811 * w + 1.0699 * w ** 2)
    c3 = w * (21.286 - 0.39232 * w + 1.2692 * w ** 2) / \
         (1 + 4.8318 * w + 1.412 * w ** 2)
    c4 = w * (0.70992 - 0.23155 * w + 0.096514 * w ** 2) / \
         (1 + 0.44907 * w + 0.75425 * w ** 2)
    return [h1, h2, c1, c2, c3, c4]


def optical_mass_rayleigh(sza, pressure_millibars):
    """Raileigh Optical Mass

    Parameters
    ----------
    sza : float or array_like
        Sun zenith angle (deg)
    pressure_millibars : float or array_like
        Surface pressure (mn)

    Returns
    -------
    float or array_like
        air mass
    References
    ----------
    .. [Gueymard_2003] Christian A. Gueymard, (2003)
        Direct solar transmittance and irradiance predictions with
        broadband models. Part I: detailed theoretical performance assessment
        Solar Energy, Volume 74, Pages 355-379,
        https://doi.org/10.1016 / S0038-092X(03)00195-6
    """
    m_r = ((pressure_millibars / STANDARD_PRESSURE_MILLIBARS) /
            ((np.cos(np.radians(sza)) + 0.48353 * sza ** 0.095846) /
             (96.741 - sza) ** 1.754))
    return m_r


def optical_mass_ozone(sza):  # from Appendix B of [Gueymard, 2003]
    """

    Parameters
    ----------
    sza : float or array_like
        Sun zenith angle (deg)

    Returns
    -------
    float or array_like
        air mass
    References
    ----------
    .. [Gueymard_2003] Christian A. Gueymard, (2003)
        Direct solar transmittance and irradiance predictions with
        broadband models. Part I: detailed theoretical performance assessment
        Solar Energy, Volume 74, Pages 355-379,
        https://doi.org/10.1016 / S0038-092X(03)00195-6
    """
    m_o = 1 / ((np.cos(np.radians(sza)) + 1.0651 * sza ** 0.6379) /
                (101.8 - sza) ** 2.2694)

    return m_o


def optical_mass_water(sza):  # from Appendix B of [Gueymard, 2003]
    """

    Parameters
    ----------
    sza : float or array_like
        Sun zenith angle (deg)

    Returns
    -------
    float or array_like
        air mass

    References
    ----------
    .. [Gueymard_2003] Christian A. Gueymard, (2003)
        Direct solar transmittance and irradiance predictions with
        broadband models. Part I: detailed theoretical performance assessment
        Solar Energy, Volume 74, Pages 355-379,
        https://doi.org/10.1016 / S0038-092X(03)00195-6
    """
    m_w = 1. / ((np.cos(np.radians(sza)) + 0.10648 * sza ** 0.11423) /
                (93.781 - sza) ** 1.9203)

    return m_w


def optical_mass_aerosol(sza):  # from Appendix B of [Gueymard, 2003]
    """Aerosol optical mass

    Parameters
    ----------
    sza : float or array_like
        Sun zenith angle (deg)

    Returns
    -------
    float or array_like
        air mass

    References
    ----------
    .. [Gueymard_2003] Christian A. Gueymard, (2003)
        Direct solar transmittance and irradiance predictions with
        broadband models. Part I: detailed theoretical performance assessment
        Solar Energy, Volume 74, Pages 355-379,
        https://doi.org/10.1016 / S0038-092X(03)00195-6
    """
    m_a = 1 / ((np.cos(np.radians(sza)) + 0.16851 * sza ** 0.18198) /
                (95.318 - sza) ** 1.9542)
    return m_a


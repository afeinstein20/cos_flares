import os, sys
import numpy as np
from astropy import units
from astropy.io import fits
from astropy import constants
from lmfit.models import Model
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.signal import find_peaks
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from astropy.table import Table, Column
from lightkurve.lightcurve import LightCurve

import spectral_utils

__all__ = ['TransitsWithCOS']

class TransitsWithCOS(object):
    """
    A class to analyze transits as observed with Hubble/COS.
    """
    def __init__(self, wavelength, flux, flux_err, time,
                 orbit, time_unit=units.s):
        """
        Initializes the class.

        Parameters
        ----------
        wavelength : np.ndarray
           An array of wavelengths. Should be the same shape
           as the flux and flux_err arrays.
        flux : np.ndarray
           An array of flux spectra. Should be the same shape
           as the wavelength and flux_err arrays.
        flux_err : np.ndarray
           An array of errors on each flux point. Should be
           the same shape as the wavelength and flux arrays.
        time : np.array
           An array of times per each spectra. Should be the
           same length as the wavelength, flux, and flux_err
           arrays.
        orbit : np.array
           An array of which orbit each specta was observed
           in.
        time_unit : astropy.units.Unit
           The units of the time array. Default is seconds.
        """

        self.wavelength = wavelength+0.0
        self.flux = flux + 0.0
        self.flux_err = flux_err+0.0
        self.time = time * time_unit
        self.orbit = orbit
        self.line_table = None
        self.width_table = Table()
        self.error_table = Table()
        self.fuv130 = None
        self.continuum_mask = None

        self.flux_units =  units.erg / units.s / units.cm**2 / units.AA

    def to_velocity(self, wave, mid=None):
        """
        Converts wavelength to velocity space given some
        reference wavelength.

        Parameters
        ----------
        wave : np.array
           Wavelength array
        mid : float, optional
           The middle wavelength to center 0 km/s on.
           Default is none and will take the center of
           the spectrum as the 0 point.

        Returns
        -------
        rv_km_s : np.array
           The velocity in km/s.
        mid : float
           Where the velocity = 0.
        """
        rv_km_s, mid = spectral_utils.to_velocity(wave, mid)


    def measure_ew(self, ion=None, line=None, vmin=None,
                   vmax=None, orbit='all', binsize=3):
        """
        Measures the equivalent width of a given line. Either the
        ion name (from self.line_table) can be passed in or the
        center wavelength, vmin (minimum velocity of the line), and
        the vmax (maximum velocity of the line) can be passed in.

        Parameters
        ----------
        ion : str, optional
           The ion name in self.line_table to use
           for the analysis.
        line : float, optional
           The center of the line to measure.
        vmin : float, optional
           The lower velocity to incorporate into the
           line measurement.
        vmax : float, optional
           The upper velocity to incorporate into the
           line measurement.
        orbit : int, optional
           Applies a mask to a specified orbit. Default
           is all orbits.
        binsize : int, optional
           Accounts for binning in the error calculation.
           If the data is not binned, use binsize=1. Default
           is 3.

        Attributes
        ----------
        width_table : astropy.table.Table
           Adds a column of measured equivalent widths to the
           attribute width_table. Columns are replaced when
           lines are re-measured.
        error_table : astropy.table.Table
           Adds a column of equivalent width errors to the
           attribute error_table. Columns are replaced when
           lines are re-measured.

        """
        wt, et = spectral_utils.measure_ew(self.orbit, self.time, self.wavelength,
                                           self.flux, self.flux_err,
                                           self.line_table, ion=ion,
                                           line=line, vmin=vmin, vmax=vmax,
                                           orbit_num=orbit, binsize=binsize,
                                           width_table=self.width_table,
                                           error_table=self.error_table)
        self.width_table = wt
        self.error_table = et

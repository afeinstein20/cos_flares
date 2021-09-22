import os
import numpy as np
from astropy import units
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column
from lightkurve.lightcurve import LightCurve

__all__ = ['FlaresWithCOS']

class FlaresWithCOS(object):
    """
    A class to analyze flares as observed with Hubble/COS.
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


    def load_line_table(self, path, fname='line_table.txt',
                        format='csv'):
        """
        Loads in a table of lines. Table is organized
        as ion, central wavelength (AA), 
        the minimum velocity of the line (vmin; km/s), 
        and
        the maximum velocity of the line (vmax; km/s).

        Parameters
        ----------
        path : str
           Where the line table is stored.
        fname : str, optional
           The name of the line table. Default is
           `line_table.txt`.
        format : str, optional
           The format the table is stored in. Default
           is `csv`.

        Attributes
        ----------
        line_table : astropy.table.Table
        """
        self.line_table = Table.read(os.path.join(path, fname),
                                     format=format)


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
        if mid == None:
            mid = int(len(wave)/2)
        else:
            mid = np.where(wave>=mid)[0][0]
        lambda0 = wave[mid] + 0.0
        rv_m_s = ((wave - lambda0)/lambda0 * 3e8)*units.m/units.s
        rv_km_s = rv_m_s.to(units.km/units.s)
        return rv_km_s, mid


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
        if orbit != 'all':
            mask = np.where(self.orbit == orbit)[0]
        else:
            mask = np.arange(0,len(self.orbit),1,dtype=int)

        if ion is not None and self.line_table is not None:
            line = self.line_table[self.line_table['ion']==ion]['wave_c']+0.0
            vmin = self.line_table[self.line_table['ion']==ion]['vmin']+0.0
            vmax = self.line_table[self.line_table['ion']==ion]['vmax']+0.0
        elif ion is not None and self.line_table is None:
            return('No table found. Please load the line table first with self.load_line_table().')

        widths = np.zeros(len(self.time[mask]))
        errors = np.zeros(len(self.time[mask]))

        for i,x in enumerate(mask):
            v, _ = self.to_velocity(self.wavelength[x], mid=line)
            reg = np.where( (v.value >= vmin) & (v.value <= vmax) )[0]
            
            widths[i] = np.nansum(self.flux[x][reg])
            errors[i] = np.nansum(self.flux_err[x][reg])/binsize

        try:
            if ion is not None:
                self.width_table.add_column(Column(widths, ion))
            else:
                self.width_table.add_column(Column(widths, str(line)))
        except ValueError:
            if ion is not None:
                self.width_table.replace_column(Column(widths, ion))
            else:
                self.width_table.replace_column(Column(widths, str(line)))

        try:
            if ion is not None:
                self.error_table.add_column(Column(errors, ion))
            else:
                self.error_table.add_column(Column(errors, str(line)))
        except ValueError:
            if ion is not None:
                self.error_table.replace_column(Column(errors, ion))
            else:
                self.error_table.replace_column(Column(errors, str(line)))
        return


    def measure_FUV130(self):
        """
        Integrates the FUV130 flux, as defined in Parke Loyd et al. (2018).
        
        Attributes
        ----------
        fuv130 : np.ndarray
           Array of flux values that constitute the FUV130 band.
        """
        fuv130 = np.zeros(len(self.time.value))

        for i in range(len(self.wavelength)):
            mask = np.where( ((self.wavelength[i] >= 1173.65) & (self.wavelength[i] <= 1198.49)) |
                             ((self.wavelength[i] >= 1201.71) & (self.wavelength[i] <= 1212.16)) |
                             ((self.wavelength[i] >= 1219.18) & (self.wavelength[i] <= 1274.04)) |
                             ((self.wavelength[i] >= 1329.25) & (self.wavelength[i] <= 1354.49)) |
                             ((self.wavelength[i] >= 1356.71) & (self.wavelength[i] <= 1357.59)) |
                             ((self.wavelength[i] >= 1359.51) & (self.wavelength[i] <= 1428.90)) )[0]

            fuv130[i] = np.trapz(self.flux[i][mask], 
                                 x=self.wavelength[i][mask])

        self.fuv130 = fuv130 + 0.0


    def measure_flare_params(self, qmask, fmask, d, flux=None):
        """
        Measures the energy and equivalent duration of the flare.
        Flare parameter equations were taken from Loyd et al. 2018.

        Parameters
        ----------
        qmask : np.array
           Mask for the regions of the FUV130 that should be considered
           the quiescent flux.
        fmask : np.array
           Mask for the regions of the FUV130 that should be considered
           the flare.
        d : float
           The distance to the star with astropy.units.Unit quantity.
        flux : np.array, optional
           The flux array to measure the flare parameters. If None, this
           function uses the FUV130 flux.

        Returns
        -------
        energy : float
           The measured energy of the flare.
        ed : float
           The measured equivalent duration of the flare.
        """
        if flux is None:
            if self.fuv130 is None:
                self.measure_FUV130()
            flux = self.fuv130 + 0.0

        Fq = np.nanmedian(flux[qmask]/10**-13) * units.erg / units.s / units.cm**2
        Ff = (flux[fmask]/10**-13) * units.erg / units.s / units.cm**2

        eng = np.trapz(Ff-Fq, x=self.time[fmask]) * 4 * np.pi * d**2
        dur = np.trapz((Ff-Fq)/Fq, x=self.time[fmask])
        return eng, dur

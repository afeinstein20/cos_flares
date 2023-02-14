import os, sys
import numpy as np
from astropy import units
from astropy.time import Time
from astropy import constants
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, Column

import spectral_utils

__all__ = ['TransitsWithCOS']

class TransitsWithCOS(object):
    """
    A class to analyze transits as observed with Hubble/COS.
    """
    def __init__(self, wavelength, flux, flux_err, time,
                 orbit):
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
        self.time = Time(time, format='mjd')
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
        return rv_km_s, mid


    def load_line_table(self, path, fname='line_table.txt',
                        format='csv', comment='#'):
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
        comment : str, optional
           If comments are present in the table, provide the
           string identifier. Default is `#`.

        Attributes
        ----------
        line_table : astropy.table.Table
        """
        self.line_table = Table.read(os.path.join(path, fname),
                                     format=format,
                                     comment=comment)


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


    def to_transit_phase(self, Tcenter, Tingress, Tegress):
        """
        Creates an array of the transit phase from the predicted transit times
        and the time array in this class.

        Parameters
        ----------
        Tcenter : np.array
           Array of astropy.time.Time variables indicative of the predicted
           transit mid-point. Transit mid-points should be in time order.

        Attributes
        ----------
        visit : np.array
           Array of integers assigned to each visit in the data set.
        phase : np.array
           Array of transit phases.
        """
        inds = np.where(np.diff(self.time.value) > 10)[0]

        visit = np.zeros(len(self.time))
        inds = np.append([0], inds)
        inds = np.append(inds, [len(self.time)])

        for i in range(len(inds)-1):
            visit[inds[i]+1:inds[i+1]+1] = i

        phase = np.zeros(len(self.time))
        for j in range(len(np.unique(visit))):
            q1 = visit == j
            phase[q1]  = (self.time[q1] - Tcenter[j].mjd).value

        self.phase = phase
        self.visit = visit

        # Defines theh orbits in each visit
        orbits = np.array([])
        transit = np.zeros(len(orbits))

        for i in range(len(np.unique(visit))):
            q = (visit == i)
            o = np.zeros(len(self.time[q]))
            t = np.zeros(len(self.time[q]))

            seps = np.where(np.diff(self.time[q]) > 0.01)[0] + 1
            seps = np.append([0], seps)
            seps = np.append(seps, len(self.time[q]))

            t_i = Time(Tingress[i]).mjd
            t_e = Time(Tegress[i]).mjd

            time_chunk = self.time[q]

            for j in range(len(seps)-1):
                o[seps[j]:seps[j+1]] += j

                if (i==3 and j < 3) or (i==4 and j<2):
                    t[seps[j]:seps[j+1]] = 2

            tt = np.where((self.time[q].value >= t_i) &
                          (self.time[q].value <= t_e))[0]
            t[tt] = 1

            orbits = np.append(orbits, o)
            transit = np.append(transit, t)

        self.orbits = orbits
        self.in_transit = transit


    def bin_in_velocity(self, cenwave, velbins, mask=None):
        """
        Bins flux in velocity space.

        Parameters
        ----------
        cenwave : float
           The central wavelength to set velocity = 0.
        velbins : np.ndarray
           Array of bin edges to evaluate over. Bins should be given in units
           of velocity (km/s).
        mask : np.array, optional
           An optional mask to apply to the data before binning. Default is None.

        Returns
        -------
        binned_data : np.ndarray
           Array of binned data fluxes.
        binned_error : np.ndarray
           Array of binned data errors.
        """
        if mask is None:
            mask = np.zeros(len(self.wavelength))

        vel1, _ = self.to_velocity(self.wavelength[mask][10],
                                   mid=cenwave)
        f, e = np.zeros(len(velbins)), np.zeros(len(velbins))

        mean = np.nanmean(self.flux[mask], axis=0)
        err  = np.sqrt(np.nansum(self.flux_err[mask]**2.0, axis=0))/len(self.flux)

        for v in range(len(velbins)-1):
            inds = ((vel1.value >= velbins[v]) &
                    (vel1.value < velbins[v+1]))
            f[v] = np.nanmean(mean[inds])
            e[v] = np.nanmedian(err[inds])

        return f, e

    def combine_lines(self, cenwaves, velmin, velmax, nbins=20, visit=0):
        """
        Transforms wavelengths into velocity space and combines multiple lines.
        Does separate analysis for data in vs. out-of transit.

        Parameters
        ----------
        cenwaves : np.ndarray
           Central wavelength values. Should be a central value per each line
           you want to combine.
        velmin : float
           The minimum velocity value to sum over.
        velmax : float
           The maximum velocity value to sum over.
        nbins : int, optional
           The number of bins to create in velocity space. Default is 20.
        visit : int, optional
           Which visit data set to evaluate. Default is 0 (the first visit).
           Other option includes 100, where 100 indicates to combine all visits.

        Returns
        -------
        outputs : astropy.table.Table
        """
        tab = Table()
        velbins = np.linspace(velmin, velmax, nbins)
        tab['velocity'] = velbins

        if type(visit) == list or type(visit) == np.ndarray:
            pass
        else:
            visit = [visit]

        keys = ['it', 'oot']

        for v in range(len(visit)):

            q_oot = (self.visit == visit[v]) & (self.in_transit == 0)
            q_it  = (self.visit == visit[v]) & (self.in_transit == 1)

            for j in range(len(cenwaves)):

                for i, q in enumerate([q_oot, q_it]):

                    f, e = self.bin_in_velocity(cenwaves[j], velbins, mask=q)

                    tab['line{0:02d}_{1}_flux_visit{2}'.format(j, keys[i],
                                                               visit[v])] = f
                    tab['line{0:02d}_{1}_error_visit{2}'.format(j, keys[i],
                                                                visit[v])]= e

        return tab

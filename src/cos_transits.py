import os, sys
import numpy as np
from astropy import units
from astropy.time import Time
from astropy import constants
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, Column

import transit_utils
import spectral_utils

import warnings
warnings.filterwarnings("ignore")

__all__ = ['TransitsWithCOS']

class TransitsWithCOS(object):
    """
    A class to analyze transits as observed with Hubble/COS.
    """
    def __init__(self, wavelength, flux, flux_err, time, Tc, ingress, egress,
                 interpolate=False):
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
        time_unit : astropy.units.Unit
           The units of the time array. Default is seconds.
        Tc : np.array
           Array of predicted transit mid-point times. Should be in the
           same units as `time`.
        ingress : np.array
           Array of predicted transit ingress times. Should be in the same
           units as `time` and the same shape as `Tc`.
        egress : np.array
           Array of predicted transit egress times. Should be in the same units
           as `time` and the same shape as `Tc`.
        interpolate : bool, optional
           An option to interpolate all the visits onto the same wavelength
           grid. Default is False.
        """

        self.wavelength = wavelength+0.0
        self.flux = flux + 0.0
        self.flux_err = flux_err+0.0
        self.time = time
        self.line_table = None
        self.Tc = Tc
        self.ingress = ingress
        self.egress = egress

        self.flux_units =  units.erg / units.s / units.cm**2 / units.AA

        self.split_visits()
        self.transit_phases()
        self.split_orbits()

        if interpolate:
            self.interpolate_grid()
            self.flux = np.copy(self.interp_flux)
            self.wavelength = np.full(self.wavelength.shape,
                                      self.interp_wave_grid)

    def split_visits(self):
        """
        Splits `self.time` by visit number.

        Attributes
        ----------
        self.visits : np.array
           Array of numbers assigned to each visit.
        self.orbits : np.array
           Array of numbers assigned to each orbit in a given visit.
        """
        visits = np.zeros(len(self.time))
        inds = np.where(np.diff(self.time) > 1)[0]+1
        inds = np.sort(np.append([0, len(self.time)], inds))

        for i in range(len(inds)-1):
            visits[inds[i]:inds[i+1]] = i+1
        self.visits = visits
        return

    def transit_phases(self):
        """
        Calculates the phase of the transit during each HST visit.

        Attributes
        ----------
        self.phase : np.array
           Array of phase values. Should be of shape `(n_visits, len(time))`.
        """
        phases = np.zeros((len(self.Tc), len(self.time)))

        for i in range(len(self.Tc)):
            phases[i] = self.time - self.Tc[i]

        self.phases = phases
        return

    def split_orbits(self):
        """
        Splits each visit by orbit number.

        Attributes
        ----------
        self.orbits : np.array
           Array of numbers assigned to each orbit in a given visit.
        """
        orbits = np.zeros((len(np.unique(self.visits)), len(self.time)))

        for i in range(len(self.phases)):
            inds = np.where(np.diff(self.phases[i]) > 0.01)[0]+1
            inds = np.sort(np.append([0, len(self.time)], inds))
            for j in range(len(inds)-1):
                orbits[i][inds[j]:inds[j+1]] = j+1
        self.orbits = orbits
        return

    def interpolate_grid(self):
        """
        Interpolates data onto the same wavelength grid.

        Attributes
        ----------
        interp_wave_grid : np.ndarray
           Wavelength grid interpolated onto.
        interp_flux : np.ndarray
           Flux interpolated onto `interp_wave_grid`.
        """
        delta = np.diff(self.wavelength[0])
        wave_soln = np.copy(self.wavelength[0])

        interp_flux = np.zeros(self.flux.shape)

        for i in range(len(self.wavelength)):

            for j in range(len(wave_soln)):
                try:
                    inds = np.where((self.wavelength[i] >
                                     wave_soln[j]-delta[0]/2.0) &
                                    (self.wavelength[i] <
                                     wave_soln[j]+delta[0]/2.0))[0][0]

                    interp_flux[i][j] = self.flux[i][inds]
                except IndexError: # for wavelengths that may not exist?
                    interp_flux[i][j] = np.nan

        self.interp_flux = interp_flux
        self.interp_wave_grid = wave_soln
        return

    def define_in_vs_out(self, inbounds, outbounds):
        """
        Defines the indices for each visit that should be considered in-transit
        and out-of-transit. This is computed in phase-space.

        Parameters
        ----------
        inbounds : np.array
           Start and stop values (in units of phase) that should be considered
           in-transit. Should be shape `(2,)`.
        outbounds : np.array
           Start and stop values (in units of phase) that should be considered
           out-of-transit. Should be shape `(2,)`.

        Attributes
        ----------
        inbounds_idx : np.array
        outbounds_idx : np.array
        """
        inbounds_idx = []
        outbounds_idx = []

        for i in range(len(self.Tc)):
            itrans = np.where( (self.phases[i] >= inbounds[0]) &
                               (self.phases[i] <= inbounds[1]) )[0]
            otrans = np.where( (self.phases[i] >= outbounds[0]) &
                               (self.phases[i] <= outbounds[1]) )[0]

            inbounds_idx.append(itrans)
            outbounds_idx.append(otrans)

        self.inbounds = inbounds
        self.outbounds = outbounds
        self.inbounds_idx = inbounds_idx
        self.outbounds_idx = outbounds_idx
        return

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
        wt, et = spectral_utils.measure_ew(self.time, self.wavelength,
                                           self.flux, self.flux_err,
                                           self.line_table, ion=ion,
                                           line=line, vmin=vmin, vmax=vmax,
                                           orbit_num=orbit, binsize=binsize,
                                           width_table=self.width_table,
                                           error_table=self.error_table)
        self.width_table = wt
        self.error_table = et

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

    def combine_lines(self, cenwaves, velmin, velmax, nbins=20, visit=0,
                      flaremask=None):
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

            if flaremask is None:
                q_oot = (self.visit == visit[v]) & (self.in_transit == 0)
                q_it  = (self.visit == visit[v]) & (self.in_transit == 1)
            else:
                q_oot = ((self.visit == visit[v]) & (self.in_transit == 0) &
                         (flaremask == 0))
                q_it  = ((self.visit == visit[v]) & (self.in_transit == 1) &
                         (flaremask == 0))

            for j in range(len(cenwaves)):

                for i, q in enumerate([q_oot, q_it]):

                    f, e = self.bin_in_velocity(cenwaves[j], velbins, mask=q)

                    tab['line{0:02d}_{1}_flux_visit{2}'.format(j, keys[i],
                                                               visit[v])] = f
                    tab['line{0:02d}_{1}_error_visit{2}'.format(j, keys[i],
                                                                visit[v])]= e
        return tab

    def model_transit(self, params, type='brightening'):
        """
        Used to model both limb darkening and limb brightening transit light
        curves. The light curves are fit to the data.

        Parameters
        ----------
        params : dict
           Dictionary of orbital parameters. Should include the following keys:
           't0' (time of mid-transit), 'per' (period, units of days), 'rp'
           (radius of planet, units of Rstar), 'arstar' (semi-major axis, units
           of Rstar), 'inc' (inclination), 'ecc' (eccentricity), 'u' (limb
           darkening coefficients, should be a list), 'b' (impact parameter)
        type : str
           Which model to compute. Default is 'brightening'. Other options are:
           'darkening'. Limb darkening model is computed using `batman`.
        """
        if type == 'brightening':
            blc = transit_utils.limb_brightening(self.phase, params['rp'])
            return zarr, blc
        elif type == 'darkening':
            dlc = transit_utils.limb_darkening(self.time, params)
            return dlc
        else:
            return('Transit model not implemented. Please enter either \
                   "brightening" or "darkening".')

    def create_lightcurve(self, ion, visits='all'):
        """
        Creates an averaged light curve for each ion, as a function of visit and
        combined across visits.

        Parameters
        ----------
        ion : str
           Name of the ion to create the light curve for.
        visits : list, optional
           List of visits to create the light curves for. Default is 'all'.

        Returns
        -------
        phase : np.array
        lc : np.array
        lc_err : np.array
        avg_phase : np.array
        avg_phase_err : np.array
        avg_lc : np.array
        avg_lc_err : np.array
        visits : np.array
        """
        wave_c = self.line_table[self.line_table['ion']==ion]['wave_c']
        vmin   = self.line_table[self.line_table['ion']==ion]['vmin']
        vmax   = self.line_table[self.line_table['ion']==ion]['vmax']

        p, l, le = [], [], []
        comb_p, comb_l, comb_e = np.array([]), np.array([]), np.array([])
        comb_v = np.array([])

        if visits == 'all':
            visits = np.unique(self.visits)

        # Normalizes data per visit
        for i in visits:

                q = (self.visits == i)

                v,_ = self.to_velocity(self.wavelength[q][0], mid=wave_c)

                qq = ((v.value >= vmin) & (v.value <= vmax))

                phase = self.time[q]-self.Tc[int(i-1)]

                #### CAVEAT ####
                # this only works when there is out-of-transit baseline on one side of the transit ONLY
                norm = phase > self.outbounds[0]

                lc = np.nanmean(self.flux[q][:,qq], axis=1)
                er = np.sqrt(np.nansum(self.flux_err[q][:,qq]**2, axis=1))/len(self.flux_err[q][:,qq])

                comb_p = np.append(comb_p, phase)
                comb_l = np.append(comb_l, lc/np.nanmean(lc[norm]))
                comb_e = np.append(comb_e, er/np.nanmean(lc[norm]))
                comb_v = np.append(comb_v, np.full(len(lc), i))

        # Creates an averaged light curve as a function of phase
        arg = np.argsort(comb_p)
        p, l, le = comb_p[arg], comb_l[arg], comb_e[arg]

        diff = np.diff(p)
        brk = np.where(diff>=0.025)[0] + 1
        brk = np.sort(np.append(brk, [0, len(p)-1]))

        avg_p, avg_p_err  = np.array([]), []
        avg_lc, avg_err   = np.array([]), np.array([])

        for i in range(len(brk)-1):
            avg_p     = np.append(avg_p,
                                  (p[brk[i]] + p[brk[i+1]-1])/2.0)
            avg_p_err.append([p[brk[i]], p[brk[i+1]-1]])

            avg_lc    = np.append(avg_lc,
                                  np.nanmean(l[brk[i]:brk[i+1]]))
            avg_err   = np.append(avg_err,
                                  np.nanmean(le[brk[i]:brk[i+1]]))

        return comb_p, comb_l, comb_e, avg_p, avg_p_err, avg_lc, avg_err, comb_v


    def plot_lc(self, ion=None, x=None, lc=None, visits='all', colors=None,
                pldict=None, ax=None):
        """
        Plots any given light curve. Can either take in a pre-created light
        curve (pass in `x` and `lc`) or can make the light curve for a given
        ion on the spot (pass in `ion`).

        Parameters
        ----------
        ion : str, optional
           The name of the ion to make the light curve for.
        x : np.array, optional
           Array of time or phase values.
        lc : np.array, optional
           Array of the light curve.
        colors : list, optional
           List of colors for each visit. Default is None. Should be of len(visits).
        pltdict : dictionary, optional
           Dictionary of matplotlib keywords for plotting.
        ax : matplotlib.axes.Ax, optional
           The axes to plot the light curve on. Default is None (i.e. creates
           a new figure).
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,4))

        if ion is not None:
            outputs = self.create_lightcurve(ion=ion, visits=visits)
            for i in range(len(visits)):
                q = outputs[-1] == visits[i]
                ax.plot(outputs[0][q], outputs[1][q], color=colors[i],
                        label='Visit {}'.format(visits[i]),
                        **pldict)
            ax.errorbar(outputs[3], outputs[5], yerr=outputs[6],
                        marker='o', color='k', linestyle='')
            for i in range(len(outputs[3])):
                ax.hlines(outputs[5][i], outputs[4][i][0],
                          outputs[4][i][1], color='k', zorder=3)

        elif (x is not None) and (lc is not None):
            ax.plot(x, lc, '.', **pldict)

        else:
            return('must pass in an argument of `ion` or `x` and `lc`.')

        return

    def grab_line_profile(self, ion, visit):
        """
        Extracts the line profile from the in-transit vs. the out-of-transit
        observations.

        Parameters
        ----------
        ion : str
        visit : int

        Returns
        -------
        in_transit : np.array
        out_of_transit : np.array
        velocity : np.array
        """
        wave_c = self.line_table[self.line_table['ion']==ion]['wave_c']

        phase = self.time - self.Tc[int(visit-1)]

        #### CAVEAT ####
        # this only works when there is out-of-transit baseline on one side of the transit ONLY
        oot_q  = np.where((self.visits == visit) & (phase > self.outbounds[0]))[0]
        tns_q  = np.where((self.visits == visit) & (phase < self.inbounds[1]))[0]

        #oot_q = phase > self.outbounds[0]
        #tns_q = phase < self.inbounds[1]

        oot = np.log10(np.nanmean(self.flux[oot_q], axis=0))
        tns = np.log10(np.nanmean(self.flux[tns_q], axis=0))

        v,_ = self.to_velocity(self.wavelength[0], mid=wave_c)
        v = v.value

        return tns, oot, v, tns_q, oot_q

    def plot_profile(self, x, y1, y2, ax, color, xmin, xmax):
        """Plots the line profiles."""
        ax.plot(x, y1, 'k')
        ax.plot(x, y2, color=color)
        ax.set_ylim(np.nanmin(y1[(x>xmin) & (x<xmax)]),
                    np.nanmax(y1[(x>xmin) & (x<xmax)])+0.01)
        return

    def plot_diff_profile(self, x, y1, y2, ax, xmin, xmax):
        """Plots the difference between line profiles."""
        d = y2 - y1
        ax.plot(x, d, 'k')
        ax.set_ylim(np.nanmin(d[(x>xmin) & (x<xmax)]),
                    np.nanmax(d[(x>xmin) & (x<xmax)])+0.01)
        ax.axhline(0, color='k', alpha=0.6, linestyle='--')
        return

    def plot_div_profile(self, x, y1, y2, ax, xmin, xmax):
        """Plots the in-transit divided by the out-of-tranasit profile."""
        d = y2 / y1
        ax.plot(x, d, 'k')
        ax.set_ylim(np.nanmin(d[(x>xmin) & (x<xmax)]),
                    np.nanmax(d[(x>xmin) & (x<xmax)])+0.01)
        ax.axhline(1, color='k', alpha=0.6, linestyle='--')
        return

    def plot_line_profiles(self, ion, visits, colors, axes=None):
        """
        Plots the line profile of a specific line. This function separates out the
        in-transit and out-of-transit data to search for differences in the line
        profiles.

        Parameters
        ----------
        ion : str
        colors : list
           What color to plot the in-transit data in. Default for out-of-transit data
           is black.
        visit : np.array, list
        axes : matplotlib.axes, optional
           Group of three subplots to plot the results on. Default is 'None'.
        """
        if axes is None:
            if len(visits) > 1:
                nrows = len(visits) + 1
            else:
                nrows = 1
            fig, axes = plt.subplots(ncols=3, nrows=nrows,
                                     figsize=(14,4*len(visits)), sharex=True)
            axes = axes.reshape(-1)
            axes[0].set_title('Line Profile')
            axes[1].set_title('In/Out Transit')
            axes[2].set_title('In-Out Transit')

        x = 0
        in_transit, out_transit = np.array([], dtype=int), np.array([], dtype=int)

        for i in range(len(visits)):
            # Grabs and separates the line profiles
            tns, oot, v, tns_idx, oot_idx = self.grab_line_profile(ion, visits[i])

            # Keeps track of in- vs. out-of-transit indices for an averaged profile
            in_transit = np.append(in_transit, tns_idx)
            out_transit = np.append(out_transit, oot_idx)

            vmin = self.line_table[self.line_table['ion']==ion]['vmin']
            vmax = self.line_table[self.line_table['ion']==ion]['vmax']

            # Plots the line profile
            self.plot_profile(v, oot, tns, axes[x], colors[i], vmin, vmax)

            # Plots the in-transit/out-of-transit profile
            self.plot_div_profile(v, oot, tns, axes[x+1], vmin, vmax)

            # Plots the in-transit - out-of-transit profile
            self.plot_diff_profile(v, oot, tns, axes[x+2], vmin, vmax)

            x += 3

        if len(visits) > 1:
            # Combines profiles from all visits
            oot_avg = np.log10(np.nanmean(self.flux[out_transit], axis=0))
            tns_avg = np.log10(np.nanmean(self.flux[in_transit] , axis=0))

            self.plot_profile(v, oot_avg, tns_avg, axes[-3], colors[-1], vmin, vmax)
            self.plot_div_profile(v, oot_avg, tns_avg, axes[x+1], vmin, vmax)
            self.plot_diff_profile(v, oot_avg, tns_avg, axes[x+2], vmin, vmax)

        for idx in [-1, -2, -3]:
            axes[idx].set_xlabel('Velocity [km s$^{-1}$]')
        for idx in np.arange(0, nrows*3, 3, dtype=int):
            axes[idx].set_ylabel('Flux Density')

        axes[0].set_xlim(vmin, vmax)
        return

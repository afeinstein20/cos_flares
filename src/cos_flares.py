import os
import numpy as np
from astropy import units
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.optimize import minimize
from scipy.interpolate import interp1d
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
                self.width_table.add_column(Column(widths, str(np.round(line,3))))
        except ValueError:
            if ion is not None:
                self.width_table.replace_column(ion, widths)
            else:
                self.width_table.replace_column(str(np.round(line,3)), widths)

        try:
            if ion is not None:
                self.error_table.add_column(Column(errors, ion))
            else:
                self.error_table.add_column(Column(errors, str(np.round(line,3))))
        except ValueError:
            if ion is not None:
                self.error_table.replace_column(ion, errors)
            else:
                self.error_table.replace_column(str(np.round(line,3)), errors)
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


    def flare_model(self, amp, t0, rise, decay):
        """
        Models the flare with an array of times (for double-peaked
        events). Uses the standard model of a Gaussian rise and 
        exponential decay.
        
        Parameters
        ----------
        amp : np.array
           Array of amplitudes to fit.
        t0 : np.array
           Array of peak times of flare to fit.
        rise : np.array
           Array of Gaussian rise factors to fit to the flare.
        decay : np.array
           Array of exponential decay factors to fit to the flare.
        
        Returns
        -------
        model : np.array
           Flare model.
        """
        gr = np.zeros(len(self.time))
        ed = np.zeros(len(self.time))
        flux = np.zeros(len(self.time))
        
        for i in range(len(amp)):
            g = amp[i] * np.exp( -(self.time[self.time<t0[i]] - t0[i])**2.0 / (2.0*rise[i]**2.0) ) 
            g += flux[self.time<t0[i]]
            gr[self.time<t0[i]] += g
            e = amp[i] * np.exp( -(self.time[self.time>=t0[i]] - t0[i]) / decay[i] ) + flux[self.time>=t0[i]]
            ed[self.time>=t0[i]] += e

        return gr + ed

        
    def load_lsf_model(self, fname):
        """
        Convolves the line spread function with a gaussian to
        fit line profiles.

        Parameters
        ----------
        fname : str
           The path + name of the line spread function file.
        
        Attributes
        ----------
        lsf_table : astropy.table.Table
           A table of convolved models. The models are created
           in steps of 5 Angstroms.
        """

        lsf = Table.read(fname, format='ascii')
        
        lsf_table = Table()

        for key in lsf.colnames:
            name = lsf[key][0]
            data = lsf[key][1:-1] # removes bad rows from the LSF
            lsf_table.add_column(Column(data, name))

        self.lsf_table = lsf_table


    def model_line_shape(self, ion, mask, shape='gaussian',
                         ext=100):
        """
        Takes an ion from the line list and fits a convolved Gaussian
        with the line spread function. Line profiles are fit by conducting
        a chi-squared fit.

        Parameters
        ----------
        ion : str
           The ion in the line list to fit a line to.
        mask : np.ndarray
           A mask for the out-of-flare observations to create a 
           template from.
        shape : str, optional
           The profile shape to convolve with the line spread
           function. Default is `gaussian`.
        ext : float, optional
           Addition to the vmin and vmax of a given ion to ensure the
           line profile can be well fit to the data. Default = 100 [km/s].
        """
        def gaussian(x, mu, std, f):
            """ A gaussian profile model.
            """
            exp = -0.5 * (x-mu)**2 / std**2
            denom = std * np.sqrt(np.pi * 2.0)
            g = f / denom * np.exp(exp)
            return g 

        def conv_model(lsf, gauss):
            """ Convolves LSF with line profile.
            """
            conv = np.convolve(lsf, gauss, mode='same')
            return conv

        def chiSquare(var, x, y, yerr, lsf):
            """ ChiSquare fit of the convolved line
                profile with the data.
            """
            mu, std, f, cf, af = var
            gmodel = gaussian(x, mu, std, f)
            conv = conv_model(lsf*cf, gmodel)
            conv /= af
            return np.nansum( (y-conv)**2.0 / yerr**2.0 )

        
        wc   = self.line_table[self.line_table['ion']==ion]['wave_c'][0]
        vmin = self.line_table[self.line_table['ion']==ion]['vmin'][0]
        vmax = self.line_table[self.line_table['ion']==ion]['vmax'][0]

        # finds the line spread profile closest to the ion in question #
        argmin = np.argmin(np.abs([float(i) for i in self.lsf_table.colnames] - wc))
        lsf = self.lsf_table[self.lsf_table.colnames[argmin]].data
        lsf /= np.nanmax(lsf)

        velocity, _ = self.to_velocity(self.wavelength[0], wc)
        velocity    = velocity.value + 0.0
        reg = np.where( (velocity >= vmin-ext) & (velocity <= vmax+ext) )[0]

        # interpolate to the same length as the line spread profile #
        wave = np.linspace(self.wavelength[0][reg][0], 
                           self.wavelength[0][reg][-1], 
                           len(lsf))

        f = interp1d(self.wavelength[0][reg], np.nanmedian(self.flux[mask,:][:,reg], axis=0))
        
        ferr = interp1d(self.wavelength[0][reg], np.sqrt( np.nansum( self.flux_err[mask,:][:,reg]**2, axis=0) ))

        f = f(wave)/1e-14
        ferr = ferr(wave)/1e-14/len(reg)

        # initial guess for the scipy.optimize.minimize function
        x0 = [wc, 0.1, 0.3, 10, 100]
        x = minimize(chiSquare, x0=x0,
                     bounds=((wave.min(), wave.max()),
                             (0.1,100),
                             (1,20),
                             (1,100),
                             (1,300)),
                     args=(wave,
                           f,
                           ferr,
                           lsf),
                     method='L-BFGS-B')

        c = np.convolve(gaussian(wave, x.x[0], x.x[1], x.x[2]), lsf*x.x[3], 'same')/x.x[4]
        print(x.x)
        return wave, f, ferr, c